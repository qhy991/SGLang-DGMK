# GLM-5.2 算子集成 SGLang 与 CUDA Graph Replay 收益消失分析

## 1. 结论先行

此次最典型的现象是：

- `fused_qkv_a_proj` 在 eager 下，GEMM leaf、FP8 linear apply 和完整局部
  region 分别测到约 `1.174x`、`1.091x` 和 `1.115x`；
- 将相同 stock/candidate 计算序列分别 capture 成 CUDA Graph 后，replay
  只剩约 `1.015x`、`1.005x` 和 `1.010x`。

主要原因不是候选在 Graph 中失效，也不是 Graph replay 自动切回 stock，而是
eager 计时包含了 CPU/Python/custom-op 到 GPU 的提交间隙。候选的 direct
DeepGEMM 调用比 stock 包装路径更快地提交到 GPU，因此 eager CUDA Event
把一部分 host submission gap 计成了候选收益。CUDA Graph capture 将这些
host 侧工作执行一次并记录为固定节点，replay 时 stock 和 candidate 都通过
相同的 graph launch 路径提交，原有 host 路径差异被大幅归一化，最终暴露出
设备 kernel 本身只有约 `0.5%--1.5%` 的差异。

因此，正确表述应当是：

> CUDA Graph 没有让一个原本明显更快的 GPU kernel 变慢；它消除了每次
> eager 调用中的 Python、wrapper 和 CPU→GPU 提交差异。消除这些差异后，
> 两个 GPU kernel 的设备关键路径接近，所以大部分 eager “加速”消失。

需要同时注意：Graph replay 不是所有场景下都比 eager “更真实”。应该以
该 bucket 在生产 SGLang 中实际使用的 execution mode 为准。对于 Graph
decode，Graph replay 是必要性能门槛；对于实际仍为 eager 的大 M prefill，
eager containing-region 结果仍可能有生产意义，Graph 性能只是诊断项。

## 2. 集成产物与范围

公开集成分支：

- branch：`agent/infini-kernel-glm52-e2e`
- commit：`2786889e7`
- repository：`git@github.com:qhy991/SGLang-DGMK.git`
- GitHub：
  `https://github.com/qhy991/SGLang-DGMK/tree/agent/infini-kernel-glm52-e2e`
- 原始集成说明：
  [`infini_kernel_fixed_nk_e2e.md`](infini_kernel_fixed_nk_e2e.md)

该分支通过现有 `glm52_opt` dispatcher 注册三个默认关闭的固定 N/K
DeepGEMM 候选：

| Registry op | phase | 精确本地 shape | profiler identity |
|---|---|---|---|
| `index_q_upproj` | decode | M=16/32, N=4096, K=2048 | `infini_kernel_glm52_index_q_upproj_decode_nk` |
| `o_proj` | decode | M=16/32, N=6144, K=16384 | `infini_kernel_glm52_attn_o_decode_nk` |
| `fused_qkv_a_proj` | prefill | M=4096, N=2624, K=6144 | `infini_kernel_glm52_fused_qkv_a_prefill_nk` |

候选和 stock 共享以下生产条件：

- 相同 `Fp8LinearMethod` 调用位置；
- 相同 BF16 输入与动态 activation quantizer；
- 相同 FP8 E4M3 activation/weight；
- 相同 packed int32 UE8M0 activation/weight scales；
- 相同 BF16 输出；
- 相同 stream、scheduler 和非目标 GPU 工作。

候选 GEMM 的核心变化是：

```python
deep_gemm.fp8_gemm_nt(
    (x_fp8, x_scale),
    (w_fp8, w_scale),
    out,
    compiled_dims="nk",
)
```

## 3. SGLang 中的注册和替换链路

整体链路如下：

```text
GLM-5.2 Linear 层 prefix
  -> prefix_to_op_name()
  -> op_context(op)
  -> Fp8LinearMethod / production dynamic FP8 quantizer
  -> try_dispatch_fp8_gemm()
  -> lookup(op, phase, M)
  -> ForwardMode + shape + dtype + packed-scale ABI 检查
  -> 命中 fixed_nk candidate
       -> deep_gemm.fp8_gemm_nt(..., compiled_dims="nk")
  -> 未命中
       -> stock w8a8_block_fp8_matmul_deepgemm()
```

### 3.1 层名映射

`python/sglang/srt/layers/glm52_opt/context.py` 将模型中的真实层 prefix
映射成 registry 使用的稳定名称，例如：

- `fused_qkv_a_proj_with_mqa` -> `fused_qkv_a_proj`
- `wq_b` -> `index_q_upproj`
- `o_proj` -> `o_proj`

这里使用最后一个完整 prefix component 精确匹配，而不是 substring
匹配，避免把 shape 或 TP 拓扑不同的层错误路由到同一个 kernel。

### 3.2 Registry

`python/sglang/srt/layers/glm52_opt/registry.py` 中的 `KernelSpec` 记录：

- op；
- phase；
- implementation；
- profiler name；
- 允许的 M bucket；
- 精确 N/K。

`e2e_candidates` profile 只从独立的 E2E 表选择候选，避免历史 archive
kernel 或同名 MoE 路径被意外启用。`fused_qkv_a_proj` 和
`index_q_upproj` 还是 explicit-only：只有明确写入
`SGLANG_GLM52_OPT_OPS` 才会启用。

### 3.3 生产 FP8 入口

`python/sglang/srt/layers/quantization/fp8_utils.py` 先执行生产
per-token/group-128 FP8 量化，然后调用 `try_dispatch_fp8_gemm()`。

候选返回 tensor 时直接结束；候选没有被选择时，继续调用 stock：

```python
output = w8a8_block_fp8_matmul_deepgemm(...)
```

这使 stock/candidate A/B 共享相同的生产 quantizer，而不是拿一个已经量化
好的 synthetic tensor 绕开 SGLang 接口。

### 3.4 Fail-closed ABI

`dispatch.py` 在选择 fixed-N/K candidate 前检查：

- 精确 registry op 和 M bucket；
- decode 必须是 `ForwardMode.DECODE`；
- prefill 必须是 `ForwardMode.EXTEND`；
- 精确 N/K 和 128x128 block recipe；
- activation/weight 必须是 contiguous CUDA FP8 E4M3；
- activation/weight scale 必须是 CUDA int32 packed UE8M0；
- 精确 scale shape 和 column-major stride；
- tensor 必须位于同一 device；
- 输出必须是 BF16；
- 不允许 bias。

任何不支持的 mode、shape、ABI 或 dtype 都在候选选择前返回 stock。候选
一旦被选择，DeepGEMM 异常直接传播，不允许再 fallback 到 stock 后将结果
记录成一次候选成功。

### 3.5 Nsys 中的 `infini_kernel` 名称

`infini_kernel...` 是默认关闭的 NVTX range，而不是修改后的 CUDA device
symbol。真实 device symbol 仍为：

```text
deep_gemm::sm100_fp8_fp4_gemm...
```

Nsys 中会看到类似：

```text
infini_kernel_glm52_fused_qkv_a_prefill_nk[M=4096,N=2624,K=6144]
  / void deep_gemm::sm100_fp8_fp4_gemm...
```

不增加 marker kernel，是为了避免额外 launch 扭曲 5--25 微秒的 decode
kernel。正式性能 A/B 必须关闭 NVTX：

```bash
export SGLANG_GLM52_INFINI_KERNEL_NVTX=0
```

## 4. 公平测试采用的边界

测试没有只测一个脱离生产接口的 GEMM，而是逐层扩大边界：

```text
packed FP8 GEMM leaf
  -> BF16 quantize + packed GEMM apply
  -> projection + split + two RMSNorm containing region
  -> checkpoint-backed SGLang serving E2E
```

严格 Task02 测试要求：

- stock/candidate 使用相同 tensor ABI；
- 两条 JIT 路径都先完成 warmup；
- graph capture 和 JIT compilation 不进入 timed replay；
- 三个独立 series；
- 每个 series 50 对 AB/BA 交替样本；
- pooled、order-balanced、AB median、BA median 四个 estimator 都必须
  至少为 `1.03x`；
- timing 前后以及 fresh input 都做 bitwise stock comparison；
- stock/candidate graph 独立 capture；
- graph replay 检查输入 mutation、输出 poison overwrite、稳定 pointer；
- 检查 graph kernel node identity，禁止 adapter/copy/fallback 节点；
- 保留 HIT、Nsys、PTX、SASS 和 NCU 证据。

严格结果位于本机 campaign worktree：

```text
/home/qinhaiyan/glm52-v2-goal-runs/worktrees/
  02-attn-fused-qkv-a-prefill/kernel-harness/serving_native/evidence/
  glm52_prod_02_attn_fused_qkv_a_prefill/FINAL_RESULT.md
```

## 5. 为什么 CUDA Event 会包含 host submission gap

常见误区是认为使用 CUDA Event 就一定只测到 device kernel。对于下列计时
方式，这个结论不成立：

```python
start.record()
candidate_or_stock_api()
end.record()
end.synchronize()
latency = start.elapsed_time(end)
```

eager 时间线可能是：

```text
CPU:
  start.record()
  Python dispatch
  custom-op / TVM-FFI wrapper
  shape checks / allocation / JIT-cache or config lookup
  cudaLaunchKernel
  end.record()

GPU:
  execute start event
  -------- GPU idle, waiting for CPU submission --------
  execute GEMM kernel
  execute end event
```

如果 GPU 已经执行完 start event，而 CPU 仍在准备和提交 kernel，那么 GPU
等待期间仍处于两个 Event 之间。因此：

```text
T_eager
  = stream 可见的 CPU->GPU 提交空洞
  + GPU kernel 执行时间
  + 多 kernel 之间的提交空洞
```

它是一次 eager API 的 stream-visible latency，但不能直接等同于 device
kernel duration。

## 6. CUDA Graph Replay 改变了什么

Graph capture 时，Python、dispatcher、custom-op、DeepGEMM kernel 选择和
参数准备执行一次，最终记录为固定节点：

```text
Graph:
  quantizer node
    -> DeepGEMM node
    -> q RMSNorm node
    -> k RMSNorm node
```

replay 时不会重新运行 Python registry、ABI 检查、custom-op wrapper 或
DeepGEMM 的 Python 选择逻辑，而是：

```text
cudaGraphLaunch()
  -> replay already-resolved CUDA nodes
```

因此 Graph replay 会大幅归一化：

- Python dispatcher 成本；
- custom-op、TVM-FFI 或 direct helper 的包装差异；
- DeepGEMM kernel-key/config 查找；
- 参数封装与部分 allocator bookkeeping；
- 多 kernel 之间的 CPU 提交空洞；
- stock/candidate 不同 host launch path 的差异。

Graph replay 本身仍有 graph launch 成本，但 stock 和 candidate 使用相同
入口，因此它更适合比较已经 capture 好的 GPU node 序列。

## 7. `fused_qkv_a_proj` 的量化证据

下面的绝对时间是三个 series median 的平均值，speedup 是正式 pooled
estimator，因此显示时间相除与 pooled ratio 会有轻微差异。

| 测试边界 | eager stock -> candidate | eager speedup | graph stock -> candidate | graph speedup |
|---|---:|---:|---:|---:|
| packed GEMM leaf | 85.931 -> 73.168 us | 1.173894x | 48.901 -> 48.267 us | 1.015262x |
| quantize + GEMM apply | 135.040 -> 123.600 us | 1.091133x | 65.008 -> 64.661 us | 1.004949x |
| projection + split + 2 RMSNorm | 181.088 -> 161.376 us | 1.114767x | 72.160 -> 71.445 us | 1.010302x |

按绝对节省计算：

| 边界 | eager 节省 | graph 节省 | graph 中保留的比例 |
|---|---:|---:|---:|
| leaf | 12.763 us | 0.635 us | 约 5.0% |
| apply | 11.440 us | 0.347 us | 约 3.0% |
| region | 19.712 us | 0.715 us | 约 3.6% |

换言之，约 95%--97% 的 eager 绝对差异在 Graph 将 host 提交路径归一化
后消失。剩余设备侧差异只有约 0.3--0.7 微秒，低于此次 `3%` promotion
门槛。

## 8. 为什么可以排除“Graph 没有捕获候选”

Graph introspection 证明：

- stock 和 candidate 使用独立 graph capture；
- leaf 两侧均为 1 个 kernel node；
- apply 两侧均为 2 个 node：相同 quantizer + 各自 GEMM；
- region 两侧均为 4 个 node：相同 quantizer、各自 GEMM、相同两个
  RMSNorm；
- candidate GEMM symbol 中 N=2624/K=6144 是编译期常量；
- stock GEMM symbol 对应动态 N/K；
- capture 后修改输入，replay 输出会正确改变；
- poisoned 输出会被 replay 完整覆盖；
- input/output pointer 稳定；
- 所有输出与 stock bitwise 相等；
- 没有 fallback、reference delegation、copy 或 adapter node。

因此 candidate 确实存在于 CUDA Graph 中。Graph non-win 不是由注册丢失、
错误 fallback、旧输出或空 graph 导致的。

另一个容易误判的现象是：Graph replay 不会重新运行 Python dispatcher，
所以 HIT counter 和 capture-time NVTX range 不会按 replay 次数重复增长。
不能因为 replay 阶段没有新的 Python HIT/NVTX 就判断 candidate 未执行；
应以 graph node identity 和输入 mutation replay 为准。

## 9. 为什么固定 N/K 没有显著缩短设备关键路径

`compiled_dims="nk"` 确实生成了不同 kernel symbol，并将 N/K 从运行时值
变成编译期常量，但它没有改变主要调度结构：

- tile：`128x224x128`
- pipeline stages：6
- grid：`[148,1,1]`
- block：`[256,1,1]`
- cluster：`[2,1,1]`
- registers：54
- dynamic shared memory：约 210 KB
- 同样的 2-CTA `tcgen05` 路径

NCU 测得 compiled-N/K candidate kernel 约为 `46.688 us`，与 graph leaf
约 `48.267 us` 接近，说明 graph replay 已经接近其实际设备关键路径。

PC sampling 的 2334 个样本中：

- 1453 个为 long-scoreboard stall；
- 474 个为 barrier stall。

真正瓶颈主要是 TMA、barrier、tensor pipeline 和 persistent-wave 调度，而
不是 N/K 分支或少量整数地址计算。编译期固定 N/K 可以减少代码和 host
选择成本，但并没有减少主要 TMA/MMA 数据流，所以代码更短不等价于 kernel
执行明显更快。

还有一项直接证据：同一候选通过 generic integrated custom-op wrapper 时，
leaf 只有 `0.991x`；换成 direct helper 后恢复到约 `1.189x`。GPU 数学没有
产生相应幅度的变化，结果却随 host wrapper 明显变化，说明 eager 测量对
host invocation path 非常敏感。

## 10. 更典型的反例：`q_b_proj` 虚假加速

`q_b_proj` 的 packed-native 实验给出了更直接的证据，详见：

[`history/e2e_candidates_20260723/_negative/14_attn_q_b_packed_REPORT.md`](history/e2e_candidates_20260723/_negative/14_attn_q_b_packed_REPORT.md)

脱离完整生产层的 eager 微基准看起来有：

```text
1.40x -- 1.46x
```

但 NCU 直接测量 device kernel：

| M | stock kernel | candidate kernel |
|---:|---:|---:|
| 16 | 12.896 us | 14.528 us |
| 32 | 13.312 us | 13.824 us |

candidate device kernel 实际更慢。Graph replay 也得到：

| M | graph speedup |
|---:|---:|
| 16 | 0.9622x |
| 32 | 0.9802x |

原因是实验 candidate 的 TVM-FFI 入口比 stock custom-op wrapper 更快提交
kernel。由于 Event 在 Python/custom-op launch 之前开始，submission gap
被错误归因成了 kernel 加速。CUDA Graph replay 和 NCU 分别从 replay
边界和纯 device kernel 边界消除了该假象。

## 11. Graph 不是统一的“真相”，必须匹配生产模式

不能将所有 eager winner 一律用 Graph non-win 淘汰，因为 Graph 和 eager
代表不同生产执行模式。

| 生产执行方式 | 性能主判据 | Graph 的作用 |
|---|---|---|
| decode bucket 实际 Graph replay | graph leaf + graph containing region + E2E | 必须通过的性能门槛 |
| 大 M prefill 实际 eager | eager containing region + E2E | correctness/liveness 与诊断 |
| 同一 op 混合 eager/graph | 分 phase、M bucket 分别测试 | 两种模式都要保留 |

`index_q_upproj` 证明 Graph 不会固定“抹掉收益”：

- eager full apply：`0.952--0.971x`
- CUDA Graph replay：`1.247--1.250x`

这说明 Graph 会重新排列候选优先级，而不是单向降低候选性能。真正原则是：

> 用与生产 bucket 相同的 execution mode 作为性能 oracle。

Task02 的 production trace 证明 `fused_qkv_a_proj M4096` prefill 当时实际
为 eager，因此其 Graph `1.004949x` 被正式分类为 diagnostic non-win，
不能单独用于拒绝 eager candidate。但如果未来 SGLang 将该 bucket 纳入
Graph，那么 replay 结果必须重新成为 promotion gate。

## 12. 为什么仍然必须做端到端测试

Graph replay 可以筛掉 submission-gap 类型的虚假加速，但不能代替完整
SGLang E2E，因为 E2E 还会检验：

- 候选 bucket 在真实 scheduler 中是否真正命中；
- local M/N/K 和并行拓扑是否与 harness 一致；
- quantizer、attention、MoE、通信与 allocator 是否改变关键路径；
- kernel 是否处于端到端 critical path；
- TTFT、TPOT、throughput、SLA 是否稳定改善；
- 多 GPU 时是否出现 overlap、同步或通信回归。

即使局部 region 真有 `1.115x`，如果该 region 只占端到端时间的约
`1%--2%`，Amdahl 上限也只有约 `0.1%--0.2%` 的整体收益，很容易被
run-to-run 噪声覆盖。

历史 E2E 数据已经出现过一次无法复现的情况：

[`e2e_gain_ops_repro_SUMMARY.md`](e2e_gain_ops_repro_SUMMARY.md)

先前 seq=2048 的 TTFT 表面改善，在相同 knobs 的复测中变为 flat/noise
或更慢。因此当前候选保持 default-off，并要求在具有正确 GLM-5.2 FP8
checkpoint 和生产 GPU 拓扑的环境中完成独立 server restart、AB/BA 和
checkpoint-backed E2E 后，才能称为 production win。

## 13. 给协作者的一段简明解释

可以直接用下面这段话解释：

> eager 测试从 CUDA Event 开始以后才进入 Python、SGLang custom-op 和
> DeepGEMM launch。GPU 先执行了 start event，随后可能空等 CPU 提交
> kernel，这个空等时间也会被 Event 计入。候选用 direct helper 提交得
> 更快，所以 eager 看起来快很多。CUDA Graph capture 把这些 Python 和
> launch 准备工作只做一次；replay 时 stock 和 candidate 都直接重放已经
> 解析好的 GPU 节点。此时 host 提交差异消失，只剩实际 GPU kernel，而
> 两个 kernel 的 TMA/MMA/barrier 关键路径几乎一样，因此收益只剩约
> 0.5%--1.5%。这不是候选没注册成功，而是 eager 的大部分收益原本不是
> device kernel 算得更快。对 Graph decode 应将其判为虚假生产加速；对
> 实际 eager 的 prefill，它仍可能是框架级收益，但必须做 containing-region
> 和端到端验证。
