# GLM-5.2 算子注册、真实加速边界与端到端验收

> Kernel-Harness 全量历史正收益、production-native 复核和最终注册等级见
> [`glm52_kernel_harness_registration_audit_zh.md`](glm52_kernel_harness_registration_audit_zh.md)。
> 本文主要解释当前四条精确 hook 的实现与使用方式。

本文给出本分支中 GLM-5.2 算子的最终注册状态、启用方法、能够成立的
加速条件，以及 FlashMLA PTX/SASS 和 MoE W2 实验为什么不能被宣传为生产
加速。

结论先行：

| 算子 | 当前状态 | 是否随裸 `hotspot_candidates` 选择 | 本地证据 |
|---|---|---:|---|
| fused QKV-A prefill direct-N/K | 外部验收候选，默认关闭 | 不适用，使用独立开关 | 完整 eager region 最弱估计量 1.078588x |
| MoE fused W13 decode BM16/2-SM | 外部验收候选，默认关闭 | 是，但仍必须提供精确 provider | 完整 region 最弱估计量 1.034255x |
| FlashMLA KV sparse decode PTX/SASS composite | 仅显式诊断，`no-replacement` | 否 | leaf eager 通过；graph 和 containing region 失败 |
| MoE W2 decode BM16 | 仅显式诊断，`no-replacement` | 否 | device kernel 1.138x；完整 API 路径 0.801642x |

“外部验收候选”不等于生产默认。QKV-A 和 W13 仍缺少真实 checkpoint 的
TP8/DP8/EP8 服务验收，所以所有新路径都保持 default-off。

## 1. 什么情况下可以说“获得了加速”

一次结果只有同时满足下面的边界，才可以归因给目标算子：

1. 模型指纹、并行拓扑、phase、M/N/K、dtype、scale 表示、shape、stride、
   storage offset 和输出所有权与生产路径完全一致。
2. stock 与 candidate 使用相同输入字节、当前 CUDA stream、SM 预算、PDL
   状态和输出分配策略。
3. JIT、provider 初始化、CUDA Graph capture 和一次性 buffer 分配都在计时
   之外；每次生产调用实际必须承担的 dispatch/enqueue 工作则留在计时内。
4. leaf、包含目标算子的完整 region，以及生产实际使用的 eager 或 CUDA
   Graph 路径都通过正确性和性能门槛。
5. 至少三组独立的 AB/BA 交替序列；pooled、order-balanced、AB median 和
   BA median 四个估计量全部不低于 1.03x。
6. candidate 命中后若失败必须直接报错，不能偷偷回退 stock 再记录为命中。
7. Nsys 中目标 `__global__` symbol 或其外层 NVTX 使用
   `infini_kernel_...` 名称，并且 graph node、输入 mutation 和输出 poison
   检查证明 replay 的确执行了 candidate。

对于生产 decode 是 CUDA Graph 的 bucket，graph leaf 和 graph containing
region 是强制门槛。对于当前生产仍为 eager 的 M4096 prefill，eager region
是性能门槛，graph 主要用于证明捕获兼容性和语义活性。

更完整的 graph replay 解释见
[`glm52_opt/sglang_integration_cuda_graph_replay_analysis_zh.md`](../glm52_opt/sglang_integration_cuda_graph_replay_analysis_zh.md)。

## 2. fused QKV-A prefill direct-N/K

### 注册位置

这条路径不经过旧的通用 archive registry，而是在 GLM-5.2 的
`fused_qkv_a_proj_with_mqa` 层上绑定私有 runner。启动时只有精确模型指纹
匹配才会安装；forward 时只在 `ForwardMode.EXTEND`、local M4096 且无 LoRA
时发布短生命周期 context。

本分支会拒绝同时启用以下两条路由：

- 独立开关 `SGLANG_OPT_GLM52_FUSED_QKV_A_PREFILL_DIRECT_NK=1`；
- `SGLANG_GLM52_OPT` 通用 registry 中的同一个
  `fused_qkv_a_proj/prefill/M4096` 条目。

这样不会出现“两个注册器都声称命中、实际只有后者执行”的歧义。

### 唯一可声明加速的 bucket

- 架构：`GlmMoeDsaForCausalLM`，`model_type=glm_moe_dsa`；
- hidden 6144、q-LoRA rank 2048、KV-LoRA rank 512、rope head dim 64；
- attention TP size 1，即每个 rank 上该 projection 为 replicated；
- `M=4096, N=2624, K=6144`；
- FP8 E4M3 contiguous weight `[2624,6144]`；
- packed int32 UE8M0 weight scale `[2624,12]`，stride `[1,2624]`；
- packed int32 activation scale `[4096,12]`，stride `[1,4096]`；
- block recipe `[128,128]`，BF16 output，无 bias、无 LoRA；
- B200/sm_100，DeepGEMM 支持 `compiled_dims="nk"`。

candidate 只把 DeepGEMM 模板 key 固定为 N/K；量化、输出分配、split 和两个
RMSNorm 都与 stock 一致。

本地 B200 三组 50-pair AB/BA 结果：

| 边界 | pooled speedup | 该边界所有序列中的最弱估计量 |
|---|---:|---:|
| packed GEMM leaf | 1.173894x | 1.131806x |
| BF16 quantize + packed GEMM | 1.091133x | 1.078588x |
| projection + split + two RMSNorm | 1.114767x | 1.083918x |

生产 prefill 当前是 eager，因此上表可以作为本地晋级证据。独立 graph capture
通过 bitwise correctness 和 mutation replay，但 graph 性能约 1.005–1.015x，
只作为诊断数据，不能替代 eager 结论，也不能外推到 decode。

### 启用

```bash
unset SGLANG_GLM52_OPT
unset SGLANG_GLM52_OPT_PROFILE
unset SGLANG_GLM52_OPT_OPS

export SGLANG_OPT_GLM52_FUSED_QKV_A_PREFILL_DIRECT_NK=1
python -m sglang.launch_server <原有 GLM-5.2 参数>
```

任一模型、shape、phase、dtype、scale layout、topology 或 LoRA 条件不匹配时，
都会在 candidate launch 之前走 stock。

## 3. fused W13 decode BM16/2-SM

### 注册位置和精确 ABI

SGLang 生产路径将 gate 与 up 合成一个 W13 grouped GEMM；不能把它当成两个
历史 N2048 GEMM。注册名和 profiler 名称为：

```text
user alias:      moe_w13
registry op:     moe_gate_proj
NVTX:            infini_kernel_glm52_moe_w13_decode
CUDA symbols:    infini_kernel_glm52_moe_w13_decode_em{4,5,8,9}_bm16_2sm
```

允许的唯一 ABI：

- `ForwardMode.DECODE`，local M16/M32；
- M16 对应 expected-M 4/5，M32 对应 expected-M 8/9；
- E32、固定 expert slab M1024、K6144、N4096；
- contiguous FP8 E4M3 activation/weight；
- TMA-aligned packed int32 UE8M0 scale；
- contiguous BF16 caller-owned output 和 contiguous int32 E32 mask；
- B200/sm_100、148 SM、PDL=true、tc-util=100；
- candidate config `(BM,BN,BK,stages,cluster-N)=(16,128,128,12,2)`。

PTX 中有 16 个 `tcgen05.mma.cta_group::2`，SASS 中有 16 个
`UTCQMMA.2CTA`；35 registers/thread、零 stack/local/spill。它不是只改名的
stock kernel。

### 本地收益边界

- W13 leaf 约 1.043x；
- W13→stock SwiGLU/packed quant→stock W2 完整 region 的最弱估计量
  1.034255x；
- Nsys W13 device p50：145.744 us → 139.168 us，1.047256x；
- Nsys 完整 device critical span：226.176 us → 218.816 us，1.033636x；
- eager/graph、expected-M 4/5/8/9 共 16 个性能 lane 全部通过 1.03x。

这个结论只覆盖上述本地单 B200 精确 ABI。没有通过 checkpoint-backed
TP8/DP8/EP8 之前，仍不能设为生产默认。

### 在另一台机器构建

精确 candidate 差分已随本分支打包并由 SHA-256 校验，不再依赖原机器的
`/home/qinhaiyan/...` 路径。准备一个干净的 DeepGEMM checkout，包含 pinned
base 和子模块 commit：

```bash
git clone https://github.com/sgl-project/DeepGEMM.git /path/to/DeepGEMM
git -C /path/to/DeepGEMM checkout 731e7c7a97d269e4b9f482ea18d0e709a948f293
git -C /path/to/DeepGEMM submodule update --init --recursive

CUDA_VISIBLE_DEVICES='' \
python third_party/deepgemm_w13/build_variants.py \
  --source /path/to/DeepGEMM \
  --audit-materialization

CUDA_VISIBLE_DEVICES='' CUDA_HOME=/usr/local/cuda MAX_JOBS=4 \
python third_party/deepgemm_w13/build_variants.py \
  --source /path/to/DeepGEMM \
  --force
```

默认产物在仓库 `.cache/glm52_w13_variants/`。构建器会同时重建同源 stock 和
candidate，验证 patch、source tree、compiler plan、DSO 和 JIT identity。

### 启用

```bash
export SGLANG_GLM52_OPT=1
export SGLANG_GLM52_OPT_PROFILE=hotspot_candidates
export SGLANG_GLM52_OPT_OPS=moe_w13
export SGLANG_GLM52_OPT_M_BUCKETS='moe_gate_proj:16|32'
export SGLANG_GLM52_W13_MANIFEST="$PWD/.cache/glm52_w13_variants/manifest.json"
export SGLANG_GLM52_HOTSPOT_MODULE="$PWD/third_party/deepgemm_w13/provider_bm16_2sm.py"

python -m sglang.launch_server <原有 GLM-5.2 参数>
```

provider 在 worker GPU 分配后、graph capture 前完成 DSO 绑定和四个
expected-M warmup。热路径只有一次 current-stream launch、写 caller-owned
output、返回 `None`。

## 4. FlashMLA PTX/SASS 结果为什么只注册为诊断

SGLang 中保留了严格的 API-v1 hook，便于继续 PTX/SASS 实验，但裸
`hotspot_candidates` 不会选择它。必须显式写
`SGLANG_GLM52_OPT_OPS=flashmla_sparse_decode` 才会要求 provider。

精确实验边界：

- M16：Q `[16,1,64,576]` BF16，KV `[2049,64,1,656]` FP8；
- M32：Q `[32,1,64,576]` BF16，KV `[4097,64,1,656]` FP8；
- sparse top-k 2048、page size 64、V dim 512、scale 0.0625；
- output `[M,1,64,512]` BF16；
- LSE `[M,64,1]` FP32 contiguous；
- production `flash_mla_with_kvcache` main 加原有 BF16 combine。

最终 composite 的静态 SASS 从 4128 条降到 3336 条，并产生直接
`F2FP.BF16.E4M3`，但仍是 168 registers、16 barriers、零 spill。说明 PTX
修改确实进入了最终机器码。

公平性能结果：

| bucket | leaf eager | leaf graph | containing eager | containing graph |
|---|---:|---:|---:|---:|
| M16 | 1.1435–1.2636x | 1.0056–1.0633x | 0.7610–0.8634x | 1.0189–1.0989x |
| M32 | 1.1402–1.2428x | 1.0082–1.0578x | 0.7929–0.8694x | 1.0131–1.0851x |

每个范围是三组序列中四个必需估计量的最小值到最大值。graph 最小值低于
1.03，包含生产 wrapper 的 eager 路径明显回退，所以最终结论是
`no-replacement`。leaf eager 的好看数字主要受 candidate 预分配输出和
不同 host ownership 影响，不能代表稳定 device chain。

因此本分支只提交：

- 精确 page-count/LSE/shape/stride 的 fail-closed 注册；
- `infini_kernel` profiler 命名；
- 显式诊断入口。

被拒绝的 FlashMLA binary 不随生产分支打包，也不得作为端到端候选启用。
下一轮 PTX 优化应直接降低 main kernel 的 scoreboard/barrier critical path，
并把 graph containing-region 作为第一轮筛选，而不是继续优化 Python wrapper
或只看 eager leaf。

## 5. W2 为什么 device kernel 快，完整路径却低于 baseline

W2 BM16 candidate 的设备 kernel 的确更快：

| 分量 | stock | candidate | 差值 |
|---|---:|---:|---:|
| Nsys/CUDA profiler 中目标 kernel | 77.659 us | 68.223 us | candidate 节省 9.436 us |
| 完整 selected API-v1 interval | 93.760 us | 116.960 us | candidate 多 23.200 us |
| interval 减去 kernel | 16.101 us | 48.737 us | candidate 多 32.636 us |

所以不存在“kernel 测错”这一矛盾。等式非常直接：

```text
完整收益 = kernel 节省 9.436 us - 额外 enqueue/dispatch gap 32.636 us
         = 回退约 23.2 us
```

最终 pooled speedup 是 `93.760 / 116.960 = 0.801642x`。

已经被源码和实验确认的事实：

- stock 是 BM128/stage8，candidate 是 BM16/stage12；
- candidate 将 TMA output store 从 16 个降到 2 个，kernel duration 因而下降；
- candidate 热路径经过 SGLang dispatch、Python provider callback、额外的
  DeepGEMM candidate selector/FFI 参数和 enqueue；
- provider 还维护 attempted/completed Python counters；
- 去掉两次每调用的 counter-dictionary materialization 约追回 8 us，但最终
  仍只有 0.801642x。

尚未做 Nsys 全路径分段，因此不能把剩余 32.636 us 精确分配给某一行 Python
或某一个 C++ helper。根据 CUDA event interval 与单 kernel profiler 的差值，
最合理的解释是：GPU 在 candidate kernel 真正 enqueue 前等待了更多 host
dispatch、descriptor/config 和 FFI 工作；这些空档也会进入两端 CUDA event
之间的完整调用时间。这一部分是推断，不能伪装成逐函数实测。

这也是 W2 不随裸 profile 启用的原因。它保留为显式诊断 registration，是为了
后续直接攻击 enqueue/dispatch 边界；当前 candidate 不具备端到端测试价值。

## 6. 端到端验收时必须记录什么

在外部 8-GPU 环境至少保留以下四组结果：

1. stock 与单算子 candidate 的 checkpoint output correctness；
2. TP8/DP8/EP8 每个 rank 的命中/回退计数和 rank-max region latency；
3. 固定请求集下的 TTFT、TPOT、throughput、graph capture/replay 状态；
4. Nsys 中实际 `infini_kernel` symbol、目标 graph node 和完整 critical span。

QKV-A 与 W13 必须分别单独 A/B，先不要同时开启。只有单算子通过后才做组合
测试，否则无法区分增益抵消、拓扑不匹配和 graph recapture。
