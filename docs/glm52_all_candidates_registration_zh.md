# GLM-5.2 全候选算子注册与公平 E2E 测试

## 1. 目标与边界

`diagnostic_all` 是一个穷举诊断档位：只要当前 SGLang 调用点能够在候选
launch 之前精确识别算子、shape、dtype、stride、phase 和 forward M，就为它
提供注册入口。它同时包含正收益、持平、回退、图内收益消失以及正确性失败过
的候选。

这不是新的 serving 默认值：

- `serving_safe` 行为不变；
- 单独设置 `SGLANG_GLM52_OPT_PROFILE=diagnostic_all` 不会选择任何算子；
- 性能实验必须显式设置 `SGLANG_GLM52_OPT_OPS`；
- 只有显式写 `all` 才会同时打开全部注册，且只建议用于 reachability smoke；
- 候选一旦通过 ABI 检查并开始调用，异常直接向上抛出，不能再偷偷运行 stock；
- 不满足 ABI 的请求在候选 launch 之前返回 stock 路径，并记录 miss reason。

## 2. 已接入的完整矩阵

共有 15 个逻辑算子、29 个 phase 注册（decode 15 个，prefill 14 个）。

| 注册名 | decode 固定边界 | prefill 固定边界 | 实现入口 |
|---|---|---|---|
| `fused_qkv_a_proj` | M=16/32, N=2624, K=6144 | M=4096 | 内置 DeepGEMM `compiled_dims="nk"` |
| `q_b_proj` | M=16/32, N=16384, K=2048 | M=4096 | 内置 fixed-N/K |
| `o_proj` | M=16/32, N=6144, K=16384 | M=4096 | 内置 fixed-N/K |
| `dense_gate_up_proj` | M=16/32, N=4096, K=6144 | M=4096 | 内置 fixed-N/K |
| `dense_down_proj` | M=16/32, N=6144, K=2048 | M=4096 | 内置 fixed-N/K |
| `index_q_upproj` | M=16/32, N=4096, K=2048 | M=4096 | 内置 fixed-N/K |
| `index_k_proj` | M=16/32, N=128, K=6144 | M=4096 | 内置 fixed-N/K |
| `index_weights_proj` | BF16 M=16/32, N=32, K=6144 | BF16 M=4096 | 内置 graph replay 诊断 |
| `index_wk_weights_proj` | BF16 M=16/32, N=160, K=6144 | BF16 M=4096 | provider |
| `dsa_decode_attn` | FlashMLA sparse decode，topk=2048 | 不接入 | provider |
| `moe_gate_proj` | W13，G=32，slab=1024，N=4096，K=6144 | M=4096 contig PSUM | decode provider / prefill 内置 |
| `moe_swiglu_quant` | BF16 `[32,1024,4096]`，forward M=16/32 | BF16 `[35200,4096]`，forward M=4096 | provider |
| `moe_down_proj` | W2，G=32，slab=1024，N=6144，K=2048 | M=4096 contig PSUM | decode provider / prefill 内置 |
| `router_logit_gemm` | BF16×BF16→FP32，M=16/32，N=256，K=6144 | M=4096 | provider |
| `router_sigmoid_topk` | FP32 `[M,256]`→FP32/int32 `[M,8]` | M=4096 | provider |

这里的 “provider” 表示调用点和 ABI 已经进入 SGLang，但候选二进制仍由
`SGLANG_GLM52_HOTSPOT_MODULE` 指定。这样 PTX/SASS、CuTe/CUTLASS 或 Triton
实现不需要再次修改模型代码，也不会把实验构建强行变成 SGLang 安装依赖。

`index_weights_proj` 的历史候选会创建并 replay 自己的 CUDA Graph。它不能
嵌套在 SGLang 的外层 CUDA Graph 中；诊断档位在检测到外层 capture 时会明确
报错，而不是改跑 stock。测试它时必须关闭 SGLang CUDA Graph。这个限制本身
就是它不能成为默认 graph-serving replacement 的结论之一。

## 3. 这次新增了哪些真实调用点

### 3.1 普通 FP8 linear

所有 block-FP8 `LinearBase` 继续经过原有 `op_context`。新增的关键修正是把
standalone shared expert 与 grouped MoE 分开：

- `*.shared_experts.gate_up_proj` → `dense_gate_up_proj`
- `*.shared_experts.down_proj` → `dense_down_proj`
- routed experts 的 W13/W2 仍由 grouped-GEMM 调用点显式标记

因此相同的 `gate_up_proj/down_proj` 叶子名不再误用不兼容 ABI。

### 3.2 fused indexer WK+weights

`wk_weights_proj` 的三个生产调用点统一经过
`try_dispatch_index_wk_weights_proj()`。候选必须返回连续 BF16 `[M,160]`，
之后仍由 stock 代码切分为 WK `[M,128]` 和 head weights `[M,32]`。

### 3.3 router GEMM 与 router top-k

- `DeepseekV2MoEGate` 的 BF16 router GEMM 经过
  `router_linear_bf16_fp32()`；
- unified Triton `moe_fused_gate()` 在分配 stock 输出之前尝试
  `try_dispatch_router_sigmoid_topk()`。

router top-k 除 shape 外还锁定 sigmoid、topk=8、无 fused shared expert、
renormalize、无输出内 routed scaling、无 softcap、单 expert group。任何语义
字段不同都在 launch 前 miss。

### 3.4 SwiGLU + packed UE8M0 quant

两个生产边界分别注册：

- masked decode：`[32,1024,4096]` → FP8 `[32,1024,2048]` +
  int32 scale view `[32,1024,4]`；
- contiguous prefill：`[35200,4096]` → FP8 `[35200,2048]` +
  int32 scale view `[35200,4]`。

两者都要求 group size 128、普通 GLM-5.2 SwiGLU、无 swizzle/clamp，并直接
产出 W2 消费的 packed UE8M0 layout，禁止把 scale 转换开销排除在测试外。

## 4. Provider API v1

模块必须声明：

```python
INFINI_KERNEL_API_VERSION = 1
```

只需要实现本次 `OPT_OPS` 实际选中的 callback。可选的
`initialize(gpu_id=...)` 在 worker 获得 CUDA device 后、warmup/graph capture
之前执行。

| 注册名 | callback | 返回契约 |
|---|---|---|
| `dsa_decode_attn` | `flashmla_sparse_decode(**stock_kwargs)` | stock `(output, lse)` |
| `moe_gate_proj` | `moe_w13(lhs, rhs, out, masked_m, expected_m)` | 原地写 `out`，返回 `None` |
| `moe_down_proj` | `moe_w2(lhs, rhs, out, masked_m, expected_m)` | 原地写 `out`，返回 `None` |
| `index_wk_weights_proj` | `index_wk_weights_proj(x, weight, phase)` | BF16 `[M,160]` |
| `router_logit_gemm` | `router_logit_gemm(hidden_states, router_weight, phase)` | FP32 `[M,256]` |
| `router_sigmoid_topk` | `router_sigmoid_topk(**semantic_kwargs)` | `(FP32 weights, int32 ids)` |
| `moe_swiglu_quant` | `moe_swiglu_quant(phase=..., **exact_kwargs)` | `(FP8 output, int32 scales)` |

如果 callback 缺失，worker 启动失败；如果 callback 已经被选中并在运行时
报错，请求失败。两种情况都不会伪装成候选命中。

## 5. 单算子启动方法

复制模板：

```bash
cp glm52_opt/runtime.env.diagnostic_all /tmp/glm52-one-op.env
export SGLANG_GLM52_ENV_FILE=/tmp/glm52-one-op.env
```

内置 fixed-N/K 示例：

```bash
export SGLANG_GLM52_OPT=1
export SGLANG_GLM52_OPT_PROFILE=diagnostic_all
export SGLANG_GLM52_OPT_OPS=q_b_proj
export SGLANG_GLM52_OPT_M_BUCKETS='q_b_proj:16|32'
export SGLANG_GLM52_INFINI_KERNEL_NVTX=1
export SGLANG_GLM52_OPT_HIT_FILE=/tmp/q_b_proj_hits.json
```

provider 示例：

```bash
export SGLANG_GLM52_OPT_OPS=router_logit_gemm
export SGLANG_GLM52_OPT_M_BUCKETS='router_logit_gemm:16|32'
export SGLANG_GLM52_HOTSPOT_MODULE=/absolute/path/provider.py
```

`SGLANG_GLM52_OPT_OPS=all` 会要求 provider 同时实现所有 provider-backed
callback；它只用于确认 29 个注册是否可达，不用于性能结论。

## 6. 公平测试规则

每个算子独立执行 baseline 与 candidate：

1. 相同 SGLang commit、模型、权重、量化格式、TP/DP/EP、batch、输入和输出；
2. baseline 设置 `SGLANG_GLM52_OPT=0`，candidate 只选择一个 `OPT_OPS`；
3. 两边都完成相同 warmup 和 CUDA Graph capture；
4. 使用 ABBA/BAAB 顺序平衡，至少三组独立进程；
5. 同时报告 eager leaf、独立 graph replay、包含上下游的 region、服务 E2E；
6. 读取 hit JSON，candidate 必须出现预期 hit；只有 miss 的运行判为无效实验；
7. nsys 中必须看到对应 `infini_kernel_glm52_*` range/内核；
8. 正确性、输出顺序、scale layout、stream 和 graph node 数都先于性能结论；
9. 报告完整中位数和速度比，包括 `<1.0x` 的回退，禁止只保留最好 shape；
10. graph-only 或 leaf-only 收益不能自动晋级为 serving replacement。

建议结果表：

| op | phase/M | correctness | eager leaf | graph leaf | containing region | E2E | hit count | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|

## 7. 已有负结果如何解释

V2 campaign 已经记录了大量 `no-replacement`，现在这些结果不再导致注册入口
被删除。典型例子包括：

- q_b decode/prefill、attention O prefill、shared-expert gate/up 与 down、
  indexer WQ-B、WK+weights：最终均为 `no-replacement`；
- decode SwiGLU+quant 的 leaf graph 可达约 1.507–1.635x，但最强包含区域聚合
  只有约 1.0204x，因此不晋级；
- W13 prefill PSUM 聚合约 1.031x，但存在单系列门槛失败；
- W2 prefill PSUM leaf graph 约 1.058x，包含区域约 1.011x；
- router logit prefill graph leaf 约 1.11x、region 约 1.09x，但 eager leaf 明显
  回退，因此没有 broad replacement；
- router top-k decode/prefill 没有通过 graph/region 门槛；
- fused WK+weights decode 曾出现候选约 12.1 µs，而 stock 合并路径约
  6.7 µs，并且候选没有通过 exact-BF16 correctness。

这些入口的意义不是暗示它们会加速，而是让外部完整模型环境能够复测并形成
明确的正、平、负 E2E 结论。`serving_safe` 只应接收后来通过全部门槛的子集。

本分支上 21 个 fixed-N/K bucket 的 4×B200、3 独立进程 ABBA 实测表见
[`glm52_diagnostic_fixed_nk_results_20260730.md`](glm52_diagnostic_fixed_nk_results_20260730.md)。

## 8. SGLang 历史 PR 审查记录

本次接入在修改生产调用点后，额外审查了 SGLang 的 GLM-5/5.1/5.2 PR
历史以及 2024–2026 年人工 review corpus。读取的模型历史文件为：

```text
/home/qinhaiyan/AI-Infra-Auto-Driven-SKILLS/model-pr-optimization-history/
  sglang/glm5-glm51/README.en.md
```

影响本次设计的主要证据如下：

| 历史证据 | 对本次接入的约束 |
|---|---|
| PR #18521、#18804：GLM-5 继承 `DeepseekV2ForCausalLM`，并修正 fused shared expert | shared expert 与 routed expert 必须按完整 prefix 区分，不能只按 `gate_up_proj/down_proj` 叶子名分发 |
| PR #22850：indexer 的 weights/K-cache fusion | 新入口必须接在当前 fused `wk_weights_proj`，不能恢复已经过时的两个独立 GEMM ABI |
| PR #25821：NSA→DSA 重命名与实现迁移 | 只修改 `dsa/dsa_indexer.py`；旧 `nsa` 路径只是兼容 re-export |
| PR #27053：GLM-5 piecewise CUDA Graph 验证 | leaf graph replay 不是最终证据；外部环境还必须跑 PCG/BCG 或明确关闭 graph 的对应实验 |
| Router GEMM 人工 review（如 PR #17707、#9834） | 锁定 256 experts 和 BF16×BF16→FP32，不能因默认 dtype 或继承模型而放宽 |
| MoE/graph 人工 review（如 PR #18213、#23882） | 保留 stream、EPLB/EP、packed scale 与 layout 语义；显式候选失败不得静默回退 |

人工 review corpus 的穷举筛选覆盖了本次修改的每个 SGLang 路径，并以
`cuda graph`、`fp8`、`moe`、`router`、`fallback`、`stream`、`shape`、
`stride`、`dtype` 等关键词扫描全部 32639 个 review thread；共匹配 429 个
thread、252 个 PR 和 815 条人工评论。最终实现据此保持以下原则：

1. 默认关闭，且 `diagnostic_all` 裸 profile 不选择任何候选；
2. 精确锁定 phase/M/shape/dtype/stride/storage offset 和语义参数；
3. 只有候选 launch 前的 ABI miss 才允许走 stock；
4. 候选一旦被选中，callback、输出 ABI 或运行时错误必须直接暴露；
5. graph leaf、eager leaf、containing region 和服务 E2E 分层报告，不能用
   graph-only 数字晋级生产替换。
