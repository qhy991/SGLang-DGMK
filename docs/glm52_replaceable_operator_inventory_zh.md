# GLM-5.2 可替换算子完整清单

更新日期：2026-07-30

## 1. 结论

当前分支已经为 **15 个逻辑算子、29 个 decode/prefill 边界**提供可控替换
入口。这里的“可替换”只表示：

1. SGLang 能在候选 launch 前识别真实生产调用点；
2. 能精确检查 phase、forward M、shape、dtype、stride、layout 和语义参数；
3. ABI 不匹配时在候选 launch 前走 stock；
4. ABI 匹配并选择候选后，运行错误和输出 ABI 错误会直接暴露；
5. 有 hit/miss 计数以及可选的 `infini_kernel_*` NVTX 名称。

它不表示这些算子都能加速。当前没有任何算子可以直接成为生产默认值。

## 2. 完整替换矩阵

| # | 注册名 | SGLang 生产调用点 | decode 边界 | prefill 边界 | 候选交付方式 | 当前本地结论 |
|---:|---|---|---|---|---|---|
| 1 | `fused_qkv_a_proj` | GLM DSA attention fused QKV-A projection | M16/32, N2624, K6144 | M4096, N2624, K6144 | 内置 fixed-N/K；另有独立 direct-N/K 开关 | diagnostic 完整 eager 回退；独立 prefill direct-N/K 仍是外部 E2E 候选 |
| 2 | `q_b_proj` | attention Q-B projection | M16/32, N16384, K2048 | M4096 | 内置 fixed-N/K | decode eager 回退；prefill 基本持平；不晋级 |
| 3 | `o_proj` | attention output projection | M16/32, N6144, K16384 | M4096 | 内置 fixed-N/K | diagnostic eager 回退/持平；只保留外部逐 shape 复测 |
| 4 | `dense_gate_up_proj` | standalone shared expert gate+up | M16/32, N4096, K6144 | M4096 | 内置 fixed-N/K | eager 回退；不晋级 |
| 5 | `dense_down_proj` | standalone shared expert down | M16/32, N6144, K2048 | M4096 | 内置 fixed-N/K | eager 回退；不晋级 |
| 6 | `index_q_upproj` | DSA indexer `wq_b` | M16/32, N4096, K2048 | M4096 | 内置 fixed-N/K | eager 回退；历史 graph/consumer 正确性也未过门槛 |
| 7 | `index_k_proj` | DSA indexer non-fused WK | M16/32, N128, K6144 | M4096 | 内置 fixed-N/K | prefill graph-only 1.178x，但完整 eager 0.895x；不晋级 |
| 8 | `index_weights_proj` | DSA indexer standalone head-gate projection | BF16 M16/32, N32, K6144 | BF16 M4096 | 内置 graph-replay 诊断 | 不能嵌套 SGLang CUDA Graph；只用于 graph-off 诊断 |
| 9 | `index_wk_weights_proj` | 当前 CUDA fused WK+head-weights projection | BF16 M16/32, N160, K6144 | BF16 M4096 | provider callback | 接口已接入；历史候选更慢且 exact-BF16 未通过 |
| 10 | `dsa_decode_attn` | FlashMLA sparse DSA decode | M16/32, topk2048, H64, QK576, V512 | 无 | provider callback | PTX/SASS leaf 曾有收益，graph/containing region 消失；诊断级 |
| 11 | `moe_gate_proj` | routed experts fused W13 grouped GEMM | E32, slab1024, M16/32, N4096, K6144 | M4096 对应 aligned M35200 contig PSUM | decode provider；prefill 内置 PSUM | W13 decode 是外部 E2E 候选；prefill PSUM region 未稳定过线 |
| 12 | `moe_swiglu_quant` | W13→W2 之间的 SwiGLU + packed UE8M0 quant | `[32,1024,4096]`，forward M16/32 | `[35200,4096]`，forward M4096 | provider callback | decode/prefill 均为 no-replacement；接口与 packed scale ABI 已验证 |
| 13 | `moe_down_proj` | routed experts W2 grouped GEMM | E32, slab1024, M16/32, N6144, K2048 | M4096 对应 aligned M35200 contig PSUM | decode provider；prefill 内置 PSUM | decode 完整 API 约0.802x；prefill region 约1.011x；诊断级 |
| 14 | `router_logit_gemm` | `DeepseekV2MoEGate` BF16 router linear | BF16 M16/32, N256, K6144 | BF16 M4096 | provider callback | prefill graph/region 有局部收益，但 eager 明显回退；诊断级 |
| 15 | `router_sigmoid_topk` | unified Triton sigmoid/no-aux router | FP32 `[M,256]`→FP32/int32 `[M,8]` | M4096 | provider callback | decode/prefill graph/region 门槛均未通过；诊断级 |

decode 有 15 个注册；`dsa_decode_attn` 没有 prefill 替换，因此 prefill 有
14 个注册，总计 29 个。

## 3. 当前可直接运行的内置候选

以下候选不需要额外 Python provider：

| 类型 | 算子 |
|---|---|
| fixed-N/K FP8 | `fused_qkv_a_proj`、`q_b_proj`、`o_proj`、`dense_gate_up_proj`、`dense_down_proj`、`index_q_upproj`、`index_k_proj` |
| BF16 graph replay | `index_weights_proj` |
| contiguous PSUM | prefill `moe_gate_proj`、prefill `moe_down_proj` |

其中 7 类 fixed-N/K 已完成 21 个 bucket 的 4×B200、3 独立进程 ABBA
测试。21/21 exact、21/21 graph exact、21/21 有真实 hit，但没有一个凭本轮
结果晋级 serving replacement。

## 4. 需要 provider 的候选

这些生产调用点已经接入，但候选二进制必须通过
`SGLANG_GLM52_HOTSPOT_MODULE` 提供：

| 注册名 | provider callback |
|---|---|
| `dsa_decode_attn` | `flashmla_sparse_decode(**stock_kwargs)` |
| decode `moe_gate_proj` | `moe_w13(lhs, rhs, out, masked_m, expected_m)` |
| decode `moe_down_proj` | `moe_w2(lhs, rhs, out, masked_m, expected_m)` |
| `index_wk_weights_proj` | `index_wk_weights_proj(x, weight, phase)` |
| `router_logit_gemm` | `router_logit_gemm(hidden_states, router_weight, phase)` |
| `router_sigmoid_topk` | `router_sigmoid_topk(**semantic_kwargs)` |
| `moe_swiglu_quant` | `moe_swiglu_quant(phase=..., **exact_kwargs)` |

仓库已经内置 W13 BM16 provider；其他 PTX/SASS、CuTe/CUTLASS 或 Triton
实验实现可通过同一 API 注入。选中 provider-backed op 但 module/callback
缺失时，worker 启动失败，不会把 stock 当成 candidate。

## 5. 仍值得外部端到端测试的精确路径

这三条路径的优先级最高，但仍保持默认关闭：

| 优先级 | 精确路径 | 启用入口 | 外部环境必须验证 |
|---:|---|---|---|
| P0 | fused QKV-A prefill direct-N/K，M4096/N2624/K6144 | `SGLANG_OPT_GLM52_FUSED_QKV_A_PREFILL_DIRECT_NK=1` | TTFT、prefill throughput、rank-max region、真实 checkpoint exactness |
| P1 | fused W13 decode BM16 two-SM | `hotspot_candidates + moe_w13 + bundled provider` | TPOT、decode throughput、所有 rank hit、W13→activation→W2 region |
| P2 | attention O decode，逐 bucket 复测 | `diagnostic_all + o_proj + 精确 M_BUCKETS` | M16/M32 分开报告，不得用 graph leaf 代替完整 wrapper/E2E |

P2 在旧 exact apply 实验中 M16 曾有正收益，但本次包含注册和命中记账的完整
wrapper 结果为回退，所以只能作为“冲突证据待外部 E2E 解决”，不能宣传为
已经确认的加速。

## 6. 单算子替换模板

内置算子：

```bash
export SGLANG_GLM52_OPT=1
export SGLANG_GLM52_OPT_PROFILE=diagnostic_all
export SGLANG_GLM52_OPT_OPS=index_k_proj
export SGLANG_GLM52_OPT_M_BUCKETS='index_k_proj:4096'
export SGLANG_GLM52_INFINI_KERNEL_NVTX=1
export SGLANG_GLM52_OPT_HIT_FILE=/tmp/index_k_proj_hits.json
```

provider 算子：

```bash
export SGLANG_GLM52_OPT=1
export SGLANG_GLM52_OPT_PROFILE=diagnostic_all
export SGLANG_GLM52_OPT_OPS=router_logit_gemm
export SGLANG_GLM52_OPT_M_BUCKETS='router_logit_gemm:16|32'
export SGLANG_GLM52_HOTSPOT_MODULE=/absolute/path/provider.py
export SGLANG_GLM52_INFINI_KERNEL_NVTX=1
export SGLANG_GLM52_OPT_HIT_FILE=/tmp/router_logit_hits.json
```

`SGLANG_GLM52_OPT_OPS=all` 只用于 reachability smoke，不用于性能结论。
公平测试必须一次只选择一个 op，并按 M bucket 分开报告。

## 7. 结果与接口文档

- 全候选注册设计：
  [`glm52_all_candidates_registration_zh.md`](glm52_all_candidates_registration_zh.md)
- 21 个 fixed-N/K B200 结果：
  [`glm52_diagnostic_fixed_nk_results_20260730.md`](glm52_diagnostic_fixed_nk_results_20260730.md)
- 既有正收益审计：
  [`glm52_kernel_harness_registration_audit_zh.md`](glm52_kernel_harness_registration_audit_zh.md)
- 部署入口：
  [`glm52_opt_deploy.md`](glm52_opt_deploy.md)
