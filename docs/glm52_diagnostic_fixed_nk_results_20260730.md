# GLM-5.2 `diagnostic_all` fixed-N/K 实测结果（2026-07-30）

## 结论

本次把 7 类 FP8 GEMM 的 decode M16/M32 与 prefill M4096 全部经过新的
SGLang 注册/dispatch 接口实测，共 21 个 bucket：

- 21/21 生产 packed-UE8M0 ABI 命中；
- 21/21 eager 输出与 stock 逐位一致；
- 21/21 可独立 CUDA Graph capture/replay，graph 输出逐位一致；
- 每个独立进程的每个 bucket 都至少记录 65 次真实 candidate hit；
- **没有一个 bucket 可以仅凭这次结果晋级 serving replacement**。

decode 的 fixed-N/K kernel 在独立 graph leaf 中全部显示 1.116–1.576x，但在
完整 SGLang eager wrapper 中全部回退到 0.895–0.921x。prefill 除
`index_k_proj` graph leaf 为 1.178x 外基本打平；eager 仍多为 0.886–0.916x。
这组结果直接证明：graph leaf 很快并不等于包含 dispatch、上下游和服务请求的
完整路径会更快。

## 方法

- GPU：4× NVIDIA B200，每个 shard 固定在一张物理 GPU；
- 同一 SGLang commit、同一输入/权重/packed scale；
- baseline：`SGLANG_GLM52_OPT=0`；
- candidate：`diagnostic_all`，每次只选一个 op 和一个 M bucket；
- 调用生产函数
  `deepgemm_w8a8_block_fp8_linear_with_fallback()`，量化和 dispatch 均在计时内；
- 每个结果来自 3 个独立 Python 进程；
- 每个进程 3 个 ABBA/BAAB series，每个 lane 10 次；
- 表中时间是跨进程中位数，speedup=`stock/candidate`；
- eager 计时会看到候选 registry/hit-accounting 带来的真实 host enqueue 间隙；
- graph 计时只 replay 已 capture 的完整 leaf graph。

复现：

```bash
PYTHONPATH=python \
python scripts/bench_glm52_diagnostic_fixed_nk.py \
  --shard-index 0 --num-shards 4 --repeats 10 --series 3
```

多个独立进程的串联 JSON 可用
`scripts/aggregate_glm52_diagnostic_fixed_nk.py` 从 stdin 聚合。

## Decode M16/M32

| op | M | eager stock→cand (µs) | eager speedup | graph stock→cand (µs) | graph speedup | 结论 |
|---|---:|---:|---:|---:|---:|---|
| fused_qkv_a_proj | 16 | 99.192→110.482 | 0.898x | 16.774→12.752 | 1.316x | eager 回退 |
| fused_qkv_a_proj | 32 | 94.317→104.331 | 0.904x | 16.765→12.610 | 1.332x | eager 回退 |
| q_b_proj | 16 | 94.339→105.005 | 0.898x | 10.675→8.763 | 1.218x | eager 回退 |
| q_b_proj | 32 | 96.725→110.486 | 0.904x | 10.680→9.570 | 1.116x | eager 回退 |
| o_proj | 16 | 90.989→99.226 | 0.915x | 33.054→23.573 | 1.402x | eager 回退 |
| o_proj | 32 | 89.629→97.142 | 0.921x | 33.142→27.045 | 1.226x | eager 回退 |
| dense_gate_up_proj | 16 | 92.909→103.267 | 0.909x | 16.717→10.718 | 1.560x | eager 回退 |
| dense_gate_up_proj | 32 | 90.565→99.762 | 0.908x | 16.718→11.661 | 1.435x | eager 回退 |
| dense_down_proj | 16 | 88.926→99.354 | 0.896x | 10.566→8.578 | 1.232x | eager 回退 |
| dense_down_proj | 32 | 87.205→97.010 | 0.906x | 10.576→8.590 | 1.231x | eager 回退 |
| index_q_upproj | 16 | 86.915→97.139 | 0.895x | 10.576→8.619 | 1.225x | eager 回退 |
| index_q_upproj | 32 | 88.357→98.296 | 0.907x | 10.558→8.566 | 1.225x | eager 回退 |
| index_k_proj | 16 | 88.853→97.499 | 0.908x | 16.725→10.603 | 1.576x | eager 回退 |
| index_k_proj | 32 | 87.557→96.216 | 0.906x | 16.722→10.629 | 1.566x | eager 回退 |

所有 decode bucket 的跨进程最差 eager speedup 也小于 1.0。它们可以保留为
CUDA-Graph/包含区域实验入口，但不能因为 graph leaf 的正收益直接晋级。

## Prefill M4096

| op | eager stock→cand (µs) | eager speedup | graph stock→cand (µs) | graph speedup | 结论 |
|---|---:|---:|---:|---:|---|
| fused_qkv_a_proj | 95.963→107.310 | 0.895x | 61.920→61.781 | 1.002x | 无收益 |
| q_b_proj | 114.898→114.749 | 0.998x | 108.474→107.419 | 1.010x | 基本持平 |
| o_proj | 322.726→323.053 | 0.999x | 322.435→321.498 | 1.003x | 基本持平 |
| dense_gate_up_proj | 95.770→102.757 | 0.916x | 85.128→85.050 | 1.002x | eager 回退 |
| dense_down_proj | 89.477→98.214 | 0.888x | 43.920→43.574 | 1.006x | eager 回退 |
| index_q_upproj | 88.024→99.390 | 0.886x | 33.195→33.138 | 1.001x | eager 回退 |
| index_k_proj | 87.837→98.133 | 0.895x | 29.570→25.069 | 1.178x | graph-only 正收益 |

`index_k_proj` 是本轮最典型的“一个 shape 有 graph 正收益，但仍不能直接替换”
案例：leaf graph 为 1.178x，完整 eager wrapper 为 0.895x，而且尚未通过包含
indexer 上下游和模型 E2E 的门槛。

## 如何使用这些负结果

这些结果不会从 registry 删除。保留它们有三个用途：

1. 在外部 GLM-5.2 TP/DP/EP 环境逐算子做 E2E，确认 graph leaf 收益是否能覆盖
   dispatch、重叠、通信和其他层；
2. 在优化 dispatch 或候选 launch ABI 后，用同一表直接检测负收益是否消失；
3. 避免以后只看到 1.2–1.5x graph leaf 就重复宣称 production 加速。

本机没有 GLM-5.2 checkpoint，因此本文不伪造服务 E2E 数字。最终 E2E 需要按
[`glm52_all_candidates_registration_zh.md`](glm52_all_candidates_registration_zh.md)
的单算子 ABBA 方法在可运行模型的环境完成。
