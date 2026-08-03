# 已作废：B300 Masked MoE SwiGLU+Quant valid-CTA 实验

> **状态：INVALIDATED（2026-08-03）。本页所有性能数字都不得作为 winner
> 或生产晋级依据。** 旧实现把源 rank 的 `M*topk` 当作目的 rank 的工作量
> 上界；EP 路由偏斜时 `sum(masked_m) > M*topk`，会漏写合法 routed rows。
> 此外，直接把单工作 CTA 改成循环 CTA 后，PDL trigger 也必须推迟到每个
> CTA 的最后一轮，否则下游 W2 可能在后续 row 写完前启动。

旧测试只构造了 `sum(masked_m)=M*topk` 的单 rank 代理，又只跑了每个启动
顺序一条请求，因此未暴露这两个正确性缺口。下面保留原文仅用于事故追溯；
14.72% TPOT、14.06x activation 和 2.84x Harness 等数字全部作废。替代实现
是 `cuda_grid_stride`：host-known `M*topk` 只是 CTA pool，每个 CTA 在设备端
循环直到完整消费 `sum(masked_m)`，且只在自己的最后一个 work item 前触发
PDL。新的偏斜/热点 Harness、8 卡 nsys 和多批次固定-KV结果记录在
`MOE_SWIGLU_B300_GRID_STRIDE.md`。

---

## 以下是已作废的历史记录

- 日期：2026-08-02
- 机器：B300-M2，8×NVIDIA B300 SXM6 AC，TP8/DP8/EP8
- 模型：`/mnt/b300-shared/models/GLM-5.2-FP8`
- 基线 commit：`fc4b5d22f2dbf5e82e6a5012cda9765de272daa6`
- 全 graph-bucket 候选 commit：`f17d164fa7a4aee42c98064a1bea418308daf918`
- 候选：`SGLANG_OPT_MOE_SWIGLU_QUANT_VARIANT=cuda_valid_cta`
- 默认行为：**关闭**；任一契约不匹配即回退原始实现

## 1. 修改是什么

B300 的 DeepEP low-latency decode 保留物理 expert slab
`gateup_output=[32,8192,4096]`，但 global BS=128 / DP8 时每卡只有
`M=16`，即 `M*topk=128` 个真实 routed assignment。

原 wrapper 按物理容量启动 `8192*8=65536` 个 CTA；大部分 CTA 读取
`masked_m` 后立即退出。候选完整复用 stock CUDA kernel body、FP32 SiLU、
FP8 E4M3 输出以及 packed int32 UE8M0 写法，只把 grid 改为
`num_real_tokens*topk`。在稳态 M=16 时 grid 从 65,536 降到 128；在
M=1 尾批次时降到 8。因此它不是近似算法，也没有缩小生产 buffer。

自动选择同时要求：

- compute capability `(10,3)` 与物理 slab `T=8192`；
- BF16 contiguous `[32,8192,4096]` 输入、contiguous int32 `[32]` mask；
- `group_size=128`、`topk=8`、host-known CUDA Graph bucket
  `M in {1,2,4,8,12,16,32}`；
- 无 clamp、swizzle、`gemm1_alpha`；
- `combined_winners`/专用 profile、显式 op allowlist 和显式 variant。

## 2. Kernel Harness 结果

生成任务 `moe_swiglu_quant_b300_decode` 保留完整 T=8192 物理 slab，
只把 `masked_m[e]` 行定义为 consumer-visible，并将 packed UE8M0 的四个
exponent byte 纳入正确性门禁。候选写预分配 production ABI；计时使用
cold-L2 CUDA event、相邻平衡 R/C 与 C/R、warmup=8、repeat=10、每 sample
30 次 inner iteration。

| 项 | stock | valid-CTA | 结果 |
|---|---:|---:|---:|
| generated task | 46.176 us | 16.144 us | median 2.8571×；保守 p10 2.8466× |
| serving-native eager（含输出分配） | 54.288 us | 27.584 us | 1.9489×；保守 p10 1.8194× |
| serving-native CUDA Graph | 41.984 us | 11.184 us | 3.7479×；保守 p10 3.6741× |

三条 lane 均通过初始 seed 和未见 post-timing seed；FP8 与 packed UE8M0
均 byte-exact，`calc_diff=0`。原始结果：

- `/mnt/b300-shared/home/qinhaiyan/wwxq/Kernel-Harness-moe-swiglu-b300-test/runs/glm52/moe_swiglu_quant_b300_decode/20260801T163402Z-ffd477/result.json`
- `/mnt/b300-shared/home/qinhaiyan/wwxq/bench_results/moe_swiglu_b300_harness_20260802b/serving_native_result.json`

配对 trace 随后暴露出 CUDA Graph 尾批次：稳态 M=16 已命中候选，但最后
一个 DP0 iteration 收缩到 M=1 并回退 stock。因此 Harness sweep 扩展到服务端
真实 bucket `1/2/4/8/12/16/32`。扩展后的 B300 物理-slab 门禁为：

| bucket | stock | valid-CTA | median speedup |
|---:|---:|---:|---:|
| M=1 | 45.648 us | 16.000 us | 2.8463× |
| M=2 | 44.848 us | 16.096 us | 2.7917× |
| M=4 | 45.952 us | 16.032 us | 2.8650× |
| M=8 | 44.784 us | 16.024 us | 2.7940× |
| M=12 | 46.064 us | 16.080 us | 2.8614× |
| M=16 | 46.112 us | 16.176 us | 2.8605× |
| M=32 | 45.600 us | 16.064 us | 2.8320× |

`7/7` shapes 均为 byte-exact、`calc_diff=0`、未见 seed 复验通过且无回归；
geomean speedup 为 `2.8357×`。原始结果：

- `/mnt/b300-shared/home/qinhaiyan/wwxq/Kernel-Harness-moe-swiglu-b300-test/runs/glm52/moe_swiglu_quant_b300_decode/20260801T182130Z-855918/result.json`

## 3. 32K 固定 KV、global BS=128 端到端结果

两轮采用相反 service 启动顺序；每个请求先构造同一 deterministic random-id
prefix，只留下最后 64 个 prompt token 未命中。两边 cache hit 都是 0.998，
`output_len=240`，不开 multi-batch；指标为
`TPOT=(latency-last_ttft)/output_len`。

| 顺序 | winners TPOT | +SwiGLU TPOT | speedup | TPOT 降低 |
|---|---:|---:|---:|---:|
| winners → SwiGLU | 32.6154 ms | 27.8592 ms | 1.1707× | 14.58% |
| SwiGLU → winners | 32.6729 ms | 27.8213 ms | 1.1744× | 14.85% |
| 两顺序均值 | 32.6442 ms | 27.8402 ms | 1.1726× | 14.72% |

反序轮日志确认候选在 DP0–DP7 各选择一次，基线选择 0 次。首轮候选
selection log 因旧脚本把共享日志 symlink 到 label archive 后又 truncate 而丢失；
结果行仍保留，但不把该轮当作 selection-count 证据。脚本现已去掉共享日志
symlink，并要求候选恰好在 8 个 rank 全部命中。

原始目录：

- `/mnt/b300-shared/home/qinhaiyan/wwxq/bench_results/moe_swiglu_b300_fixedkv240_20260801d`
- `/mnt/b300-shared/home/qinhaiyan/wwxq/bench_results/moe_swiglu_b300_fixedkv240_reverse_20260801e`

两种顺序高度一致，但每种顺序只有一个 request-level 结果行；这证明收益不依赖
单一启动顺序，尚不能替代高 N 置信区间。候选继续保持默认关闭，等待 code review
与新的 8 卡 nsys containing-region trace。

## 4. 配对 nsys containing-region 结果

profiler 只在同一 fixed-KV invocation 完成精确 prefix warmup 后启动。候选与
stock 使用同一模型、S=32768、global BS=128、local M=16、TP8/DP8/EP8、
output=48 和相同服务参数。每条 trace 都包含
`75 layers × 47 decode iterations × 8 ranks = 28,200` 个
W13→SwiGLU→W2 triple。

| 指标（中位数） | stock | valid-CTA | 变化 |
|---|---:|---:|---:|
| SwiGLU activation | 42.304 us | 3.008 us | 14.0638× |
| 可移除 activation critical path | 41.152 us | 1.760 us | -95.72% |
| W13→activation→W2 region | 194.849 us | 132.578 us | 1.4697× / -31.96% |

首条候选 trace 的 28,200 次 activation 中，28,125 次为 grid=128；剩余
75 次全部位于 DP0 最后一个 iteration，且仍为 stock grid=65,536。这个形态
与未审计的小/尾 batch 一致，而服务启动时确实会 capture M=1 bucket，因此它
直接驱动了上述全 bucket 扩展；不能再把稳态 M=16 的成功等同于完整 graph 覆盖。

扩展后重抓的 28,200 次 activation 全部为 grid=128，8 个 rank 各 3,525 次，
physical-slab grid=65,536 为 0。该 measured window 没有再次出现小尾批，因此
不把 grid=8 作为每条 trace 必须出现的条件；M=1 由 `8 ranks` 的启动选择日志、
完整物理 ABI Harness 门禁和 CUDA Graph capture 覆盖。新 trace 相对同一 stock
denominator 的 region 为 194.849→133.152 us，即 1.4634×、降低 31.66%，与
上一条候选 trace 的 31.96% 降幅一致。

原始配对证据：

- `/mnt/b300-shared/home/qinhaiyan/wwxq/bench_results/nsys_moe_swiglu_valid_cta_20260802c`
- `/mnt/b300-shared/home/qinhaiyan/wwxq/bench_results/nsys_moe_swiglu_stock_paired_20260802a`
- `/mnt/b300-shared/home/qinhaiyan/wwxq/bench_results/nsys_moe_swiglu_paired_20260802a/containing_region.json`
- `/mnt/b300-shared/home/qinhaiyan/wwxq/bench_results/nsys_moe_swiglu_paired_20260802a/decode_bottlenecks.json`
- `/mnt/b300-shared/home/qinhaiyan/wwxq/bench_results/nsys_moe_swiglu_graph_buckets_20260802d`
- `/mnt/b300-shared/home/qinhaiyan/wwxq/bench_results/nsys_moe_swiglu_graph_buckets_paired_20260802d/containing_region.json`

同一 decode window 的占比还显示 W13 21.19%、其它 DeepGEMM 20.31%、
DeepEP 17.07%、W2 11.39%、FlashMLA 6.65%。DeepEP low-latency dispatch
跨 rank 启动偏斜 p50=4.784 us、p99=280.974 us，是 graph-bucket 闭环后的
下一个分布式优化目标。NCCL 的 duration 不能直接相加当作 critical path，
后续必须使用 rank-max/containing-region/端到端门禁。

## 5. 被拒绝的基础设施实验

- serving-native 首次启动在 CUDA 前失败：production config 探测无权限的
  `/home/ubuntu/wwxq/cache/sglang/glm52_opt.env`。Harness 现在先固定自己的
  `reference_glm52_opt.env`，同时保护 stock denominator 不继承候选开关。
- 生成任务首次启动误用无 Torch 的 system Python。所有生成 `run.sh` 现在支持
  `KERNEL_HARNESS_PYTHON=/absolute/production/python`，并由 selftest 强制检查。
- 更早的 fixed-KV 尝试中，跨 invocation 预热会被 one-batch flush；以及
  `chunked_prefill_size=16384` 使 token capacity 不足。两类结果均未进入性能结论。

## 6. 复现

```bash
# 固定 KV 端到端；同一 winners stack，只增加一个显式候选
N_RUNS=1 OUT_LEN=240 GLOBAL_BS_LIST=128 LABELS="winners swiglu" \
  PORT=30002 SGLANG_CUDA_GRAPH_MAX_BS=16 CHUNKED_PREFILL_SIZE=2048 \
  bash glm52_opt/scripts/run_moe_swiglu_b300_ab.sh

# Kernel Harness serving-native eager + graph
CUDA_VISIBLE_DEVICES=0 \
KERNEL_HARNESS_PYTHON=/path/to/production/python \
SGLANG_ROOT=/path/to/this/checkout \
  serving_native/run.sh b300_moe_swiglu_quant_decode_m16 \
  --candidate serving_native/candidates/moe_swiglu_valid_cta.py \
  --execution-mode both --warmup 8 --repeat 10
```

全 bucket 扩展已经通过 8 卡 nsys containing-region 门禁并消除了该 replay
中的 65,536-CTA fallback。下一步用 production `low_latency` DeepEP、32K KV、
global BS=128 重跑端到端；只有该门禁才能判断更完整的 graph 覆盖是否对真实
TPOT 有额外可测收益。
