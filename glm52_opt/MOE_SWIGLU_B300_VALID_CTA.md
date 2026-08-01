# B300 Masked MoE SwiGLU+Quant：valid-CTA 优化与固定 KV 验证

- 日期：2026-08-02
- 机器：B300-M2，8×NVIDIA B300 SXM6 AC，TP8/DP8/EP8
- 模型：`/mnt/b300-shared/models/GLM-5.2-FP8`
- 基线 commit：`fc4b5d22f2dbf5e82e6a5012cda9765de272daa6`
- 候选：`SGLANG_OPT_MOE_SWIGLU_QUANT_VARIANT=cuda_valid_cta`
- 默认行为：**关闭**；任一契约不匹配即回退原始实现

## 1. 修改是什么

B300 的 DeepEP low-latency decode 保留物理 expert slab
`gateup_output=[32,8192,4096]`，但 global BS=128 / DP8 时每卡只有
`M=16`，即 `M*topk=128` 个真实 routed assignment。

原 wrapper 按物理容量启动 `8192*8=65536` 个 CTA；大部分 CTA 读取
`masked_m` 后立即退出。候选完整复用 stock CUDA kernel body、FP32 SiLU、
FP8 E4M3 输出以及 packed int32 UE8M0 写法，只把 grid 改为
`num_real_tokens*topk=128`。因此它不是近似算法，也没有缩小生产 buffer。

自动选择同时要求：

- compute capability `(10,3)` 与物理 slab `T=8192`；
- BF16 contiguous `[32,8192,4096]` 输入、contiguous int32 `[32]` mask；
- `group_size=128`、`topk=8`、host-known `M=16|32`；
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

## 4. 被拒绝的基础设施实验

- serving-native 首次启动在 CUDA 前失败：production config 探测无权限的
  `/home/ubuntu/wwxq/cache/sglang/glm52_opt.env`。Harness 现在先固定自己的
  `reference_glm52_opt.env`，同时保护 stock denominator 不继承候选开关。
- 生成任务首次启动误用无 Torch 的 system Python。所有生成 `run.sh` 现在支持
  `KERNEL_HARNESS_PYTHON=/absolute/production/python`，并由 selftest 强制检查。
- 更早的 fixed-KV 尝试中，跨 invocation 预热会被 one-batch flush；以及
  `chunked_prefill_size=16384` 使 token capacity 不足。两类结果均未进入性能结论。

## 5. 复现

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

下一门禁是新的 8 卡 nsys：比较 W13→activation→W2 containing region，而不是
只看 leaf。旧 winners trace 的 activation median 为 42.305 us、region 为
194.722 us，约 41.153 us/layer 位于可移除 critical path。
