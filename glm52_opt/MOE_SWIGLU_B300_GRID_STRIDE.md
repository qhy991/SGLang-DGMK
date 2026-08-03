# B300 MoE SwiGLU+Quant route-complete grid-stride 优化

日期：2026-08-03

机器：B300-M2，8×NVIDIA B300 SXM6 AC，TP8/DP8/EP8

模型：`/mnt/b300-shared/models/GLM-5.2-FP8`

## 结论范围

本实现替代已作废的 `cuda_valid_cta`。旧实现把源 rank 的 `M*topk` 当成
目的 rank 的工作上界；真实 EP all-to-all 后，每个目的 rank 的工作量是设备端
`sum(masked_m)`。在 source M16/topk8 的生产诊断中，8 个目的 rank 分别收到
`[143,124,114,137,133,126,114,133]` 个 assignment，因此只执行 128 个
非循环 CTA 会漏算合法 row。

新 variant 名为 `cuda_grid_stride`：

1. host 仍按 CUDA Graph 已知的 `num_real_tokens*topk` 启动静态 CTA pool；
2. CTA 以 `work_id += gridDim.x` 循环消费设备端 prefix scan 给出的全部
   `sum(masked_m)` work；
3. CTA pool 只控制吞吐，不再充当正确性上界；
4. PDL wait 每个 CTA 只执行一次，但 trigger 必须放在该 CTA 最后一个 work
   的最终 store 前，即 `work_id + gridDim.x >= total_work`。否则 W2 可能在
   同一 CTA 的后续 row 尚未写完时启动。

这不是把同样数量的 CTA 拆成多轮 launch，也没有记住某一批激活过的专家。
每次调用、每个 CTA 都重新扫描当前设备上的 `masked_m` prefix sum，专家 ID
和 row ID 可随 layer、request 与 batch 改变。典型 143-row 情况只向 scheduler
提交 128 个 CTA，其中 15 个 CTA 多执行一轮；不是提交 65,536 个 CTA 后让
绝大部分自行退出。单专家 1024-row 压测则由同一 128-CTA pool 各执行八轮。

stock 的 BF16 输入、FP32 SiLU、FP8 E4M3 输出、packed int32 UE8M0 scale
和预分配 output ABI 均保持不变。候选默认关闭，只有 B300 `(10,3)`、生产
`[32,8192,4096]` slab、允许的 graph bucket 和显式 allowlist/variant 同时
满足时才选择。

## 正确性与 leaf 门禁

独立 B300 validator 对 active FP8 code 和 packed scale byte 做逐字节比较：

| 目的 rank work | CTA pool | candidate | stock | mismatch |
|---:|---:|---:|---:|---:|
| 128 | 128 | 14.336 us | 45.760 us | 0 |
| 143 | 128 | 13.856 us | 44.640 us | 0 |
| 256 | 128 | 14.176 us | 45.024 us | 0 |
| 1024 | 128 | 17.184 us | 45.088 us | 0 |

Kernel Harness 新增两个独立任务，而不是修改原有 source-balanced proxy：

| route profile | work 定义 | 7-shape geomean | 最小保守 speedup |
|---|---|---:|---:|
| `ep_skew` | M16 目的 work=143 | 9.1493x | 8.4163x |
| `ep_hotspot` | M16 单专家 work=1024 | 3.4216x | 3.2098x |

两项均覆盖 M=`1/2/4/8/12/16/32`，7/7 WIN、`calc_diff=0`、初始与未见
seed 正确、无回归。Harness 会在 timing 前冻结 bridge、Python dispatch、
候选 `.cuh` 和 stock `.cuh` 共 5 个文件，避免结果只识别桥接脚本。

serving-native 的 eager/CUDA Graph 双 lane 也通过：偏斜为
1.9492x/3.6900x（paired p10 1.8082x/3.5583x），单专家热点为
1.9826x/3.2690x（paired p10 1.8555x/3.1110x）。这些仍是 leaf 结论，不能
直接外推 32K KV 下的 TPOT。

## 8 卡、32K KV profiler

工作负载为 S=32768、global BS=128、local M16、output=48，prefix cache
命中率 0.998。28,200 个 W13 -> activation -> W2 triple 的结果为：

| 中位数 | stock | `cuda_grid_stride` | 变化 |
|---|---:|---:|---:|
| activation | 42.304 us | 3.776 us | 11.2034x |
| 可移除 activation critical path | 41.152 us | 2.144 us | -94.79% |
| W13-to-W2 containing region | 194.849 us | 155.489 us | 1.2531x / -20.20% |

grid audit 为 28,125 次 grid=128、75 次尾批 grid=8、0 次 grid=65,536。
因此新 trace 同时验证稳态与尾批 graph bucket，且没有退回物理 slab launch。

优化后剩余的高占比方向是 W13 23.34%、其它 DeepGEMM 19.37%、DeepEP
16.02%、W2 12.43%、FlashMLA 6.33%。DeepEP dispatch 的 rank duration CV
为 15.59%，rank start skew 为 p50 4.57 us、p99 295.59 us，并出现 4.90 ms
最大 outlier；这是后续通信/计算负载均衡调查的首要对象。上述 summed duration
包含 overlap/wait，只用于排序，不能代替 rank-max 与端到端门禁。

远程证据：

- `/mnt/b300-shared/home/qinhaiyan/wwxq/bench_results/nsys_moe_swiglu_grid_stride_pdl_safe_20260803b`
- stock denominator：`/mnt/b300-shared/home/qinhaiyan/wwxq/bench_results/nsys_moe_swiglu_stock_paired_20260802a`

## 32K KV 端到端门禁

当前晋级实验严格固定为 B300-M2 8 卡、S=32768、global BS=128、local M16、
output=48、32704/32768 prefix cache hit。启动顺序为
`winners_before -> swiglu -> winners_after`，每臂独立启动服务、2 个 warmup、
30 个不同 suffix 的 128-request 批次。三个 arm 的 30 个 prompt-set hash 必须
逐项相同，另用未计时的 128×48 greedy output probe 比较全部 6,144 个 token ID。

第一轮 random-ID 30-batch A/B/A 的 ITL 为 31.4401 -> 28.8896 ->
31.4481 ms，中点下降 8.1240%，但三臂的 6,144-token hash 全部不同，且两个
baseline 也不相同；旧 runner 只保留 hash，无法定位，因此该轮只保留为
performance-only 证据。

随后用 B300-M2 上 94,145 条 ShareGPT 对话构建固定 token plan。seed=42
reservoir sample 8,192 条，实际 tokenize 3,869 条；每个 arm 使用 128 条
32,704-token 自然 prefix、2 个 warmup、30 个测量 suffix 和 1 个正确性
suffix。plan SHA256 为
`a4e1b1668ef1121242492439094c8c53fc72aef97e6aff213339fdfa6abc0fea`，
每个测量 batch 的 cache hit 都精确为 0.998046875。

自然数据 ITL 为 33.3026 -> 30.7472 -> 33.2900 ms，A/B/A 中点下降
2.5491 ms（7.6559%），paired-median bootstrap CI95 为
[2.5495, 2.5639] ms。自然 token 的 baseline 比 random ID 慢约 1.86 ms，
实证说明内容/路由分布会改变真实负载，不能只测一组 synthetic IDs。

完整输出 ID 显示 baseline-before 与 baseline-after 自身相差 4,584/6,144
token；candidate 到两侧 baseline 分别相差 4,551 和 4,697，first-token
mismatch 分别为 14、17，而 baseline 自身为 18。candidate 偏离处在 baseline
非确定性的同一量级，不能从 hash 把错误归因到本 kernel；但 exact gate 仍未
通过。因此 7.6559% 只能作为强 performance/headroom 结果，
`cuda_grid_stride` 继续保持 opt-in，尚不进入生产 winner 集合。

## 测试脚本来源

当前 fixed-KV A/B/A 的流量脚本不是 SGLang upstream 自带 benchmark：编排入口
是项目内的 `glm52_opt/scripts/run_moe_swiglu_b300_ab.sh`，请求发生器是
`run_fixed_kv_decode_series.py`，ShareGPT token plan 也由项目脚本生成。服务端则
是实际的 `sglang serve`，使用真实权重、TP8/DP8/EP8、FP8 KV、FlashMLA 和
DeepEP，并通过公开的 `/generate`、`/flush_cache`、`/metrics`、`/server_info`
接口测量。

自定义 runner 的 latency、last-request TTFT、output throughput 和 metrics
cache-hit 算法与仓库内置的
`python -m sglang.benchmark.one_batch_server` 一致；新增部分是一次构建 KV 后
连续测量不同 suffix、严格 A/B/A 配对、逐请求 cache counter 和完整 output ID
门禁。baseline 为之前已经晋级的 `combined_winners`，不是纯 upstream stock；
只有中间臂额外打开本次 SwiGLU variant。另保留一轮内置
`one_batch_server` 同条件结果作为脚本来源交叉检查，但单行结果不替代 30-batch
门禁。

该交叉检查实际直接调用
`python -m sglang.benchmark.one_batch_server`；解析到的模块为
`python/sglang/benchmark/one_batch_server.py`，49,609 bytes，SHA256
`fae46de1b87447bc637dead8618c662cf59c18b9e29b5999223ffc5beb9fd139`。
在同一 B300 TP8/DP8/EP8、S=32768、global BS=128、out=48 条件下，
`combined_winners` 与候选的单行 ITL 分别为 31.765 ms 和 29.233 ms，候选低
2.531 ms（7.97%，1.0866x）。方向和 30-batch 自然 token 结果一致；因为缺少
A/B/A、请求多样性和完整输出门禁，它仍只作为来源与量级交叉验证。

## 50 题标注质量门禁

为避免只比较无标签 output hash，另从 upstream GSM8K `test.jsonl` 固定抽取
5-shot 之后的前 50 题（dataset index 5--54）。数据集 1,319 行、749,738
bytes，SHA256 为
`3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`。
请求显式固定 temperature=0、top-p=1、max tokens=512、50 threads 和
`enable_thinking=false`；任何空 response 都使整臂失效。三臂均无空 response。

| arm | GSM8K | elapsed（仅诊断） |
|---|---:|---:|
| `winners_before` | 49/50（98%） | 10.366 s |
| `swiglu` | 48/50（96%） | 7.101 s |
| `swiglu` 独立复跑 | 48/50（96%） | 6.054 s |
| `winners_after` | 49/50（98%） | 7.282 s |

两个 baseline 虽有 32/50 条完整文本不同，但抽取答案和 correctness 逐题
50/50 相同，均只错 index 12。两次候选的完整文本也只有 15/50 完全相同，
但抽取答案逐题 50/50 相同，均额外把 index 37 的正确答案 2 回答为 0。
所以在这个 50 并发协议下，`98% -> 96% -> 98%` 是可复现的候选相关差异，
不能称为与 baseline 指标一致。

进一步只重放 index 37、`num_threads=1`、CUDA Graph max BS=1；三臂都回答
2，得分均为 1/1。候选日志确认 8 个 rank 都实际选择 M1 grid-stride。
这排除了该问题上的确定性单请求数学错误，说明 50 并发差异依赖 continuous
batching 的入批/尾批 shape 演化：候选改变推进速度后，后续 batch 组成与其它
kernel shape 可发生变化，并放大模型数值非确定性。50 题不足以证明总体准确率
下降，但 exact metric 门禁没有通过，因此生产晋级仍保持阻断。该轮 GSM8K
latency 也不是 32K KV 性能协议，不能替代前述 ITL 结果。

## 复现

```bash
# 设备端 route-completeness validator
python glm52_opt/scripts/validate_moe_swiglu_route_complete_b300.py

# 8 卡 fixed-KV nsys
RUN_ID=nsys_moe_swiglu_grid_stride_pdl_safe_20260803b \
SWIGLU_MODE=candidate GLOBAL_BS=128 S=32768 OUT_LEN=48 \
bash glm52_opt/scripts/run_moe_swiglu_b300_nsys.sh

# 30-batch A/B/A
python glm52_opt/scripts/build_fixed_kv_sharegpt_plan.py \
  --dataset /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
  --model-path /path/to/GLM-5.2-FP8 \
  --output /path/to/fixed_kv_sharegpt_plan.bin \
  --batch-size 128 --input-len 32768 --prefix-len 32704 \
  --runs 30 --warmup-runs 2 --seed 42

RUN_ID=moe_swiglu_grid_stride_pdl_safe_sharegpt_fixedkv_aba30_20260803a \
N_RUNS=30 FIXED_KV_SERIES=1 FIXED_KV_WARMUP_RUNS=2 \
FIXED_KV_TOKEN_PLAN=/path/to/fixed_kv_sharegpt_plan.bin \
OUT_LEN=48 GLOBAL_BS_LIST=128 \
LABELS="winners_before swiglu winners_after" \
SGLANG_CUDA_GRAPH_MAX_BS=16 \
bash glm52_opt/scripts/run_moe_swiglu_b300_ab.sh

# GSM8K 50-question labeled A/B/A quality gate
GSM8K_EVAL=1 \
GSM8K_DATA_PATH=/path/to/gsm8k/test.jsonl \
GSM8K_NUM_EXAMPLES=50 GSM8K_NUM_SHOTS=5 GSM8K_START_OFFSET=0 \
GSM8K_NUM_THREADS=50 GSM8K_MAX_TOKENS=512 \
GSM8K_ENABLE_THINKING=0 \
LABELS="winners_before swiglu winners_after" \
bash glm52_opt/scripts/run_moe_swiglu_b300_ab.sh

# Isolate GSM8K dataset index 37 with one request (5 + 32 = 37)
GSM8K_EVAL=1 \
GSM8K_DATA_PATH=/path/to/gsm8k/test.jsonl \
GSM8K_NUM_EXAMPLES=1 GSM8K_NUM_SHOTS=5 GSM8K_START_OFFSET=32 \
GSM8K_NUM_THREADS=1 GSM8K_MAX_TOKENS=512 \
GSM8K_ENABLE_THINKING=0 SGLANG_CUDA_GRAPH_MAX_BS=1 \
LABELS="winners_before swiglu winners_after" \
bash glm52_opt/scripts/run_moe_swiglu_b300_ab.sh
```
