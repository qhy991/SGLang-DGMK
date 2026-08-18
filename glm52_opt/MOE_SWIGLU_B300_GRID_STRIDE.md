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

## 原理解释：不是“少算 CTA”，而是“有限 CTA 循环排空动态任务”

### 1. 这个 kernel 处理什么

它位于 MoE 的两个 GEMM 之间：

```text
W13 GEMM
    -> 对每个 routed row 计算 SiLU(gate) * up
    -> 按 128 个元素一组量化为 FP8，并写 packed UE8M0 scale
W2 GEMM
```

一个 CTA 负责一个 routed row 的 2048 个输出元素。因而 kernel 必须为目的
rank 收到的每个有效 row 写出 FP8 value 和 scale；专家 slab 中未被
`masked_m` 覆盖的 padding 不属于可观察结果，不需要计算。

### 2. 为什么 `M*topk=128` 不能作为正确性上界

在 global BS=128、DP8、topk=8 的 decode 稳态中，每个源 rank 的 local
`M=16`，所以它发出 `16*8=128` 个 assignment。EP all-to-all 只保证 8 个
rank 总计仍有 `8*128=1024` 个 assignment，并不保证每个目的 rank 恰好收到
128 个。生产 trace 的实际分布为：

```text
[143, 124, 114, 137, 133, 126, 114, 133]，总和为 1024
```

因此需要严格区分：

- `source_work = M*topk`：一个源 rank 发出的工作，也是目的 rank 接收量的均值；
- `destination_work = W = sum(masked_m)`：当前目的 rank 真正必须完成的工作；
- `G = gridDim.x`：CTA pool 大小，只是并行度选择。

旧 `cuda_valid_cta` 相当于启动 `G=128` 个 CTA，并让 CTA `b` 只处理 row
`b`。当 `W=143` 时，它只写 row 0--127，row 128--142 没有执行者，因而
性能数字建立在漏算之上，必须作废。

### 3. 当前 grid-stride 下发算法

`cuda_grid_stride` 仍从 host 启动 CUDA Graph 可知的静态 CTA pool，但每个
CTA 会循环领取同一余数类中的工作：

```cpp
G = gridDim.x;  // M16/topk8 时为 128
for (uint32_t r = blockIdx.x; r < W; r += G) {
    auto [expert_id, row_id] = prefix_lookup(masked_m, r);
    swiglu_quant_and_store(expert_id, row_id);
}
```

真实实现不在 host 读取 `W`；CTA 在设备上扫描本次调用的 `masked_m`，同时
得到 `W` 和 `(expert_id,row_id)`。这样不会引入 device-to-host 同步，也不需要
根据动态路由结果改变 CUDA Graph 的 launch shape。

`W=143,G=128` 时，CTA 0--127 先处理 row 0--127，CTA 0--14 再处理
row 128--142。`W=1024,G=128` 时，每个 CTA 各处理八个 row。这里 128 表示
同时存在的工作队伍数量，不表示最多只能完成 128 份工作；它也不是把 128 个
CTA 分成多轮 kernel launch，而是在一次 launch 内循环排空设备端任务表。

### 4. 不漏算、不重算的理由

对任意有效全局工作编号 `r`，`0 <= r < W`，存在唯一分解：

```text
b = r mod G
k = floor(r / G)
r = b + k*G
```

所以它只会被 CTA `b` 的第 `k` 次循环访问一次。不同 CTA 的余数类互不
重叠，而所有余数类的并集覆盖 `[0,W)`，因此 grid-stride 本身既不会漏算，
也不会重算。

设 `P[e] = sum(masked_m[0:e])`。专家 `e` 对应半开区间
`[P[e], P[e]+masked_m[e])`，这些区间不重叠且并集也是 `[0,W)`。每个 `r`
由此唯一映射为：

```text
expert_id = 唯一满足 P[e] <= r < P[e] + masked_m[e] 的 e
row_id    = r - P[e]
```

因此 kernel 每次都按当前 layer、当前 batch、当前 rank 的真实路由表工作，
不会记忆上一批激活过哪些专家，也没有裁剪专家集合。改变的是任务消费方式，
不是 router、专家选择或 SwiGLU 数学公式。

### 5. PDL 为什么必须随循环一起修改

该 kernel 是 W13 的消费者，也是 W2 的生产者，并使用 PDL 保持相邻 kernel
重叠。CTA 改成循环后，如果它在第一份 work 后就向 W2 发 trigger，同一 CTA
的后续 row 仍未写出，消费者可能过早推进。

当前实现让每个有工作的 CTA 只执行一次 PDL wait，并用
`r + gridDim.x >= W` 判断当前 work 是否是该 CTA 的最后一份。只有此时才在
最终 value/scale store 前发 trigger，保持与 stock 相同的 trigger/store
相对顺序。没有 work 的 CTA 直接完成，走 CUDA 的隐式完成路径。

### 6. 为什么会变快

stock 按物理 expert slab 启动 `8192*8=65,536` 个 CTA，而典型目的 rank
只有约 128--143 个有效 row。无效 CTA 虽然不做 SwiGLU 数学计算，仍需要被
GPU 下发和调度，并读取 `masked_m`、执行 prefix scan 和 block 同步后才能
发现自己无效。

新实现通常只提交 128 个 CTA；真实 work 多于 128 时，让少数 CTA 再循环，
真实数学工作量不变，但大幅减少无效 CTA 的创建、调度、扫描和退出成本。这是
一次下发算法修改，不是仅调整 launch 参数。相应地，生产 trace 中 activation
从 42.304 us 降到 3.776 us；该 kernel 在多个 MoE layer 中重复出现，最终在
固定 32K KV、global BS=128 的端到端协议中体现为约 33.30 ms 降到 30.75 ms。

### 7. 给外部专家的审查边界

当前证据支持“目的端工作覆盖正确”和“leaf 数值逐字节一致”，但不能把它直接
表述为完整生产晋级已经通过。审查时应分别检查：

1. `sum(masked_m)` 是否覆盖 W2 可观察的全部目的端 row；
2. prefix scan 到 `(expert_id,row_id)` 的映射是否唯一且不越过 expert slab；
3. `blockIdx.x + k*gridDim.x` 是否在所有 `W` 下完整覆盖且不重复；
4. PDL wait/trigger 和最终 value/scale store 的顺序是否满足消费者语义；
5. 空 CTA、尾批 M1/M2/M4/M8/M12 和 M32 CUDA Graph bucket 是否正确；
6. dispatch gate 是否只允许已审计的 B300、shape、ABI 和 variant；
7. 并发 continuous batching 下尚未闭环的质量差异是否来自该 kernel，还是由
   推进速度改变后续 batch shape、通信顺序或其它 kernel 数值轨迹导致。

因此当前结论分为两层：算法和 leaf 层已通过 route-completeness 与逐字节
门禁；端到端性能稳定观察到约 30.75 ms，但并发质量门禁尚未完全闭环，候选
继续保持 opt-in，不作为默认生产 winner。

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
