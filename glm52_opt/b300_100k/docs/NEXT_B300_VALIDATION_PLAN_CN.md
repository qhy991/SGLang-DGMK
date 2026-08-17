# 下一台 B300：安全、可证伪的复验计划

> 本文是教学型阶段计划；机器可执行的冻结参数、candidate treatment、promotion gate 与停止规则以 [`../workload_contract.json`](../workload_contract.json) 和 [`../REPRODUCTION.md`](../REPRODUCTION.md) 为唯一规范来源。

## 0. 目标与非目标

目标是把“冻结 runtime 上 accepted 的 N6”和“退化 host 上有强信号的 v5”迁移到当前 main，并得到健康机器上的可重复 E2E 结论。

第一轮不做：

- 同时升级 wheel、镜像、模型或 attention backend；
- 把真实 attention CP8 与现有 DP-attention cell 混在同一结论里；
- 先跑 profiler 再解释性能；
- 一次组合多个 research candidate；
- 因某个 leaf 快就跳过 server gate。

## 1. 阶段 A：机器健康预检

在加载大模型前完成：

1. 记录 8 张 GPU 型号、driver、CUDA、NVLink 拓扑、时钟与 ECC/Xid；
2. 记录所有 CPU logical/core/socket/SMT topology；
3. 对每个 CPU frequency domain 做短负载，确认不存在大面积约500MHz异常；
4. 检查 governor、turbo、RAPL、温度和 throttling；
5. 检查容器 IPC、shm、NUMA/cpuset；
6. 跑一个短 GPU collective smoke，确认 NVLink/NCCL/DeepEP 无错误。

如果 host CPU 频率异常，不进入正式性能阶段。修 affinity 不能替代修机器。

## 2. 阶段 B：冻结当前 main

记录：

- Git HEAD 与 `git status --short`；
- 容器 image digest；
- Python/CUDA/Triton/DeepEP/FlashMLA/DeepGEMM 版本；
- model config/index/dataset hash；
- tokenizer、quant artifacts、weight shards/revision；
- runtime source contract；
- server/client 命令与环境变量。

当前归档的 full weight/tokenizer/quant revision 仍不完整，新机器复验时应补齐。

## 3. 阶段 C：代码与路径正确性

1. 运行 `test_moe_fused_gate_pad_mask.py`；
2. 运行 `test_moe_fused_gate_static_placement.py`；
3. 校验 N6 map SHA：
   `36d13233672288317fd69495d4cedb46844b8ae99033d184d84aff0c99c68f09`；
4. baseline 启动时确认 N6 marker 不出现；
5. candidate 启动时确认 static-placement selected marker 出现；
6. 对同一输入做 exact token 与 logprob tolerance 检查；
7. 核对 server args diff 只有 treatment。

## 4. 阶段 D：先恢复健康 N6 anchor

冻结 cell：

| 项目 | 值 |
|---|---|
| GPU | 8×B300 |
| 模型 | GLM-5.2-FP8 |
| 并行 | TP8/DP8/EP8，`attn_cp_size=1` |
| 请求 | logical prefix 90000、actual hit 89984、suffix 10000、output1 |
| 压力 | 110 requests、concurrency11、request-rate888888 |
| KV/chunk | page64、global80384、per-rank ceiling10048 |
| runtime | FlashMLA-KV、DeepEP normal、mem0.82、overlap/graph off |

先跑 identity+136SM baseline。只有 baseline 同时满足：

- 110/110；
- P50≤2000 ms；
- P90≤5000 ms；
- total-token throughput≥438000；
- 路径 marker、cache hit、shape 正确；
- 重复运行无异常漂移；

才允许解释后续候选的绝对性能。否则优先诊断机器/runtime drift。

## 5. 阶段 E：当前 main 的 N6 正式复验

顺序：

1. correctness；
2. development A-B-A；
3. 若 primary P50≥1%、P90回退≤2%、吞吐回退≤1%，进入五对；
4. 五对要求至少4/5 P50 wins；
5. 再跑独立 client seed=20260813 holdout；
6. 最后按需做 matched Nsys，不能先用 profiler 数字晋级。

如果 main 的结果没有复现旧 N6，优先检查：

- selected marker 是否缺失；
- 当前 main 与冻结 22-file runtime contract 的差异；
- map 方向和 physical/logical ID 语义；
- DeepEP config 是否同时作用于 dispatch/combine；
- cache hit、chunk、M shape 是否漂移；
- host CPU/GPU 是否健康。

## 6. 阶段 F：v5 正式复验

v5 必须建立在已恢复的 N6 anchor 上，同一 affinity、同一镜像、同一 runtime：

1. control N6；
2. candidate v5 map；
3. control N6；
4. 检查 control drift；
5. correctness 使用 seed1 train、seed0 holdout、seed2 external exact replay 的既有合同；
6. 五对与独立 holdout；
7. matched Nsys 验证 notify tail 与 progress signal 是否复现。

禁止把退化 host 的 -68.89% 当预期健康收益。健康机器上很可能更小；只要稳定超过门槛且不牺牲正确性/tail，就有价值。

## 7. 阶段 G：FlashMLA band

只有 N6/v5 server path 稳定后再测：

1. 先对 8 个 admitted M 逐 shape 跑 output/LSE correctness；
2. 每个 shape 交错 stock/candidate 测量顺序；
3. 范围外必须确认走 stock fallback；
4. selected shape 异常必须停止，不允许无 marker 静默污染；
5. 再在同一健康 N6/v5 anchor 上做 no-profiler E2E。

Leaf 8.82%–12.49% 只是先验；晋级仍要求 P50 和服务门槛。

## 8. 真实 attention CP8 是新 cell

如果下一步仍要求“CP=8、EP=8”，应明确建立新的 attention CP8 合同，而不是继续沿用本报告的 DP-attention 口径。需要重新冻结：

- CP group 与 TP/DP/EP 的重叠关系；
- KV 分片与 attention collective；
- 每 rank Q/KV shape；
- cache ownership；
- chunk 和 memory capacity；
- 正确性 reference；
- baseline 绝对门槛。

现有 N6 map/router primitive可能可复用，但 100K TTFT 数字必须重新测。

## 9. 停止规则

出现任一情况立即停止当前 candidate：

- correctness 不一致、NaN、非法内存访问；
- selected marker 缺失或 fallback 未被记录；
- anchor drift 超过预设门槛；
- 前两/三对已数学上不可能达到4/5 wins；
- P50 回退、吞吐明显回退，即使 P90变好；
- profiler 扰动被误当正式性能；
- host health 不合格；
- treatment diff 包含未声明配置变化。

这种停止不是“实验失败”，而是避免把不可比较结果写进事实账本。
