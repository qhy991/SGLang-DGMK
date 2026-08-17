# GLM-5.2 / B300：N1–N40 完整实验总账

> 本文面向 CUDA 与推理优化初学者，目标不是只列“快了多少”，而是说明每个候选改了什么、为什么可能有效、证据能证明到哪一层，以及为什么最终接受或拒绝。

## 0. 权威来源与适用范围

NATIVE_OPTIMIZATION_DECISIONS_20260813.md (`private-archive:evidence/by_experiment/glm52_mok_followup_candidates_20260813/native_candidates/NATIVE_OPTIMIZATION_DECISIONS_20260813.md`) 是 N 编号、primitive、证据阶段与裁决的权威表；冻结 workload、seed 和门槛以 [`../workload_contract.json`](../workload_contract.json) 为准；性能与 correctness 数字以对应 samples/summary/correctness 文件为准。本文是从这些来源派生的教学账本，不取代原始结果。若早期笔记使用 provisional 编号或不同映射，应以权威决策表为准。

GitHub 只公开 N6/v5 的脱敏 compact evidence；其余 `private-archive:` 标识用于在私有归档中恢复历史证据，不是仓库内链接。

冻结测试合同如下：

- 模型路径：GLM-5.2 的冻结服务栈，SGLang reviewed branch `1571b72db014603ebfaad6d59cbbde1740b23b3e`，22 文件 immutable runtime contract。
- 容器：`sglang_0515_optimized`，镜像 `sha256:4484ab841baa40eb89a7ee187110877982c44cd7da06f8a8469c8e0059e28bdd`。
- 并行：TP8 / DP8 / EP8，DP attention，attention TP1；这里没有启用 CP8。因此本文数字不能被表述为“CP8 结果”。
- workload：约 90K shared prefix + 10K suffix、每请求输出 1 token，x11 concurrency、110 requests；formal client primary seed=0，holdout client seed=20260813，server seed=565849983。
- chunk：global 80384、每 rank 10048；memory fraction 0.82。
- 基线：`SGLANG_GLM52_OPT=1`、`serving_safe`、E2E prefill indexer 与 paged/clustered MQA 打开、MoK 关闭。
- 这是长上下文 **prefill / TTFT** 优化，不是 decode / ITL 优化。虽然部分候选来自 decode 经验，只有在上述 prefill 合同上重新验证后才有资格进入栈。

## 1. 如何读懂证据等级

性能优化最常见的误判，是把“小算子变快”直接写成“服务变快”。本文严格使用以下层次：

1. **静态 / Amdahl 审计**：只检查候选是否命中实际路径、理论上最多能省多少；可以不启动模型。它适合尽早淘汰错边界或收益上限过小的想法。
2. **leaf / 算子证据**：隔离一个 CUDA kernel 或算子测延迟，并检查 bit-exact 或容差。它只证明局部实现，不证明 TTFT。
3. **model correctness**：真实模型请求的 token 和 selected-token logprob 对比。它证明语义未明显改变，但仍不等于性能通过。
4. **development bracket**：通常按 `N6 → candidate → N6` 夹测，先看控制组漂移，再用约 1% P50 门槛筛选。它是开发筛选，不是正式晋级。
5. **formal**：fresh-server、预声明 AB/BA 配对，最多五对；门槛为 P50 至少 4/5 胜、P50 中位改善至少 1%、P90 回退不超过 2%、吞吐回退不超过 1%、P50≤2000 ms、P90≤5000 ms。
6. **holdout**：独立 seed 的复验，防止只对一个请求序列有效。
7. **profile**：通过门槛后再用 Nsight 做因果解释。profile 会扰动时间，因此 profiled TTFT 不进入正式性能结论。

文中“accepted”表示在当前合同下通过了相应晋级要求；“research only”表示它仍可作为后续研究线索，但不能进入已接受基线。

## 2. N1–N6：从局部想法到第一个可接受组合

### N1 — 仅做 balanced static expert placement

- **改动与机制**：静态重排专家到物理 GPU，目标是让各卡收到的 token 更均匀，降低最忙 rank 的 MoE 时间。
- **证据阶段**：五对 fresh-server screen；不是 leaf kernel 结论。
- **correctness**：权威表未列独立 token/logprob 数值，不能补写“11/11 exact”；该轮核心证据是服务性能筛选。
- **关键结果**：P50 2051.07→1962.35 ms，快 4.33%；吞吐提高 4.24%；但 P90 3055.06→3336.21 ms，慢 9.20%；P50 5/5 胜。
- **裁决**：**拒绝**，因为尾延迟明显越过 P90 门槛；不并入 baseline。
- **可学到什么**：平均负载更平衡，不代表尾部同步等待更短。分布式 MoE 的慢 rank、通信相位和偶发拥塞会支配 P90。

### N2 — DP scheduler 状态只做一次 D2H 传输

- **改动与机制**：合并 scheduler 的 device-to-host 状态读取，希望减少 CPU/GPU 控制面往返。
- **证据阶段**：五对服务 screen。
- **correctness**：权威表未给单独 token/logprob 数字；候选能运行完性能请求，但这不应被扩写为严格 correctness 证明。
- **关键结果**：P50 2047.67→2052.46 ms，慢 0.23%；P90 慢 3.16%；吞吐低 5.94%；仅 1/5 P50 胜。
- **裁决**：**拒绝**，P50、P90、吞吐均未过门槛。
- **可学到什么**：少一次 D2H 不一定减少关键路径；如果读取原本被隐藏、或合并引入等待，控制面“看起来更少”也可能更慢。

### N3 — DP-attention local control broadcast

- **改动与机制**：把 DP-attention 的局部控制状态做 broadcast，减少各 rank 重复生成或读取控制信息。
- **证据阶段**：model correctness + 五对服务 screen。
- **correctness**：11/11 token exact；selected-token logprob 最大/平均差 1.49e-4 / 2.70e-5。
- **关键结果**：P50 2014.99→2052.95 ms，慢 1.88%；P90 慢 5.24%；吞吐低 3.59%；仅 1/5 P50 胜。
- **裁决**：**拒绝**；未进入 profiler。
- **可学到什么**：broadcast 自带同步与通信成本。删除重复工作之前，要比较“被删计算”与“新同步”的实际关键路径代价。

### N4 — balanced placement 与 direct physical-ID router 原子组合

- **改动与机制**：同时使用平衡专家映射，并让 router 直接产生物理 expert ID，避免路由后再做额外 ID 置换。由于两者语义耦合，作为一个原子候选评估。
- **证据阶段**：operator/model correctness + 五对服务 screen。
- **correctness**：算子和模型正确性通过；权威表没有展开 token/logprob 数字。
- **关键结果**：arm median P50 2035.57→1993.82 ms，快 2.05%；P90 慢 0.53%；吞吐 478593.75→465113.32 token/s，低 2.82%。配对统计则显示 P50 快 1.94%、吞吐高 2.03%，暴露出明显非平稳性。
- **裁决**：**拒绝**，按预声明规则使用 arm-median 时吞吐回退超过 1%；不做 holdout/profile，也不能事后换统计口径或把收益只归因于 router。
- **可学到什么**：先冻结统计规则很重要。看到两个统计口径方向不一致时，应保留“不稳定”这一事实，而不是选择更好看的数字。

### N5 — SUM_LEN direct-output DP allgatherv + symmetric reduce-scatterv

- **改动与机制**：尝试在 attention/MoE 边界用变长 collective，避免 padding 或中间输出搬运。
- **证据阶段**：静态路径与 Amdahl 审计；未进入 GPU 性能测试。
- **correctness**：没有实现后的模型 correctness，因为静态审计已判定候选不命中有价值边界。
- **关键结果**：DeepEP sparse 与 dense `moe_dense_tp_size=1` 走 SCATTERED；attention TP1 使 78 层 attention→MLP 边界几乎无此通信；剩余 gather 位于 terminal LM-head/logits，且只有 1 token。
- **裁决**：**拒绝**。脚本保留，但候选作用在错误或可忽略的边界。
- **可学到什么**：写 CUDA 或 collective 代码前先做“路径频率 × 单次成本”审计，常常比优化实现本身更有价值。

### N6-development — 共享 DeepEP dispatch/combine SM 数的局部搜索

- **改动与机制**：在 N4 组合上调整 DeepEP normal 的共享 SM 配额。因为现有实现 dispatch/combine 共用一个配置，测试 120、144，并以 136 为锚点。
- **证据阶段**：model correctness + development bracket；这些百分比不是正式晋级数字。
- **correctness**：120/144 相对 fused136 均为 11/11 token exact。
- **关键结果**：顺序 `136,120,144,136`。相对两个 136 锚点中位数，120 的 P50 快 1.50%、P90 快 12.80%、吞吐高 9.74%；144 的 P50 慢 0.68%。
- **裁决**：开发阶段选出 **120**，随后必须回到原始 identity/136 基线做全新的 formal，而不是把 bracket 当作接受证据。
- **可学到什么**：SM 配额不是“越多越快”。更多通信SM**可能**与GEMM/attention竞争；该sweep只证明当前组合下120优于所测邻点，matched profile没有隔离SM单变量机制。

### N6-final — balanced map + physical router + DeepEP 120/120

- **改动与机制**：将 N4 的 expert map/router 原子组合与 N6-development 选出的共享 120 SM 组合，扩大优化边界，让负载、路由和通信资源配置共同优化。
- **证据阶段**：model correctness + 完整 formal (`private-archive:evidence/by_experiment/glm52_mok_followup_candidates_20260813/native_candidates/router_static_fusion_deepep120_formal/`) + 独立 holdout (`private-archive:evidence/by_experiment/glm52_mok_followup_candidates_20260813/native_candidates/router_static_fusion_deepep120_holdout_seed20260813/`)。
- **correctness**：相对两个 identity baseline 均为 11/11 token exact。
- **关键结果（formal，seed 0）**：P50 2042.12→1933.67 ms，快 5.31%；P90 3035.39→2891.13 ms，快 4.75%；吞吐 452544.29→486640.21 token/s，高 7.53%；P50 4/5 胜。
- **关键结果（holdout，seed 20260813）**：P50 2043.91→1932.27 ms，快 5.46%；P90 快 6.73%；吞吐高 7.40%；P50 5/5 胜。
- **裁决**：**accepted**，成为后续候选的控制栈。仍需注意 P90 噪声，并在更多 workload cell 验证，不能外推为所有场景都快 5%。
- **可学到什么**：局部回退可以换取更大组合空间。N1 或 N4 单独不合格，但与通信 SM 共同调优后，组合跨过了全部门槛。

### N6-profile — 通过后才做 Nsight 因果分析

- **改动与机制**：没有新增优化；对 N6-final 与基线做匹配的 10 秒 CUDA+NVTX profile，解释为什么它有效。
- **证据阶段**：profile / causal evidence (`private-archive:evidence/by_experiment/glm52_mok_followup_candidates_20260813/native_candidates/router_static_fusion_deepep120_nsys/`)；profiled TTFT 明确不计入正式性能统计。
- **correctness**：沿用 N6-final 已通过的模型 correctness；profile 本身不是新的 correctness gate。
- **关键结果**：匹配的 10 秒窗口中，候选每设备推进了 654 次 dispatch、基线 616 次，即候选多推进约 6.17%；`cached_notify_combine` P50/P90 分别低 43.65%/45.91%；combine kernel P50 基本不变；每 GPU 去掉约 316–466 次 post-router padded-ID mask 工作。
- **裁决**：因果解释 **accepted**。下一研究方向是 dispatch/combine 独立 SM，但现有 DeepEP 配置约束使其不能直接实现。
- **可学到什么**：Nsight 的作用是解释已复现的 E2E 结果，不是用 profile 中受扰动的 TTFT 替代正式 benchmark。

## 3. N7–N15：安全 leaf 优化、错误边界与 Amdahl 淘汰

### N7 — FP8 indexer-Q 线程数 128→256

- **改动与机制**：提高 FP8 indexer-Q kernel 的线程并行度，希望加速长序列索引器。
- **证据阶段**：leaf + model correctness + Amdahl 审计；未进入 formal。
- **correctness**：六个算子 case bit-exact；模型 11/11 token exact，相对 N6 及两个 identity 的最大 logprob 差不超过 2.60e-4。
- **关键结果**：B=10048 的 leaf 快 26.5%–29.1%；但 N6 trace 中该算子每设备 10 秒内只有 9.17 ms，理论最多省约 2.7 ms，达不到约 1% E2E 门槛。
- **裁决**：作为安全 primitive **保留研究**，但不启动正式 E2E 晋级。
- **可学到什么**：非常漂亮的 kernel 加速也可能没有服务价值；先乘以实际时间占比再决定是否继续。

### N8 — dispatch 与 combine 使用独立 SM 数

- **改动与机制**：首次尝试 dispatch=120、combine=136，想让两个通信阶段分别占用最合适的 SM。
- **证据阶段**：兼容性/结构审计；在 warmup 就失败。
- **correctness**：没有模型请求。`DeepEPConfig` 要求二者相等，因为共享 QP 和 buffer layout。
- **关键结果**：触发配置 assertion；不存在可报告的 N8 性能数字。
- **裁决**：**拒绝当前实现**。若要继续，需设计新的 buffer/QP 实现，属于扩大工程边界而不是删除 assertion。
- **可学到什么**：配置约束往往编码了底层内存布局不变量；绕过检查可能造成 silent corruption。

### N9 — 合法的共享 SM 搜索：112/120/128

- **改动与机制**：在 N8 不可行后，退回合法的一维共享 SM 搜索。
- **证据阶段**：model correctness + development local search。
- **correctness**：112 与 128 候选均为 11/11 token exact。
- **关键结果**：120 锚点 P50/P90/吞吐为 1881.515/2804.51 ms/503043.06 token/s。112 的 P50 慢 2.12%、P90 慢 0.78%；128 的 P50 慢 6.29%、P90 慢 9.85%、吞吐低 8.08%。原始后处理因 GNU awk 保留变量出错，修复后从既有日志重算，无需重跑。
- **裁决**：**拒绝 112/128，保持 120**。
- **可学到什么**：参数搜索应包含锚点复测；日志后处理 bug 与 GPU 实验本身要分开判断。

### N10 — 用 FlashInfer top-k 替换 DSA SGL kernel

- **改动与机制**：在 M=10048、width=100032、K=2048 的真实形状，用 FlashInfer top-k 替换 SGL radix top-k。
- **证据阶段**：leaf correctness/performance + Amdahl；未进模型服务。
- **correctness**：16 行抽样与 `torch.topk` 完全一致；FlashInfer 与近似 SGL radix 的 20.58M 输出中仅 88 个元素不同。
- **关键结果**：SGL 2.4461 ms；FlashInfer graph/eager 约 4.2940/4.2949 ms，约慢 43%。top-k 只占 N6 trace 约 2.7%，方向本身已错误。
- **裁决**：**拒绝**，不做模型/source/profile。
- **可学到什么**：更精确或更通用的库实现不一定适合特定大宽度形状；替换前必须在真实 shape 上 leaf 对照。

### N11 — 显式切换为 `flashmla_sparse`

- **改动与机制**：只改变 sparse attention backend，测试 FlashMLA sparse 是否优于当前路径。
- **证据阶段**：model correctness + development bracket。
- **correctness**：11/11 token exact；logprob 最大/平均差 1.42e-4/3.19e-5。
- **关键结果**：控制 1891.15/2826.85 ms/499383.23 token/s；候选 2090.68/3266.26 ms/445727.68 token/s，即 P50 慢 10.55%、P90 慢 15.54%、吞吐低 10.74%。
- **裁决**：**拒绝**；不做 formal/profile，继续保留 `flashmla_kv`。
- **可学到什么**：同一品牌下的不同 backend 对应不同数据布局与工作负载；不能因为 KV 路径有效就推断 sparse 路径也有效。

### N12 — 将 native fused shared expert 作为第九条本地 MoE route

- **改动与机制**：把 shared expert 融入本地 routed experts，意图用单次 fused MoE 计算减少 launch/读写。
- **证据阶段**：同合同历史服务证据；没有新 N6 重跑。
- **correctness**：权威表未列新候选 correctness；这里只能使用历史路径结果。
- **关键结果**：历史 fused P50 2078.67 ms、吞吐 442543.62；balanced non-fused 为 1909.35 ms、501987.43，fused 约慢 8.9%、吞吐低 11.8%。
- **裁决**：**拒绝**，因为它把 shared expert 推入 DeepEP，改变了错误的 ownership/通信边界。
- **可学到什么**：fusion 只有在省下的数据移动大于新增路由与通信时才有效；“少一个 kernel”不是充分理由。

### N13 — 继续优化 accepted static map

- **改动与机制**：检查 N6 的平衡映射是否仍有明显 rank 负载不均，可以继续搜索。
- **证据阶段**：静态负载/Amdahl 审计。
- **correctness**：没有新代码和模型请求。
- **关键结果**：75 个活跃层中，balanced median/max load ratio 为 1.000673/1.027191；identity 为 1.253423/1.686311。只有 layer 3 约 2.72%，大多数层已在 0.1% 内。
- **裁决**：**拒绝继续搜索**，保留当前 immutable map。
- **可学到什么**：当剩余不均衡已接近噪声，扩大搜索只会增加过拟合和维护成本。

### N14 — 从最新 main 移植 DeepEP normal dispatch

- **改动与机制**：检查较新上游是否有可直接移植的 dispatch 改进。
- **证据阶段**：静态版本/代码同一性审计。
- **correctness**：无代码差异可验证，因此没有模型 run。
- **关键结果**：上游/main `3a7f8b62` 是冻结 reviewed 分支的祖先；dispatcher 实现相同，reviewed 分支只额外包含 GLM 修改。
- **裁决**：**拒绝**，不是新 primitive；迁移反而会丢掉已验证路径。
- **可学到什么**：先做 git ancestry 和 diff，避免把“新分支名”误当成“新实现”。

### N15 — 复用 final DSA top-k 的 78.5 MiB int32 workspace

- **改动与机制**：避免 75 层重复申请大 workspace，假设 allocator 开销可观。
- **证据阶段**：leaf allocator/Amdahl 审计。
- **correctness**：未修改服务实现；无需模型 correctness。
- **关键结果**：真实 shape 上，75 层 fresh `torch.empty` 总计 2.027088 ms，固定 workspace 2.024626 ms，仅省 0.002462 ms 总量，即每层 0.00003283 ms；理论 TTFT 改善约 0.0001296%，远低于约 19 ms 的 1% 门槛。
- **裁决**：**拒绝**；allocator 本身已有缓存。
- **可学到什么**：显存容量很大不等于分配时间很大；不要用字节数代替时间占比。

## 4. N16–N24：重叠、DeepEP 参数与 FlashMLA 的服务筛选

### N16 — Blackwell single-batch overlap（SBO）

- **改动与机制**：开启 `enable_single_batch_overlap`，试图重叠计算和通信。
- **证据阶段**：model correctness + development bracket。
- **correctness**：11/11 token exact；logprob 最大/平均差 1.68e-4/3.03e-5，唯一变量是 SBO 开关。
- **关键结果**：控制 1917.79/3438.36 ms/467911.32 token/s；SBO 1951.58/3500.84 ms/467556.12，即 P50 慢 1.76%、P90 慢 1.82%、吞吐低 0.08%。
- **裁决**：**拒绝**，不做 formal/profile。
- **可学到什么**：N6 已较早启动 shared expert；SBO 延迟到 routed-down/combine 附近才重叠，没有消掉当前关键路径。

### N17 — DeepEP dispatch NVLink send token 6→7

- **改动与机制**：增加 dispatch 发送 token 配额，期望提高 NVLink 注入效率。
- **证据阶段**：model correctness + development bracket。
- **correctness**：11/11 token exact；logprob 最大/平均差 1.78e-4/2.48e-5，只有该字段变化。
- **关键结果**：控制 1907.925/2829.96 ms/502826.015 token/s；候选 1973.37/3368.38 ms/482359.13，即 P50 慢 3.43%、P90 慢 19.03%、吞吐低 4.07%。
- **裁决**：**拒绝**；保持 6，不做 formal/profile。
- **可学到什么**：通信参数会改变拥塞和调度，增加注入强度可能放大尾部争用。

### N18 — 恢复 native overlap schedule

- **改动与机制**：`disable_overlap: true→false`，同时停用零命中的 clustered-MQA 实验，测试原生 overlap。
- **证据阶段**：model correctness + path marker + development bracket。
- **correctness**：11/11 token exact；logprob 最大/平均差 2.73e-4/5.80e-5；路径 marker 正确。
- **关键结果**：控制 1908.96/3181.75 ms/477658.63 token/s；候选 2687.84/2734.50 ms/391109.32。P50 慢 40.80%、吞吐低 18.12%；较低 P90 来自整体变慢后分布被压缩，且 P50 已远超 2 秒。
- **裁决**：**拒绝**，不做 formal/profile。
- **可学到什么**：DP8 cached-prefill 需要同步推进；“overlap 打开”并不保证有有效并行，错误调度可让所有请求一起变慢。

### N19 — 联合修改 DeepEP dispatch/combine chunk

- **改动与机制**：在 SM=120 不变时设置 dispatch 32/256、combine 16/256，试图提高分块流水效率。
- **证据阶段**：model correctness + development bracket + formal early-stop。
- **correctness**：11/11 token exact；logprob 最大/平均差 2.76e-4/3.57e-5。
- **关键结果**：development 看似 P50 快 1.18%、P90 慢 0.94%、吞吐高 0.60%；formal 前两对却分别慢 1.23% 和 2.56%，0/2 胜，即使剩余全胜也最多 3/5。
- **裁决**：**拒绝**，数学 early-stop；无 holdout/profile。
- **可学到什么**：开发夹测会产生假阳性。预先定义“已不可能达到 4/5”可以节省机器时间且不损害统计规则。

### N20 — 仅把 combine send token 6→16

- **改动与机制**：隔离 N19 的 combine 侧因素，避免同时改多个参数。
- **证据阶段**：model correctness + development bracket + formal early-stop。
- **correctness**：11/11 token exact；logprob 最大/平均差 1.48e-4/2.66e-5。
- **关键结果**：development 表面 P50 快 1.77%、P90 快 10.73%、吞吐高 3.96%，但锚点有慢尾；formal 三对的候选 P50 改善为 +0.62%、-2.80%、-4.46%（后两对即明显变慢），仅 1/3 胜，且后两对吞吐更低。
- **裁决**：**拒绝**，early-stop；保留 6/6，无 holdout/profile。
- **可学到什么**：把联合候选拆成单变量有助于归因，但不能消除系统噪声，仍需 formal。

### N21 — synchronous PrefillDelayer compatibility

- **改动与机制**：启用同步 PrefillDelayer，尝试通过延迟/聚合改善尾部。
- **证据阶段**：4-process 状态机 regression + model correctness + development bracket。
- **correctness**：`mixed→delay→wait_timeout` 回归通过；模型 11/11 token exact，logprob 最大/平均差 2.19e-4/3.46e-5；只有 enable flag 变化。
- **关键结果**：控制 1945.875/3143.365 ms/485777.315 token/s；候选 2067.43/2932 ms/485211.79。P90 快 6.72%，但 P50 慢 6.25%、超过 2 秒，吞吐低 0.12%。
- **裁决**：**拒绝**，无 formal/profile。
- **可学到什么**：同步等待可能让尾部更整齐，却牺牲大多数请求的中位延迟；门槛必须同时约束 P50 和 P90。

### N22 — direct-paged DeepGEMM MQA 的 M 上限 2048→10048

- **改动与机制**：让 direct-paged MQA kernel 覆盖完整长序列 rank-local M=10048，避免 fallback。
- **证据阶段**：B300 leaf correctness/performance。
- **correctness**：FP8 E4M3、H32/D128、page64、100K context、top-k2048 的 selected-score multiset exact。
- **关键结果**：stock 5.7637 ms，direct 6.9074 ms，慢 19.84%；反向顺序仍慢 17.87%。该路径约占 3.6%，要达到 1% E2E 至少需 leaf 快约 28%，实际方向相反。
- **裁决**：**拒绝**，不进入 serving；当前 fail-closed M 限制正确。
- **可学到什么**：扩大 kernel 适用域之前，既要证明正确，还要证明新 shape 的算力/访存特征仍适合该实现。

### N23 — exact FlashMLA `b3_b5_native_exact`（M=10048）

- **改动与机制**：为当前主 bucket 使用 bit-exact FlashMLA 优化实现，减少 MQA kernel 时间。
- **证据阶段**：leaf + model correctness/path hit + development bracket。
- **correctness**：329,252,864 个 BF16 output 与 643,072 个 FP32 LSE 全部 bit-exact；模型 11/11 token exact，logprob 最大/平均差 1.68e-4/2.30e-5；path hit=702。
- **关键结果**：leaf 4.171→3.802 ms，快 8.85%。服务控制 1899.37/3123.55 ms/485845.87 token/s；候选 1881.34/2795.49 ms/504490.82，即 P50 只快 0.949%、P90 快 10.50%、吞吐高 3.84%。
- **裁决**：**拒绝 development 晋级**，因为 P50 未达到严格 1% 门槛，不能四舍五入；无 formal/profile。
- **可学到什么**：边界规则必须机械执行。0.949% 很有希望，但仍是 research candidate，不是 accepted。

### N24 — N23 + N7（indexer-Q 256 threads）

- **改动与机制**：组合两个正确的 leaf primitive，测试小收益是否可叠加跨过 E2E 门槛。
- **证据阶段**：model correctness/path hit + development retry + formal early-stop。
- **correctness**：11/11 token exact；logprob 最大/平均差 1.997e-4/3.463e-5；FlashMLA hit=702。
- **关键结果**：第一次 development 因控制漂移 4.46% 作废；重试 P50 快 1.272%。formal 三对为一胜两负：1972.94→1898.91、1921.16→1968.19、1977.07→2031.23 ms，仅 1/3 胜，且第三对候选超过 2 秒。
- **裁决**：**拒绝**，数学 early-stop；无 holdout/profile。
- **可学到什么**：两个 leaf 正收益不必然可加；共享资源、调度和测量漂移会在 E2E 层改变结果。

## 5. N25–N34：边界扩张审计、旧 MoE 想法与 FlashMLA 微优化

### N25 — 将 N23 provider 扩到 tail buckets

- **改动与机制**：覆盖 M=9472/9728/9792/9856/9920/9984 等尾部 shape。
- **证据阶段**：日志频率 + Amdahl 静态审计。
- **correctness**：未实现，因此无 correctness run。
- **关键结果**：110 请求中 M=10048 有 98 次（89.1%），其余 bucket 共 12 次。即使按 N23 的 0.949% 等比例外推，额外收益仅约 0.116%。
- **裁决**：**拒绝实现**，收益上限太小而 dispatch surface 会扩大。
- **可学到什么**：覆盖更多 shape 会增加测试矩阵和维护成本；先用频率加权收益决定是否值得。

### N26 — 旧 `combined_winners` 的 MoE alignment / contiguous PSUM

- **改动与机制**：复查 decode/小 M 场景曾探索的 W1/W2 对齐与 contiguous partial-sum，判断能否迁移到本 prefill。
- **证据阶段**：同路径 leaf 与 DeepEP-normal 服务 A/B 历史证据。
- **correctness**：路径 hit 已验证；权威表未列新的 token/logprob 数值。
- **关键结果**：M1024 时 align16/32 让 W13 111.8→307.8 μs，W2 115.7→354.3/478.7 μs，两个阶段都显著变慢；contiguous PSUM 在 M1024/2048 的公平 A/B 中 TTFT 分别慢 2.28%/1.68%。
- **裁决**：**拒绝迁移**。
- **可学到什么**：decode 小 M 的赢家不能直接迁移到长 prefill；W1 局部收益可能被 W2 或通信边界反噬。

### N27 — 移植 KDA 分支 TRT-LLM FP8 MoE

- **改动与机制**：参考 KDA / DeepSeek 类策略，考虑替换当前 DeepEP-normal + DeepGEMM 路径为 TRT-LLM routed MoE。
- **证据阶段**：静态合同与集成边界审计。
- **correctness**：没有可运行的同合同候选。
- **关键结果**：KDA HEAD `002667478` 针对 64 req、100K+1000 decode throughput 及 standard/MegaMoE；N6 的 auto+DeepEP 落到 DeepGEMM，且 `flashinfer_trtllm_routed` 没有 DeepEP-normal 注册。移植会同时替换 A2A 和 layout，而非一个正交 primitive；相关 DP local scheduler 方向 N3 已失败。
- **裁决**：**拒绝直接移植**。
- **可学到什么**：借鉴策略应迁移“问题分解方法”，而不是复制为不同模型、不同 phase 和不同通信栈写的代码。

### N28 — 官方 `sgl-deep-gemm 0.1.5.post2+cu130`

- **改动与机制**：尝试用官方新 DeepGEMM 包替换冻结栈实现。
- **证据阶段**：ABI、权重加载与 warmup compatibility。
- **correctness**：未达到可服务状态。纯 torch packer 在 5 类 weight 上 bit-exact，adapter 让 8 ranks 完成 696 次调用/11 shapes/load，但首个 GEMM warmup 仍失败。
- **关键结果**：最初在 `smxx_layout.hpp:111` 的 weight scale packing 失败；适配后在 `fp8_gemm_nt layout.hpp:15` 再失败。
- **裁决**：**拒绝**，属于 ABI/correctness 不兼容，无性能数字；继续适配会同时改变多变量。
- **可学到什么**：能 import、能 load weight、能跑第一个 GEMM 是三层不同兼容门槛；未过 correctness 前禁止讨论加速。

### N29 — FlashMLA P1：consumer-side FP32 scale gather

- **改动与机制**：把 scale gather 移到 consumer 侧，尝试改善访存与生产/消费配合。
- **证据阶段**：leaf microbenchmark。
- **correctness**：全部 output/LSE bit-exact。
- **关键结果**：三组平衡顺序均变慢约 1.42%、1.48%、1.44%，pooled 慢 1.48%。
- **裁决**：**leaf 拒绝**；不进入 source/service/model。
- **可学到什么**：减少 producer 工作可能把更多随机访存压给 consumer，整体并不一定更快。

### N30 — FlashMLA R3-A：首坐标 prefetch

- **改动与机制**：提前取首个坐标，试图隐藏地址生成或加载延迟。
- **证据阶段**：leaf microbenchmark。
- **correctness**：bit-exact。
- **关键结果**：三组分别慢 2.21%、2.29%、2.26%，pooled 慢 2.29%。
- **裁决**：**leaf 拒绝**，不进服务。
- **可学到什么**：prefetch 可能增加寄存器、指令或 cache 污染；必须测量而不能只靠直觉。

### N31 — FlashMLA B1：缩短 coordinate-ready 路径

- **改动与机制**：减少坐标准备路径中的等待/指令，希望更早发起主加载。
- **证据阶段**：leaf microbenchmark。
- **correctness**：bit-exact。
- **关键结果**：按“正值表示延迟降低”的原表口径，三组 reduction 为 +0.014%、-0.013%、-0.026%，pooled 为 -0.019%，即约慢 0.019%，统计上近乎中性。
- **裁决**：**拒绝**，没有足够 leaf leverage。
- **可学到什么**：接近计时噪声的差异不应包装成优化；即便真实，乘上 E2E 占比后也不可见。

### N32 — FlashMLA B2：NoPE producer staggering

- **改动与机制**：错开 NoPE producer 的工作时序，降低同一时刻的资源竞争。
- **证据阶段**：leaf microbenchmark；随后打包进 N35 做服务验证。
- **correctness**：bit-exact。
- **关键结果**：三组 leaf 分别快 0.849%、0.787%、1.422%；pooled 3.808288→3.773479 ms，快 0.914%。与 stock 相比的组合 leaf 可达 1.1072x，但这不是 E2E 加速。
- **裁决**：**leaf 通过，research primitive**；是否可接受由 N35 服务结果决定。
- **可学到什么**：leaf pass 只授予“进入下一关”的资格，不授予 baseline 身份。

### N33 — FlashMLA B4：first-NoPE eviction

- **改动与机制**：更早释放/驱逐首个 NoPE 相关资源，尝试减轻占用。
- **证据阶段**：leaf microbenchmark。
- **correctness**：bit-exact。
- **关键结果**：三组变化为快 0.033%、慢 1.247%、慢 1.388%；pooled 慢 1.293%。
- **裁决**：**leaf 拒绝**。
- **可学到什么**：释放资源的时点会影响重用和 cache 命中；“更早释放”不是单调更好。

### N34 — FlashMLA B4：early-warp RoPE eviction

- **改动与机制**：让早期 warp 更早驱逐 RoPE 相关状态。
- **证据阶段**：leaf microbenchmark。
- **correctness**：bit-exact。
- **关键结果**：三组分别慢 1.618%、1.641%、1.685%，pooled 慢 1.638%。
- **裁决**：**leaf 拒绝**。
- **可学到什么**：warp 级生命周期调整会改变寄存器/cache/同步平衡，必须用多顺序重复测量。

## 6. N35–N40：FlashMLA 组合、pipeline stage 与 MoE fusion

### N35 — 将 N32 打包为服务候选

- **改动与机制**：在 N23 exact FlashMLA 上加入 N32 NoPE producer staggering，形成 28 文件 contract 的服务候选。
- **证据阶段**：model correctness/path hit + development bracket；未获准进入 formal。
- **correctness**：11/11 token exact；logprob 最大/平均差 3.385e-4/4.540e-5；path hit=702。
- **关键结果**：第一次 bracket 控制漂移 +6.345%，表面快 2.588% 但作废；重试漂移 -3.060%，仅比 3% 有效阈值多 0.060 个百分点。候选 1888.87/2814.94 ms/503693.10 token/s；方向性相对锚点中位数 P50 快 2.096%、P90 快 0.042%、吞吐高 0.531%。
- **裁决**：**未准入**，不是 accepted；无 formal。保留为 research hypothesis。
- **可学到什么**：候选看起来很快也不能忽略无效 bracket。控制漂移门槛保护的是实验可解释性。

### N36 — pipeline stage 2→3，但未显式对齐

- **改动与机制**：增加 FlashMLA pipeline stage，希望用更深流水隐藏内存延迟。
- **证据阶段**：build + 首次同步 launch correctness。
- **correctness**：失败，出现 misaligned address。24-byte validity 区把 `tma_coord` 推离了 16-byte 对齐。
- **关键结果**：shared memory 预算看似低于 227 KiB，但布局对齐不满足 TMA 要求；没有合法计时和服务结果。
- **裁决**：**拒绝 N36 实现**。
- **可学到什么**：共享内存总字节数通过不代表布局正确；TMA/向量访问的地址对齐是硬不变量。

### N37 — `tma_coord` 16-byte 对齐的三阶段实现

- **改动与机制**：修复 N36 的布局，显式将 `tma_coord` 对齐到 16 bytes，保留 3-stage pipeline。
- **证据阶段**：leaf correctness/performance + Nsight leaf profile；随后打包为 N38。
- **correctness**：全部 output/LSE 相对 N35 bit-exact。
- **关键结果**：计划 shared memory 232432 B，只比 232448 B 上限少 16 B，因此无法上 4 stages。三组 leaf 快 0.524%/0.722%/0.502%；pooled 3.795478→3.764518 ms，快 0.8157%。Nsight 主 kernel 约 3.69 ms，combine 约 0.055 ms；预测 E2E 仅约 0.13%–0.19%。
- **裁决**：**leaf 通过、research primitive**，进入 N38；未被单独 accepted 为 E2E 优化。
- **可学到什么**：接近 shared-memory 上限时，对齐填充也必须计入预算；Amdahl 预估提前说明 E2E 很可能看不见。

### N38 — N23 + B2 + aligned 3-stage 服务候选

- **改动与机制**：组合 exact FlashMLA、N32 staggering 和 N37 三阶段 pipeline。
- **证据阶段**：model correctness/path hit + 有效 development bracket。
- **correctness**：11/11 token exact；logprob 最大/平均差 2.579e-4/3.868e-5；path hit=702。
- **关键结果**：控制漂移 -0.295%，bracket 有效。控制 1929.865/3189.005 ms/484953.095 token/s；候选 1931.17/2826 ms/502162.49。P50 实际慢约 0.068%，P90 快 11.38%，吞吐高 3.55%。
- **裁决**：**拒绝**，P50 未达 1%，无 formal/holdout；尾部/吞吐方向仅作为研究证据。
- **可学到什么**：局部流水优化可以改善吞吐或尾部，却不一定缩短单请求 TTFT 的中位关键路径。

### N39 — 关闭 DeepGEMM PDL

- **改动与机制**：在真实 routed-MoE shape（M=5008,E=32；W1 N4096/K6144，W2 N6144/K2048）关闭 Programmatic Dependent Launch，验证 PDL 是否带来调度负担。
- **证据阶段**：生产格式 leaf correctness + 三组 30-pair ×10 replay。
- **correctness**：FP8 quant/UE8M0/TMA scale 下，W1 的 20,512,768 和 W2 的 30,769,152 个输出均 bit-exact。
- **关键结果**：先修复了“切换全局状态时计时无效”的方法问题。合法复测中，关闭 PDL 后 W1 0.150410→0.151294 ms，慢 0.588%；W2 0.082642→0.083339 ms，慢 0.844%。
- **裁决**：**leaf 拒绝关闭 PDL**；保持 PDL on，不进模型服务。
- **可学到什么**：影响 kernel 生成/全局状态的开关必须在正确边界重建并同步，否则 microbenchmark 会测到错误对象。

### N40 — N23 + non-swizzled contiguous routed-MoE SwiGLU/FP8-quant fusion

- **改动与机制**：融合 routed-MoE 的 SwiGLU、UE8M0 FP8 quant 和 `round_to_bf16`，并使用 non-swizzled contiguous 布局，目标是删除中间张量读写与 launch。
- **证据阶段**：leaf correctness/performance + 零 token 边界修复 + model correctness/path hit + development bracket。
- **correctness**：leaf 的 10,256,384 个 FP8 code 和 20,032 个 scale word byte-exact；首次模型运行暴露 local M=0 的 zero-grid bug，加入 zero-token guard 后模型 11/11 token exact，logprob 最大/平均差 2.569e-4/4.674e-5；path hit=702。
- **关键结果**：M=5008 的 leaf 三组快 59.27%/59.31%/59.10%，pooled 0.045245→0.018443 ms。有效 bracket 漂移 2.922%；控制 1917.275/2813.54 ms/502755.05 token/s；候选 1947.18/3337.04 ms/472553.71，即 P50 慢 1.56%、P90 慢 18.61%、吞吐低 6.01%，并出现两次仅候选侧的 `empty_chunked_topk`。
- **裁决**：**development 拒绝**，无 formal/holdout；保留 leaf 实现证据，不进入 accepted stack。
- **可学到什么**：59% 的 tiny leaf 加速仍可能被 E2E 完全反转；融合改变了 launch、stream、空 rank 和调度行为，必须覆盖 M=0 等分布式边界。

## 7. 总结：哪些是已接受，哪些只是研究证据

| 类别 | 编号 | 可以安全地说什么 |
|---|---|---|
| E2E accepted | N6-final | 在冻结长上下文 prefill 合同上，formal P50 快 5.31%、吞吐高 7.53%，并在独立 seed 复现；这是当前唯一完整通过的 N 系列组合。 |
| accepted causal evidence | N6-profile | Nsight 支持“同样 10 秒推进更多 dispatch、降低通知时延并删除 padded-ID mask 工作”的因果解释；profiled TTFT 不参与加速百分比。 |
| leaf pass / research | N7、N23、N32、N37、N40 的 leaf 部分 | 局部正确且某些 kernel 更快，但没有通过 E2E 晋级；不能写成服务 accepted。 |
| development/formal rejected | N1–N4、N9、N11、N16–N21、N24、N35、N38、N40 等 | 在当前合同下未过预声明门槛，即使某个指标或某次运行更好，也不能并入 baseline。 |
| static/applicability stop | N5、N13–N15、N25 等 | 目标边界不在关键路径或Amdahl上限过小，因此在写/跑E2E前停止。 |
| compatibility/correctness stop | N8、N28、N36 等 | 协议、ABI或对齐正确性不成立，不产生合法性能数字。 |
| leaf/operator stop | N7、N10、N22、N26、N29–N34、N37、N39 等 | 在operator/leaf层反向、中性或收益上限不足；其中也保留正向research primitive，不等同完整E2E rejection。 |

最后，推荐对外使用的准确表述是：

> 在 B300、TP8/DP8/EP8（非 CP8）、90K shared-prefix + 10K suffix、x11 并发、110 请求、单 token 输出的长上下文 prefill/TTFT 测试中，N6（平衡专家映射 + 物理 ID router + DeepEP 120/120 SM）相对冻结 identity/136 基线在正式测试将 P50 TTFT 降低 5.31%、吞吐提高 7.53%，并在独立 seed 分别复现 5.46% 和 7.40%；其他 N1–N40 候选均未被promotion，其中许多在static、compatibility、correctness或leaf门即停止，并未运行完整E2E gate。

## 8. 初学者的五条实践原则

1. **先冻结 workload，再谈优化。** prefill 与 decode、TP/DP/EP/CP 的不同组合会改变瓶颈，跨合同数字不能直接比较。
2. **先做路径和 Amdahl 审计。** N5、N13–N15、N25 说明，最便宜也最可靠的优化有时是“不写那段代码”。
3. **把 correctness 分层。** bit-exact leaf、模型 token exact、logprob 容差分别回答不同问题；能启动服务不等于正确。
4. **允许局部失败，但扩大边界后重新走门槛。** N1/N4 失败、N6 组合成功，体现了“允许局部回退、换取更大优化空间”的正确用法。
5. **profile 用于归因，正式 benchmark 用于裁决。** Nsight 能告诉我们时间花在哪里，但性能晋级必须来自未 profiler 污染的 fresh-server 测试。
