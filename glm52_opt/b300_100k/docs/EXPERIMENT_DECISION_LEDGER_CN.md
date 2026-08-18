# 实验决策账本：每项修改、证据与结论

本文件是精选快速索引；每项的机制、CUDA 解释和原始样本见 [完整系统报告](GLM52_B300_SYSTEMATIC_TEACHING_REPORT_20260817_CN.md)。全部 N1–N40 与历史实验另见 [全实验主报告](ALL_EXPERIMENTS_MASTER_REPORT_CN.md) 和 [N1–N40 完整账本](N1_N40_COMPLETE_LEDGER_CN.md)。百分比中“更慢/更快”均指候选相对同合同 control。不同 workload 的数字不做横向比较。

## 1. N6 形成过程

| 候选 | 修改 | 结果 | 决策与教学结论 |
|---|---|---|---|
| N1 balanced placement | 按冻结路由统计重排每层 256 experts | 负载 median/max ratio 约 1.253/1.686→1.0007/1.0272；P50 -4.33%，吞吐 +4.24%，但 P90 +9.20% | 单独拒绝；均衡平均负载不保证 tail |
| N4 atomic placement+router | balanced map 与 router 直接 physical-ID 原子组合 | P50 -2.05%，P90 +0.53%，arm-median 吞吐 -2.82% | 单独拒绝；不能把变化只归因给 router |
| DeepEP SM sweep | normal dispatch/combine 136→120/144，再试112/128 | development 中 120 方向最好；144、112、128 均差 | 120 只作为组合选择，不宣称独立正式收益 |
| N8 split dispatch/combine SM | 尝试分别配置两者 | 共享 QP/buffer-layout assertion，启动 fail-closed | 拒绝；底层协议不支持虚构的独立自由度 |
| N6 | N4 + DeepEP 120 | independent correctness probe：token exact；formal -5.31%/-4.75%/+7.53%；independent-seed holdout -5.46%/-6.73%/+7.40% | **ACCEPTED IN FROZEN CELL** |

N6 formal 的 paired wins 为 P50 4/5、P90 3/5、throughput 4/5。P90 波动较高，但独立 holdout 5/5 P50 wins 补强了结论。

## 2. 扩大 placement 边界

| 候选 | 修改 | 结果 | 决策 |
|---|---|---|---|
| per-token recorder | 在 router Top-K 记录 token 级 expert IDs | seed0/1/2 exact replay SSOT | 研究基础设施；不放在性能 path |
| temporal v1 | 逐层 hill-climbing，优化 step-critical proxy | 退化 host 上 P50 +8.60%、P90 +6.26%、吞吐 -2.64% | REVERT |
| temporal v2 | 增加 communication-safe 约束和局部回退 | P50仍约 +5.95%；tail/吞吐方向信号不足以救 primary | REVERT |
| temporal v5 | whole-layer 二进制选择；25 行 proposal、53 行回退 N6 | 11/11 exact；退化 host A-B-A 相对 P50 -68.89%，但绝对 P50 3119.99 ms、吞吐 320683.49 | RESEARCH；需健康 host 重测 |

v5 matched Nsys：dispatch progress +15.87%，dispatch notify P90/P99 -27.81%/-65.02%，combine notify P90/P99 -39.73%/-43.10%，FlashMLA per-call 基本不变。证据支持“rank arrival/control tail 改善”，不支持“attention 变快”。

## 3. KDA / FlashMLA

| 候选 | 修改 | Leaf / correctness | Server E2E | 决策 |
|---|---|---|---|---|
| N22 direct-paged DeepGEMM MQA | 把 direct-paged admission 的 M 上限2048→10048；无 clustered kernel | selected-score multiset exact；真实 M10048 慢17.9%–19.8% | 未进入服务 | REVERT；KDA/clustered源码研究另列，不能共用这组数字 |
| N23 exact-M10048 | B3+B5 FlashMLA exact shape | 4.171→3.802 ms，-8.85%；output/LSE bit-exact | P50 -0.949%、P90 -10.50%、吞吐 +3.84% | development P50 门槛要求 ≥1%，拒绝但保留迭代基底 |
| N24=N23+N7 indexer 256 threads | N23 exact-M10048 上叠加 indexer-Q 128→256 | 六例 bit-exact；isolated indexer leaf -26.5%至-29.1%，trace 中占比很小 | 组合 formal 1/3 wins 后 early-stop | REVERT |
| N32/N35 | NoPE producer staggering | 在 N23 上 leaf 再约 -0.914% | 两 anchor drift 3.060%>3% | NO DECISION |
| N36 | index/scale/validity ring 2→3 stage | 24B validity array 破坏 16B TMA 对齐，misaligned address | 正确性门停止 | REVERT |
| N37 | 显式 16B 对齐 | bit-exact；leaf 再 -0.8157%；shared memory 232432B，距上限仅16B | 作为 N38 输入 | LEAF-ADMITTED |
| N38 | 对齐后的三 stage E2E | 正确 | P50 +0.068%，P90 -11.38%，吞吐 +3.55% | primary P50 无收益，REVERT |
| 8-shape band | 显式 admit M=9616/9728/9792/9856/9920/9984/10016/10048，范围外回 stock | 两测量顺序均 bit-exact，leaf -8.82%至-12.49% | 后续 host 退化，未形成 formal screen | LEAF-ADMITTED, NOT E2E-PROMOTED |

“production band”只是实验资产名称，不代表 production promotion。

## 4. DeepEP 参数

| 候选 | 修改 | 结果 | 决策 |
|---|---|---|---|
| N17 send7 | dispatch send 6→7 | P50 +3.43%、P90 +19.03%、吞吐 -4.07% | REVERT |
| N19 joint 32/256 + 16/256 | 同时改 dispatch/combine chunk | formal 前两对慢 1.23%/2.56%，0/2 early-stop | REVERT |
| N20 combine16 | 只改 combine send6→16 | development 好看，formal 仅1/3 wins，后两对吞吐下降 | REVERT |
| real-shape send16 | 8-rank M10048/H6144/EP256/topk8/SM120 leaf | 最慢 rank dispatch median -2.45%、wall -1.53%；P90 wall -4.48% | server 绝对门槛失败，REVERT |
| send20 / recv sweep | 扩大 chunk | 无稳定优势 | REVERT |

通信 chunk 同时影响控制开销、队头阻塞和 tail；micro 的最慢 rank 改善并不自动迁移到 server。

## 5. DP 通信、scheduler 与 overlap

| 候选 | 修改 | 结果 | 决策 |
|---|---|---|---|
| N5 variable allgatherv | 对 variable-length gather/reduce-scatter 做替代 | 静态边界已不适合冻结 cell，未 GPU 计时 | REVERT |
| exact-fill equalAG | rank 等长且 exact-fill 时 SUM_LEN all-gather；其余 fallback | model/unit/source contract通过；缺专属HIT与no-profiler E2E，不能独立复核no-op | NOT PROMOTED |
| N16 SBO | shared/routed expert 与 combine 的 stream/event 重排 | exact；P50 +1.76%、P90 +1.82%、吞吐 -0.08% | REVERT |
| N18 overlap scheduler | 开启 native overlap schedule | exact；P50 +40.80%、吞吐 -18.12% | REVERT；无 causal profile，不指定未经证实的根因 |
| N21 PrefillDelayer | DP collective 协调同步进入 prefill | exact；P90 -6.72%，但 P50 +6.25%、吞吐 -0.12% | REVERT；用 median 代价换 tail |
| overlap outside SBO | Blackwell+DeepGEMM+DeepEP-normal 下为空 hook | 无 material path change | 计时前 REVERT |

## 6. CPU affinity 与 host fault

| 项目 | 修改/观测 | 结果 | 决策 |
|---|---|---|---|
| host diagnosis | 256 logical CPU 中，仅 socket1 高半约32个 physical core+SMT 稳定约3.6GHz，其余大区约500MHz | governor/turbo/RAPL/温度正常；无 thermal/Xid/NVLink/GPU 故障证据 | 只能称 `CPU frequency-domain/platform fault` |
| old affinity audit | 旧 worker affinity 按 global CPU 编号重算 | 发现 scheduler 逃出父 cpuset | fail-closed，不跑 benchmark |
| cpuset-safe primitive | 父 mask 为 SSOT；按 package/core/SMT grouping；8份互斥；不足 fail-closed | 2 unit tests、23-file contract、process-tree audit 通过 | RESEARCH correctness primitive |
| topology all-siblings | v5+cpuset+拓扑分区 single arm | P50 2716.96、P90 2899.52、吞吐 412371.30 | 绝对 P50/吞吐失败 |
| physical-only | 每物理核一个线程 | P50 2752.50、P90 2851.30、吞吐 412677.09 | 同样失败；问题不只是 SMT contention |

这些 single-arm 同时含 v5、cpuset patch 和 topology treatment，不能估计 affinity 的独立增量。

## 7. 其他 N-series 负结果

| 候选 | 结果 | 结论 |
|---|---|---|
| N2 single-D2H | P50 +0.23%、P90 +3.16%、吞吐 -5.94%，1/5 | REVERT |
| N3 DP-attention local control broadcast | exact；P50 +1.88%、P90 +5.24%、吞吐 -3.59%，1/5 | REVERT |
| N10 FlashInfer Top-K | SGL 2.4461 ms vs FlashInfer 4.294 ms | leaf REVERT |
| N11 explicit flashmla_sparse | P50 +10.55%、P90 +15.54%、吞吐 -10.74% | REVERT |
| N12 ninth-route shared expert | 2078.67 ms/442544 vs balanced nonfused 1909.35/501987 | REVERT |
| N15 Top-K workspace reuse | 75层合计仅约0.002462 ms，Amdahl 上限太小 | 静态 REVERT |
| N28 newer DeepGEMM wheel | packing/hot-path ABI 失败 | 正确性前 REVERT |
| N40 contiguous SwiGLU+FP8 | leaf -59.1%，exact；E2E P50 +1.56%、P90 +18.61%、吞吐 -6.01% | REVERT；最重要的 leaf-win/E2E-loss 案例 |

## 8. 历史不同 workload

| 项目 | 合同与结果 | 正确表述 |
|---|---|---|
| chunk DP 二次除法修复 | runtime chunk 1024→128 的错误导致约8倍轮次；修复后 full8192 79.05→15.86s，cached64K+1024 2.14→0.65s；叠加早期 DeepEP24 后 full8192 12.05s | HISTORICAL prefill fix；说明“删轮次”比微调 kernel 更有价值 |
| decode MTP/EAGLE | JSONL可证64K、batch8、output512、TPOT 32.15→24.33 ms、吞吐248→329 token/s、candidate acc_length=1.87；draft2等来自未完整落地的handoff合同 | HISTORICAL decode；baseline length=1是语义基准，93.5%需以draft2合同为前提 |
| fused QKV-A | 两组decode development方向冲突（33.259→34.438 ms；23.814→20.512 ms）；历史leaf 2.4×/bit-exact缺本地权威载体 | NO FORMAL DECISION；未晋级 |
| MoE align16/32 | 大 M prefill 明显增加无效计算 | REVERT；decode 小 M 假设不适用 |
| MoK megakernel | screening-level progress修复后 x10/x12/x16 均可完成；x10 P50 +8.40%/吞吐 -14.42%，x12 +7.55%/-8.36%；内存约109.45→207 GiB/rank | token-exact screen通过但seed/阈值未formal冻结；稳定性成功、性能REVERT；MoK不是MTP |

## 9. true attention CP8 / EP8 后续实验

| 候选 | 修改 | 结果 | 决策 |
|---|---|---|---|
| true CP8 baseline | TP8/DP1/attention CP8/attention TP1/EP8；zigzag切10,016真实extend token | x11 arm-median P50/P90/吞吐为3755.57/3797.85 ms/291300 token/s | 新研究cell；未超过N6 |
| combined-indexer | 每层两次626-row MQA/top-k合成一次1252-row；范围外fallback | x1 P50 5/5，paired median -2.50%；x11三指标全5/5，-3.30%/-3.32%/+3.19%；11/11 token exact；Nsys调用数减半 | **CP8内部 research winner；不替换N6** |
| combined MQA SM112 | 在合并后把 MQA SM 148→112 | 最慢rank leaf -2.42%，Amdahl E2E上限约0.28% | leaf门拒绝，不跑server |
| packed KV all-gather | 先pack再传输 | MLA KV leaf约1.96×，indexer-K约0.85×更慢；x11无稳定E2E收益 | REJECT/NEUTRAL |

完整解释见 [true CP8教学报告](GLM52_TRUE_CP8_OPTIMIZATION_20260818_CN.md)。这里的3.30%只比较CP8 candidate与CP8 control；CP8 candidate绝对P50约3634.06 ms、吞吐约302172 token/s，仍慢于accepted N6的1933.67 ms与486640 token/s。

## 10. 总决策

- Accepted：只有 N6，且只针对冻结 100K cached-prefill cell。
- 下一优先级：健康 host 上复验 v5，并在不同 suffix 长度/并发下寻找 CP8 是否存在架构交叉点。
- 可作为局部 primitive 继续研究：FlashMLA 8-shape band、cpuset-safe affinity、true-CP8 combined-indexer。
- 其余候选保留负结果和因果知识，不保留为 main 中的并行 runtime path。
