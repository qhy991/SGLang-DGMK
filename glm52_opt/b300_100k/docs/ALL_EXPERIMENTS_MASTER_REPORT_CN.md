# GLM-5.2 / B300 全部实验系统报告

日期：2026-08-17

范围：B300 退役快照中 `bench_results` 的完整文件级清单，以及已经发布到 GitHub 的精选代码与紧凑证据

报告目标：不只解释最终 N6，而是保存从早期 prefill、decode、kernel、MoK、N1–N40 到 temporal placement、profiling 和主机诊断的完整实验知识

## 1. 最重要的结论

整个优化活动并不存在一个可以覆盖所有 workload 的“总冠军”。结果必须分成三类理解：

1. **历史上最大的 prefill 收益**来自修复 chunk 被 DP8 二次除法的问题：full-8192 TTFT 从 79.05 s 降到 15.86 s；叠加该历史 cell 的 DeepEP 24 SM 后降到 12.05 s。这个 4.98×–6.56× 的收益来自删除重复执行与通信轮次，不是把某个 CUDA kernel 加速六倍。
2. **历史 decode 的正结果**包括 MTP/EAGLE（TPOT 32.15→24.33 ms、吞吐 248→329 token/s，约 1.32×）以及一组 S=32768、BS128 的组合 winner（ITL median 33.116→31.243 ms，约 5.66%）。它们都不是当前 output=1 的 prefill/TTFT 结果。
3. **冻结 100K cached-prefill cell 的唯一正式 accepted 方案**是 N6：balanced static expert placement + router 直接输出 physical expert ID + DeepEP normal dispatch/combine 136→120 SM。正式 P50 -5.31%、P90 -4.75%、吞吐 +7.53%；独立 holdout 为 -5.46%/-6.73%/+7.40%。

后续 FlashMLA、DeepEP chunk、overlap、PrefillDelayer、MoK、temporal placement 和 cpuset 都产生了有价值的局部或因果证据，但没有在健康主机、相同冻结合同下取代 N6。尤其要记住：**kernel 快，不等于服务快；profile 能解释原因，不等于可以替代无 profiler E2E。**

## 2. 这次“全部整理”实际覆盖了什么

远端盘点得到 350 个顶层实验目录，另有 91 个 `bench_results` 根级文件；合计 20,752 个文件、259,480,606,210 bytes（约 241.66 GiB）。为避免把 232 GiB 左右的 raw profiler、日志和二进制再次复制到本地，本包采用“结论层完整、raw timeline 有意省略”的策略：

| 项目 | 数量/大小 | 含义 |
|---|---:|---|
| 远端顶层实验目录 | 350 | 每一个都进入目录索引，空目录也不省略 |
| 远端全部文件 | 20,752 / 241.66 GiB | 包括 raw Nsys/SQLite、日志、build/cache 和二进制 |
| 本地 compact 证据 | 7,056 / 105,935,909 bytes | summary、samples、correctness、server args、合同、hash、派生 profiler 表、实验文档与复现脚本 |
| 明确排除 | 13,696 / 259,374,670,301 bytes | 每一项都在排除账中记录原因 |
| 复制缺失/多余/大小不符 | 0 / 0 / 0 | 本地 allowlist 与远端 inventory 精确对齐 |
| raw profiler 或大二进制混入 | 0 | `.nsys-rep/.sqlite/.ncu-rep/.so` 等均未进入 compact tree |
| NCU 派生表 | 1 | 保留 `mqa_topk/details.csv`，raw `.ncu-rep` 排除 |

入口：

- [全部 350 目录逐项索引](ALL_350_DIRECTORIES_INDEX.md)
- [完整 N1–N40 决策账本](N1_N40_COMPLETE_LEDGER_CN.md)
- [全部结果文档索引](ALL_RESULT_DOCUMENTS_INDEX.md)
- [归档覆盖、排除与异常说明](ARCHIVE_COVERAGE_AND_GAPS_CN.md)
- 原始 inventory (`private-archive:inventory/remote_all_files.tsv`)
- compact allowlist (`private-archive:inventory/compact_files.tsv`)
- 完整排除账 (`private-archive:inventory/excluded_files.tsv`)

完整 `all_experiments` 证据树是**私有审计档案**：3,127 个文件、6,210 行内容仍含内部绝对路径、用户名或来源路径。扫描没有发现实际 token、私钥或已赋值秘密，但它不能未经脱敏直接推到公开 GitHub。本报告和两个公开索引只保留工作负载、裁决、相对归档 ID 与覆盖统计；`private-archive:` 不是仓库内链接。

## 3. 初学者先理解：一次推理优化到底在改什么

可以把多 GPU 推理的一次请求看成三张相互嵌套的图：

```text
计算图：Attention / Router / Expert GEMM / Shared MLP / Logits
通信图：DP 协商 / DeepEP dispatch / Expert compute / DeepEP combine / collective
控制图：CPU scheduler / batch 形状 / CUDA launch / stream-event / rank progress
```

CUDA kernel 是 GPU 上的一段并行程序；SM（Streaming Multiprocessor）是执行这些线程块的硬件资源。一个 kernel 变快，只说明计算图的某个叶子节点变快。用户看到的 TTFT 取决于三张图的关键路径：

```text
端到端时间 ≠ 所有 kernel 时间简单相加
端到端时间 ≈ 最慢依赖链 + 同步等待 + host 发射/调度 + 排队
```

MoE 又增加了一个特殊问题。每个 token 只去少量 expert，但不同 rank 收到的 token 数不一样。快 rank 完成自己的 GEMM 后，仍可能在 DeepEP combine 或 collective 处等待最慢 rank。因此 placement、arrival skew 和通信 tail 往往比单个 GEMM 的平均速度更重要。

评价证据时使用四个正交的门，而不是一条“越靠后越强”的等级链：

- **正确性门**：token、logprob、operator output/LSE、路径命中是否符合合同。
- **无 profiler E2E 门**：fresh server、paired A/B 或 A-B-A、P50/P90/吞吐、绝对 SLO 和重复性。
- **因果 profile**：Nsys/NCU/NVTX 用于回答“为什么”，不把 profiled TTFT 当性能数字。
- **可复现门**：代码 revision、模型/数据 hash、server args、runtime contract 和结果 hash。

## 4. 必须分开的 workload cells

不同 cell 的绝对数字不能横向比较；否则很容易把 decode 的 TPOT、prefill 的 TTFT、leaf 的毫秒和 profile 的累计 duration 混成一个结论。

| Cell | 主要时期 | 冻结或已知合同 | 可以回答的问题 |
|---|---|---|---|
| H1 早期 prefill | 07-24 至 07-26 | full 8192，或 cached 64K + 1024 incremental；早期 DeepEP24/低延迟路径 | chunk、轮次数和早期通信设置是否合理 |
| H2 decode/kernel | 07-29 至 08-03 | 32K/64K context，BS16–256，output 32/48/64/512；部分 CUDA graph | TPOT、decode throughput、MTP 和小 M kernel |
| F0 100K route-complete | 08-11 | 8×B300，TP8/DP1/EP8，90K+10K，output1，chunk8192，DeepEP low-latency | DSV4 workspace/indexer 思想是否迁移到 GLM |
| H3 MoK | 08-12 | TP8/DP8/EP8，90K+10K，x10/12/16，MoK M10016 | megakernel 的稳定性、内存和性能 |
| F1 N-series | 08-13 至 08-14 | 8×B300；TP8/DP8/EP8；attention TP1、`attn_cp_size=1`；90K+10K；output1；x11/110；per-rank M≤10048 | 冻结 100K cached-prefill 的正式优化 |
| F2 degraded-host | 08-15 至 08-16 | 名义上沿用 F1，但 CPU 大片频域约500MHz | 只可解释相对 A-B-A、causal profile 和 host diagnosis |
| M1 current main | 08-17 | 从冻结 runtime surgical port 的默认关闭代码与 compact evidence | 源码是否可审查；尚不能声称复现 F1 性能 |

“CP=8”在本次历史口径中容易产生误解。正式记录是 TP8/DP8/EP8，但 `attn_cp_size=1`；8 路是 DP-attention/上下文分发 rank，不是已验证的 attention context parallel size 8。

## 5. 阶段一：先找系统性浪费，而不是先写 CUDA

### 5.1 chunk 二次除法修复

早期配置想让每 rank 处理 1024 token chunk，但全局/局部换算再次被 DP8 除，实际每 rank 只得到 128。一次长 prefill 因此被切成大约八倍轮次，每一轮都重复执行 scheduler、模型层和跨 GPU 协商。

| Workload | 修复前 | 修复后 | 结果 |
|---|---:|---:|---:|
| full 8192 prefill TTFT | 79.05 s | 15.86 s | 4.98× |
| cached 64K + 1024 incremental | 2.14 s | 0.65 s | 3.3× |
| full 8192 + 该 cell 的 DeepEP24 | 79.05 s | 12.05 s | 约6.56× |

教学重点：减少一次 kernel 的 10% 通常只能作用于很小占比；把八轮重复工作恢复为一轮，可以同时删除计算、通信和 host 控制成本。这是整个活动里最强的 first-principles 优化案例，但它属于 H1，不可拿 12.05 s 与 F1 的约1.9 s 直接比较。

### 5.2 早期通信、EPLB、SBO 与 profiling

07-24 至 07-26 的目录保存了 dynamic/static one-batch、DP permutation、request profile、EPLB A/B、DeepEP normal/low-latency、SBO/overlap 和多轮 Nsys。它们主要用于回答：服务到底命中了哪条路径、通信等待在什么阶段、chunk 修复是否改变了轮次数。

这些实验的价值更偏向**重建执行图**，不是一组统一 promotion suite。许多目录只有 config、one-batch JSONL、派生 kernel 分类或空目录；因此在总索引中保留，但不强行汇总成一个“平均收益”。

## 6. 阶段二：decode、MTP 与 kernel frontier

### 6.1 MTP/EAGLE：真正的 decode 正结果

MTP 是 speculative decoding：draft 路径先预测多个 token，主模型一次验证。如果平均每次验证能提交更多 token，昂贵 forward 的摊销变好。

本地 JSONL 可直接证明的历史 cell 是64K context、batch8、output512；EAGLE、1 speculative step、Top-K=1与最多2个draft token来自历史handoff metadata，当前compact缺完整launch command：

| 指标 | Baseline | MTP | 变化 |
|---|---:|---:|---:|
| TPOT | 32.15 ms | 24.33 ms | 改善约24.3% |
| throughput | 248 token/s | 329 token/s | +约32% |
| acceptance length | 1（普通decode语义基准） | 1.87 | 每次验证提交量 +87% |

baseline JSONL 实际字段为 `acc_length=-1`（未记录/不适用），表中的1不是实测字段。若handoff的draft2合同得到恢复，则 `1.87/2=93.5%` 是draft接受比例；不能写成“相对baseline多93.5%”。这组约1.32×是decode方向的历史结果，不属于F1 output=1 TTFT；TPOT由batch/global output throughput派生。

### 6.2 decode winners 与算子替换

S=32768、output48、约100个样本的 BS128 结果中，组合 winner（FlashMLA P1+c2、o_proj、index_q_upproj、MoE M-tile align）把 median ITL 从 33.116 ms 降到31.243 ms，约5.66%。BS256 只有不完整样本，不能给同级结论。

另一个 operator-swap 实验展示了 Amdahl 定律：

- decode `o_proj` leaf 约快10.6%；
- `index_q_upproj` leaf 约快13.7%；
- 二者在该 decode trace 中只占全部 GPU kernel 约1%；
- `all_infini` E2E 只有约+0.5%到+0.6%；
- prefill fused-QKV 没有命中目标 M4096，TTFT反而慢2.1%到2.9%。

因此“专用 kernel 存在并 HIT”仍不够；还要证明它覆盖真实 shape、位于关键路径，并且没有通过额外 dispatch、同步或内存压力抵消收益。

### 6.3 64K/短上下文 gain-ops matrix

这组矩阵同时暴露了 workload sensitivity：

- 64K BS16 中 fused QKV 的 TPOT 31.29→36.27 ms，明显更慢；index_q 只有约1.6%方向改善。
- 64K BS32 中三种候选 TPOT 大约改善12%–13%，但 TTFT 仍慢约0.7%–1.1%，且样本不构成当前正式门。
- seq1024/2048/4096 的 TPOT 基本在±1%–3%；仅 seq2048 TTFT 有约7%–9%方向改善，邻近 shape 不复现。

这说明 shape、batch 和阶段必须成为 admission guard 的组成部分，不能把“在 M16 快”扩写成“对整个模型快”。

### 6.4 MoE kernel、alignment、PTX 与 PSUM

- 小 expert load 的 decode 中，把 alignment 从128缩到16可少算 padding，leaf 约1.13–1.14×；但在 prefill 大 M 中全局启用会严重回退。
- N26 的同路径数据更直接：M1024 的 W13 111.8→307.8 µs；W2 115.7→354.3/478.7 µs（align16/32），均更慢。
- 手写 PTX 候选约152–329 µs，参考约74 µs，并存在错误，直接放弃。
- contiguous PSUM 在 M1024/2048 的 fair DeepEP-normal A/B 中 TTFT 回退2.28%/1.68%。

低层代码不是天然更快。寄存器压力、shared memory、tile 选择、额外 padding、编译器 pipeline 和调用边界都可能抵消“少一次写回”的直觉收益。

### 6.5 DeepEP low-latency 与 graph 实验

08-01 至 08-03 保存了 dispatch/combine wait、QP40、route normalization、graph correctness、graph multiset、rank duration、MoE graph bucket、FlashMLA R2A、GSM8K A-B-A 等大量 leaf/trace 结果。它们帮助发现等待、图捕获和小 M 的局部规律，但 workload 与 F1 的 DeepEP normal 100K cached-prefill 不一致。

因此这些目录全部保留在历史 compact archive 中，但不会把 low-latency/decode 的累计 kernel duration拿来证明 F1“通信占比”。跨 GPU、跨 stream 的累计 duration也不是 wall critical path。

## 7. 阶段三：100K 迁移与 MoK

### 7.1 100K route-complete/indexer 迁移

这一轮没有直接复制 DeepSeek-V4 的 clustered MQA，因为 DSV4 的 H64/top-k512/C4/Q16 合同与 GLM 的 H32/top-k2048/ragged contiguous KV 不兼容。实际迁移的是 workspace/output 简化；GLM scheduler chunk M 可到8192，并用显式 path marker 证明8个rank都命中。

冻结负载是 TP8/DP1/EP8、90K prefix + 10K suffix、output1、chunk8192，x表示客户端并发、请求数为x×10：

| x | Control P50 | Candidate P50 | P50变化 | P90变化 | 吞吐变化 | 结论 |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 999.61 ms | 1010.89 ms | -1.13% | -0.84% | -0.85% | 更慢 |
| 2 | 1417.17 ms | 1390.21 ms | +1.90% | +7.17% | +2.43% | 单点正信号 |
| 3 | 2254.02 ms | 2252.79 ms | +0.05% | +0.04% | +0.11% | 打平且P50超2s |
| 4 | 2822.35 ms | 2836.38 ms | -0.50% | -1.33% | -0.73% | 更慢 |

相邻并发不复现，故结论是“代码迁移并真实命中，但没有稳定 E2E 收益”。它也揭露了一类 benchmark 陷阱：跨轮不 flush cache 会把后续请求变成接近100K全命中，产生看似很好的假结果。

### 7.2 MoK megakernel：稳定性成功，性能失败

MoK 不是 MTP。MoK 把 MoE 计算/通信做成更大 GPU 边界，使用 MXFP8、symmetric memory 等。原实现面对 ragged DP rank 时，在 MoK 与 DeepEP 两种 collective family 之间切换会出现 timeout/OOM。

修复包括：EP-wide arm consensus、模式切换前 drain+barrier、约1.08 GiB workspace预分配。稳定性从x10 15/100、x12 1/120、x16 12/160，恢复到100/100、120/120、160/160；这是实质性的progress/stability修复。它仍是screening：相邻arm的framework random seed未完全冻结，多数点仅一个样本；token-exact screen通过，但logprob阈值未预先声明，不是formal correctness/performance promotion。

性能却没有晋级：

| x | P50变化 | P90变化 | 吞吐变化 |
|---:|---:|---:|---:|
| 10 | +8.40%更慢 | +31.29%更慢 | -14.42% |
| 12 | +7.55%更慢 | -1.43%（更快） | -8.36% |
| 16 run1 | +3.00%更慢 | -7.39%（更快） | +4.31% |
| 16 run2 | +5.91%更慢 | +4.04%（更慢） | -3.50% |

x16 的方向不能复现；模型内存约109.45→207 GiB/rank，KV capacity 2,103,808→205,888。正确的裁决是“保留稳定性/debug 分支，拒绝性能 promotion”。

## 8. 阶段四：冻结 100K cell 的 N1–N40

N-series 是本次证据纪律最完整的一段：先冻结目标和门，再逐项处理 compute、communication、control 和 dependency boundary。完整逐项教学见 [N1–N40 决策账本](N1_N40_COMPLETE_LEDGER_CN.md)，权威原始决策文件也已保存。

### 8.1 N6 为什么成功

N6 由三项原子组合构成：

1. balanced static placement：让256个 expert 在8个 EP rank 间更均匀，降低最慢 rank skew；
2. router 直接 physical-ID：静态重排后直接产生 DeepEP 可消费的 expert ID，删去二次 remap/mask；
3. DeepEP normal dispatch/combine 共用120 SM，而不是136，改变通信与其他计算对 SM 的资源平衡。

正式五对结果：

| 指标 | Baseline | N6 | 变化 |
|---|---:|---:|---:|
| P50 TTFT | 2042.12 ms | 1933.67 ms | -5.31% |
| P90 TTFT | 3035.39 ms | 2891.13 ms | -4.75% |
| token throughput | 452544.29/s | 486640.21/s | +7.53% |

P50与吞吐各4/5 paired wins，P90是3/5。独立correctness probe相对两个reference的generated tokens exact；随后独立client seed holdout复现P50/P90/吞吐 -5.46%/-6.73%/+7.40%，P50 5/5 wins。两者不是同一组运行。正式数字只能归因于三项组合，不能把N1、N4、SM sweep的百分比相加。

matched Nsys 支持的组合因果是：per-device dispatch progress 616→654（+6.17%），`cached_notify_combine` P50/P90 -43.65%/-45.91%，combine 主 kernel P50基本不变，且每 GPU 删除约316–466个 post-router mask。它说明 rank progress/control tail 改善；没有隔离证明“120 SM单项让 attention/GEMM 更快”。

### 8.2 N1–N40 的整体模式

N1–N40 大致分为五类：

- **删错误边界或重复工作**：N2/N3/N5/N14/N15 多数在静态审计或 E2E 被拒；边界不在关键路径时，代码再漂亮也没有价值。
- **MoE/DeepEP 资源与 chunk**：N6 accepted；N8协议不支持；N9、N17、N19、N20均回退或不重复。
- **Attention/FlashMLA**：N22反向，N23 leaf -8.85%但E2E P50只有-0.949%，N24正式1/3，N29–N34多数反向或中性，N35因anchor drift no-decision，N38 P50略慢。
- **依赖栈与算子融合**：N28新DeepGEMM wheel ABI失败；N39保留PDL；N40 leaf快59.1%但E2E P50/P90/吞吐分别回退1.56%/18.61%/6.01%。
- **scheduler/control**：N16 SBO、N18 overlap scheduler、N21 PrefillDelayer都正确但E2E主指标失败。

N40是最值得初学者反复看的反例：一个 byte-exact、leaf 快59%的 fusion，进入多 rank、多 stream、DeepEP 服务后仍全面回退。这不是 leaf 测量“错了”，而是 leaf 回答的问题太窄。

## 9. 阶段五：扩大边界、允许局部回退

### 9.1 FlashMLA 八 shape band

最终 development band 显式 admit `M=9616,9728,9792,9856,9920,9984,10016,10048`；范围外走 registry-local stock fallback，selected shape 异常则停止。8个 shape 在两种顺序下 output/LSE bit-exact，leaf median 快8.82%–12.49%。

这是“扩大优化空间但保持局部回退”的正确软件结构：不是把所有 M 都强行导向一个 kernel。但后续 host 已退化，没有形成健康主机 formal E2E，所以不能把8.82%–12.49%写成 TTFT 收益，也不能称 production-promoted。

### 9.2 temporal placement v1/v2/v5

aggregate placement 只看整场总 token；temporal placement 尝试减少“同一时刻哪个 rank 最慢”，并允许只替换有证据的层，其余局部回退 N6。

- v1 与 v2 在 degraded host 的 development bracket 中没有晋级；不能和健康 N6 绝对值横比。
- v5 用 seed1训练、seed0 holdout、seed2外部 exact replay，在78行 map 中替换25行，其余53行保留 N6；按 active MoE 计是25/75。
- 外部 proxy：compute mean约-1.05%、max-channel mean约-0.505%、send count不增。
- 正确性：两份 reference 均11/11 token exact；max/mean logprob 1.5565e-4/3.4867e-5。
- degraded-host A/v5/C 相对 control 中位：P50 -68.89%、P90 -79.38%、吞吐 +257.53%；但候选绝对 P50=3119.99 ms、吞吐=320683.49/s，未达到 F1 门。

matched Nsys 显示 FlashMLA per-call基本不变，而 dispatch/combine notify 的P90/P99明显缩短。因此 v5 的合理因果是跨 rank arrival/control-plane tail 改善，不是 attention kernel 变快。正式状态是 `RELATIVE_WIN_BLOCKED_BY_HOST_ANCHOR`，不是 promotion。

### 9.3 DeepEP send16、equalAG 与 overlap

real-shape send16 leaf probe 中，最慢rank dispatch median -2.45%、wall -1.53%，P90 wall -4.48%；server absolute gate仍失败。equal exact-fill all-gather通过model/unit/source contract，但本地compact缺专属path-hit与no-profiler E2E，因此不能申报收益，也不能独立复核“实际no-op”的更强归因。SBO、native overlap scheduler、PrefillDelayer分别展示了“正确但E2E主指标没缩短”“改变batch推进导致严重回退”“用median代价换tail”的不同失败模式。

这些负结果共同说明：通信参数对队头阻塞、控制开销和 rank tail 极敏感，不能从单 rank microbenchmark直接推导 server TTFT。

### 9.4 CPU frequency-domain fault 与 affinity

08-15后，256个 logical CPU 中 socket0全部128加socket1低半64在独立负载下约500MHz；只有socket1高半64 logical（约32 physical core+SMT）稳定约3.6GHz。governor、turbo、RAPL、温度正常，未见 thermal/Xid/NVLink/GPU故障。证据只允许写“CPU frequency-domain/platform fault”，不能声称已定位 BIOS 或驱动根因。

旧 worker affinity 会按 global CPU号重算并逃出父 cpuset。修复候选让父 mask 成为 SSOT，按 `(socket,core)` 分组 SMT sibling，给8个 worker平衡且互斥的 partition，不足则 fail closed。2个 unit tests、23-file runtime contract和process-tree audit通过。

但 degraded-host single arm 仍未过门：all-siblings P50/P90/吞吐=2716.96/2899.52 ms/412371.30，physical-only=2752.50/2851.30/412677.09。它证明隔离 primitive 更正确，却没有修复平台频域故障；而且 treatment 同时含 v5+cpuset+topology，不能估算 affinity 的独立收益。

## 10. Profiling 的正确使用方式

本档保留 Nsys/NCU 的派生 JSON/CSV，不保留 raw `.nsys-rep/.sqlite/.ncu-rep`。这样仍可复核 kernel call、分位数、scheduler sync、NVTX 和 NCU counter，但不能重新打开完整 timeline 做任意新查询。

三条规则必须遵守：

1. profiled TTFT 只用于定位，不与 no-profiler baseline 比性能；工具本身会显著扰动时序。
2. 多 GPU、跨 stream 的 kernel duration sum 不能当 wall time，也不能直接称 critical-path 占比。
3. NCU 适合已经缩小到单 kernel 的瓶颈；如果 Systems 显示问题是 rank arrival/notify tail，就不应为了“有 NCU 数字”而强行分析一个非瓶颈 kernel。

本活动唯一 raw NCU 报告被排除，但保留了 `glm52_native_mqa_topk_ncu/.../details.csv`，包含两次 launch 的 duration、occupancy、register、shared-memory 和 SOL 指标。它只适用于对应 MQA/top-k leaf case，不是 N6/v5 E2E authority。

## 11. 最终决策矩阵

| 等级 | 候选 | 当前正确表述 |
|---|---|---|
| F1 accepted | N6 | 只在冻结 100K cached-prefill cell 通过 correctness、重复性、绝对门和 holdout；尚非所有 workload 的 production replacement |
| 历史正结果 | chunk fix、DeepEP24组合、decode MTP、decode BS128 winners | 各自在自己的 H1/H2 cell 有效；不得与 N6 数字相加 |
| research priority | temporal v5、FlashMLA 8-shape band、cpuset-safe affinity | 已有 correctness/leaf/relative causal 信号；需健康主机重新完成无 profiler E2E |
| stability/debug | MoK fix | 稳定性显著修复，但性能和内存/KV容量失败 |
| E2E promotion未获准/被拒 | N1–N5、N7–N40除N6，以及早期大量kernel/overlap/通信候选 | 很多候选在static、ABI、correctness或leaf层停止；正向leaf/research证据仍逐项保留，不能统称“全是负结果” |
| current main | default-off N6 runtime port + maps/tools/docs | source-reviewed；未在 B300 main HEAD 上复现旧数字 |

## 12. 下一台健康 B300 的最小复验顺序

1. 先恢复 F1 精确合同：8×B300、TP8/DP8/EP8、attention TP1、90K+10K、output1、x11/110、page64、chunk80384/10048、FlashMLA-KV、DeepEP normal、overlap/CUDA graph关闭。
2. 先跑 identity/136 baseline，要求110/110、P50≤2s、P90≤5s、吞吐≥438k/s、prefix/path marker和噪声稳定；不恢复 anchor，就停止解释候选绝对值。
3. 在冻结 runtime 重放 N6 formal+holdout，确认 map SHA和selected marker。
4. 在 current `main` 单独做 source-port correctness和同合同性能复现，不能用旧 revision的数字替代。
5. 以 N6 为 control，先测 v5 同 affinity A-B-A+correctness；再测 FlashMLA band。cpuset patch先做process-tree audit，再作为固定环境，不与候选同时变化。
6. 只有 Systems 把回退缩小到一个具体 kernel 后，才用 NCU；否则继续分析 rank progress、notify tail、stream/event和host launch。

## 13. 如何继续查阅

- 想看每个 N 的修改和数字：读 [N1–N40 完整账本](N1_N40_COMPLETE_LEDGER_CN.md)。
- 想找任一早期目录：查 [350目录索引](ALL_350_DIRECTORIES_INDEX.md)。
- 想知道某文件为什么没复制：查 排除账 (`private-archive:inventory/excluded_files.tsv`)。
- 想核验本地字节：查 compact SHA256SUMS (`private-archive:validation/SHA256SUMS`)。
- 想看 JSON 历史格式异常或隐私边界：读 [归档覆盖说明](ARCHIVE_COVERAGE_AND_GAPS_CN.md)。

这份总账的核心不是把所有实验压成一个百分比，而是保留每个实验回答了什么、没有回答什么，以及为什么 accepted、rejected、no-decision 或 blocked。这样下一轮优化才能从已有证据继续，而不是从目录名和单次好看的数字重新猜一次。
