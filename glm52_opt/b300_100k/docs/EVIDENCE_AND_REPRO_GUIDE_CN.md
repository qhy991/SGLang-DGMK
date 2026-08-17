# 证据阅读、引用与复现指南

## 1. 哪个文件是权威来源

| 事实 | SSOT |
|---|---|
| 冻结 workload、seed、门槛 | [`../workload_contract.json`](../workload_contract.json) |
| N6 正式结论 | [`../evidence/n6/formal/`](../evidence/n6/formal/) |
| N6 holdout | [`../evidence/n6/holdout/`](../evidence/n6/holdout/) |
| N6 正确性 | [`../evidence/n6/correctness.json`](../evidence/n6/correctness.json) |
| N6 因果摘要 | [`../evidence/n6/causal_summary.json`](../evidence/n6/causal_summary.json) |
| v5 A-B-A | [`../evidence/v5/bracket_summary.json`](../evidence/v5/bracket_summary.json) |
| v5 正确性与因果 | [`../evidence/v5/`](../evidence/v5/) |
| accepted expert map | [`../../glm52_100k_x11_static_expert_map.json`](../../glm52_100k_x11_static_expert_map.json) |
| v5 frozen map | [`../../research/temporal_placement/maps/glm52_100k_x11_seed1_p50_rank_v5.json`](../../research/temporal_placement/maps/glm52_100k_x11_seed1_p50_rank_v5.json) |
| main runtime code | [`moe_fused_gate.py`](../../../python/sglang/jit_kernel/moe_fused_gate.py)、[`topk.py`](../../../python/sglang/srt/layers/moe/topk.py) 与 [`environ.py`](../../../python/sglang/srt/environ.py) |
| 完整实验叙述 | [`GLM52_B300_SYSTEMATIC_TEACHING_REPORT_20260817_CN.md`](GLM52_B300_SYSTEMATIC_TEACHING_REPORT_20260817_CN.md) |

其他文档应引用这些文件，而不是手工维护第二份数值真相。

## 2. Public compact 与 private compact

### Public compact

来自已发布 GitHub main，已经移除内部绝对路径、主机标识、raw profile 和日志。它足以支持公开结论，优先使用。

### Private compact

从迁移归档主机的 B300 快照直接复制，保留更多：

- N6 五对与 holdout 的原始 server args/验证摘要；
- N6 baseline/candidate 的 Nsys 派生 JSON/CSV；
- v5 correctness、A-B-A 与 matched Nsys 派生表；
- FlashMLA 八 shape 原始 leaf JSON；
- DeepEP send16 development/causal 摘要；
- cpuset process-tree audit；
- source contracts。

它可能出现内部挂载路径、用户名、容器路径或 SQLite 来源路径，只供本地审计，不能直接推公开仓库。

## 3. 为什么没有复制 raw Nsight

有意排除：

- `*.nsys-rep`、`*.sqlite`、`*.qdrep`、`*.ncu-rep`；
- server/launch/driver/orchestrator logs；
- 模型权重、tokenizer、数据集、容器层和预编译 `.so`；
- per-token recorder tensor；
- 完整 `bench_results`。

理由不是这些文件“没价值”，而是 compact 派生表已经足以复核当前因果结论；raw profile 大、可能含内部信息，而且不能作为 no-profiler 正式性能数字。完整原始归档仍保存在私有迁移主机的归档根目录下。

## 4. 如何读 N6 formal

先读脱敏 samples，再读 summary：

```text
glm52_opt/b300_100k/evidence/n6/formal/
├── samples.tsv
└── summary.json
```

`samples.tsv` 保留五对 baseline/candidate 的顺序和每臂指标；`summary.json` 给出聚合、wins和请求完成数。不要只抄 summary 百分比；需要核对：

- 是否确实 5 对；
- P50 wins 是否 4/5；
- P90 wins 实际是 3/5；
- 吞吐 wins 是否 4/5；
- 请求是否 110/110。

fresh-server 生命周期来自 runner/orchestration provenance，不是 `samples.tsv` 的字段。correctness 也不能从 formal/holdout samples 推出，必须单独读取：

```text
glm52_opt/b300_100k/evidence/n6/correctness.json
```

它记录 candidate 相对两个 reference 的 generated tokens exact；correctness probe 与性能 holdout 是两组独立运行。

Holdout 使用独立 client seed 20260813，而 primary client seed 是 0；server seed 是 565849983。三种 seed 不得混写。

## 5. 如何读 Nsys

Nsys 文件只能回答“为什么可能更快/更稳”，不能替换无 profiler E2E：

- `per_gpu_kernel_analysis.json`：每 GPU 的 kernel 次数、分位数、累计时长；
- `scheduler_sync_analysis.json`：scheduler/同步事件派生；
- `cuda_gpu_kern_sum*.csv`：kernel 汇总；
- `nvtx_sum*.csv`：NVTX 范围；
- `causal_comparison.json`：matched 两臂比较。

注意：

1. notify/wait kernel 的长时间常表示等待其他 rank，不代表它做了很多算术；
2. 跨 GPU、跨 stream 的 kernel sum 不能相加当 wall time；
3. profile 下的 TTFT/吞吐被 profiler 严重扰动，不能和正式无 profiler 数字横比；
4. call count 上升可能是 progress signal，也可能只是 capture 进度不同，必须结合 trace span 和 calls/s。

## 6. 如何读 FlashMLA/DeepEP leaf

`private_compact/flashmla_leaf/` 的九个 JSON 记录：

- 8 个真实 M shape；
- 两种测量顺序；
- 原始 timing samples；
- output/LSE bit-exact；
- tensor/ABI contract。

其中 `extension.path` 可能含内部路径，所以这些文件保持 private。8.82%–12.49% 是 leaf median reduction，不是 TTFT。

`private_compact/deepep_send16/` 保存 development samples、summary 和 paired causal summary。它支持“最慢 rank leaf 有信号”，不支持 server promotion。

## 7. Git 可恢复性

N6 runtime、map/tooling 和第一版 compact evidence 分别由 Git 提交 `01749bb`、`644547a` 和 `38fc37c` 保存。Git 历史本身是公开代码 provenance；私有归档另保留 branch/worktree/bundle 恢复材料，但不把内部路径、未审查二进制或许可边界不清的完整实验树复制到公开仓库。

## 8. 复现时的最小命令语义

Baseline：

- identity expert placement；
- router static placement fusion off；
- DeepEP normal dispatch/combine 136 SM。

Candidate N6：

- accepted map；
- static EP dispatch；
- `SGLANG_GLM52_ROUTER_STATIC_PLACEMENT_FUSION=1`；
- DeepEP normal dispatch/combine 120 SM。

除 treatment 外，模型、镜像、TP/DP/EP、attention backend、chunk、page、cache hit、memory fraction、overlap、CUDA Graph、请求与 seed 都必须冻结。

## 9. 复验判定条件

一次 N6 复验只有同时满足下列条件才有效：

1. main revision 与 dirty state 已记录；
2. 两个 CUDA tests 通过；
3. accepted map SHA 正确；
4. selected marker 出现；
5. 110/110 请求完成；
6. cache hit=89984、per-rank M/chunk 落在合同内；
7. baseline 首先恢复 P50≤2000 ms、P90≤5000 ms、吞吐≥438000；
8. A/B treatment diff 只有 map/router fusion/DeepEP SM；
9. exact generated tokens；
10. 五对与 holdout 通过后才可把旧结论迁移到新 main。

## 10. 哈希验证

从 `glm52_opt` 目录执行：

```bash
shasum -a 256 -c b300_100k/SHA256SUMS
```

`b300_100k/SHA256SUMS` 覆盖公开 workload、报告与 compact evidence；任何被覆盖文件修改后都应重建并复核清单。私有归档有独立根清单，不与公开 manifest 混用。
