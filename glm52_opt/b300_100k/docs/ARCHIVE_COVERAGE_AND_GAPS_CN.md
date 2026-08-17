# 全实验证据覆盖、排除与缺口说明

## 1. 覆盖结论

私有 `all_experiments/evidence/by_experiment` 与源端 inventory 生成的 allowlist 精确一致：

| 检查 | 结果 |
|---|---:|
| 远端顶层实验目录 | 350 |
| 根级审计条目 | 1（`__root_files__`） |
| 远端文件 | 20,752 |
| 远端字节 | 259,480,606,210 |
| 应复制 compact 文件 | 7,056 |
| 实际本地文件 | 7,056 |
| 本地 compact 字节 | 105,935,909 |
| 缺失 | 0 |
| 多余 | 0 |
| 大小不符 | 0 |
| raw/binary/超过2MiB混入 | 0 |

私有归档的 `validation/SHA256SUMS` 为每个 compact 文件重新计算 SHA-256；它是归档包的字节级 SSOT。原实验目录中的 `results.sha256` 仍按原字节保留，但它可能引用被有意省略的 log/raw profile，因此不能把“原 manifest 存在”误解为原目录完整复制。

## 2. 为什么没有复制 241.66 GiB 全量

本包的目标是保留结论闭环，而不是复制可以重新打开完整timeline的所有原始采集。第二轮审查把此前误漏的CSV、诊断TXT、校验清单、环境配置和runner补入后，排除账仍完整记录13,696个文件、259,374,670,301 bytes：

| 原因 | 文件数 | 字节数 | 解释 |
|---|---:|---:|---|
| 单文件超过2MiB | 315 | 258,902,564,462 | 主要是 raw Nsys/SQLite、archive、大 tensor；这是空间主体 |
| source/build/dependency | 7,756 | 136,273,853 | cache、build、site-packages、csrc 等；代码另由 Git/source snapshot保存 |
| log/runtime/pointer | 4,320 | 250,576,926 | server/launch/stdout/stderr/PID/env/run path，噪声与隐私风险高 |
| raw profile/binary | 164 | 59,151,067 | 小于2MiB但扩展名仍为 `.ncu-rep/.so/.pt/...`，继续排除 |
| 非结果文本类型 | 899 | 26,086,036 | 主要是编译中间格式、CUDA cache与非结果文件；runner/config已提升到compact |
| 非结果 TXT | 242 | 17,957 | 仅PID、rep-base、poll和latest-prefix等运行指针 |

完整逐文件理由在 excluded_files.tsv (`private-archive:inventory/excluded_files.tsv`)，不是用一条 glob 删除后不留记录。

## 3. 保留了哪些证据

allowlist 保留所有不超过2MiB、具有结果语义的：

- Markdown 报告、README、EXPERIMENT、决策账本；
- E2E `samples.tsv`、summary、paired order、server args；
- correctness、token/logprob comparison、operator output/LSE gate；
- workload/config/runtime source contract、manifest、SHA/verify；
- map、selection summary、offline replay、temporal placement派生结果；
- Nsys 的 per-GPU kernel、scheduler sync、CUDA API、NVTX、causal comparison；
- NCU 唯一导出 `glm52_native_mqa_topk_ncu/.../details.csv`；
- 小型 JSONL/CSV 的早期 A/B、decode TPOT 和 leaf timing。
- 历史 `.env`、`.sh`、`.py` reproduction assets；它们不算独立measurement，但保存运行合同与分析逻辑。

这套规则比“只保留 accepted”更宽：失败启动、negative E2E、no-decision、静态拒绝和参考实现也保留，从而避免未来重复试错。

## 4. 六个空目录

以下目录在迁移后的 `bench_results` 顶层存在，但 inventory 中为0文件：

1. `b300_deepep_ll_dispatch_wait_stats_20260802i`
2. `b300_w2_deepep_zero_copy_stride_gate_20260802d`
3. `glm52_large_boundary_nsys_20260815`
4. `glm52_m1024_timeline_20260724T1312`
5. `m3_glm52_64k_breakdown_20260724T1811Z`
6. `nsys_glm52_originalflag_m1024_20260724T1317`

它们在 350目录索引中明确标为“空目录/证据缺口”。目录名可以证明曾计划或创建过这个实验位置，不能证明实验完成，更不能补造结果。

其中 `m3_glm52_64k_breakdown` 相关的旧容器 tar 曾因权限阻断未完整迁移；它与当前 N6/v5 结论无关，但该历史 cell 不能仅靠现有包完整复跑。

## 5. 50 个历史 `.json/.jsonl` 格式异常

7,056个文件中有50个无法按严格JSON/JSONL parser解析。本包没有悄悄修复或删除，因为修改会失去source fidelity：

- 15个是空/空白响应；
- 27个是命令输出或纯文本却使用`.json`后缀；
- 8个是JSON-like内容后带额外尾部、截断或拼接垃圾，严格parser报`Extra data`等错误。

逐项见 json_validation_failures.tsv (`private-archive:validation/json_validation_failures.tsv`)。这些文件可作为历史命令/命中证据阅读，但在程序消费前必须按实际格式解析，不能假设扩展名就是 schema。

## 6. 隐私与公开边界

扫描结果：

- 私钥、GitHub token、AWS key、credential URL 或已赋值 secret：0个候选；
- 含内部绝对路径、用户名、jump host或私网地址：3,127个文件、6,210条模式命中。

因此整个 `all_experiments` 树被标为 **PRIVATE AUDIT ARCHIVE**。它适合本地审计，不适合原样推 GitHub。可公开内容仍应使用根目录已有的 `evidence/public_compact` 和 GitHub main 中的脱敏 bundle。

内部路径本身通常不是 credential，但会暴露用户名、挂载结构、容器位置和机器拓扑。若以后要公开某个历史实验，应逐文件：

1. 删除/规范化内部路径和 host identity；
2. 保留原始 SHA 到 private provenance；
3. 为 sanitized 文件重新生成 SHA；
4. 说明性能 claim scope 没有因脱敏而变化。

## 7. raw profiler 的能力缺口

不复制 raw `.nsys-rep/.sqlite/.ncu-rep` 的代价是：无法在本地重新打开 timeline、重跑任意 SQL、查看未导出的 CUDA event 或重新选择 NCU metric。保留的派生表足以复核当时写下的 causal claim，但不适合提出全新的 profiler 问题。

已保留的边界是：

- N6 matched Nsys：per-GPU kernel、scheduler sync、CUDA/NVTX summary、router comparison；
- v5 matched Nsys：两臂派生表和 causal comparison；
- DeepEP send16及其他实验的 compact causal/summary；
- MQA/top-k NCU `details.csv`，raw `.ncu-rep`排除。

如果未来新问题必须依赖完整 timeline，应回到私有 raw archive；GitHub compact 不能假装具备该能力。

## 8. 重复文件与 canonical source

历史 runner 复制了大量相同 map和帮助文件。例如 accepted N6 map 在远端多个实验目录重复出现，其 canonical SHA256 是：

```text
36d13233672288317fd69495d4cedb46844b8ae99033d184d84aff0c99c68f09
```

v5 map canonical SHA256 是：

```text
570e58026ae890bfd294a34786a095142ca71d14747c87606a536b20c415620a
```

本档为了保持每目录字节闭环而保留重复副本；运行时/文档 SSOT 仍是 GitHub main 的 canonical map。重复副本不能被解释为更多独立实验样本。

## 9. 仍然无法完全冻结的要素

- 有 config/index/dataset hash，但 full weight shards、tokenizer、quant artifacts和稳定 model revision仍不完整；
- 容器 image digest 已记录，但不是一个可独立获取、验证全部层的发布来源；
- current GitHub main 是 surgical port，尚未在 B300复现冻结 runtime的 N6数字；
- F2 的 CPU frequency-domain/platform fault 未定位到具体 BIOS/firmware/driver根因；
- raw profile和大日志被有意省略，不能在本地重新做任意因果挖掘；
- 某些早期实验只保留单臂/小样本/不同 seed，不满足后来 N-series 的 formal gate。

这些缺口不会否定已经通过的局部正确性或 F1 N6结论，但限制了结论的可推广范围。

## 10. 私有归档校验入口

- coverage_summary.json (`private-archive:inventory/coverage_summary.json`)
- validation_summary.json (`private-archive:validation/validation_summary.json`)
- compact SHA256SUMS (`private-archive:validation/SHA256SUMS`)
- missing_files.tsv (`private-archive:validation/missing_files.tsv`)
- unexpected_files.tsv (`private-archive:validation/unexpected_files.tsv`)
- size_mismatches.tsv (`private-archive:validation/size_mismatches.tsv`)
- raw_or_oversize.tsv (`private-archive:validation/raw_or_oversize.tsv`)
- secret_scan_candidates.tsv (`private-archive:validation/secret_scan_candidates.tsv`)
- internal_path_files.tsv (`private-archive:validation/internal_path_files.tsv`)

“完整”在这里的精确定义是：350目录全部入账、20,752个源文件全部被`copy`或带理由`exclude`、7,056个allowlist文件逐字节落地且零差异；它不等于241.66GiB raw archive的第二份镜像。`bench_results`之外，私有归档另保存了KDA source/recovery bundle、11份patch、B300环境元数据、17.8MiB固定decode workload和本地分析快照；这些资产由私有根级`PROVENANCE.tsv`与`SHA256SUMS`追踪，不属于公开仓库。
