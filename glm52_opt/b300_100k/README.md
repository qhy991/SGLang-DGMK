# GLM-5.2 100K cached-prefill optimization on 8×B300

This directory is the public, compact source of truth for the 2026-08 B300
optimization campaign. It contains the frozen workload contract, the accepted
N6 result, research-only v5 evidence, and a beginner-first Chinese learning
path that also records the wider prefill, decode, kernel, MoK, N1–N40,
profiling, and host-diagnostic campaign.
Raw Nsight reports, server logs, model assets, host metadata, and private
archive locations are intentionally excluded.

## Accepted result

N6 combines three interacting changes:

1. balanced static MoE expert placement;
2. direct physical expert IDs from the fused router;
3. DeepEP normal dispatch/combine SMs reduced from 136 to 120.

In the frozen no-profiler server test, N6 changed median P50 TTFT from
2042.12 ms to 1933.67 ms (-5.31%), median P90 TTFT from 3035.39 ms to
2891.13 ms (-4.75%), and median total-token throughput from 452544.29 to
486640.21 token/s (+7.53%). An independent client-seed holdout confirmed
-5.46% P50, -6.73% P90, +7.40% throughput, and 5/5 P50 pair wins.

This is an accepted result only for the frozen cell described in
[`workload_contract.json`](workload_contract.json). It is not evidence for
decode, arbitrary context lengths, or deployment-wide replacement.

The router implementation in this publication is a source-reviewed port from
the frozen accepted runtime and remains default-off. The formal numbers are
pinned to that frozen runtime revision; this newer `main` integration has not
itself been rerun on B300 and must reproduce the anchor before broader use.

## Start here

If you are new to CUDA or distributed inference, read in this order:

1. [`docs/LEARNING_PATH_CN.md`](docs/LEARNING_PATH_CN.md): terminology,
   evidence rules, and a 30-minute/half-day/full-study route.
2. [`docs/CUDA_AND_EXECUTION_DAG_CN.md`](docs/CUDA_AND_EXECUTION_DAG_CN.md):
   threads, warps, SMs, memory, streams, Tensor Cores, attention, MoE,
   TP/DP/EP/CP, Nsys, NCU, and the three execution DAGs.
3. [`docs/GLM52_B300_SYSTEMATIC_TEACHING_REPORT_20260817_CN.md`](docs/GLM52_B300_SYSTEMATIC_TEACHING_REPORT_20260817_CN.md):
   the complete corrected system report.
4. [`docs/MAIN_CODE_CHANGE_WALKTHROUGH_CN.md`](docs/MAIN_CODE_CHANGE_WALKTHROUGH_CN.md):
   accepted code path, invariants, tests, and default-off admission.
5. [`REPRODUCTION.md`](REPRODUCTION.md): the normative N6 rerun contract.

## Layout

- [`docs/EXPERIMENT_DECISION_LEDGER_CN.md`](docs/EXPERIMENT_DECISION_LEDGER_CN.md): compact cross-campaign decision index.
- [`docs/ALL_EXPERIMENTS_MASTER_REPORT_CN.md`](docs/ALL_EXPERIMENTS_MASTER_REPORT_CN.md): historical prefill/decode/MTP/MoK/N-series/large-boundary narrative.
- [`docs/N1_N40_COMPLETE_LEDGER_CN.md`](docs/N1_N40_COMPLETE_LEDGER_CN.md): every N1–N40 candidate explained by mechanism, evidence, result, decision, and lesson.
- [`docs/ALL_350_DIRECTORIES_INDEX.md`](docs/ALL_350_DIRECTORIES_INDEX.md): directory-level coverage index for every archived experiment directory.
- [`docs/ALL_RESULT_DOCUMENTS_INDEX.md`](docs/ALL_RESULT_DOCUMENTS_INDEX.md): catalog of 188 historical result documents in the private compact archive.
- [`docs/ARCHIVE_COVERAGE_AND_GAPS_CN.md`](docs/ARCHIVE_COVERAGE_AND_GAPS_CN.md): exact archive coverage, exclusions, known malformed historical files, and capability gaps.
- [`docs/EVIDENCE_AND_REPRO_GUIDE_CN.md`](docs/EVIDENCE_AND_REPRO_GUIDE_CN.md): how to read samples, correctness, Nsys, hashes, and claim boundaries.
- [`RESULTS.md`](RESULTS.md): compact decision ledger.
- [`workload_contract.json`](workload_contract.json): benchmark and promotion SSOT.
- [`REPRODUCTION.md`](REPRODUCTION.md): exact N6 treatment, host-health gate, safe rerun order, and stop rules.
- [`evidence/n6/`](evidence/n6/): formal, holdout, correctness, and causal summaries.
- [`evidence/v5/`](evidence/v5/): research-only correctness, A-B-A, and Nsys-derived evidence.
- [`../glm52_100k_x11_static_expert_map.json`](../glm52_100k_x11_static_expert_map.json): accepted N6 map.
- [`../research/temporal_placement/`](../research/temporal_placement/): frozen v5 map and offline analysis tools; not promoted.

Leaf/operator speedups are never added together or reported as TTFT gains.
Only fresh-server, no-profiler E2E measurements can promote a serving path;
Nsys exports are causal evidence only.

## Public repository versus private archive

The repository intentionally does not contain raw `.nsys-rep`, profiler
SQLite databases, server logs, model/data assets, container layers, host
identities, or unreviewed binary/worktree snapshots. It publishes the
accepted runtime path, canonical maps, frozen contract, compact N6/v5 evidence,
corrected teaching reports, and directory/result catalogs.

The private retirement archive retains 350 experiment directories, a 7,056-file
compact evidence tree, raw profiler/log assets, and recovery material. Catalog
entries prefixed with `private-archive:` identify that recovery source; they are
not broken GitHub links. This boundary preserves all experiment history without
publishing internal paths or treating profiler artifacts as official E2E
measurements.
