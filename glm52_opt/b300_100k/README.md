# GLM-5.2 100K cached-prefill optimization on 8×B300

This directory is the public, compact source of truth for the 2026-08 B300
optimization campaign. It contains the frozen workload contract, the accepted
N6 result, research-only v5 evidence, and a detailed Chinese teaching report.
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

## Layout

- [`docs/GLM52_B300_SYSTEMATIC_TEACHING_REPORT_20260817_CN.md`](docs/GLM52_B300_SYSTEMATIC_TEACHING_REPORT_20260817_CN.md): detailed CUDA-oriented report.
- [`RESULTS.md`](RESULTS.md): compact decision ledger.
- [`workload_contract.json`](workload_contract.json): benchmark and promotion SSOT.
- [`REPRODUCTION.md`](REPRODUCTION.md): exact N6 treatment and safe rerun order.
- [`evidence/n6/`](evidence/n6/): formal, holdout, correctness, and causal summaries.
- [`evidence/v5/`](evidence/v5/): research-only correctness, A-B-A, and Nsys-derived evidence.
- [`../glm52_100k_x11_static_expert_map.json`](../glm52_100k_x11_static_expert_map.json): accepted N6 map.
- [`../research/temporal_placement/`](../research/temporal_placement/): frozen v5 map and offline analysis tools; not promoted.

Leaf/operator speedups are never added together or reported as TTFT gains.
Only fresh-server, no-profiler E2E measurements can promote a serving path;
Nsys exports are causal evidence only.
