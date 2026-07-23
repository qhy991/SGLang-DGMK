# GLM-5.2 production W2 decode optimization report

Date: 2026-07-22

## Disposition

**No replacement.** DeepGEMM alignment 16 is a reproducible, correct
single-B200 leaf improvement for all four named packed-ABI workloads, but the
alignment selector is process-global rather than a fail-closed per-bucket
SGLang oracle. This four-GPU host also cannot run the unchanged
TP8/DP8/EP8 containing-region and SGLang end-to-end gates. No goal-07 bucket is
enabled; stock SGLang/DeepGEMM remains both the active implementation and the
rollback.

This is not a frozen f32-scale E8 result. The measured leaf is the reachable
production symbol, `grouped_gemm_nt_f8f8bf16_masked`, with E32, slab1024,
K2048, N6144, FP8 inputs, packed int32 UE8M0 scales, and BF16 output. The
deterministic masks are exact production-ABI test data, not a live EP8 router
trace.

## Paired leaf result

The canonical campaign used three same-session alternating sessions of 30
pairs per alignment and workload, after five warmups, on physical B200 GPU
`GPU-30b619de-87f2-1862-0d07-a595da8fe417`. Alignment 16 was best on every
named workload:

| Workload | Stock p50 (ms) | BM16 p50 (ms) | Paired p10 / p50 / p90 | Fresh active-row correctness | Production decision |
|---|---:|---:|---:|---|---|
| M16, plan `expected_m=4` | 0.098208 | 0.091376 | 1.020861x / 1.080470x / 1.139754x | exact; `calc_diff=0` | disabled |
| M16, current-source `expected_m=5` | 0.103024 | 0.095040 | 0.983099x / 1.087436x / 1.229160x | exact; `calc_diff=0` | disabled |
| M32, plan `expected_m=8` | 0.103376 | 0.096368 | 1.015899x / 1.075564x / 1.152016x | exact; `calc_diff=0` | disabled |
| M32, current-source `expected_m=9` | 0.100640 | 0.096752 | 0.938654x / 1.062069x / 1.136703x | exact; `calc_diff=0` | disabled |

Each row contains 90 pairs. Pre-timing and fresh post-timing checks used
separately allocated, NaN-poisoned outputs and independently allocated fresh
inputs. Correctness covers active rows; inactive slab rows are not claimed.
Alignment 32 also cleared the 3% median gate on all four rows but was slower
than alignment 16. Alignment 64 cleared only the two M16 rows; alignments 96
and 128 cleared none. The alignment-128 stock-vs-stock controls had paired
medians between 0.992249x and 0.994566x.

Raw results and all negative variants are preserved in
[`paired_alignment_summary.json`](paired_alignment_summary.json) and the
per-run JSON files under
[`profile/moe-w2-alignment16/analysis/microbench/`](../../profile/moe-w2-alignment16/analysis/microbench/).

## Why BM16 is faster

The runtime-selected stock configuration is BM128/BN128/BK128 with load-M64,
eight stages, and 213804 bytes of configured shared memory. The selected
experiment is BM16/BN128/BK128 with load-M8, twelve stages, and 230188 bytes.
Both keep 1,536 logical tile tasks, 11 logical scheduler waves over 148 SMs,
and a 56-task final wave. The persistent kernel itself launches 148 blocks;
NCU therefore reports one launch wave per SM, not 11 launched CTA waves.

For the fixed masks, BM16 removes seven eighths of the padded M and epilogue
surface without changing the static tensor-MMA or input-TMA instruction count.
The generated-code comparison is:

| Evidence | Stock BM128 | Candidate BM16 |
|---|---:|---:|
| ptxas registers / spills / barriers | 36 / 0 / 9 | 34 / 0 / 9 |
| PTX instructions | 1122 | 909 |
| SASS instructions | 1416 | 1112 |
| `tcgen05.mma` / `UTCQMMA.2CTA` | 16 | 16 |
| input `UTMALDG.2D` | 10 | 10 |
| PTX TMEM loads | 32 | 4 |
| SASS output `UTMASTG.2D` | 16 | 2 |
| `UTMACMDFLUSH` | 8 | 1 |
| deferred blocking barriers | 19 | 5 |

These are static instruction counts, not dynamic executed counts. Exact
generated PTX/SASS/cubin hashes and template fields are in
[`jit_inventory.json`](../../profile/moe-w2-alignment16/analysis/jit_inventory.json).

Independent NCU captures agree with that mechanism. Across the four workloads,
stock kernel duration was 75.520--76.256 microseconds and BM16 was
68.576--70.112 microseconds. DRAM reads changed only from about 414.28 MB to
406.90 MB, while DRAM writes fell from 40.41--40.67 MB to 7.94--8.34 MB and
tensor-pipe active percentage fell from 30.41--31.00% to 4.18--4.23%. The
remaining kernel is more latency/coordination limited: eligible warps fell from
about 0.073 to 0.059--0.060 per scheduler and the long-scoreboard ratio rose
from 15.77--16.04 to 21.47--21.62. SourceCounters map the largest samples to
barrier/TMA wait paths rather than showing a register spill or local-memory
problem.

The complete stock and BM16 profile matrices are under these immutable attempt
directories:

- [`stock profile`](../../profile/moe-w2-packed-baseline/analysis/profiles/profile_glm52-w2-alignment-2bae536257aa929b957fdb28_20260722T172516Z_656663_30552/)
- [`BM16 profile`](../../profile/moe-w2-alignment16/analysis/profiles/profile_glm52-w2-alignment-2bae536257aa929b957fdb28_20260722T180259Z_1117292_24148/)

The deterministic cross-attempt reduction is
[`stock128_vs_bm16_profile_comparison.md`](../../profile/moe-w2-packed-baseline/analysis/stock128_vs_bm16_profile_comparison.md),
with the complete machine-readable record in the adjacent JSON file.

Nsight Systems independently captured exactly one selected leaf kernel per
workload, with stock durations of 74.687--76.544 microseconds and BM16
durations of 65.664--66.816 microseconds. The first strict post-collection
audit exposed informational `nsys stats` preamble lines before the CSV header;
the raw reports and extracts were preserved rather than rewritten. The parser
now locates the real header, correlates the API, GPU, and kernel-execution
exports, and both immutable attempt directories pass the strict validator.
The repair and checks are recorded in [`validation.md`](validation.md).

## Correctness and graph boundary

[`leaf_validation_summary.json`](leaf_validation_summary.json) is a strict
PASS for four CUDA Graph artifacts, four edge artifacts, and twelve edge cases.
All four graph captures observed capture active during launch, replayed 30
times deterministically, matched eager output exactly (`max_abs=0`,
`calc_diff=0`), and preserved the return contract. Graph p50 was
0.083968--0.084064 ms.

The edge suites cover empty experts and counts
`0,15,16,17,31,32,33,127,1024`; every active-row comparison passed exactly.
This proves a single-GPU leaf graph and correctness contract only. It does not
prove performance for those edge masks, an overlap-enabled graph, or a full
serving graph.

The separate four-rank attempt
[`tp4_20260722T185932Z_1629770_10420`](tp4_diagnostic/tp4_20260722T185932Z_1629770_10420/summary.json)
also passed strict post-collection audit for 18 stock baselines and six Nsys
reports. It contains no candidate. Its MoE-region check is lower-level eager,
no-overlap EP4 structural/repeatability validation with no independent math
oracle; it is neither candidate region acceptance nor a production graph.

## Reachability and production boundary

Current source reaches W2 after DeepEP low-latency dispatch, fused W13, and
fused SwiGLU plus packed UE8M0 quantization, then feeds DeepEP combine. For the
measured current B200 path, recipes and overlap arguments are null, PDL is
enabled, the eager stream is the legacy default stream (`cudaStream_t == 0`),
and DeepGEMM uses 148 SMs. Plan hints 4/8 and current-source-derived hints 5/9
remain separately named; `expected_m` does not change the selected layout for
these shapes.

Current Blackwell source disables routed W2 down-GEMM/combine two-stream
overlap. The pinned post1 grouped-masked Python API also lacks
`enable_overlap`, `max_block_n`, and `signal`, so this environment cannot
certify an overlap-enabled lane. SGLang commit
`49dc279b59753485c9ad6fa366d289018ecc41d3` adds four CPU contract tests for
recipe/overlap bypass, SM scoping, return identity, replacement success, and
stock fallback. It is test-only: no production SGLang source path was changed.

## Gate disposition

| Deliverable or gate | Result |
|---|---|
| Static call trace and exact packed ABI | complete |
| Single-B200 runtime input/config/cache capture | complete |
| Three-session alternating portfolio | complete; BM16 wins all four leaf rows |
| Nsys, NCU full/PM-sampling/SourceCounters | complete; raw and derived artifacts preserved |
| ptxas/PTX/SASS comparison | complete |
| Fresh-input correctness | passed on all 20 alignment/workload rows |
| CUDA Graph and edge-mask leaf checks | strict PASS |
| TP4/DP4/EP4 diagnostic | strict PASS for stock-only eager/no-overlap diagnostic; [exact attempt](tp4_diagnostic/tp4_20260722T185932Z_1629770_10420/summary.json) |
| Live EP8 masks and exact eight-rank launch contract | BLOCKED on this four-GPU host |
| TP8 full dispatch -> W13 -> SwiGLU+quant -> W2 -> combine | BLOCKED on this four-GPU host |
| Eight-rank SGLang end-to-end decode | BLOCKED on this four-GPU host |
| Production enablement | none; stock remains active |

Every TP4 workload log records the default 20-SM DeepEP communication config,
failed IBGDA transport initialization, and ProcessGroupNCCL device-ID inference
from global rank. Its timing is therefore fallback-environment diagnostic data,
not a tuned communication result; the reported ranges preserve the large
dispatch-M16 and combine-M32 trial excursions.

The local leaf gain is not deployable as measured: changing DeepGEMM alignment
is process-global and could affect other grouped GEMMs and masks, while a hot
path host read or synchronization to choose it is forbidden. The required
eight-rank region and end-to-end checks are also unavailable. The exact
all-disabled policy is in
[`enable_fallback_policy.md`](enable_fallback_policy.md), and the unchanged
external gate is in
[`external_validation_blocker.md`](external_validation_blocker.md).

## Provenance caveats

The measurement campaign is pinned to Kernel-Harness commit
`ac9d47fe3731bca46ee2de2f77b004e124352934`, SGLang commit
`49dc279b59753485c9ad6fa366d289018ecc41d3`, and campaign
`glm52-w2-alignment-2bae536257aa929b957fdb28`. SGLang was clean. The
Kernel-Harness status recorded only untracked in-campaign evidence, reports,
and caches; the exact harness hashes are in
[`alignment_campaign_manifest.json`](alignment_campaign_manifest.json).

Measurements resolve an isolated `sgl-deep-gemm==0.1.4.post1` overlay at
upstream commit `edcf77b276965de8f03cdc47c23f01b08bf7c7ab`; the shared venv
was not overwritten. Wheel, extension, source, CUTLASS/fmt, and build/import
identity are in
[`stock_deep_gemm_provenance.json`](stock_deep_gemm_provenance.json).

The final locked `python3 testbench/bin/verify_harness.py` run exited 0. It
reported selftest 24 tasks/0 problems, knowledge lint 13 entries/0 errors,
current index/distillation, all 24 task directories in sync, a clean diff
check, and zero invalid audit-sweep results. Pointer audit was advisory only:
`runs/index.jsonl` is absent and counted as one malformed pointer. Task,
serving-native, goal-specific contract, and custom evidence checks are recorded
in [`validation.md`](validation.md).
