# GLM-5.2 fused W13 prefill final report

## Disposition: no production replacement

The library-native PSUM row layout is a validated component improvement for
the frozen production ABI, but it is not enabled in SGLang. Production remains
on the stock contiguous DeepGEMM path because this four-GPU host cannot run the
required TP8/DP8/EP8 containing-region and end-to-end gates, and the target
GLM-5.2 FP8 checkpoint is absent.

This is an externally blocked no-replacement disposition, not a claim that the
component has no headroom. Stock fallback is the only active production path.

## Reachability and ABI

Current SGLang source disproves the plan's stale masked-entry premise. Normal
DeepEP prefill sets `use_masked_gemm=False`, converts dispatch output to an
aligned expert-major buffer with `ep_scatter`, and reaches one fused
`grouped_gemm_nt_f8f8bf16_contig` call with N4096. No separate N2048 gate/up
kernel is substituted.

The frozen component uses:

- 32 local experts, K6144, fused N4096;
- 32,982 valid and 35,200 aligned rows from provisional EP8 replay rank 5;
- FP8 E4M3 activation and weights;
- packed `int32` UE8M0 activation/weight scales with production strides;
- int32 row layout or 32-element PSUM cumulative endpoint layout;
- BF16 output, PDL enabled, recipes `None`, current CUDA stream;
- valid-row W13 correctness plus the following SwiGLU boundary.

The fixture is explicitly
`provisional_ep8_router_contract_replay_not_live`; it does not masquerade as a
checkpoint-derived capture. Full details are in [reachability.md](reachability.md)
and [shape_abi_capture_status.md](shape_abi_capture_status.md).

## Source and integration delta

SGLang commit
`7f0365a457379b8f6d435f55fcd6d16b817f733a` exposes four keyword-only
DeepGEMM controls through the contiguous wrapper. Default callers forward an
empty keyword dictionary, and no registry entry, environment switch, shape
branch, or production call site selects PSUM.

Kernel-Harness adds the exact serving-native workload, frozen fixture,
production-ABI builder, NaN-poisoned output, transitive source hashes,
fail-closed candidates, paired validators, Nsys/NCU collection, and EP4
diagnostics. No installed package, frozen task, oracle, timing code, generated
task metadata, legacy file, or historical knowledge entry was changed.

DeepGEMM itself is unchanged. Profiling uses fresh line-info JIT cubins for the
upstream E1 and E5 specializations. The EP4 diagnostic resolves an isolated
copy of the exact stock SGLang-pinned DeepEP wheel; build/import hashes are in
[source_build_record.md](source_build_record.md).

## Paired component result

All rows use the same physical B200 0 under one all-GPU reservation, three
warmups, ten interleaved pairs, and correctness before timing.

| Arm | Reference p50 | Candidate p50 | Paired speedup | p10--p90 | Correct |
|---|---:|---:|---:|---:|---:|
| identity 1 | 0.646144 ms | 0.645088 ms | 0.999950x | 0.997324x--1.009217x | yes |
| identity 2 | 0.644848 ms | 0.641824 ms | 1.003516x | 0.999950x--1.011677x | yes |
| identity 3 | 0.643840 ms | 0.643360 ms | 1.000099x | 0.997176x--1.006438x | yes |
| selected PSUM | 0.646128 ms | 0.614240 ms | 1.052746x | 1.047795x--1.057321x | yes |

The no-expected-M arm measured 1.053284x and is indistinguishable from the
selected explicit-M1024 arm at this noise floor. Output-gap zeroing measured
1.039054x; `compiled_dims=mnk` measured 1.050259x. Neither adds a justified
delta. Raw JSON and the complete table are in
[paired_w13_table.md](paired_w13_table.md).

## Profiler diagnosis

NCU measures 629.824 -> 599.328 us (1.050884x), while five matching Nsys
launches have 633.117 -> 601.277 us medians (1.052954x). The trace contains
exactly five stock E1 and five PSUM E5 launches.

PSUM keeps input TMA bytes/instructions unchanged, removes 94.128% of scalar
global-load instructions, and reduces UTCQMMA work, TMEM loads, and output TMA
stores by 5.545% each. Long-scoreboard and barrier stalls improve. It trades
that for 38 -> 50 registers, +25.247% branch instructions, +20.533%
shared-load conflicts, and +43.476% shared-store conflicts. Shared memory
still limits both variants to one CTA per SM, so the register increase causes
no occupancy loss or spill.

The remaining kernel is compute/latency/synchronization limited: tensor-pipe
activity is about 84% of elapsed cycles, DRAM-active cycles are below 32%, and
the dominant source hotspot is `mbarrier.try_wait.parity`. A speculative CuTe
rewrite was rejected before modification because the existing library
specialization already clears the component gate while the real endpoint
adapter and containing-region boundary remain unvalidated.

The complete six-dimension analysis, timing caveat, source/SASS mappings, PM
series, raw reports, and hashes are in
[the profiler report](../../../profile/moe-w13-prefill-psum-vs-stock-20260723b/REPORT.md).

## Communication, region, and end-to-end status

The separately labeled EP4 diagnostic passed three stock identity controls
each for normal DeepEP dispatch and combine. Dispatch p50s are approximately
2.08--2.21 ms and combine p50s approximately 1.72--1.77 ms; identity ratios
span 0.986847x--1.018670x. DeepEP used its default 20 communication SMs and
reported failed IBGDA initialization, both preserved in the log.

These rows establish only local EP4 communication behavior. They use four
ranks, 8,192 local tokens, and 64 local experts per rank. They are not added
to the single-GPU component latency, are not called a containing-region
measurement, and are never relabeled TP8.

The required region

```text
DeepEP dispatch -> W13 -> SwiGLU+quant -> W2 -> DeepEP combine
```

and complete SGLang prefill remain unmeasured at TP8/DP8/EP8. The host has
four B200s, and only a GLM-5.2 NVFP4 checkpoint is present; NVFP4 is not a
surrogate for the target FP8 packed-scale ABI. See
[deepep_region_table.md](deepep_region_table.md),
[end_to_end_table.md](end_to_end_table.md), and
[external_validation_blocker.md](external_validation_blocker.md).

## Enable/fallback policy

Current production policy is unchanged:

```text
SGLANG_GLM52_OPT=0
grouped_gemm_nt_f8f8bf16_contig(..., m_indices)
use_psum_layout=False
```

A future integration may select PSUM only for the exact named W13 prefill,
packed-FP8, normal-DeepEP, TP8 bucket after checkpoint-derived endpoints,
full-output and downstream correctness, graph/stream/recipe/signal/overlap
semantics, the full TP8 region, and end-to-end accuracy/performance all pass.
Unsupported cases must fall back without a device-to-host decision. The exact
policy is in [final_policy.md](final_policy.md).

## Validation and provenance

- `serving_native/selftest.py`: 40 fixed workloads, passed.
- `testbench/bin/selftest.py`: 24 tasks, 0 problems.
- Evidence validators: 11 tests, passed.
- Direct SGLang wrapper/registry tests: 10 tests, passed.
- Knowledge lint/index/distill: 13 entries, 0 problems, generated views fresh.
- Locked B200 `verify_harness.py`: passed; all 24 task directories are in sync
  with `glm52_ops`.
- Component JSON validator: all seven raw runs and source manifests passed.
- EP4 JSON validator: all six results, topology/import identities, and source
  manifests passed.
- Profile assertions: five E1/five E5 launches, full/source reports, 120/137
  mapped source lines, and both WarpStates PM series passed.
- Both worktrees were clean before each cited campaign.
- `verify_harness.py --pointer-report` retains one advisory corpus issue:
  `runs/index.jsonl` is absent. No result pointer is cited.

The append-only attempt history, including rejected paths and the failed
`20260723a` Nsys export attempt, is in [attempt_ledger.md](attempt_ledger.md).
The append-only knowledge entry is
`glm52--moe_w13_prefill_production--b200--20260723a`.

## Completion boundary

All work possible on this host is complete: current-path reachability, exact
ABI workload, correctness, paired component baselines and variants, Nsys/NCU,
EP4 communication diagnostics, source review, provenance, policy, tests, raw
artifacts, and knowledge capture.

The unweakened TP8/DP8/EP8 region and end-to-end gate remains external. Until
that gate can run with the target FP8 checkpoint, there is no production
replacement and stock fallback remains active.
