# GLM-5.2 MoE W2 prefill optimization report

Status: complete — no replacement; TP8/DP8/EP8 acceptance externally blocked.

## Outcome

The locked single-B200 production-ABI replay showed a 1.064482x paired-median
PSUM component speedup (0.336816 -> 0.316400 ms), with
1.059181–1.065541x p10–p90 and correctness on valid rows consumed by
`ep_gather`. Three identity controls bracketed 0.999100–1.000713x paired
medians, so the component effect is outside the measured local noise.

This is development evidence, not a production win. EP4 normal-DeepEP identity
diagnostics passed under the pinned stock overlay, but they neither exercised
the candidate nor the full MoE region. No checkpoint-derived EP8 capture,
eight-rank containing-region result, or SGLang end-to-end result exists.
Therefore no production caller is enabled and stock row-wise contiguous W2
remains active.

## Reachability and ABI pivot

The plan's masked-GEMM description is stale for current normal-DeepEP prefill.
`DeepEPMode.AUTO` selects normal dispatch for extend batches; `ep_scatter`
produces a flat expert-major tensor plus row-wise `int32 m_indices`; W2 reaches
`grouped_gemm_nt_f8f8bf16_contig` and then
`deep_gemm.m_grouped_fp8_gemm_nt_contiguous`. The masked path and
`_varlen_deep_gemm_silu_mul_quant` belong to low-latency decode. The full
source trace and conditional event semantics are in
[reachability.md](reachability.md).

The added `moe_w2_grouped_prefill_m4096` workload preserves the source-locked
normal-prefill ABI: 32 local experts, K=2048, N=6144, FP8 E4M3 inputs, packed
`int32` UE8M0 scales, BF16 output, 128-row expert alignment, no recipe
override. The component run separately observed PDL=true, 148 SMs, and
tensor-core-util=100; those are not live EP8 control values. The workload
replays one named rank from a deterministic EP8 router-contract fixture: 32982
valid rows in 35200 allocated rows. The fixture is explicitly
`provisional_ep8_router_contract_replay_not_live`; exact shapes, strides, and
blocked live fields are separated in
[shape_abi_capture_status.md](shape_abi_capture_status.md).

## Attempt and paired component results

The component fixture constructs cumulative endpoints matching the values a
normal `ep_scatter` invocation would produce, then passes them to DeepGEMM's
PSUM layout with `compiled_dims=nk`, expected M 1024, and output gap zeroing
disabled. It adds no timed pack, allocation, copy, host read, or synchronization
in the component runner; retaining/passing real scatter endpoints is
unvalidated. An untimed preflight fails closed unless the exact
shape/dtype/stride/packed-scale ABI, PSUM API, PDL, 128-row alignment, endpoint
tensor, and recipe contract match.

All four PSUM variants passed valid-row correctness and the 3% component gate:

| Variant | reference/candidate p50 | paired p50 (p10–p90) | Decision |
|---|---:|---:|---|
| primary, expected M 1024, `nk`, no gap zeroing | 0.336816/0.316400 ms | 1.064482x (1.059181–1.065541x) | retain as development candidate |
| expected M unset | 0.336960/0.316624 ms | 1.064358x (1.058173–1.071949x) | indistinguishable from primary |
| `compiled_dims=mnk` | 0.336816/0.316448 ms | 1.064222x (1.059469–1.064967x) | no benefit; dynamic-M risk |
| zero output gaps | 0.336880/0.319440 ms | 1.054704x (1.050629–1.060917x) | directionally slower |

The top three distributions overlap; the evidence does not show that expected
M or `mnk` specialization causes the gain. Zero-gap correctness is not a
general proof of gap semantics because the comparison intentionally covers
only rows consumed by `ep_gather`. Raw paths and exact identity controls are in
[paired_w2_table.md](paired_w2_table.md), with the complete hypothesis/risk/
rollback history in [attempt_ledger.md](attempt_ledger.md).

## Profiler diagnosis

Source analysis proves `ceil(raw_m/128) == aligned_m/128` for every expert, so
PSUM cannot remove full M tiles. The measured diagnosis agrees:

- NCU duration: 331.296 -> 313.312 us (1.057400x).
- Nsys five-launch kernel medians: 332.159 -> 311.967 us (1.064725x), across
  one correctness launch, three warmups, and one timed launch per arm.
- Same 148-block, one-wave, one-CTA-per-SM grid; 38 -> 50 registers, unchanged
  214828 B shared memory, and no local traffic/spill.
- Input TMA bytes are unchanged. Scalar/global loads fall 264000 -> 31006;
  UTCQMMA, TMEM loads, and output TMA stores each fall 5.545%; DRAM reads fall
  23.641%.
- Long-scoreboard-per-issue falls 6.312 -> 4.070 and barrier falls
  4.669 -> 3.892; tensor-active elapsed rises 70.29% -> 71.41%.

This supports reduced row-layout loads and tail-validity/output work inside the
same tile count, not fewer tiles or higher occupancy. WarpStates PM timelines
are present; base-utilization PM instance series are unavailable through the
NCU Python API and no such timeline is claimed. The one-repeat profiled runner's
1.257x CUDA-event ratio is rejected because Nsys shows that PSUM host-side
launch preparation overlaps the preceding untimed output-poison fill before
the start marker completes, whereas stock launch preparation occurs after its
start marker. The full six-dimension analysis,
source/SASS hotspots, raw-report hashes, and limitations are in
[the profiler report](../../../profile/moe-w2-prefill-psum-vs-stock-20260722b/REPORT.md).

## EP4 diagnostic and preserved failures

The immutable first campaign passed component, profile, serving-native
selftest, and configured verifier, but its EP4 step failed before correctness
or timing. A synchronous `async_finish=False` DeepEP call returned an
`EventOverlap` wrapper without an inner event, and the diagnostic runner
incorrectly called `current_stream_wait()`. Commit `6180228` now waits only for
asynchronous completion and fails closed if an asynchronous call lacks a real
event. The original log and failed manifest remain preserved.

The clean locked retry used physical GPUs 0–3 and the pinned stock DeepEP
overlay. It passed three dispatch and three combine identity pairs, the
40-workload serving-native selftest, and the configured verifier. Dispatch
paired medians range 0.990684–1.022721x; combine medians range
0.995847–0.999845x. The logs retain default 20-communication-SM and NVSHMEM
IBGDA warnings. This synchronous eager EP4 lane uses local tokens 8192 and does
not include W13, SwiGLU+quant, W2, PSUM, overlap, or the full region. Exact
rows and artifacts are in [deepep_region_table.md](deepep_region_table.md).

The first strict profile analysis is also preserved: it failed because optional
base-utilization PM series were absent. The validated successor records that
absence explicitly and uses the available WarpStates series.

## Validation and provenance

- Kernel-Harness measurement code: `83e0352`; validated profile analysis:
  `bcfc77a`; event fix: `6180228`; EP4 evidence: `382e1f5`.
- SGLang opt-in wrapper/test commit: `07802235`; existing callers still pass
  the exact stock `{}` DeepGEMM kwargs.
- Strict component validator: seven measured JSONs passed; strict EP4
  validator: six measured JSONs passed. Their adversarial tests pass 7/7 and
  6/6.
- Serving-native selftest: 40 workloads. Testbench selftest: 24 tasks,
  0 problems. Knowledge lint: 13 entries, 0 problems; generated index and
  distillation are fresh.
- Ten SGLang forwarding/registry test functions pass directly. Python and shell
  syntax checks and `git diff --check` pass.
- The configured verifier passed under the locked campaign. Its pointer audit
  retains the pre-existing missing `runs/index.jsonl` as advisory.

The exact source/package/build/import/JIT/report hashes are in
[source_build_record.md](source_build_record.md). `audit_result.py` is not
applicable to serving-native JSON; the strict serving-native validators are the
relevant evidence checks. The related frozen E=8 task remains deliberately
unmeasured because it is the wrong callable/ABI, as recorded in the append-only
knowledge entry.

## External acceptance blocker and final policy

This host has four B200s and no GLM-5.2 checkpoint. It cannot produce live EP8
routing, eight-rank dispatch/combine baselines, the eight-rank
dispatch -> W13 -> SwiGLU+quant -> W2 -> combine result, or TP8/DP8/EP8 SGLang
prefill correctness/latency/throughput. No end-to-end number is invented; the
blocked table is [end_to_end_table.md](end_to_end_table.md).

The eight-rank gate is unchanged and documented in
[external_validation_blocker.md](external_validation_blocker.md). No bucket is
enabled. The generic wrapper's keyword-only controls remain inert, every
production caller stays on stock row-wise contiguous W2, and future promotion
requires endpoint retention without adapter tax plus live EP8 correctness,
graph/stream/overlap, full-region, and end-to-end wins. The exact policy is
[final_policy.md](final_policy.md).
