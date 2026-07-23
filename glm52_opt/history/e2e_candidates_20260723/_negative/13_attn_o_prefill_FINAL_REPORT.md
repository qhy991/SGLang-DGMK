# GLM-5.2 production attention O-projection prefill report

## Disposition

**No replacement.** The exact production packed-FP8 workload was reached,
measured, profiled, and subjected to an isolated DeepGEMM source change. No
candidate produced a deployable 3% paired-p50 gain. The five-stage source
experiment regressed the exact leaf and the complete `Fp8LinearMethod.apply`
region in all three controlled series, so it was reverted. Stock six-stage
DeepGEMM remains active.

## Scope locked to production

The named workload is `linear_attn_o_prefill_m4096`, with rank-local
`M=4096, N=6144, K=16384`. It preserves the production ABI:

- BF16 caller activation;
- FP8 E4M3FN checkpoint weight;
- SGLang per-token FP8 activation quantization;
- packed `int32` UE8M0 activation and weight scales with column-major
  TMA-aligned strides;
- BF16 output;
- DeepGEMM PDL enabled.

The proven call chain is
`self.o_proj` → `RowParallelLinear` → `Fp8LinearMethod.apply` →
`deepgemm_w8a8_block_fp8_linear_with_fallback` →
`w8a8_block_fp8_matmul_deepgemm` →
`deep_gemm_fp8_fp8_bf16_nt` →
`deep_gemm.fp8_gemm_nt`. The runtime and ABI evidence is detailed in
`REACHABILITY.md` and `../production_20260723d/reachability_runtime.json`.
The frozen float32-scale `o_proj_prefill` task was not modified or used as the
production denominator.

## Performance result

The identity calibration stayed near 1.0× across three 30-pair series. The
pre-existing compiled-NK dispatcher was neutral:

| Candidate | Three paired-p50 speedups | Decision |
|---|---|---|
| Existing dispatcher, exact leaf | 1.001274×, 0.994634×, 1.006626× | below gate |
| Existing dispatcher, full FP8-linear region | 1.009760×, 1.001332×, 0.997545× | below gate |
| Five-stage source delta, fair exact leaf | 0.973239×, 0.978281×, 0.972641× | regress |
| Five-stage source delta, full FP8-linear region | 0.961955×, 0.972547×, 0.976800× | regress |

All controlled source-experiment correctness checks passed. The complete
sample table and full-precision machine-readable summary are in
`PAIRED_SUMMARY.md` and `paired_summary.json`.

## Profiler diagnosis

The stock kernel is bound by its tensor pipeline, not by HBM or launch latency.
Nsight Compute reports 91.28% elapsed tensor-pipe activity, about 22.55% peak
DRAM-read throughput, 42 registers/thread, zero local/spill traffic, one
CTA/SM, and 12.5% occupancy. It launches 148 CTAs and DeepGEMM models six
internal waves with a 124-tile last wave.

The compiled-NK dispatcher removes about 12% of average executed instructions
and one register, but changes NCU duration only from 256.864 µs to 256.130 µs.
Its tensor utilization stays near 92%, while long-scoreboard and barrier ratios
increase. Nsight Systems likewise shows only a 0.255 µs median difference
between dynamic and compiled-NK kernels in the paired dispatcher trace.

The five-stage change reduces dynamic shared memory from 209.7 KB to 176.4 KB,
but it still cannot admit a second CTA. NCU duration rises from 251.3 µs to
259.6 µs, elapsed tensor activity falls from 92.15% to 87.54%, and
long-scoreboard/barrier ratios worsen. Registers remain 42 and spills remain
zero. Exact cubin disassembly has the same 1,967 static instructions and tensor/
TMA operation counts; only minor synchronization scheduling differs. The
removed buffer stage therefore starves the tensor pipeline rather than
relieving a binding resource limit.

The profiler reports, exported metric tables, source-correlated stalls, cubins,
ptxas resources, and SASS are under:

- `../../../../profile/attn-o-prefill-packed-production-20260723d/`
- `../../../../profile/attn-o-prefill-packed-production-20260723e-stage5/`

## Source experiment and rollback

The exact-shape heuristic change was committed in SGLang as
`68e047c9a` and built into a separate `deep_gemm_experimental` overlay. It is
based on DeepGEMM v0.1.4, upstream commit
`731e7c7a97d269e4b9f482ea18d0e709a948f293`. Build/import/JIT provenance,
including the overlay `_C.so` SHA-256, is in `overlay_provenance.json`,
`overlay_manifest.json`, `source_sha256.txt`, and `BUILD_PROVENANCE.md`; the
installed stock package was untouched.

The rejected change was reverted in SGLang commit `8f450dbdf`. Validation shows
the final heuristic source has the same SHA-256 as the starting
`f93f8867b` source and the SGLang worktree is clean. The exact rejected diff is
preserved in `source_experiment.patch`; the attempt rationale and all supported
configuration trials are in `ATTEMPT_LEDGER.md`.

## Validation

- `testbench/bin/selftest.py`: 24 tasks, 0 problems.
- `serving_native/selftest.py`: 40 fixed workloads.
- Experiment Python compilation and shell syntax checks: passed.
- Eleven CPU-only registry/policy checks: passed, including default and
  OPT=1/default-profile stock fallback for O-projection prefill M4096.
- `testbench/bin/verify_harness.py`: passed; its missing historical
  `runs/index.jsonl` pointer is advisory and the command exited successfully.
- All raw and profile JSON artifacts parse successfully.
- SGLang source rollback diff and worktree check: passed.

The logs are in `validation/`.

## Unavailable production gates

The host has four B200s and no usable GLM-5.2 FP8 checkpoint. SGLang’s current
acceptance test is an eight-GPU TP8 lane with a TP8+DP8 variant. Consequently,
the full attention-layer and SGLang prefill baselines could not be executed
honestly, and a TP4 result was not relabeled as production evidence. The empty
local NVFP4 directory also cannot substitute for the required FP8 model.

This does not leave a candidate awaiting only external acceptance: the
five-stage candidate already fails the micro and containing-region gates. The
unmodified eight-rank gate and its exact future requirements are recorded in
`EXTERNAL_VALIDATION_BLOCKER.md`.

## Final enable policy

No new kernel is promoted. Default `SGLANG_GLM52_OPT=0` and
OPT=1 with the default `serving_safe` profile both select stock. The existing
compiled-NK route remains an explicit full-profile ablation only; the
five-stage overlay is never selected by production dispatch. Exact details are
in `ENABLE_FALLBACK_POLICY.md`.

## Evidence roots

- Baseline, reachability, configuration sweep, stock/dispatcher profiles:
  `../production_20260723d/`
- Five-stage paired series, build provenance, rollback, and validation:
  this directory
- Stock/dispatcher profiler bundle:
  `../../../../profile/attn-o-prefill-packed-production-20260723d/`
- Stock/five-stage profiler bundle:
  `../../../../profile/attn-o-prefill-packed-production-20260723e-stage5/`
