# GLM-5.2 attention q_b decode packed-native result

Date: 2026-07-23

Disposition: **NO REPLACEMENT**. The packed-native implementation is correct,
graph-safe, and adapter-free, but it is slower than stock DeepGEMM in both
production CUDA-graph buckets. No bucket is enabled in the default
`serving_safe` profile; stock SGLang remains the production path.

## Reachability and ABI

The current call chain is:

`Fp8LinearMethod.apply` →
`deepgemm_w8a8_block_fp8_linear_with_fallback` →
`w8a8_block_fp8_matmul_deepgemm` →
`deep_gemm.fp8_gemm_nt`.

The runtime trace in `baseline_20260723c/reachability_runtime.json` observed
`q_b_proj` in eager execution and CUDA Graph capture on a non-default graph
stream. Both local buckets use `N=16384, K=2048` and BF16 output:

| Bucket | caller | activation scale | weight scale | output |
|---|---|---|---|---|
| M16 | BF16 `[16,2048]` | int32 `[16,4]`, stride `[1,16]` | int32 `[16384,4]`, stride `[1,16384]` | BF16 `[16,16384]` |
| M32 | BF16 `[32,2048]` | int32 `[32,4]`, stride `[1,32]` | int32 `[16384,4]`, stride `[1,16384]` | BF16 `[32,16384]` |

The int32 tensors are DeepGEMM's MN-major, TMA-aligned packed UE8M0
representation. SGLang expands each 128-row weight-scale block before packing.
The candidate consumes these tensors directly: no unpack, f32 temporary,
conversion kernel, or adapter allocation occurs.

## Reference and synthetic gap

Three production identity series in `baseline_20260723c/pairs/` established the
local noise floor. The median-of-series speedups were 0.9880 at M16 and 0.9839
at M32, with no identity run satisfying the 3% production gate.

The related frozen `q_b_decode` task is deliberately a different ABI: its
scales are f32 `[M,16]` and `[128,16]`. The final frozen run
`runs/glm52/q_b_decode/20260723T102514Z-de110a/result.json` was correct, neutral
at both shapes, and had no wins or regressions. Its audit is PROVISIONAL solely
because the evidence tree was uncommitted during the run; the candidate itself
and frozen harness were not dirty. See `frozen_result_audit.txt`.

## Source/build change

The isolated DeepGEMM overlay adds
`fp8_fp4_gemm_nt_packed_warp`/`fp8_gemm_nt_packed_warp`, validates the packed
dtype/shape/stride contract, and adds a compile-time kernel path in which warp 2
stages already-packed scales while warp 0 streams A/B. The final v3 variant:

- prefetches activation and weight packed words before the A/B full-barrier wait;
- retains activation words in registers;
- loads one uniform weight-scale word in lane 0, broadcasts it within warp 2,
  and uses one aligned `uint4` shared store per lane;
- preserves the stock 148-SM launch policy and the existing UTCCP,
  fence/barrier, MMA, and epilogue sequence;
- has no extra kernel and no local-memory spills.

SGLang loads the fork side-by-side, mirrors stock `num_sms`, `tc_util`, and PDL,
routes only a packed int32 pair to the new entry, and returns `False` to the
stock caller for a missing overlay, mixed ABI, rejected shape, or runtime error.

Build provenance:

| Field | Value |
|---|---|
| Upstream | `https://github.com/sgl-project/DeepGEMM`, tag `v0.1.4` |
| Upstream commit | `731e7c7a97d269e4b9f482ea18d0e709a948f293` |
| Final source variant | `c3ac853756fdeed770406e2d64fbb2b2ee53c489` |
| Local SGLang commit | `84b9ffd30` |
| Overlay | SGLang `third_party/DeepGEMM-GLM52/overlays/c3ac853756fdeed770406e2d64fbb2b2ee53c489` |
| Extension SHA256 | `da5bd99b379564df09e6783606aa0d7c16ef6c42f26e894c2da15f7965e3eaed` |
| Python / Torch / CUDA | 3.12.13 / 2.11.0+cu130 / 13.0 |
| Stock package | untouched |

The CUDA template is JIT-compiled from the overlay headers, so the unchanged
host extension hash across variants is expected. Exact source hashes and build
logs are under `overlay_build*`.

## Paired microbenchmark

Each row is the median of three independent same-process AB/BA series, each with
10 warmups and 100 repeats on one wrapper-selected physical B200.

| Variant | M | stock µs | candidate µs | apparent speedup |
|---|---:|---:|---:|---:|
| v1 direct warp loads | 16 | 51.952 | 37.136 | 1.4170x |
| v1 direct warp loads | 32 | 52.416 | 37.168 | 1.4355x |
| v2 prefetch/vector store | 16 | 50.704 | 36.112 | 1.4458x |
| v2 prefetch/vector store | 32 | 52.688 | 37.632 | 1.4401x |
| v3 lane-0 broadcast | 16 | 53.584 | 37.712 | 1.4552x |
| v3 lane-0 broadcast | 32 | 52.144 | 36.912 | 1.3997x |

These are not deployable wins. The event starts before the Python/custom-op
launch, so it includes the CPU-to-GPU submission gap. The experimental TVM-FFI
entry submits faster than SGLang's stock custom-op wrapper even though its
device kernel is slower. CUDA Graph replay and NCU remove that artifact.

## Production layer and CUDA Graph

`packed_broadcast_v3_20260723a/production_graph.json` runs the real
`Fp8LinearMethod` caller with BF16 input, dynamic packed-UE8M0 quantization, the
q_b context tag, and independent stock/candidate graph captures. Each timing is
100 AB/BA repeats.

| M | eager stock µs | eager cand. µs | speedup | graph stock µs | graph cand. µs | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 121.840 | 130.560 | 0.9332x | 13.440 | 13.968 | 0.9622x |
| 32 | 107.744 | 112.448 | 0.9582x | 13.472 | 13.744 | 0.9802x |

Both eager and graph outputs match stock exactly (`max_abs_diff=0`). The trace
records 113 packed-native hits per bucket. With policy `q_b_proj:16`, an M32
call records `no_spec:q_b_proj:decode:m32` and matches stock exactly, proving
immediate selective fallback.

## Profiler evidence

Full/source NCU reports were collected in the same wrapper invocation as each
paired series. Key final metrics:

| M | path | kernel µs | grid | regs | shared B | DRAM read % | SM % | global loads | shared stores |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | stock | 12.896 | 148 | 37 | 231212 | 34.36 | 11.04 | 0 | 1172 |
| 16 | v3 | 14.528 | 148 | 38 | 231212 | 30.32 | 10.04 | 1024 | 2196 |
| 32 | stock | 13.312 | 148 | 39 | 224044 | 33.35 | 11.38 | 0 | 1172 |
| 32 | v3 | 13.824 | 148 | 40 | 224044 | 31.97 | 10.68 | 1024 | 2196 |

Stock is a one-CTA-per-SM, shared-memory-limited persistent kernel. It is
scheduler-starved rather than at the HBM roof: very few warps are eligible, and
long-scoreboard/barrier waits dominate. Moving scale transfer from TMA to warp
instructions increases instruction and shared-store work. V3 improves v2 at
M32 but cannot beat the stock TMA path; at M16 the shuffle adds work and makes
the result slightly worse. SASS/resource dumps confirm no spills and the
intended 148-CTA grid.

## Enable/fallback policy

Production policy is **enable no q_b bucket**:

- default `SGLANG_GLM52_OPT_PROFILE=serving_safe` with no q_b allowlist resolves
  directly to stock;
- `SGLANG_GLM52_OPT=0` is the explicit reference and emergency rollback;
- the packed entry is retained only for reproducible, explicit experiments;
- unsupported/missing/mixed/error cases fail closed to stock;
- the f32 adapter is never enabled.

## Full-decode and distributed validation

Full SGLang decode could not be run on this host. The expected checkpoint
directory `/mnt/OS-oKqEXySb/models/GLM-5.2-NVFP4` exists but is empty, and no
other local GLM-5.2 checkpoint was found. The host also exposes four B200s,
while the production acceptance contract is TP8/DP8/EP8. A TP4 diagnostic
cannot load without weights and would not substitute for TP8 in any case.

No end-to-end or production-win claim is made. The exact external blocker and
acceptance policy are recorded in `EXTERNAL_VALIDATION_BLOCKER.md`. Because the
local graph/region gate already fails, resolving that external blocker cannot
promote this candidate without a new winning implementation.

## Validation

- SGLang packed-route unittest: 4/4 pass.
- Production eager and graph correctness: exact at M16/M32.
- Hit counter and M32 selective-fallback proof: pass.
- DeepGEMM v3 overlay build and JIT: pass.
- `serving_native/selftest.py`: 39 workloads, pass.
- `testbench/bin/selftest.py`: 24 tasks, pass.
- Ruff on changed Python: pass after excluding two pre-existing warnings in
  the touched legacy files; formatting passes.
- `verify_harness.py`: all checks before task projection passed, then the
  configured no-CUDA import lane reported all generated tensor-table files
  stale. No generated/frozen files were edited, as required.
- Frozen result audit: PROVISIONAL due uncommitted evidence-tree provenance;
  internally consistent, candidate and harness clean.

Raw evidence is preserved under this directory and
`profile/q-b-decode-packed-*`.
