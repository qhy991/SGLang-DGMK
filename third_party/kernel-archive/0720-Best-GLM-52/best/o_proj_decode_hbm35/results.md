# Results — GLM-5.2 o_proj decode 35% HBM campaign

Authoritative evidence is the `RESULT_JSON` block from `run.sh --candidate` under the
default protocol (warmup=3, repeat=10, iterations=30, CUPTI cold-L2 device-kernel median)
on B200 GPU id 1.

Target (AC-3): `bw_util ≥ 0.35` on BOTH shapes, i.e. `candidate_us ≤ 36.13` (M=16) and
`≤ 36.29` (M=32). `bytes_hbm`: 101,154,816 (M=16), 101,621,760 (M=32).

## Attempt ledger

| # | Candidate | M=16 us | M=16 HBM% | M=32 us | M=32 HBM% | Correct | Verdict vs 35% |
|---|-----------|--------:|----------:|--------:|----------:|---------|----------------|
| A0 | **Seed** (Triton split-K M=16 + DeepGEMM ref M=32) | 45.50 | 27.79% | 53.26 | 23.85% | PASS (diff 4.3e-9 / 0) | MISS both |
| A1 | **Direction A** — per-call fused packed-UE8M0 repack + `fp8_gemm_nt(compiled_dims='nk')` (both shapes) | **33.04** | **38.27%** | **33.84** | **37.54%** | PASS (diff 0 / 0) | **MEET both** (triple-gate median) — **SELECTED WINNER** |
| B0 | Direction B — single-pass Triton fp8 block-scaled GEMM (no split-K round-trip), both shapes | 69.41 | 18.22% | 73.92 | 17.18% | PASS (diff 3.2e-9) | MISS (rejected: ~2.1× slower than A1) |
| C0 | Direction C — custom SM100 CUDA weight-streaming GEMM (one thread/column), both shapes | 3961.3 | 0.32% | 5145.7 | 0.25% | PASS (diff 3.2e-9) | MISS (rejected: ~120-150× slower than A1) |

**TARGET MET.** A1 clears ≥35% HBM on both frozen shapes on the three-gate per-shape median,
with calc_diff 0 (pre- and post-timing) and no reference fallback. Directions B and C were
built and gated (correct on both shapes) purely as comparison attempts — both are far slower,
so **A1 remains the selected winner**. Final candidate SHA
`054cf91a7cf5beaa886f5c6c34ed60a36afd7a0f6db30048aa3d64196c258551`
(+ `scale_pack.cu` SHA `c8106d4227a351c9341f74b5b9acf9f1e3ebc9c7da340de6ce1dcc1d78bef821`).

## A0 — Seed baseline (run_id `20260718T045740Z-66a2fa`)

Seed SHA `d98f2710b59a0cc5c3ba197495c9a4405235958e01807cf615c2dfe1011a5b7b`.

```
  shape    ok  calc_diff   cand_us    ref_us  speedup  sp_cons  verdict      AI   bound      GB/s      BW   reward
   M=16  PASS   4.30e-09     45.50     51.79   1.138x   1.135x      WIN    31.8  memory    2223.4  27.79%   0.2779
   M=32  PASS   0.00e+00     53.26     52.87   0.993x   0.975x  neutral    63.4  memory    1908.2  23.85%   0.2385
```

Per-shape detail:
- **M=16**: cand 45.50 µs (lo/hi 45.30/45.57, p90 45.55), ref 51.79 µs. bw_util 27.79%. WIN
  vs reference, but **misses 35%** (need ≤ 36.13 µs, a further ~20.6% cut). Seed path =
  Triton split-K (SPLIT_K=2, fp32 partials written then reduced to bf16).
- **M=32**: cand 53.26 µs (lo/hi 52.88/54.77), ref 52.87 µs. bw_util 23.85%. `neutral` —
  the seed falls back to `deep_gemm.fp8_gemm_nt` (the f32-scale reference path), so it just
  reproduces the reference. **Misses 35%** (need ≤ 36.29 µs, a ~31.9% cut).

Harness verdict: exit 0 (1 WIN, 0 regress, 1 neutral) — passes the Harness gate but **does
NOT** meet the campaign's 35% HBM bar on either shape.

### Reading against the existence proof
The Harness `BASELINE_CAVEAT` states production SGLang dispatch (int32-packed UE8M0 scales
→ `w8a8_block_fp8_matmul_deepgemm`) runs **33.1 µs vs 53.3 µs at M=16** on this exact
protocol — i.e. ~38% HBM, already above the 35% bar. The reference f32-blockwise-scale path
is ~1.6× slower purely because of the scale representation. This is the basis for
**Direction A**: per-call lossless repack of the f32 UE8M0-valued scales into DeepGEMM's
packed-UE8M0 layout, then dispatch the faster kernel, for BOTH shapes.

---

## A1 — Direction A: per-call packed-UE8M0 dispatch (TARGET MET)

Candidate `candidate/candidate.py` (final SHA `054cf91a7cf5…`; the Round-0 confirm below used
the behavior-identical `92ab42a2ea43…`, differing only by a build-failure warning) +
`candidate/scale_pack.cu` (SHA `c8106d4227a3…`). Mechanism: a single fused CUDA kernel (built once at import via
`cpp_extension.load`, outside the timed window) losslessly repacks the float32 UE8M0-valued
scales into DeepGEMM's packed-int32 layout (activation packed directly; weight block scale
N-expanded then packed, matching SGLang's production weight requant), then dispatches the
same `deep_gemm.fp8_gemm_nt(compiled_dims='nk')`. Stateless: output buffer freshly
allocated and fully overwritten each call; no cross-call caches.

### Why it works — decomposition (rough cold-L2 medians, probe timer)
| Component | M=16 | M=32 |
|---|--:|--:|
| Float32-scale reference | 52.1us (24.2%) | 53.1us (23.9%) |
| **Packed-UE8M0 GEMM alone** (pre-packed scales) | 32.4us (39.1%) | 33.4us (38.1%) |
| … with `compiled_dims='nk'` | 30.2us (41.9%) | 31.3us (40.5%) |
| Naive per-call repack (torch index_select+C packer) | +19us | +20us |
| Fused single-kernel repack (this candidate) | ~+2us | ~+2.5us |

The packed GEMM itself clears 35% comfortably; the whole engineering problem was making the
per-call repack cheap. A naive repack (expand weight block scale to a 3 MB float32
intermediate across several eager kernels) cost ~19us and its host-launch gaps — measured as
the CUPTI device span `max(kernel.end) − min(kernel.start)` — pushed the full pipeline to
~63us (worse than reference). Collapsing the repack into ONE fused CUDA-kernel launch with
ONE allocation removed those gaps and lands the full pipeline at ~33-34us.

### Authoritative triple-gate (AC-3, AC-4) — default protocol, B200 GPU 1
Final artifact (with `C10_CUDA_KERNEL_LAUNCH_CHECK` and corrected byte-accounting docstring).
Run IDs `20260718T065002Z-915fcd`, `20260718T065010Z-c77425`, `20260718T065018Z-df3b8d`.
All three: `VERDICT: CORRECT`, 2/2 shapes WIN, 0 regress, calc_diff 0 (pre & post timing),
`is_reference_fallback=False`. (An earlier identical triple-gate on the pre-launch-check
build — SHAs `74f922a7…`/`de778253…`, run IDs `…92aef9/…4b1dae/…f2a7a9` — gave the same
medians: M=16 33.024us, M=32 33.864us.)

| Shape | gate1 | gate2 | gate3 | **median** | HBM% @ median | ceiling | margin |
|---|--:|--:|--:|--:|--:|--:|--:|
| M=16 | 32.960 | 33.024 | 33.040 | **33.024us** | **38.29%** | 36.13us | 3.11us |
| M=32 | 33.808 | 33.824 | 33.872 | **33.824us** | **37.56%** | 36.29us | 2.47us |

Per-shape median ≤ ceiling on BOTH shapes ⇒ **≥35% HBM confirmed on both frozen shapes**.
Representative gate-1 table:
```
   M=16  PASS   0.00e+00     33.03     52.07   1.577x   1.558x      WIN    31.8  memory    3063.0  38.29%   0.3829
   M=32  PASS   0.00e+00     33.82     53.02   1.568x   1.561x      WIN    63.4  memory    3005.0  37.56%   0.3756
```

### Headroom vs the 8.0 TB/s roofline
At the median, achieved read bandwidth is ~3.06 TB/s (M=16) / ~3.00 TB/s (M=32) against the
8.0 TB/s peak. AI≈32 vs ridge 562 ⇒ deeply memory-bound; 35% HBM = 2.8 TB/s. The candidate
sits ~7-8% above the bar. Remaining headroom to the roofline exists but is not required by
the campaign target and would demand a custom SM100 weight-stream kernel (Direction C),
which is not pursued because Direction A already clears both shapes with margin.

## Directions B and C — built and gated in Round 1
In Round 0 they were deferred under evidence-driven escalation (A cleared both shapes). The
Round-0 review's no-deferral rule required them to be built and measured, so they were
implemented and gated as concrete rejected attempts — see the **Round 1** section below and
`docs/no_go_disposition.md`. Both are correct and far slower than A1, confirming A1 as the
single sufficient mechanism.

---

## Round 1 — Directions B & C built and gated (rejected attempts)

Per the Round-0 review's no-deferral rule, Directions B and C were implemented as real
task-local candidates and run through `run.sh --candidate` on both frozen shapes. Both are
correct; both are far slower than A1, so A1 remains the selected winner.

### B0 — Direction B (single-pass Triton, no split-K round-trip)
`variants/directionB/candidate.py`. One Triton kernel accumulates the whole K reduction in
registers and writes bf16 `out` directly (no fp32 partial round-trip, no second kernel).
Gate (`artifacts/directionB/gate_clean.log`, GPU 1 idle): M=16 69.41us/18.22%,
M=32 73.92us/17.18%, calc_diff 3.2e-9, VERDICT CORRECT, both REGRESS vs reference.
NCU (`artifacts/directionB/ncu_M16.txt`): grid 24 CTAs, block 128, occupancy 6.25%,
DRAM 10.1%, SM 4.3% — **latency/occupancy-bound**: the autotuned tile (BLOCK_N=256) leaves
only 24 CTAs on 148 SMs and the tiny BLOCK_M=16 underfills the fp8 MMA, so it cannot approach
the DRAM roofline. Rejected: ~2.1× slower than A1.

### C0 — Direction C (custom SM100 CUDA weight-streaming GEMM)
`variants/directionC/candidate.py` + `scale_gemm.cu` (built at import via cpp_extension.load).
One thread owns one output column and streams that weight column once; the activation tile is
staged in shared (converted to float once per 128-K block) with fp32 accumulation and per-block
scale application. Plain CUDA-core stream (no tcgen05/TMEM).
Gate (`artifacts/directionC/gate1.log`, GPU 1 idle): M=16 3961us/0.32%, M=32 5146us/0.25%,
calc_diff 3.2e-9, VERDICT CORRECT, both REGRESS.
NCU (`artifacts/directionC/ncu_M16.txt`): grid 96 CTAs, block 64, occupancy 3.12%, DRAM 0.35%,
SM 9.2%, kernel 3.97ms — **ALU-bound at catastrophic occupancy**: per-element fp8→float
conversion plus the serial 32-wide M inner loop dominate; the weight is streamed but at 0.35%
of DRAM peak. Rejected: ~120-150× slower than A1. This is exactly why the plan gates custom
SM100 work on NCU evidence — a naive custom kernel is far worse than DeepGEMM's tuned tcgen05
path; matching it would require the full TMA/TMEM/tcgen05 route, which is unnecessary since A1
already clears the target.

### Selection & A1 re-confirmation
Fastest correct candidate by the three-gate median = **A1** (33.04 / 33.84us). A1 was
re-confirmed after a compliance-hardening edit (build-failure now warns loudly; behavior on the
fast path unchanged). Final-artifact SHA `054cf91a7cf5…`. Re-confirm run IDs
`20260718T083015Z-f580e8`, `…538a24`, `…0a1bda`:

| Shape | gate1 | gate2 | gate3 | median | HBM% | ceiling | verdict |
|---|--:|--:|--:|--:|--:|--:|---|
| M=16 | 33.040 | 33.040 | 33.032 | **33.040us** | 38.27% | 36.13 | PASS (margin 3.09us) |
| M=32 | 33.856 | 33.824 | 33.835 | **33.835us** | 37.54% | 36.29 | PASS (margin 2.45us) |

All three: CORRECT, 2/2 WIN, calc_diff 0 pre & post, no reference fallback.
