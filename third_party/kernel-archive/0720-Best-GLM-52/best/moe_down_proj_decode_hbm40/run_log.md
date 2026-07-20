# Run Log — GLM-5.2 MoE Down-Projection Decode, 40% HBM Campaign

Task: `glm52/moe_down_proj_decode` (masked FP8 grouped GEMM, E=8, K=2048, N=6144, top_k=8).
Target: ≥40% HBM utilization on **both** decode shapes M∈{16,32} through the frozen harness gate.

## Environment (captured 2026-07-18)

| Item | Value |
|------|-------|
| GPU | NVIDIA B200 (148 SMs, L2 = 132,644,864 B ≈ 126.5 MB, 191.5 GB) |
| Driver / CUDA | 610.43.02 / nvcc 13.2 (V13.2.78) |
| Python / torch | 3.12.13 / 2.11.0+cu130 (cuda 13.0) |
| deep_gemm / triton / cupti | 0.1.4 / 3.6.0 / available (CUPTI timing active) |
| Harness git | 7d79e5ecbb52f2d943ed6644b8fa9f06a3c52c2d |
| Worktree git (base) | 5bf958d9c06a818a3ecae818631f8f2c733ad5d6 |
| Host | dry-vm-embraces-fin-03 |
| GPU pin | REMOTE_GPU_ID=0, CUDA_VISIBLE_DEVICES=0 |

## Candidate provenance (AC-3)

- Seed: `Kernel-Harness/testbench/tasks/glm52/moe_down_proj_decode/candidate.py`
- Copied byte-for-byte to `candidate/candidate.py`.
- Clean-seed SHA-256: `02cedf67bdfea14d0ca0bd767130a75781eac9b1494800137bbad58b5352651e` (both files `cmp`-identical at copy time).
- Working candidate is then optimized in place; its SHA-256 is recorded per gate run in the run headers below (v2 = `29a603ab8a8b…`).

## Baseline (AC-1, AC-5) — seed candidate == reference

GPU 0 verified idle (0% util, 0 MiB) immediately before.

| M | cand µs | ref µs | HBM% | calc_diff | verdict |
|---|--------:|-------:|-----:|----------:|---------|
| 16 | 47.02 | 46.94 | 27.26% | 0.00e+00 | neutral |
| 32 | 47.20 | 47.14 | 27.64% | 0.00e+00 | neutral |

Matches prior baseline (~27.28% / 27.69%). Both memory-bound (AI 31.4 / 61.7 ≪ 562.5 ridge). Raw: `docs/raw/baseline_seed_full.log`.

## Bottleneck diagnosis — NCU (AC-6)

NCU (2026.1.1.0) on the reference masked GEMM, profiling only the GEMM calls
(`cudaProfilerStart/Stop`). The single `fp8_m_grouped_gemm_nt_masked` call decomposes
into a **kernel chain** per call (NCU serialized durations — inflated, but the split and
per-kernel DRAM% are the signal). Raw: `docs/raw/ncu_ref_m16_gemm.txt` (M=16),
`docs/raw/ncu_ref_m32_gemm.txt` (M=32).

| Kernel | ~dur M16 / M32 (NCU) | DRAM% | role |
|--------|---------------------:|------:|------|
| `transpose_and_pack_fp32_into_ue8m0` (×2) | 4.8+6.9 / 4.9+7.1 µs | ~0.2–6% | f32→ue8m0 scale pack |
| `arange` + `div_floor` + `scatter_gather` | 4.1+6.3+8.8 / 4.3+6.1+8.8 µs | <1% | grouped SF-layout build |
| **`sm100_fp8_fp4_gemm_1d1d_impl`** | **24.29 / 24.22 µs** | **57.4 / 57.8%** | the matmul, reads ~107 MB |

**Key finding (both shapes):** the matmul already streams the expert weights at ~57% of DRAM peak
(≈45–53% HBM by the harness byte model). The baseline sits at 27% only because ~30 µs of per-call
**scale-transform + grouped-layout overhead** runs ahead of every GEMM. The matmul time is ~24 µs at
**both** M=16 and M=32 (weight-read-dominated; tiny-M activation growth is negligible), which is why
both shapes land near 31 µs end-to-end once the overhead is removed. The optimization is to eliminate
that overhead, not to write a faster matmul.

`masked_m` distributions (seeded, fixed): M=16 → [14,27,11,19,16,16,11,14] (Σ=128);
M=32 → [36,39,29,31,33,31,27,30] (Σ=256). `expected_m`=128 for both; no slab overflow.

## Optimization ladder (attempt ledger — see docs/attempt_ledger.md)

The decisive lever: the frozen scales are built with `use_ue8m0=True` (powers of two), so
they can be **losslessly** reinterpreted as the int32-packed ue8m0 scale-factor layout that
DeepGEMM's masked kernel consumes with `disable_ue8m0_cast=True` — skipping the internal
transform. Measured GEMM-only (scales pre-packed) = **28.3 µs both shapes** (≈45% HBM).
The remaining problem was packing the scales **cheaply** inside the timed window:

| Pack approach | pack µs | end-to-end µs (M16 / M32) | HBM% |
|---------------|--------:|--------------------------:|------|
| library `transform_sf_into_required_layout` (timed) | ~19 | 47.4 / 47.2 | ~27% (no win) |
| `get_mn_major…packed_ue8m0` + repeat_interleave (3 kernels) | ~11 | 39.5 / 39.4 | ~33% |
| custom Triton, 2 kernels (x, w-expand) | ~4.7 | 33.1 / 33.5 | ~39% |
| custom Triton, 1 fused kernel (grid E×K4) | ~4.3 | 32.9 / 32.3 | 39% / 40.4% |
| **custom Triton, fused grid E×K4×NB, x folded into b==0** | **~2.8** | **31.3 / 30.6** | **40.9% / 42.7%** |

The pack is **byte-identical** to `get_mn_major_tma_aligned_packed_ue8m0_tensor` (verified
`torch.equal` on both x and w scales), so correctness is a bit-exact reinterpretation.

## Candidate (adopted)

`candidate/candidate.py`: one fused Triton kernel packs both scales (weight scale packed +
128×-expanded across N-rows; activation scale folded into block-0 programs), then dispatches
`fp8_m_grouped_gemm_nt_masked(..., disable_ue8m0_cast=True)`. Stateless; all pack work is
inside the timed window; defensive fallback to the reference transform on any off-contract
input shape.

## Gate — provisional (v2, sha 29a603ab8a8b)

Correctness verified directly (load-independent): both shapes pass, calc_diff=0, cosine=1.0,
zero element failures, inputs provably unmutated across calls (stateless).

| M | cand µs | ref µs | HBM% | reward | speedup | verdict |
|---|--------:|-------:|-----:|-------:|--------:|---------|
| 16 | 31.30 | 46.99 | 40.94% | 0.4094 | 1.501× | WIN |
| 32 | 31.33 | 47.10 | 41.64% | 0.4164 | 1.503× | WIN |

VERDICT CORRECT, 2/2 WIN, performance_gate MET. Raw: `docs/raw/candidate_v2_provisional_gate.log`.

**NOT AUTHORITATIVE:** the harness flagged timing instability (M=16 spread 3.10×, a 96.7 µs
outlier sample; M=32 flagged too). A resident 21 GB process (pid 483197, 0% util) was on
GPU 0 immediately before this run (shared-GPU campaign). Per AC-5, authoritative confirmation
requires **three** interference-free runs on an exclusively-idle GPU 0 with per-shape medians
≥40%. Those runs are pending an exclusive-idle window and are recorded below when taken.

## Candidate NCU — genuine result, not a byte-model artifact (AC-4.1, AC-6)

NCU on the adopted candidate, both shapes (`cudaProfilerStart/Stop` scope). Raw:
`docs/raw/ncu_cand_m16.txt` (M=16), `docs/raw/ncu_cand_m32.txt` (M=32).
The candidate runs **exactly two kernels per call** — the reference overhead chain is gone:

| Kernel | ~dur M16 / M32 (NCU) | DRAM% M16 / M32 | bytes | grid |
|--------|---------------------:|----------------:|------:|-----:|
| `_pack_ue8m0_scales` (fused pack) | ~5.0 / ~5.0 µs | ~0.25% / ~0.25% | 93.7 KB | 1536 |
| `sm100_fp8_fp4_gemm_1d1d_impl` (same matmul as ref) | 24.2–24.7 / 24.3–24.6 µs | **56.4–57.6% / 56.8–57.4%** | ~107 MB | 148 |

Both shapes: the matmul streams weights at ~57% of DRAM peak (physical, ≤100%); the pack moves
trivial data (93.7 KB). Reported achieved 3275 GB/s (M16) / 3325 GB/s (M32) ≪ 8 TB/s peak; rewards
0.409 / 0.416 < 1.0 (no >100% anomaly). The end-to-end harness bw_util (40.9% / 41.6%, byte model
~102–104 MB / ~31 µs) is genuine on both shapes — the ~30 µs reference overhead chain
(transpose_and_pack ×2 + arange/div_floor/scatter_gather) is entirely absent. The **same**
scale-transform removal explains **both** M=16 and M=32: identical matmul kernel, identical ~57% DRAM,
identical ~24 µs, so the only delta from the ~47 µs reference is the removed overhead.

## Authoritative confirmation (AC-5) — TARGET MET

Three interference-free runs, GPU 0 **exclusively idle verified before AND after each**
(0 MiB, 0 compute procs on GPU-0 UUID; the 21 GB resident process was on GPU 1).
Harness defaults warmup=3, repeat=10. Raw: `docs/raw/candidate_auth_run{1,2,3}.log`.

| Run | M=16 µs | M=16 HBM% | M=32 µs | M=32 HBM% | correctness | verdict |
|-----|--------:|----------:|--------:|----------:|-------------|---------|
| 1 | 31.36 | 40.87% | 31.39 | 41.56% | calc_diff=0 | 2/2 WIN |
| 2 | 31.30 | 40.94% | 31.33 | 41.65% | calc_diff=0 | 2/2 WIN |
| 3 | 31.31 | 40.93% | 31.39 | 41.56% | calc_diff=0 | 2/2 WIN |
| **per-shape median** | **31.31** | **40.93%** | **31.39** | **41.56%** | calc_diff=0 | **TARGET MET** |

Both per-shape medians ≥40% HBM and within the latency ceilings (M16 31.31 ≤ 32.041 µs;
M32 31.39 ≤ 32.617 µs). Every individual run also clears 40% on both shapes (min 40.87%).
The candidate's own samples are tight (31.2–31.6 µs); the "unstable timing" flag on run 1 is a
**reference-side** CUPTI outlier (ref 47–139 µs), which does not affect the candidate's reward.

## Interference log (AC-5)

| Time (UTC) | Event | GPU0 mem | Action |
|------------|-------|---------:|--------|
| 08:44 | pre-baseline | 0 MiB | idle → baseline taken |
| 09:48 | pre v1 gate | 21042 MiB (pids 392686, 459459) | busy; v1 had a compile bug anyway |
| 09:57 | v2 provisional gate | transient during run | UNSTABLE → treated provisional, re-run required |
| ~10:xx | 3 authoritative runs | 0 MiB before/after each (procs on GPU 1 only) | CLEAN → TARGET MET |
| R1 | M=32 NCU (ref + candidate) | 0 MiB before/after (0 procs on GPU-0 UUID) | CLEAN → both-shape NCU complete |

## Verdict

**TARGET MET** — stateless `candidate/candidate.py`, correct on both shapes (calc_diff=0, before and
after timing), reaches ≥40% HBM on both M=16 (40.93%) and M=32 (41.56%) by the per-shape median of
three interference-free authoritative runs on exclusively-idle GPU 0, within the latency ceilings,
NCU-grounded, harness untouched (only the 3 pre-existing dirty siblings).

