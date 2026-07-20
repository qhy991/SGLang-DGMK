# RESULT — GLM-5.2 moe_gate_proj_decode HBM >=40%: **TARGET MET**

## Verdict

Both M=16 and M=32 reach **HBM >=40%** with equality passing, across four
authoritative idle-GPU gates. Correctness is bit-exact (`calc_diff = 0.0`).

| shape | baseline (ref) | candidate median | HBM% (median) | 40% limit | worst-gate median | margin |
|------:|---------------:|-----------------:|--------------:|----------:|------------------:|-------:|
| M16 | 46.657 us / 27.33% | **30.944 us** | **41.21%** | 31.882240 us | 31.050 us | 0.83 us |
| M32 | 46.601 us / 27.72% | **31.028 us** | **41.64%** | 32.299520 us | 31.048 us | 1.25 us |

Packed-path speedup ~1.50–1.51x over the frozen f32-scale reference. Every gate
returned `exit_code 0` (CORRECT + performance_gate met). The AC-6 anchor is the
median of the per-gate medians; the worst individual gate median on each shape
still clears the inclusive limit.

## Per-gate evidence (`docs/attempts/final_gate_{1,2,3,4}.json`)

| gate | M16 us | M16 HBM% | M32 us | M32 HBM% | calc_diff | verdict |
|-----:|-------:|---------:|-------:|---------:|----------:|:-------:|
| 1 | 30.888 | 41.29 | 30.896 | 41.82 | 0.0 | CORRECT/0 |
| 2 | 31.000 | 41.14 | 31.040 | 41.62 | 0.0 | CORRECT/0 |
| 3 | 31.050 | 41.07 | 31.048 | 41.61 | 0.0 | CORRECT/0 |
| 4 | 30.800 | 41.41 | 31.016 | 41.66 | 0.0 | CORRECT/0 |

Gates 1–3 ran back-to-back; gate 4 is a confirmation run on a re-verified fully
idle GPU 2 (4 MiB / 0% util) after a transient co-resident process (pid 762983,
0% compute, 1.1 GB idle allocation) was observed and confirmed gone. Candidate
timing spread is ~1.01x on the stable gates; the `timing_unstable` flags that
appear are driven by cold-L2 outliers in the **reference** samples (the f32-scale
path spikes to 47–98 us on an occasional first sample) and by one cold candidate
sample in gate 2 M32 (p90 62 us) — none touch the candidate **median**, which is
the quantity the 40% target checks and which is tight at ~31 us across all four.

## Mechanism (the ported winner)

Same-family `moe_up`/`moe_down` lever, ported 1:1 (moe_gate shares `moe_up`'s exact
shapes, scales and masks — verified in `docs/baseline/baseline.md`):

1. **Fused exact UE8M0 pack** (`candidate/scale_pack.cu`): one CUDA launch, one
   allocation, repacks the frozen f32 UE8M0 block scales (exact powers of two) into
   DeepGEMM's packed-int32 MN-major layout for **both** operands and **all 8 experts**.
   The weight scale is expanded to per-N-row **inline** (`row>>7`), the activation
   scale stays per-token. Bit-exact vs `get_mn_major_tma_aligned_packed_ue8m0_tensor`
   (verified: `xp (8,128,12)`, `wp (8,2048,12)` equal on both shapes).
2. **`disable_ue8m0_cast=True`** dispatch of the identical
   `fp8_m_grouped_gemm_nt_masked`, so DeepGEMM consumes the pre-packed scales and
   reaches its Blackwell fast kernel `sm100_fp8_fp4_gemm_1d1d_impl` (NCU-confirmed),
   skipping the per-call f32→packed transform chain that drags the reference to 27%.
3. **PDL** (`deep_gemm.set_pdl(True)`) around the GEMM so it overlaps the pack tail
   inside the CUPTI span, with **save/restore** so the harness's separate reference
   timing is untouched (no denominator manipulation).

Everything is **stateless per call**: the packed scales are rebuilt into a fresh
buffer every invocation; no cached weights/scales, no input mutation, no precomputed
output, no timer exploit, no target fallback. The CUDA extension builds once at import
(outside the timed window); a pure-torch pack is retained only as a loud correctness
fallback.

## AC status

- **AC-1** ✓ Seed sha256 `08a8d674…fa6a8` matches; three stable idle baselines
  (M16 46.657 us/27.33%, M32 46.601 us/27.72%). `docs/baseline/`.
- **AC-2** ✓ Exact frozen masked semantics, `calc_diff = 0.0` (≤5e-6), stateless.
- **AC-3** ✓ Ported exact fused UE8M0 pack incl. N-row expansion + `disable_ue8m0_cast=True`;
  all work inside the timed span (single fused pack launch).
- **AC-4** ✓ Pack/GEMM/e2e floors + NCU traffic/waves/scheduler. `docs/floors/floors_ncu.md`.
- **AC-5** ✓ (resolved as *no source edit needed*) NCU shows GEMM mem SoL ~56% with
  residual headroom, but of the public DeepGEMM knobs only PDL is material and it is
  already applied; both shapes clear 40% with margin, so no task-local/isolated
  DeepGEMM source fork was warranted. No source edits made; stock oracle frozen.
- **AC-6** ✓ Four authoritative gates; per-shape medians satisfy inclusive limits.
- **AC-7** ✓ This document + `run_log.md` + `attempt_dag.md`.

## Environment

Idle NVIDIA B200, `CUDA_VISIBLE_DEVICES=2`. Harness `git_sha 7d79e5e` (main).
torch 2.11.0+cu130, deep_gemm 0.1.4, sgl_kernel 0.4.4, CUDA 13.0.
Protocol: CUPTI cold-L2 device-kernel median, warmup=3, repeat=10, iterations=30.
