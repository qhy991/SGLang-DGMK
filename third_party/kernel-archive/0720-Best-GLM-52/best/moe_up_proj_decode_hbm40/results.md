# Results — glm52 moe_up_proj_decode, HBM ≥ 40%

## Verdict

**TARGET MET** — HBM ≥ 40% (inclusive) on both frozen decode shapes, M16 and M32,
across three authoritative idle-GPU gates. Correctness PASS with `calc_diff = 0`.

Three CUPTI cold-L2 device-kernel-median gates on NVIDIA B200
(`docs/attempts/final_gate_{1,2,3}.json`, candidate `sha256 3940c89c…`), median across gates:

| M | median cand µs | 40% ceiling µs | median HBM% | worst gate µs (HBM%) | pass |
|---:|---:|---:|---:|---:|:---:|
| 16 | 30.855 | ≤ 31.882240 | 41.33% | 30.936 (41.22%) | yes |
| 32 | 30.952 | ≤ 32.299520 | 41.74% | 31.000 (41.68%) | yes |

All six shape-gates clear the inclusive 40% bar with ≥ 1.2 pt margin (≥ 41.2% HBM);
even each shape's worst single gate stays under the ceiling. Speedup vs the frozen
reference ≈ **1.53×** (conservative q=0.9 ≥ 1.50×); reference median ≈ 47.2 / 47.4 µs
(27.0% / 27.3% HBM).

### Per-gate detail (authoritative `final_gate_*`)

| gate | run_id | M16 µs | M16 HBM% | M32 µs | M32 HBM% | correct | calc_diff |
|---|---|---:|---:|---:|---:|:---:|---:|
| 1 | 20260719T170637Z-fb0615 | 30.855 | 41.33 | 31.000 | 41.68 | yes | 0.0 |
| 2 | 20260719T170647Z-a8f9a2 | 30.808 | 41.39 | 30.839 | 41.89 | yes | 0.0 |
| 3 | 20260719T170656Z-b13c05 | 30.936 | 41.22 | 30.952 | 41.74 | yes | 0.0 |

`final_gate_2` flagged `timing_unstable` on M16 (spread 1.523) but its median 30.808 µs
still clears with margin and `post_timing_correct = true`; the other five shape-gates are
timing-stable.

## Mechanism

The frozen reference calls `deep_gemm.fp8_m_grouped_gemm_nt_masked` with **unpacked
float32 UE8M0** block scales, which forces DeepGEMM's slow f32-scale path (~47 µs,
~27% HBM). SGLang's production dispatch hands the same kernel **packed int32 UE8M0**
scales and runs the identical math far faster. The candidate reproduces that packed
dispatch statelessly, per call:

1. **Fused exact UE8M0 pack** (`candidate/scale_pack.cu`, built once at import, outside
   the timed window). A single CUDA kernel losslessly repacks the f32 UE8M0 scales
   (exact powers of two) into DeepGEMM's packed-int32 MN-major layout — both operands,
   all 8 experts, one launch, one allocation. The weight scale is expanded to per-N-row
   inline (`row >> 7`, no `(E,N,K)` tensor materialized); the activation scale stays
   per-token. Fusing to one launch keeps the CUPTI device span tight (the span counts
   every kernel `run()` launches, plus host gaps).
2. **Packed dispatch** — the identical `fp8_m_grouped_gemm_nt_masked` with
   `disable_ue8m0_cast=True`, so DeepGEMM consumes the pre-packed scales directly and
   skips its internal f32→packed cast. Same math, same kernel, faster scale path.
3. **PDL** — DeepGEMM Programmatic Dependent Launch (`set_pdl(True)`) around the GEMM so
   it begins as the pack drains, shaving ~0.5–0.7 µs of host launch gap inside the span.
   A knob cross-product (`compiled_dims × num_sms × tc_util × pdl`) found PDL the **only**
   material, robust knob; `num_sms`/`compiled_dims` differences were within noise. The
   PDL global is **saved and restored** around the call, so nothing leaks into the
   harness's separate reference timing (no denominator manipulation).

A pure-torch pack is retained as a correctness-only fallback (not the fast path); a build
failure of the CUDA extension is surfaced loudly rather than silently degrading.

### Why this clears 40% (NCU, `docs/floors/floors_ncu.md`)

- **Pack is negligible:** ~549 KB total traffic, 4.6% mem SoL, launch/occupancy-bound
  (~1–2 µs real contribution back-to-back). No material floor to recover there.
- **GEMM streams the weight once:** DRAM read ~107.8 MB ≈ the 100.66 MB `w_fp8`
  (constant across M) + activations + scales — memory-bound (mem SoL 56–62% ≫ sm SoL
  20–24%), matching the byte model. The packed path removes the f32-scale overhead that
  kept the reference at 27% HBM.

## Correctness

Frozen masked-grouped compare on valid rows only (`masked_m` per expert). Output buffer
NaN-poisoned before `run()` and again before timing; `calc_diff = 0`, `max_abs_err = 0`,
`max_rel_err = 0`, cosine = 1.0 on every shape/gate; post-timing recheck on fresh inputs
PASS. Inputs are never re-quantized, re-seeded, cached, or mutated.

## Artifacts

- Authoritative gates: `docs/attempts/final_gate_{1,2,3}.json` (candidate `3940c89c…`).
- Preliminary gates (pre-PDL candidate `f59c9e71…`): `docs/attempts/gate_{prelim,1,2,3}.json`.
- Knob sweep (PDL isolation): `docs/attempts/knob_sweep.py`, `knob_sweep_out.txt`.
- Baseline: `docs/baseline/baseline.md`, `run_{1,2,3}.json`.
- Floors + NCU: `docs/floors/floors_ncu.md`, `probe_pack.py`, `verify_pack.py`;
  `docs/ncu/ncu_m{16,32}.csv`, `ncu_m{16,32}.log`, `ncu_driver.py`.
- Winner: `candidate/candidate.py` (`sha256 3940c89c…`), `candidate/scale_pack.cu`.
- Ledger: `docs/attempt_dag.md`; run log: `docs/run_log.md`.

## Environment

NVIDIA B200 (cap 10.0, 191.5 GB), idle GPU (`CUDA_VISIBLE_DEVICES=3`). torch 2.11.0+cu130,
deep_gemm 0.1.4, sgl_kernel 0.4.4, CUDA 13.0, python 3.12.13. Harness git_sha `7d79e5e`,
branch `main`. Protocol: CUPTI cold-L2 device-kernel median, warmup 3, repeat 10,
iterations 30. Cost model: HBM peak 8.0e12 B/s; 40% limit = bytes / (0.40 × 8e12) →
M16 31.882240 µs, M32 32.299520 µs.

Seed integrity: harness `candidate.py` seed `sha256 2259efd1…` matches the pinned hash in
`docs/prior_knowledge.md`. No Kernel-Harness or DeepGEMM source was modified — the stock
reference remains the untouched oracle.
