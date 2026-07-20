# Run log — glm52 moe_up_proj_decode (HBM ≥ 40%)

Campaign resumed at task8 after a prior RLCR session hit API disconnect / 100% context
post-task7 (`docs/CONTINUE.md`). No restart — finalize the existing TARGET-MET evidence.

- **2026-07-19 — Baseline (AC-1).** Verified harness seed `sha256 2259efd1…` matches the
  pinned hash. Three idle-GPU CUPTI baselines (reference = f32-scale
  `fp8_m_grouped_gemm_nt_masked`): M16 47.209 µs / 27.01% HBM, M32 47.368 µs / 27.28% HBM.
  Gap to 40%: needs 1.481× (M16) / 1.466× (M32). `docs/baseline/baseline.md`.
- **Candidate (AC-2, AC-3).** Ported the same-family moe_down/gate winner: fused exact
  per-call UE8M0 pack (`scale_pack.cu`) with per-N-row weight-scale expansion, then
  `disable_ue8m0_cast=True`. Masked-grouped correctness PASS, `calc_diff = 0`. Harness WIN
  ~40–41% HBM.
- **Floors + NCU (AC-4).** Per-kernel NCU: pack negligible (~549 KB traffic, 4.6% mem SoL,
  ~1–2 µs back-to-back); GEMM weight-streaming, memory-bound (mem SoL 56–62% ≫ sm SoL
  20–24%), DRAM read ~107.8 MB ≈ 100.66 MB `w_fp8`. GEMM not at HBM roofline (occ 12.4%,
  1 wave / 148 SMs) → DeepGEMM public launch knobs are a justified, NCU-gated mechanism.
  `docs/floors/floors_ncu.md`.
- **Preliminary gates (pre-PDL candidate `f59c9e71…`).** `gate_{prelim,1,2,3}`: M16
  ~31.4–31.9 µs / ~40.5% HBM, M32 ~31.5–32.0 µs / ~40.5–41.0%. Cleared 40% but thin —
  `gate_2` M16 dipped to **39.93%** (31.936 µs), i.e. a sub-40% excursion under noise.
  Margin too tight to ship.
- **Knob sweep (AC-5).** Cross-product `compiled_dims × num_sms × tc_util × pdl`, CUPTI
  device-span timed to match the gate (`docs/attempts/knob_sweep.py` + `_out.txt`).
  **PDL = True is the only material, robust knob** (~0.5–0.7 µs); `num_sms` /
  `compiled_dims` differences were within noise. Baked PDL into the candidate with
  global save/restore so it never touches the reference timing.
- **Final gates (AC-6, AC-7).** Candidate `sha256 3940c89c…`. Three authoritative idle
  gates (`docs/attempts/final_gate_{1,2,3}.json`): median **M16 30.855 µs / 41.33% HBM**,
  **M32 30.952 µs / 41.74% HBM**; all six shape-gates ≥ 41.2% HBM, every gate correct with
  `calc_diff = 0`. Both medians clear the inclusive limits (31.882240 / 32.299520 µs) with
  ≥ 1.0 µs headroom. Speedup ≈ 1.53× (conservative ≥ 1.50×). **TARGET MET.**
- **Finalize (task8).** Wrote `docs/results.md`, `docs/run_log.md`, `docs/attempt_dag.md`;
  confirmed `candidate/` is the runnable winner; committed worktree artifacts. No new
  optimization rounds opened.
