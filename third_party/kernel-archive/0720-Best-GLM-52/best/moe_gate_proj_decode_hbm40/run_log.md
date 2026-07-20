# Run Log — moe_gate_proj_decode >=40% HBM (resume session)

Chronological log of the resume session. Prior session died early (API connection
closed) while porting the moe_down winner; this session resumed from
`docs/CONTINUE.md` / `docs/refined-plan.md` and did **not** recreate scaffolding.

All GPU work on idle NVIDIA B200, `CUDA_VISIBLE_DEVICES=2`. Harness at
`/home/qinhaiyan/Kernel-Harness`, task `testbench/tasks/glm52/moe_gate_proj_decode`.
Runner helper: `docs/run_gate.sh <prefix> [run.sh args]` → writes `<prefix>.log`
(full table) + `<prefix>.json` (result JSON between RESULT_JSON markers).

## 1. Context recovery
- Read `docs/CONTINUE.md`, `docs/refined-plan.md`, `.humanize/kernel-agent/refined-plan.md`,
  RLCR `state.md` (round 0), config.toml, prompt.md.
- Confirmed target: masked grouped FP8, E=8 K=6144 N=2048 M={16,32}, expected_m=128.
  Inclusive 40%: M16 <=31.882240 us, M32 <=32.299520 us.
- Reference port: `…/moe-up-decode-…/candidate/{candidate.py,scale_pack.cu}` (winner:
  M16 41.33% / M32 41.74%). Confirmed moe_gate == moe_up shapes/scales/masks.

## 2. AC-1 — seed + three baselines
- `sha256(seed candidate.py) = 08a8d674…fa6a8` == pinned hash. ✓
- deep_gemm 0.1.4 has get_pdl/set_pdl. GPU 2 idle (0%, 4 MiB).
- `run.sh --json` (no --candidate) x3 → reference medians M16 46.657 us/27.33%,
  M32 46.601 us/27.72%, calc_diff 0. `docs/baseline/run_{1,2,3}.{json,log}`,
  `docs/baseline/baseline.md`.

## 3. AC-2/AC-3 — port the fused UE8M0 pack winner
- `candidate/scale_pack.cu` — byte-identical to the moe_up winner (generic; reads
  E/M/N/K from tensor shapes).
- `candidate/candidate.py` — ported from moe_up; only diffs: docstring label,
  extension name `glm52_moe_gate_decode_scale_pack` (avoids torch-ext cache collision),
  warning text. Fused pack + `disable_ue8m0_cast=True` + PDL save/restore.
- Validation sweep (`docs/attempts/validate.*`): M16 30.952 us/41.20%,
  M32 30.968 us/41.72%, calc_diff 0, `is_reference_fallback:false`, exit 0.
- Extension built: `~/.cache/torch_extensions/…/glm52_moe_gate_decode_scale_pack.so`.
- Pack bit-exactness vs `get_mn_major_tma_aligned_packed_ue8m0_tensor`:
  `xp (8,128,12)` and `wp (8,2048,12)` **equal** on both shapes. ✓

## 4. AC-4 — floors + NCU
- Crude CUDA-event floor probe (`docs/floors/measure_floors.py`) — kept for provenance
  only; inflated by launch latency + 256 MB L2-flush tail (pack 7.8 us, M32 e2e 65 us),
  NOT authoritative.
- NCU (`docs/ncu/ncu_driver.py`, NVTX-scoped, GPU 2) → `docs/ncu/ncu_m{16,32}.csv`:
  - `fused_pack_grouped_kernel`: ~5.5 us isolated, 225 KB DRAM, 4.4% mem SoL, 0.69
    waves → real back-to-back contribution ~1-2 us (negligible).
  - `sm100_fp8_fp4_gemm_1d1d_impl` (DeepGEMM Blackwell packed fast kernel): ~26 us,
    DRAM read 107.8 MB (weight-streaming), mem SoL ~56% >> sm SoL ~25%, occ 12.4%,
    grid 148 = num_sms, 1 wave. Memory-bound; residual roofline headroom but only PDL
    is a material public knob (already applied) → no source fork warranted (AC-5).
  - `docs/floors/floors_ncu.md`.

## 5. AC-6 — authoritative gates
- Observed a transient 37% util / 1.1 GB snapshot before gate 1; traced to co-resident
  pid 762983 (0% compute, idle allocation) which then exited (GPU back to 4 MiB / 0%).
- Gate 1: M16 30.888/41.29%, M32 30.896/41.82% — exit 0.
- Gate 2: M16 31.000/41.14%, M32 31.040/41.62% — exit 0.
- Gate 3: M16 31.050/41.07%, M32 31.048/41.61% — exit 0.
- Gate 4 (confirmation, re-verified idle GPU): M16 30.800/41.41%, M32 31.016/41.66% — exit 0.
- `docs/attempts/final_gate_{1,2,3,4}.{json,log}`. All calc_diff 0.
- Median-of-medians: M16 30.944 us/41.21%, M32 31.028 us/41.64%. Worst-gate medians
  (31.050 / 31.048 us) clear the limits (31.882 / 32.300 us).

## 6. AC-7 — finalize
- `docs/results.md`, this `docs/run_log.md`, `docs/attempt_dag.md`.
- RLCR round-0 summary + goal-tracker updated. `docs/CONTINUE.md` marked complete.
