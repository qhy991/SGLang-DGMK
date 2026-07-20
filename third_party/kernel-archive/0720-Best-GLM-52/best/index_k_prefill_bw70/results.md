# Results — glm52 index_k_prefill 70% HBM campaign

**FINAL VERDICT: TARGET NOT MET — evidence-backed no-go, bounded by the DeepGEMM GEMM
kernel floor. Best correct candidate preserved: ~84.1µs / ~64.4% HBM (1.19x over
baseline), gate exit 0, calc_diff=0 on every label.**

- Target: every workload label (M∈{1024,2048,4096}) at **≥70% HBM (≤77.29µs)** per-label
  median across three authoritative runs. All labels are the same physical GEMM
  (S=65536, N=128, K=6144); 432,799,936 HBM bytes; 8.0 TB/s peak.
- Baseline (reference call): ~100µs / ~54% HBM.
- **Best correct candidate: ~84.1µs / ~64.4% HBM** — a real 1.19x, but short of the
  1.29x needed.

## Three authoritative runs (final candidate, GPU 2, per-label medians)

| label | run1 | run2 | run3 | median | HBM% | target ≤77.29µs |
|---|---:|---:|---:|---:|---:|---|
| M=1024 | 83.98 | 84.09 | 84.12 | **84.09** | 64.3% | FAIL |
| M=2048 | 84.10 | 84.15 | 84.11 | **84.11** | 64.3% | FAIL |
| M=4096 | 83.98 | 84.05 | 83.87 | **83.98** | 64.4% | FAIL |

All correct (3 layers, calc_diff=0 pre- and post-timing), gate exit 0 (3 wins,
0 regress), stable to ±0.15µs.

**AC-7 preflight (Round 1):** these acceptance-critical measurements were re-run with a
GPU 2 idleness check captured immediately before each command (0-1% util, 0 MiB, 0
compute processes; clocks recorded, not locked). The reruns reproduce the numbers within
noise — GEMM-only floor 80.26µs/67.4% and per-label medians 84.07/84.13/83.94µs — so the
no-go verdict stands. See `docs/run_log.md` "Authoritative measurement preflight checks".

## Progress (single-run gate, GPU 2, CUPTI cold-L2)

| Candidate | ~µs | HBM% | correctness | gate |
|---|---:|---:|---|---|
| Baseline (reference call) | 100.1 | 54.0% | PASS (0.0) | exit 1 |
| `compiled_dims="nk"`/`"mnk"` | 98.9 | 54.7% | PASS (0.0) | exit 0 |
| `mnk`+`pdl` | 97.5 | 55.4% | PASS (0.0) | exit 0 |
| `mnk`+`pdl`+scale prepack (helper) | 93.1 | 58.0% | PASS (0.0) | exit 0 |
| **`mnk`+`pdl`+`num_sms=128`+fused Triton pack (BEST)** | **84.1** | **64.4%** | **PASS (0.0)** | **exit 0** |

## Root cause and the named no-go bound (NCU + decomposition backed)

1. The harness times the device-kernel **span over all kernels in `run()`**
   (`testbench/harness/timing.py:132`).
2. DeepGEMM's default f32-scale path runs a **~29µs, 5-kernel** f32→UE8M0 scale
   transform every call; NCU shows the GEMM itself is already memory-bound (~74% on its
   own traffic). Pre-packing removed most of that overhead.
3. A per-call **fused Triton UE8M0 pack** kernel (byte-identical to DeepGEMM's own packed
   layout; scales are exact powers of two so the exponent pack is lossless) cut the
   scale-prep to a near-optimal ~3µs contribution and reached **~84µs / ~64.4%**.
4. **The wall is the GEMM kernel itself.** With scales pre-packed and zero scale-prep in
   the timed window, `fp8_gemm_nt` on the exact shape S=65536, N=128, K=6144 runs at
   **≥80.19µs (≤67.5% HBM)** — best over `compiled_dims∈{mnk,nk,""}`, `pdl∈{T,F}`,
   `num_sms∈{64..148}`. Since 80.19µs > 77.29µs, **no scale-preprocessing optimization
   can reach ≥70% HBM.**

**Named bound:** `T_total ≥ T_gemm_floor(≥80.19µs / ≤67.5% HBM)`. Reaching ≥70% would
require a custom SM100 GEMM that beats vendor-tuned DeepGEMM's tcgen05/TMEM kernel on a
narrow-N=128 memory-bound shape — the highest-risk path (flagged by Codex and the plan),
disproportionate to the memory-bound knob/scale-prep scope, and unlikely to beat
DeepGEMM meaningfully at N=128. Recorded as an evidence-backed no-go per AC-8 / DEC-1.

## Evidence artifacts
- `docs/run_log.md` — env, hashes, full attempt ledger (A0–A6), NCU breakdowns, GEMM-floor
  decomposition, nsys timeline, 3-run confirmation.
- `docs/candidate_dag.md` — attempt DAG.
- `docs/ncu/*.ncu-rep` (on disk, gitignored) — `mnk_pdl_M1024`, `prepack_M1024`,
  `fusedpack_v1_M1024`, `fusedpack_v2_M1024`; `docs/ncu/task7_codex_analysis.md`.
- `variants/` — every attempted candidate (nk, mnk, mnk_pdl, mnk_sms132, prepack,
  fusedpack) + `measure.py`.

## Rule compliance
- Correctness: all 3 layers pass, calc_diff=0 pre+post on every label.
- Stateless: no caching / persisted operands / input mutation; the scale pack is per-call
  and lossless (byte-identical to DeepGEMM's own f32→UE8M0 cast; calc_diff=0), not
  re-quantization.
- No fake M dispatch: `compiled_dims="mnk"` bakes the genuinely-constant physical shape;
  all labels run the identical path.
- Harness tree untouched (`index_k_prefill/candidate.py` CLEAN); measured only on GPU 2;
  NCU evidence captured before the custom kernel; GPU clocks not locked (DEC-3); public
  DeepGEMM APIs only (DEC-4).
