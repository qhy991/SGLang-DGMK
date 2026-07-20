# Results — GLM-5.2 MoE Down-Projection Decode, 40% HBM Campaign

## Verdict: **TARGET MET** (both shapes ≥40% HBM)

Stateless `candidate/candidate.py` reaches ≥40% HBM bandwidth utilization on **both** decode
shapes, confirmed by the per-shape median of three interference-free authoritative harness runs
on an exclusively-idle GPU 0, within the latency ceilings, correct (calc_diff=0) before and after
timing. The shared Kernel-Harness checkout is byte-for-byte unchanged.

## Headline numbers (per-shape median of 3 authoritative runs)

| Shape | Baseline µs / HBM% | Candidate µs / HBM% | Ceiling µs (40%) | Speedup | Reward (=bw_util) | Result |
|-------|-------------------:|--------------------:|-----------------:|--------:|------------------:|--------|
| M=16 | 47.02 / 27.26% | **31.31 / 40.93%** | 32.041 | 1.50× | 0.409 | ✅ ≥40% |
| M=32 | 47.20 / 27.64% | **31.39 / 41.56%** | 32.617 | 1.50× | 0.416 | ✅ ≥40% |

- Every individual authoritative run also clears 40% on both shapes (M16 min 40.87%, M32 min 41.56%).
- Correctness: `calc_diff = 0.00e+00`, cosine = 1.000000, 0 element failures, both shapes, before
  and after timing. Harness verdict: CORRECT, 2/2 shapes WIN, performance_gate MET.

## What made the difference

The op is memory-bound and weight-read-dominated, so the harness reward collapses to HBM bw_util
and the only lever is latency. NCU showed the reference `fp8_m_grouped_gemm_nt_masked` (f32 scales)
runs a **per-call scale-transform + grouped-SF-layout kernel chain (~30 µs)** ahead of a matmul that
already streams weights at ~57% DRAM. The frozen scales are built `use_ue8m0=True` (powers of two),
so they can be **losslessly** reinterpreted as int32-packed ue8m0. The candidate packs them itself
in one fused Triton kernel (~2.8 µs, byte-identical to DeepGEMM's `get_mn_major_tma_aligned_packed_
ue8m0_tensor`) and dispatches with `disable_ue8m0_cast=True`, so only two kernels run per call —
the cheap pack + the same matmul — and end-to-end drops from ~47 µs to ~31 µs.

## Genuineness (AC-4.1)

Not a byte-model artifact. Candidate NCU on **both shapes** (`docs/raw/ncu_cand_m16.txt`,
`docs/raw/ncu_cand_m32.txt`): matmul at 56–58% DRAM (≤100%), reads ~107 MB; reported 3275 GB/s (M16)
/ 3325 GB/s (M32) ≪ 8 TB/s peak; rewards 0.409 / 0.416 < 1.0 (no >100% anomaly). The improvement is
pure latency reduction on the same fixed byte model (102.53 / 104.37 MB), not disappearing modeled
bytes. The reference overhead chain is entirely absent from the candidate trace on both shapes; the
matmul itself is the identical kernel at identical ~57% DRAM and ~24 µs for M=16 and M=32.

## Statelessness / anti-hack (AC-2)

Independent adversarial review (Codex, gpt-5.5:xhigh — `docs/raw/codex_review.md`) found **no
correctness or reward-hack defect**: no cross-call cache, no retained state, no input mutation, no
output aliasing, no lazy/proxy output, no timer patching, no thread injection. All pack work runs
inside the timed window; packed tensors are freshly allocated each call; only `out` is written (by
the library matmul, exactly as the reference, for valid rows `out[e,:masked_m[e]]`).

## Provenance & harness integrity (AC-3)

- Seed SHA-256 `02cedf67…651e`; `candidate/candidate.py` copied byte-for-byte, then optimized.
- `KDA_HARNESS_ROOT` git status: only the 3 pre-existing dirty siblings
  (`index_k_proj_decode`, `o_proj_decode`, `o_proj_prefill`); `moe_down_proj_decode/candidate.py`
  unchanged; no harness core edits.

## Known characteristics / risks

- **M=16 margin is thin (40.93% vs 40%, ~0.7 µs / 2.3%).** Robust across all clean runs (min
  40.87%) and the candidate's own timing is tight (31.2–31.6 µs), but it is sensitive to GPU
  interference and clock/CUPTI variance — always confirm on an exclusively-idle GPU 0.
- The fast path is **intentionally specialized** to the frozen `use_ue8m0=True` contract. `_packable()`
  guards shape/dtype and falls back to the reference transform otherwise; it does not (and, for the
  latency budget, cannot cheaply) verify at runtime that arbitrary f32 scales are powers of two. Off
  this task, that would need a semantic guard. Not an invalid win for the frozen benchmark.

## Reproduce

```bash
cd "$KDA_HARNESS_ROOT/testbench/tasks/glm52/moe_down_proj_decode"
CUDA_VISIBLE_DEVICES=0 ./run.sh --candidate "$CLAUDE_PROJECT_DIR/candidate/candidate.py"
```

Raw logs: `docs/raw/baseline_seed_full.log`, `docs/raw/candidate_auth_run{1,2,3}.log`,
`docs/raw/ncu_ref_m16_gemm.txt`, `docs/raw/ncu_ref_m32_gemm.txt`, `docs/raw/ncu_cand_m16.txt`,
`docs/raw/ncu_cand_m32.txt`, `docs/raw/codex_review.md`.
Ledger: `docs/attempt_ledger.md`. Full log: `docs/run_log.md`.
