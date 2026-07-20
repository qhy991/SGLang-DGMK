# No-Go Disposition — GLM-5.2 o_proj decode 35% HBM

## Verdict: NO-GO PATH NOT APPLICABLE — a confirmed winning candidate exists

The plan (AC-6) requires an evidence-backed no-go dossier **only if ≥35% HBM is physically
unreachable on both shapes**. It is not. A confirmed candidate (A1) clears the bar on both
frozen shapes on the authoritative three-gate per-shape median, so the campaign terminates on
the **confirm** path, and the no-go dossier is not produced.

## Why the target is reached (confirm-path evidence)
| Shape | A1 median | HBM% | ceiling (35%) | result |
|---|--:|--:|--:|---|
| M=16 | 33.040us | 38.27% | ≤36.13us | PASS (margin 3.09us) |
| M=32 | 33.835us | 37.54% | ≤36.29us | PASS (margin 2.45us) |

Correctness: calc_diff 0 pre- and post-timing on both shapes; no reference fallback
(`is_reference_fallback=false`). Full evidence in `docs/results.md`, `docs/run_log.md`,
`docs/attempt_dag.md`, `artifacts/ncu/NCU_REPORT.md`, `docs/compliance_review.md`.

## The three directions, all measured (no unexplored path)
Every planned direction was built and gated, so "unreachable" is not merely asserted — the
winning path is demonstrated and the alternatives are shown inferior:

| Direction | Built & gated | M=16 / M=32 | Correct | Outcome |
|---|---|---|---|---|
| A — packed-UE8M0 DeepGEMM dispatch | yes (`candidate/`) | 33.04 / 33.84us | yes (diff 0) | **SELECTED — meets ≥35% both shapes** |
| B — single-pass Triton (no split-K) | yes (`variants/directionB/`) | 69.41 / 73.92us | yes (diff 3.2e-9) | rejected (~2.1× slower; occupancy-bound) |
| C — custom SM100 CUDA weight stream | yes (`variants/directionC/`) | 3961 / 5146us | yes (diff 3.2e-9) | rejected (~120-150× slower; ALU-bound) |

## Had the target been unreachable (for completeness)
If A1 had failed and B/C were the best correct candidates, the no-go dossier's four required
elements already exist and would read:
- **Best correct candidate**: A1 (or, absent A, Direction B at ~18% HBM).
- **NCU bottleneck**: for the packed path, DRAM-bandwidth-bound at ~52% of peak
  (`artifacts/ncu/NCU_REPORT.md`); for B, occupancy-bound (24 CTAs, 6.25%); for C, ALU-bound
  (occupancy 3.12%, DRAM 0.35%).
- **Named floor**: HBM read bandwidth — the ideal read-once 100.66 MB weight over 8.0 TB/s
  peak is 12.6us, and 35% of peak = 2.8 TB/s / 36.2us; the DeepGEMM SM100 kernel extracts
  ~52% of peak DRAM in its active window.
- **Rejected-attempt ledger**: `docs/attempt_dag.md` (naive torch repack, fused-Triton repack,
  2-kernel-CUDA repack, Direction B, Direction C).

None of this is needed for acceptance because the confirm path succeeded.
