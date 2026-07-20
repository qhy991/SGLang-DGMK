# Attempt ledger — glm52 moe_up_proj_decode (HBM ≥ 40%)

Frozen contract: independent up-projection masked-grouped FP8 GEMM (E=8, K=6144, N=2048,
expected_m=128). Inclusive 40% limits: M16 ≤ 31.882240 µs, M32 ≤ 32.299520 µs.

| ID | Change | Evidence | Outcome |
|---|---|---|---|
| A0 | Seed-hash check + three idle CUPTI baselines (f32-scale reference) | seed `2259efd1…` matches; M16 47.209 µs/27.01%, M32 47.368 µs/27.28%; needs 1.48×/1.47× | keep (AC-1) |
| A1 | Candidate = fused exact per-call UE8M0 pack (`scale_pack.cu`, N-row expand) + `disable_ue8m0_cast=True` | masked correctness `calc_diff=0`; harness WIN ~40–41% HBM | keep (AC-2, AC-3) |
| A2 | Per-kernel NCU floor decomposition | pack negligible (549 KB, 4.6% mem SoL); GEMM memory-bound (mem SoL 56–62% ≫ sm SoL 20–24%), 1 wave/148 SMs, occ 12.4% → launch-knob headroom | diagnose (AC-4) |
| A3 | Preliminary 3-gate round on pre-PDL candidate `f59c9e71…` | M16 ~31.4–31.9 µs/~40.5%; `gate_2` M16 **39.93%** (sub-40% excursion) — margin too thin | revise |
| A4 | Knob cross-product `compiled_dims × num_sms × tc_util × pdl`, gate-matched CUPTI span | **PDL=True only material+robust knob** (~0.5–0.7 µs); nsms/cdim within noise | select (AC-5) |
| A5 | Final candidate `3940c89c…` = A1 pack + `disable_ue8m0_cast` + PDL save/restore; three authoritative idle gates | median M16 30.855 µs/41.33%, M32 30.952 µs/41.74%; all 6 shape-gates ≥ 41.2%; `calc_diff=0`; ~1.53× | **ship** (AC-6, AC-7) |

No GEMM rewrite and no DeepGEMM/Harness source edit were needed: the fused packed
dispatch plus the PDL knob cleared both inclusive 40% limits with ≥ 1.0 µs margin. The
stock f32-scale reference remains the untouched correctness oracle.

## DAG

```
A0 baseline ─▶ A1 packed candidate ─▶ A2 NCU floors ─▶ A3 prelim gates (thin, 39.93% dip)
                                                             │
                                                             ▼
                                              A4 knob sweep → PDL is the robust win
                                                             │
                                                             ▼
                                       A5 final candidate (pack + PDL) ─▶ 3 gates ✅ TARGET MET
```
