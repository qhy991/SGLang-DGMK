# Attempt DAG — moe_gate_proj_decode >=40% HBM

Single-mechanism campaign: the same-family packed-UE8M0 lever was a known winner on
identical shapes, so the DAG is a straight port + verify chain with one branch
(source-fork) that was evaluated and correctly pruned.

```
[baseline: stock reference, f32 UE8M0 scales]
   M16 46.657us / 27.33%   M32 46.601us / 27.72%   calc_diff 0
   gap to 40%: 1.463x (M16), 1.443x (M32)
        │
        ▼
[A1] PORT moe_up winner  ── fused UE8M0 pack (CUDA, 1 launch) + disable_ue8m0_cast + PDL
   │   pack bit-exact vs DeepGEMM reference packer (xp/wp equal, both shapes)
   │   validate sweep: M16 30.952us/41.20%  M32 30.968us/41.72%  calc_diff 0  exit 0
   │   ✓ ACCEPTED — clears both 40% limits on first run
   │
   ├─▶ [A2] NCU floor characterization (gate for source work)
   │      pack ~5.5us isolated / ~1-2us real, 225KB, negligible
   │      GEMM sm100_fp8_fp4_gemm_1d1d: 26us, 107.8MB read, mem SoL 56%, 1 wave
   │      → memory-bound; residual headroom exists BUT…
   │
   └─▶ [B1] BRANCH: isolated DeepGEMM source fork (AC-5)  —— PRUNED
          rationale: moe_up knob cross-product already established PDL is the only
          material public knob (num_sms/tc_util/compiled_dims = noise); PDL applied.
          Both shapes clear 40% with margin (0.83/1.25us). A source fork adds risk
          with no target need. NOT PURSUED. Stock oracle frozen.
        │
        ▼
[GATES] four authoritative idle-GPU runs on winner A1
   g1 M16 30.888/41.29  M32 30.896/41.82   exit 0
   g2 M16 31.000/41.14  M32 31.040/41.62   exit 0
   g3 M16 31.050/41.07  M32 31.048/41.61   exit 0
   g4 M16 30.800/41.41  M32 31.016/41.66   exit 0   (confirmation, verified-idle GPU)
        │
        ▼
[TARGET MET]  median-of-medians M16 30.944us/41.21%  M32 31.028us/41.64%
              both clear inclusive 40% (31.882 / 32.300 us); calc_diff 0 throughout
```

## Fastest correct candidate

`candidate/candidate.py` + `candidate/scale_pack.cu` (A1). This IS the final winner;
no faster correct candidate exists in the DAG (the pruned B1 branch was never built
because A1 already met the target with margin and NCU showed no material public knob
beyond the PDL already in A1).

## Nodes rejected / pruned

- **B1 (DeepGEMM source fork):** pruned pre-implementation. NCU-gated per AC-5; the gate
  (material headroom recoverable by a source edit *and* a target still unmet) was not
  triggered — target met, only-material-knob (PDL) already in the winner.
- **CUDA-event floor probe:** measurement artifact (launch latency + L2-flush tail);
  superseded by NCU + CUPTI. Retained in `docs/floors/measure_floors.py` for provenance.
