# Candidate Attempt DAG / Ledger — o_proj decode 35% HBM

Lineage of the candidate (recorded byte-for-byte descendant of the Harness seed), with every
attempted mechanism and its measured outcome. Authoritative latencies are CUPTI cold-L2
medians via `run.sh`; exploratory rows use the same CUPTI timer in `artifacts/probe_*.py`.

```
A0  SEED (copied byte-for-byte from Harness; SHA d98f2710…a5b7b)
│     Triton split-K (M=16) + DeepGEMM float32-scale reference fallback (M=32)
│     AUTHORITATIVE: M=16 45.50us/27.79%  M=32 53.26us/23.85%   → MISS both
│     NCU: split-K main kernel 35% DRAM + wasteful _reduce_splitk partial round-trip (M=16);
│          reference launches ~5 internal f32→ue8m0 transform kernels + 43% DRAM GEMM (M=32)
│
├─ (explore) packed-UE8M0 GEMM with PRE-packed scales (not per-call)
│     probe: M=16 32.4us/39.1%  M=32 33.4us/38.1%  → clears 35% ⇒ mechanism validated
│     +compiled_dims='nk': M=16 30.2us/41.9%  M=32 31.3us/40.5%  → faster codegen chosen
│
├─ (reject) per-call repack via torch index_select + DeepGEMM C packer + gemm
│     probe: full ~63us (WORSE than reference). Cause: weight block scale expanded to a 3MB
│     float32 intermediate across several eager kernels; host-launch gaps counted by the
│     CUPTI device span. REJECTED.
│
├─ (reject) fused Triton packers (2 launches) + gemm
│     probe: full ~45-49us. Better but still 2 Triton launches + allocs + views ⇒ host gaps.
│     REJECTED (Triton launch host-overhead too high).
│
├─ (reject) raw-CUDA 2-kernel repack (separate x,w launches) + gemm
│     probe: M=16 35.55us/35.6% (OK)  M=32 36.91us/34.4% (MISS by 0.6us). REJECTED for M=32.
│
└─ A1  WINNER — single FUSED CUDA kernel (x+w in one launch, one allocation) + fp8_gemm_nt(nk)
      candidate/candidate.py SHA 92ab42a2… , candidate/scale_pack.cu SHA c8106d42…
      (pre-hardening build 74f922a7…/de778253… measured identically)
      AUTHORITATIVE triple-gate median: M=16 33.024us/38.29%  M=32 33.824us/37.56%  → MEET both
      calc_diff 0 (pre & post timing), 2/2 WIN, no reference fallback, stateless.
      NCU: sm100_fp8_fp4_gemm_1d1d at ~52% peak DRAM (compute idle) ⇒ DRAM-bandwidth-bound;
           fused_pack_kernel moves ~37KB (negligible). Named floor = HBM read bandwidth.
```

## Rejected-attempt ledger (why each was dropped)
| Attempt | Full latency (probe) | Verdict | Reason dropped |
|---|--:|---|---|
| torch index_select + C packer + gemm | ~63us | reject | 3MB f32 intermediate + eager-kernel host gaps |
| fused Triton 2-launch + gemm | ~45-49us | reject | Triton per-launch host overhead + allocs/views |
| raw-CUDA 2-kernel + gemm | 35.6 / 36.9us | reject | M=32 misses by ~0.6us (2nd launch gap) |
| **fused single-kernel + gemm(nk)** | **33.0 / 33.8us** | **ACCEPT** | both shapes clear 35% with 2.5-3.1us margin |

## Directions B and C — built and gated as rejected attempts (Round 1)
Round 0 deferred B/C under evidence-driven escalation; the Round-0 review's no-deferral rule
required them built and measured. Both were implemented as standalone task-local candidates,
gated on both shapes (correct), and rejected in favor of A1:

```
B0  variants/directionB/candidate.py — single-pass Triton fp8 block-scaled GEMM (no split-K)
      GATE (idle GPU): M=16 69.41us/18.22%  M=32 73.92us/17.18%  calc_diff 3.2e-9  CORRECT/REGRESS
      NCU: 24 CTAs, occ 6.25%, DRAM 10.1% ⇒ latency/occupancy-bound. ~2.1x slower than A1. REJECT.

C0  variants/directionC/{candidate.py,scale_gemm.cu} — custom SM100 CUDA weight-streaming GEMM
      GATE (idle GPU): M=16 3961us/0.32%  M=32 5146us/0.25%  calc_diff 3.2e-9  CORRECT/REGRESS
      NCU: 96 CTAs, occ 3.12%, DRAM 0.35% ⇒ ALU-bound naive fp8 convert. ~120-150x slower. REJECT.

FINAL SELECTION (fastest correct, three-gate median): A1 (packed DeepGEMM).
  A1 re-confirmed after compliance-hardening edit → candidate.py SHA 054cf91a7cf5…
  (behavior identical to 92ab42a…; only adds a loud build-failure warning).
  Re-confirm median: M=16 33.040us/38.27%  M=32 33.835us/37.54%. Both ≤ ceilings.
```

The plan's escalation logic (custom kernels only when NCU justifies) is vindicated by
measurement: a naive Triton/CUDA weight stream is far worse than DeepGEMM's tuned SM100
tcgen05 kernel. Matching or beating A1 would require the full TMA/TMEM/tcgen05 route — which
is unnecessary because A1 already clears ≥35% on both shapes with margin.
