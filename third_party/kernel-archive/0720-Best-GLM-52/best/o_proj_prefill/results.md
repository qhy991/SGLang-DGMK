# Results — GLM-5.2 o_proj prefill FP8 GEMM (Kernel-Harness bridge)

**Terminal verdict: PARTIAL WIN delivered · EVIDENCE-BACKED NO-GO on the all-shape hard target.**

The vendored candidate passes the authoritative harness (exit 0: correct on all three shapes, 3 wins, 0 regressions, median-of-3) but misses every hard per-shape latency/utilization target. Committed NCU artifacts show DeepGEMM's SM100 FP8 GEMM kernel is already 81–96% tensor-pipe-active, and a genuine fused custom-kernel spike (Triton, task8) is correct but ~6× slower — so the hard targets require exceeding an already near-speed-of-light vendor kernel, which is not achievable under the frozen f32-block-scale contract and the stateless (DEC-3) constraint.

## Environment
Idle B200 GPU 0 · driver 610.43.02 · CUDA 13.0 (nvcc 13.2) · torch 2.11.0+cu130 · deep_gemm 0.1.4 · harness `7d79e5e`. Full packet in `run_log.md`. Implied FP8 dense peak = 4500 TFLOPS.

## Final candidate (vendored: `candidates/candidate_final.py`, live at task `candidate.py`)
`deep_gemm.fp8_gemm_nt(..., compiled_dims="mnk")` + `deep_gemm.set_pdl(True)` at import; 148 SMs; stateless (no cached/transformed operand copies, DEC-3-compliant).

## Authoritative median-of-3 gate (candidate.py, exit 0 ×3, all correct pre+post-timing)

Numbers below are the median-of-3 from `docs/artifacts/gate/final_cand{1,2,3}.log` (the re-confirmation gate). The initial gate `docs/artifacts/gate/cand{1,2,3}.log` agrees within run-to-run noise (87.753 / 139.416 / 268.834 µs; 1.032 / 1.020 / 1.013×) — both sets exit 0 with 3 wins / 0 regressions.

| M | baseline median µs | candidate median µs | speedup (live) | user-approx target speedup | util% | target µs | target util | **meets hard target?** |
|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| 1024 | 90.592 | 87.704 | **1.033×** | ~1.38× | 52.24 | 65.4 | 70% | ❌ |
| 2048 | 142.161 | 139.464 | **1.019×** | ~1.17× | 65.70 | 122.2 | 75% | ❌ |
| 4096 | 272.424 | 268.991 | **1.013×** | ~1.20× | 68.13 | 229.1 | 80% | ❌ |

Harness verdict: 3 wins / 0 regressions → exit 0 (PARTIAL WIN). Both denominators reported: live same-run baseline (90.59/142.16/272.42 µs) and the user-stated approximate targets (~1.38/1.17/1.20×). Required same-run speedups to MEET target: 1.385× / 1.163× / 1.189×.

### Note on latency ⟺ utilization
At the fixed 4500 TFLOPS peak, util = FLOPs/(time·peak), so each latency target maps exactly to its util target (65.4µs↔70.05%, 122.2µs↔74.98%, 229.1µs↔79.99%). Meeting one meets the other; DEC-2 ("both strictly required") is satisfied by construction. Gate on latency.

## Named bound (NCU application-replay on the isolated DeepGEMM GEMM kernel + nsys per-call breakdown)

`fp8_gemm_nt` per call runs a set of f32→ue8m0 scale-layout transform kernels (`transpose_and_pack_fp32_into_ue8m0`, `pack_fp32_into_ue8m0`, `vectorized_gather`, `arange`, `div_floor`; each ~60 instances in the committed nsys tables) **then** the GEMM. Cleanly bounded from committed numbers, the per-call transform+gaps ≈ harness total − isolated GEMM ≈ 90.6−67.5=**23µs** (M=1024), 142.2−121.9=**20µs** (M=2048), 272.4−251.1=**21µs** (M=4096) — roughly M-independent, so ~25% of the small-M span but <8% at M=4096. This, not GEMM slowness, is why overall util rises with M. Both candidate and reference pay it identically.

Isolated GEMM kernel `sm100_fp8_fp4_gemm_1d1d_impl` (128×128 tile, grid 148 = 1 persistent CTA/SM), NCU application-replay — **committed artifacts `docs/artifacts/ncu/gemm_sol_m{1024,2048,4096}.csv`**:

| M | isolated-GEMM µs | tensor-pipe active% | SM throughput% | occupancy% | dram% | achieved FP8 util% |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 67.5 | 81.4 | 76.2 | 12.4 | 33.4 | 67.9 |
| 2048 | 121.9 | 92.5 | 86.8 | 12.5 | 25.1 | 75.2 |
| 4096 | 251.1 | **95.7** | 91.5 | 12.5 | 25.3 | 73.0 |

(achieved FP8 util = FLOPs/(dur·4500 TFLOPS). Occupancy ~12.5% is expected for these few-warp persistent tcgen05 kernels that keep the tensor pipe fed; the binding resource is the tensor pipe, not warp occupancy.)

### Timing methodology reconciliation (Round 0 review item)
Three numbers, three different measurements — reconciled here:
- **Harness authoritative total** (gates the campaign): `timing.py` records `max(kernel.end) − min(kernel.start)` over all kernels in one `run()` call, cold-L2, median over iterations. This is the **full per-call device span = transform + inter-kernel gaps + GEMM**: 90.6 / 142.2 / 272.4 µs.
- **NCU isolated GEMM** (SOL efficiency, above): a single GEMM-kernel launch in isolation — 67.5 / 121.9 / 251.1 µs. Used only for the tensor-pipe-active / SM-throughput percentages, not as a standalone latency claim.
- **nsys in-loop GEMM median** (`docs/artifacts/nsys/kern_sum_m*.csv`): the GEMM kernel timed back-to-back in a 60-call loop; its median runs slightly higher than the isolated NCU launch (e.g. ~275 µs at M=4096) because of warm-cache/clock and back-to-back scheduling effects. The nsys tables also confirm the per-call transform kernels (`transpose_and_pack_fp32_into_ue8m0`, `pack_fp32_into_ue8m0`, `vectorized_gather`, `arange`, `div_floor`), each ~60 instances, i.e. one set per call.

Per-shape bound (using the NCU isolated GEMM as the optimistic floor a custom kernel would have to reach):
- **M=1024** — even the isolated GEMM (67.5 µs) already exceeds the 65.4 µs target. Meeting it needs beating DeepGEMM's GEMM by >3% *and* removing all transform overhead. Tensor pipe 81.4% active leaves some room, but small-M scheduling + scale work make it fragile.
- **M=2048** — isolated GEMM (121.9 µs) is essentially at the 122.2 µs target. The full per-call span (142 µs) exceeds target only because of the transform + gaps. So M=2048 is reachable *only* if the transform is removed without slowing the GEMM — i.e. a fused custom kernel matching DeepGEMM's 92.5%-tensor-active GEMM, or (DEC-3-forbidden) cached packed scales.
- **M=4096** — GEMM is 95.7% tensor-pipe-active. The 73.0%→80% achieved-util gap is not recoverable slack; the pipe is nearly continuously active. Needs a ~9% faster-than-DeepGEMM GEMM.

## task8 — measured in-kernel fused-scale spike (Round 1 built, Round 2 corrected)
A genuine, stateless (DEC-3-preserving) fused **Triton** blockwise-scaled FP8 GEMM was built and measured to test the fusion hypothesis. `run()` passes the **raw f32** `inputs["x_scale"]`/`inputs["w_scale"]` directly into the kernel; the reference-equivalent ue8m0 rounding is done **in-kernel** via the exact bit manipulation `deep_gemm.ceil_to_ue8m0` uses (`bitcast f32→int; exp=((bits>>23)&0xFF)+(mantissa!=0); clamp[1,254]; (exp<<23)→float`). No `ceil_to_ue8m0`/prepacking/transform helper is called before launch, so the separate f32→ue8m0 transform kernels are genuinely eliminated. Source: `candidates/task8_triton_spike/candidate.py`; probe log: `docs/artifacts/task8/triton_fused_inkernel_probe.log`.

| M | fused-spike µs (best of 32 autotuned cfgs) | vs DeepGEMM candidate | correct? | MFU% |
|---:|---:|---:|:--:|---:|
| 1024 | 364.4 | 0.249× | ✅ calc_diff 2.86e-9 | 12.6 |
| 2048 | 675.6 | 0.210× | ✅ calc_diff 2.86e-9 | 13.6 |
| 4096 | 1338.4 | 0.203× | ✅ calc_diff 2.84e-9 | 13.7 |

**Result:** the in-kernel fused approach is **correctness-viable** (in-kernel ue8m0 rounding matches the reference to 2.86e-9) and it *does* remove the separate transform launches (it is faster than an earlier variant that pre-rounded scales outside the kernel: ~0.155×→~0.21×, confirming transform removal helps some). But the hand-written kernel still reaches only ~12–14% MFU — **~4–5× slower** than DeepGEMM. Removing the transform is not enough when the GEMM body cannot match DeepGEMM's 92–96% tensor-active tcgen05 pipeline. This resolves the M=2048 "only blocked by the transform" nuance with a measured artifact: the transform can be fused away (correctly, in-kernel), yet no hand-authored kernel available here matches DeepGEMM's GEMM throughput, so the net result is far worse. A `.cu` CUTLASS/tcgen05 kernel that both matches DeepGEMM and fuses scaling is the sole remaining theoretical path; NCU shows DeepGEMM is already ~SOL, so beating it is not credible in this campaign.

Also tested and rejected (Round 1): passing DeepGEMM-**pre-packed** int32 ue8m0 scales (`get_mn_major_tma_aligned_packed_ue8m0_tensor`) to `fp8_gemm_nt` — fails a DeepGEMM layout assertion (`sf.size(-2) == ceil_div(mn, gran_mn)`); `fp8_gemm_nt` always re-transforms f32 scales internally. Computing the packing per call is the same transform (no win); caching it is DEC-3-forbidden and is exactly the "fake win" the harness caveat rejects.

## Escalation decision (Codex, `docs/artifacts/codex/escalation_decision.txt`)
**NO_GO.** Do not continue knobs (ceiling ~1.01–1.03× vs required 1.16–1.39×). Do not escalate to a custom SM100 kernel as the campaign path: it would have to match/beat an already 92–96% tensor-pipe-active DeepGEMM kernel on all three shapes under DEC-3 — not a credible all-shape median-of-3 strategy. The suggested M=2048 fusion spike **was executed** this round (task8, above): correct but ~6× slower, confirming the path.

## DEC-3 impact (surfaced for user reconsideration)
DEC-3 (stateless; no cached packed ue8m0 scales) is the sole blocker for **M=2048** (GEMM-only 121.8 µs < 122.2 µs target). Relaxing DEC-3 to allow precomputing the packed scales once per frozen shape (outside the timed region; correctness re-checked pre+post-timing) would plausibly bring M=2048 to target. **However, relaxing DEC-3 does not unblock M=1024 (still needs a faster-than-DeepGEMM GEMM) or M=4096 (near-saturated tensor pipe).** See Pending User Decision DEC-4 in `.humanize/kernel-agent/refined-plan.md`/goal-tracker.

## Attempt ledger / candidate DAG

| # | Config | Shapes result (median/probe) | Correct | Exit | Disposition |
|---|--------|------------------------------|:--:|:--:|---|
| A0 | reference `fp8_gemm_nt()` defaults (baseline) | 90.59/142.16/272.42 µs (median-of-3) | ✅ | 1 | baseline denominator |
| A1 | `compiled_dims=""` | 0.999/1.001/1.002× | ✅ | 1 | neutral — rejected |
| A2 | `compiled_dims="n"` | 0.999/1.001/1.002× | ✅ | 1 | neutral — rejected |
| A3 | `compiled_dims="k"` | 1.012/1.006/1.005× | ✅ | 0 | small win, superseded |
| A4 | `compiled_dims="nk"` | 1.011/1.006/1.005× | ✅ | 0 | small win, superseded |
| A5 | `compiled_dims="mn"` | 1.001/1.002/1.003× | ✅ | 1 | neutral — rejected |
| A6 | `compiled_dims="mnk"` | 1.016/1.008/1.006× | ✅ | 0 | best compiled_dims |
| A7 | `mnk` + `disable_ue8m0_cast=True` | — | ❌ | 2 | **correctness fail** (calc_diff vs ue8m0 reference) — rejected |
| A8 | `""` + `disable_ue8m0_cast=True` | — | ❌ | 2 | correctness fail — rejected |
| A9 | `mnk` + PDL (148 SMs) | 1.031/1.019/1.011× | ✅ | 0 | **best overall** |
| A10 | `mnk` + `num_sms=132` | 0.963/0.943/0.949× | ✅ | 0(w) | regressive vs A9 — rejected (compute-bound; SM loss unrecovered) |
| A11 | `mnk` + `num_sms=128` | 0.954/0.919/0.966× | ✅ | 0 | wave-quant hypothesis falsified — rejected |
| A12 | `mnk` + `num_sms=96` | 0.816/0.794/0.797× | ✅ | 0 | much worse — rejected |
| **FINAL** | **`mnk` + PDL, 148 SMs (median-of-3)** | **1.033/1.019/1.013×** | ✅ | **0** | **vendored** (final_cand*.log; initial cand*.log agrees: 1.032/1.020/1.013×) |
| A13 | pre-packed int32 ue8m0 scales → `fp8_gemm_nt` (Round 1) | — | ❌ build | — | DeepGEMM layout assertion; no pre-packed path — rejected |
| A14 | task8 fused Triton blockwise FP8 GEMM, in-kernel ue8m0 rounding, autotuned (R1 built, R2 corrected) | 0.249/0.210/0.203× | ✅ (2.86e-9) | 1 | correct but ~4–5× slower (12.6–13.7% MFU) — custom-kernel path closed |

## Reproduce
```bash
cd $KDA_HARNESS_ROOT/testbench/tasks/glm52/o_proj_prefill
CUDA_VISIBLE_DEVICES=0 REMOTE_GPU_ID=0 ./run.sh        # exit 0, 3 wins, 0 regressions
```
Artifacts: `docs/artifacts/{baseline,gate,ncu,nsys,codex}/`.
