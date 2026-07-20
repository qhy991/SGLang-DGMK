# Run Log — GLM-5.2 o_proj prefill FP8 GEMM (Kernel-Harness bridge)

All authoritative measurements: `cd $KDA_HARNESS_ROOT/testbench/tasks/glm52/o_proj_prefill && CUDA_VISIBLE_DEVICES=0 REMOTE_GPU_ID=0 ./run.sh` (CUPTI cold-L2, warmup=3 repeat=10 iterations=30). Pinned idle B200 GPU 0.

## Reproducibility packet (2026-07-18)

| Field | Value |
|---|---|
| host | dry-vm-embraces-fin-03 |
| GPU | NVIDIA B200 (GPU 0), cc 10.0, 191.5 GB, 148 SMs |
| GPU idle at capture | 0% util, 0 MiB used, persistence on, sm clock 120→1965 MHz max |
| driver | 610.43.02 |
| CUDA (torch) | 13.0 · nvcc 13.2 |
| torch | 2.11.0+cu130 |
| deep_gemm | 0.1.4 |
| sgl_kernel | 0.4.4 |
| python | 3.12.13 |
| KDA_HARNESS_ROOT | /home/qinhaiyan/Kernel-Harness |
| harness git_sha | 7d79e5e (branch main, dirty=True) |
| implied FP8 dense peak | 4500 TFLOPS (from achieved_tflops / compute_util) |

## Baseline — reference candidate (deep_gemm.fp8_gemm_nt, all defaults), median-of-3

| M | run1 us | run2 us | run3 us | median us | util%(median) | bound | correct | exit |
|---:|---:|---:|---:|---:|---:|---|:--:|:--:|
| 1024 | 90.560 | 90.592 | 90.656 | **90.592** | 50.57 | compute | yes | 1 |
| 2048 | 141.909 | 142.282 | 142.161 | **142.161** | 64.45 | compute | yes | 1 |
| 4096 | 271.014 | 272.538 | 272.424 | **272.424** | 67.27 | compute | yes | 1 |

Default candidate == reference ⇒ 0 wins / 0 regressions / 3 neutral, exit 1 (correct, not faster). Correctness (calc_diff=0, post-timing recheck) passes on all shapes.

### Key insight: latency target ⟺ utilization target (same constraint)
FP8 util = FLOPs / (time × peak), with FLOPs and peak (4500 TFLOPS) fixed. The latency targets map exactly onto the util targets: 65.4µs→70.05%, 122.2µs→74.98%, 229.1µs→79.99%. So a candidate that hits a shape's latency target automatically hits its util target and vice versa — DEC-2 ("both required") is satisfied by construction. Gate on latency.

### Required same-run speedup vs live baseline (median-of-3 denominator)
| M | median baseline us | target us | speedup needed | user-stated approx |
|---:|---:|---:|---:|---:|
| 1024 | 90.592 | 65.4 | **1.385×** | ~1.38× |
| 2048 | 142.161 | 122.2 | **1.163×** | ~1.17× |
| 4096 | 272.424 | 229.1 | **1.189×** | ~1.20× |

## DeepGEMM 0.1.4 knob surface (introspected)
`fp8_gemm_nt(a, b, d, c=None, recipe=None, recipe_a=None, recipe_b=None, compiled_dims='', disable_ue8m0_cast=False)`; globals `set_num_sms` (default 148), `set_pdl`, `set_tc_util`.

## Knob sweep (task4, authoritative ./run.sh, single-run probes)
compiled_dims screen (speedup M1024/M2048/M4096): ""=0.999/1.001/1.002 (exit1); "n"=neutral; "k"=1.012/1.006/1.005 (exit0); "nk"=1.011/1.006/1.005 (exit0); "mn"=neutral; **"mnk"=1.016/1.008/1.006 (exit0, best)**. Second batch: **mnk+PDL=1.031/1.019/1.011 (best overall)**; mnk+num_sms132=0.96/0.94/0.95 (worse); num_sms128=0.95/0.92/0.97; num_sms96=0.82/0.79/0.80; **disable_ue8m0_cast=True → exit 2 (correctness fail)**. Full ledger in `results.md`.

## Profiling (task6)
- nsys per-call breakdown: `fp8_gemm_nt` = fixed ~20µs f32→ue8m0 scale-transform pipeline + GEMM; transform ~M-independent (≈22% of M=1024 runtime, ≈7% of M=4096). Artifacts: `docs/artifacts/nsys/`.
- NCU application-replay, isolated `sm100_fp8_fp4_gemm_1d1d_impl`: M1024 67.6µs/81.7% tensor-active/67.8% util; M2048 121.8µs/91.7%/75.2%; M4096 250.7µs/95.6%/73.1%. Named bound + per-shape analysis in `results.md`.
- NCU kernel-replay could not intercept the JIT GEMM kernel by name; application-replay (`--replay-mode application`) succeeded.

## Final gate (task5/task9) + verdict (task7/task10)
candidate.py = `compiled_dims="mnk"` + `set_pdl(True)`, stateless. Median-of-3 authoritative gate: exit 0 ×3, all correct pre+post-timing, 3 wins/0 regressions → M1024 87.75µs(1.032×), M2048 139.42µs(1.020×), M4096 268.83µs(1.013×). All hard targets missed. Codex escalation (task7) = NO_GO. **Terminal: PARTIAL WIN delivered · EVIDENCE-BACKED NO-GO on the all-shape hard target.** See `results.md`.

## Timeline
- 04:01Z — reproducibility packet captured; GPU 0 idle confirmed.
- 04:02–04:07Z — baseline ×3 (task1), default-candidate correctness confirmed (task2).
- 04:07Z — introspected deep_gemm API; Codex knob-matrix analysis (task3).
- 04:12–04:20Z — knob sweep (task4): compiled_dims, disable_ue8m0_cast, PDL, num_sms.
- 04:23–04:35Z — nsys + NCU profiling (task6); named the bound.
- 04:45Z — median-of-3 gate of final candidate (task5/task9): exit 0, PARTIAL WIN.
- 04:33Z — Codex escalation decision (task7): NO_GO. task8 (custom kernel) deferred.

## Round 1 (review follow-up)
Addressed the Round 0 review's two blocking gaps:
- **task6 evidence re-captured & committed**: NCU application-replay for all 3 shapes saved to `docs/artifacts/ncu/gemm_sol_m{1024,2048,4096}.csv` (isolated GEMM 67.5/121.9/251.1µs; tensor-pipe-active 81.4/92.5/95.7%; SM 76/87/92%; occ ~12.5%). nsys M=2048 split added (`docs/artifacts/nsys/kern_sum_m2048.csv`). The Round 0 committed CSV that said "No kernels were profiled" (NCU kernel-replay could not intercept the JIT GEMM) was replaced.
- **Timing methodology reconciled**: harness `timing.py` measures `max(kernel.end)−min(kernel.start)` per `run()` call (full per-call span incl. transform), cold-L2, median. NCU isolated GEMM (single launch) and nsys in-loop median measure different things; the ~250 vs ~275 vs ~272µs numbers at M=4096 are explained in `results.md`. Transform+gaps ≈ harness_total − isolated_GEMM ≈ 23/20/21µs.
- **task8 executed (not deferred)**: fused Triton blockwise FP8 GEMM with **in-kernel** ue8m0 rounding on raw f32 scales (Round 2 corrected an earlier variant that pre-rounded scales outside the kernel) — correct (calc_diff 2.86e-9) but ~4–5× slower (0.249/0.210/0.203×, 12.6–13.7% MFU). `candidates/task8_triton_spike/candidate.py`, `docs/artifacts/task8/triton_fused_inkernel_probe.log`. Pre-packed-scale path rejected (DeepGEMM layout assertion).
- **Final gate re-confirmed** (`docs/artifacts/gate/final_cand{1,2,3}.log`): exit 0 ×3, all correct, 3 wins/0 regressions, 1.033/1.019/1.013×.
- Verdict unchanged and now fully artifact-backed: PARTIAL WIN delivered · EVIDENCE-BACKED NO-GO on the all-shape hard target.
