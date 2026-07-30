# Decode q_b_proj: graph-only fixed-N/K dispatch (no-replacement)

## What this is

`KernelSpec.graph_only` also restricts the decode `q_b_proj` fixed-N/K candidate
(`_E2E_DECODE["q_b_proj"]`, `compiled_dims="nk"`, N=16384, K=2048) to CUDA-graph
capture. Outside capture, `try_dispatch_fp8_gemm` calls the generic
`_graph_only_declines(spec)` right after the spec lookup — before the ABI check
and the hit/miss lock — so eager decode returns stock
(`None` → `w8a8_block_fp8_matmul_deepgemm`) with no candidate launch. Graph
capture selects the candidate.

`q_b_proj` is **explicit-only** (`_E2E_EXPLICIT_OPS`): it is never in the default
`e2e_candidates` set, so o_proj's default behavior is unchanged. Select it with
`SGLANG_GLM52_OPT_OPS=q_b_proj`. `SGLANG_GLM52_Q_B_PROJ_GRAPH_ONLY` defaults on
(`=0` forces a diagnostic eager leaf). The `fixed_nk` path in
`fp8_gemm.run_fp8_gemm` is checked **before** the historical q_b DeepGEMM fork,
so the candidate is the clean `compiled_dims="nk"` specialization — **not** the
goal-14 packed/source fork (negative evidence, not re-enabled).

## Why this shape was tried

Copy of the o_proj graph-only mechanism onto the sibling decode q_b GEMM. The
candidate (DeepGEMM `fp8_gemm_nt` with N=16384, K=2048 baked) is bit-identical to
stock and genuinely faster on the memory-latency-bound decode shape; graph-only
keeps that device win off the eager path (whose glm52_opt dispatch tax lost every
eager paired session — the goal-14 negative route).

## Verified behaviour (two B200s, single GPU per dataset)

`Kernel-Harness serving_native/q_b_graph_only_gpu_contract.py`, run 1 (3×40, GPU
`…f7ae`) and run 2 (5×100, GPU `…a595`):

| Property | Result |
| --- | --- |
| eager, graph-only on | declines; returns None; no fixed-N/K hit; == stock (both M, both GPUs) |
| eager, `=0` | selects the candidate; exactly one hit |
| capture, either setting | one clean GEMM node; identity **differs** from stock (N/K baked); identical on vs off |
| correctness (eager + graph + region) | max abs err 0.0 (M16, M32, both GPUs) |
| CUDA-graph **leaf** vs stock | M16 1.13–1.20× (pass); **M32 1.01–1.16×** (run 2 series dips to 1.0138 < 1.03) |
| CUDA-graph **containing region** (quant+GEMM) | M16 1.11–1.16× (pass); **M32 1.02–1.12×** (run 1 series dips to 1.0175 < 1.03) |
| eager identity lane (informational) | ~0.92× — expected eager dispatch tax, avoided in graph replay |

## Status of the candidate

**No-replacement.** M16 is a strong, robust graph win (leaf 1.13–1.20×, region
1.11–1.16×) on both B200s. But at the required **M32** bucket the win is only
~1.06× central and does **not** robustly clear the ≥1.03 gate on all estimators —
and *which* required lane dips below 1.03 flips by GPU (region on `…f7ae`, leaf on
`…a595`; the two B200s run ~7% apart). Root cause: the M32 win is simply small —
candidate advantage 0.67–1.63 µs on a ~11 µs kernel — so the noisy paired
`ab_median` estimator dips below the 3% floor under two runs / five series. This
is **not** because q_b's read (33.5 MB) is smaller than o_proj's 100.7 MB
(`index_q_upproj`, 8.4 MB, cleared); the win magnitude tracks the
`compiled_dims="nk"` instruction reduction, set by k-loop length (q_b's K=2048 is
short vs o_proj's K=16384, 1.39–1.44× at M16) and diluted at M32 by N=16384. o_proj's
own M32 was also marginal (leaf 1.064–1.077, region 1.086–1.122) but got a clean
single 3-series pass; q_b did not.

The candidate stays **registered, default-off, explicit-only** as a documented
diagnostic (mirrors `index_q_upproj`). No SGLang serving default changed;
`serving_safe` stays empty and stock remains active. No TP8 claim (q_b_proj is
rank-local under DP attention; TP8/DP8/EP8 acceptance is unreachable on this
4-GPU, weightless host). Full evidence:
`glm52-hotspot-goal-runs/tasks/fp8_q_b_decode_ptx_r1/evidence/03_FINAL_REPORT.md`.
