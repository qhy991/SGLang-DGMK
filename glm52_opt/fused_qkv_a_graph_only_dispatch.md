# Decode fused_qkv_a_proj: graph-only fixed-N/K dispatch

## What this is

`KernelSpec.graph_only` now also restricts the decode `fused_qkv_a_proj`
fixed-N/K candidate (`_E2E_DECODE["fused_qkv_a_proj"]`, `compiled_dims="nk"`,
N=2624, K=6144) to CUDA-graph capture. Outside capture,
`try_dispatch_fp8_gemm` calls the generic `_graph_only_declines(spec)` right
after the spec lookup — before the ABI check and the hit/miss lock — so eager
decode returns the stock path (`None` → `w8a8_block_fp8_matmul_deepgemm`) with no
candidate launch and no glm52_opt dispatch tax. Graph capture selects the
candidate. No `dispatch.py` change was needed (the guard was added by the o_proj
round). It is an **explicit-only** e2e op: it never joins `_E2E_DEFAULT_OPS` and
arms only under `SGLANG_GLM52_OPT_OPS=fused_qkv_a_proj`.

`SGLANG_GLM52_FUSED_QKV_A_GRAPH_ONLY` defaults on. Set it to `0` to force eager
selection for a diagnostic leaf measurement. Independent of the o_proj/W2 envs.

## Why

The decode candidate device kernel (DeepGEMM `fp8_gemm_nt` with N=2624, K=6144
baked as compile constants) is genuinely faster than stock on the
memory-latency-bound decode shape, but the eager glm52_opt dispatch path costs
real host time (the same tax that vetoes o_proj's eager arm). Production decode
is CUDA-graph-bound and graph replay executes no Python, so restricting selection
to capture keeps the device win and leaves eager on stock. Mirrors decode
`o_proj` and `moe_down_proj` (W2). The FP8 path is what production reaches: the
M≤16 `dsv3_fused_a_gemm` fast path in `prepare_qkv_latent` is inert for an FP8
weight (`use_min_latency_fused_a_gemm` needs a BF16 weight).

## Verified behaviour (one B200, single GPU, paired 3 series)

`Kernel-Harness serving_native/fused_qkv_a_graph_only_gpu_contract.py`:

| Property | Result |
| --- | --- |
| eager, graph-only on | declines; returns None; no fixed-N/K hit; == stock |
| eager, `=0` | selects the candidate; exactly one hit |
| capture, either setting | one clean GEMM node; identity **differs** from stock (`…Lj2624ELj6144E…` vs `…Lj0ELj0E…`); identical on vs off |
| correctness (eager + graph + region) | max abs err 0.0 (M16, M32) |
| CUDA-graph leaf vs stock | M16 1.28–1.32×, M32 1.32–1.35× (≥1.03 all series/estimators) |
| CUDA-graph containing region (prepare_qkv_latent quant+GEMM) | M16 1.27–1.32×, M32 1.32–1.39× (≥1.03 all series/estimators) |
| eager identity lane (informational) | ~0.92× — the expected eager dispatch tax, avoided in graph replay |

## Status of the candidate

Kept as an **external-acceptance-candidate**, default off. The containing-region
gate clears with wide headroom (≥1.27 at both M16 and M32) — a *stronger* region
result than o_proj (whose M32 region was 1.106) because this is an even smaller
latency-bound kernel (16.1 MB weight, ~2 µs floor), so the `compiled_dims="nk"`
instruction-issue reduction is a larger fraction of the total and the cheap FP8
quant does not dilute it. Promotion still requires checkpoint-backed TP8/DP8/EP8
acceptance, unreachable on this 4-GPU weightless host (this op is rank-local
under DP). The `e2e_candidates` profile stays default-off and stock remains
active. Full evidence:
`glm52-hotspot-goal-runs/tasks/fp8_fused_qkv_a_decode_ptx_r1/evidence/03_FINAL_REPORT.md`.
