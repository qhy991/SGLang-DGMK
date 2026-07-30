# Decode o_proj: graph-only fixed-N/K dispatch

## What this is

`KernelSpec.graph_only` now also restricts the decode `o_proj` fixed-N/K
candidate (`_E2E_DECODE["o_proj"]`, `compiled_dims="nk"`) to CUDA-graph capture.
Outside capture, `try_dispatch_fp8_gemm` calls the generic
`_graph_only_declines(spec)` right after the spec lookup — before the ABI check
and before the hit/miss lock — so eager decode returns the stock path
(`None` → `w8a8_block_fp8_matmul_deepgemm`) with no candidate launch and no
glm52_opt dispatch tax. Graph capture selects the candidate.

`SGLANG_GLM52_O_PROJ_GRAPH_ONLY` defaults on. Set it to `0` to force eager
selection for a diagnostic leaf measurement.

## Why

The decode o_proj candidate device kernel (DeepGEMM `fp8_gemm_nt` with N=6144,
K=16384 baked as compile constants) is genuinely faster than stock on the
memory-latency-bound decode shape — but the eager glm52_opt dispatch path
(is_enabled + op_name + lookup + alloc + hit-accounting) costs real host time and
lost every eager paired session in goal-10 (M16 0.89–1.01×, M32 0.92–0.98×).
Production decode is CUDA-graph-bound and graph replay executes no Python, so
restricting selection to capture keeps the device win and leaves eager on stock.
Mirrors `moe_down_proj` (W2) and FlashMLA `dsa_decode_attn`.

## Verified behaviour (one B200, single GPU, paired 3 series)

`Kernel-Harness serving_native/o_proj_graph_only_gpu_contract.py`:

| Property | Result |
| --- | --- |
| eager, graph-only on | declines; returns None; no fixed-N/K hit; == stock |
| eager, `=0` | selects the candidate; exactly one hit |
| capture, either setting | one clean GEMM node; identity **differs** from stock (N/K baked); identical on vs off |
| correctness (eager + graph + region) | max abs err 0.0 (M16, M32) |
| CUDA-graph leaf vs stock | M16 1.39–1.44×, M32 1.06–1.08× (≥1.03 all series) |
| CUDA-graph containing region (quant+GEMM) | M16 1.32–1.37×, M32 1.09–1.12× (≥1.03 all series) |
| eager identity lane (informational) | ~0.92× — the expected eager dispatch tax, avoided in graph replay |

## Round 2: is the *schedule* that candidate compiles to the right one? (no)

Round 2 asked whether the round-1 candidate's DeepGEMM layout leaves throughput
on the table, since the captured kernel is `BLOCK_M=16 BLOCK_N=128 stages=12
cluster=2` — only `ceil(6144/128) = 48` tiles on a 148-SM B200 at M=16, and at
M=32 a second `BLOCK_M` row that re-requests the whole weight matrix. The
2-CTA cluster shares no weight bytes at all (`sm100_fp8_fp4_gemm_1d1d.cuh`
hard-codes `num_tma_multicast = 1`; it only enables the 2-SM UMMA).

`SGLANG_GLM52_O_PROJ_DECODE_SCHED` selects the schedule so the arms can be A/B'd
in one process: `nk148` (round-1 candidate, the **default**), `nk94` (the best
the stock package can reach), `bn64` (round-2 identity, needs the task's
DeepGEMM overlay). Unknown names raise instead of silently timing stock.

**Outcome: `no-replacement`.** The `bn64` identity forces
`BLOCK_M=32 BLOCK_N=64 stages=16 cluster=1` — 96 tiles and one weight sweep at
both M, bit-identical output — and still measures **0.9822 graph leaf / 0.9970
graph region at M=16** against the round-1 candidate (M=32 clears at
1.057/1.067, but the gate needs both buckets). Nsight Compute shows why: all
four arms read 101.7–102.0 MB from DRAM (1.011× the irreducible weight matrix)
and all sit at **48–56% of peak DRAM throughput**, regardless of tiles, BLOCK_N,
stages or cluster. M=32's "second sweep" never reaches DRAM — it is L2 traffic
(12.13 M vs 7.15 M L2 lookups for identical DRAM bytes), which is why removing
it is worth ~6% and not ~2×. The residual is DRAM efficiency on a `BLOCK_K=128`
strided read, and `BLOCK_K` is the UE8M0 scale granularity, so changing it
forfeits bit-exactness.

Full evidence:
`glm52-hotspot-goal-runs/tasks/o_proj_decode_r2/evidence/02_FINAL_REPORT.md`.

## Status of the candidate

Kept as an **external-acceptance-candidate**, default off. Unlike MoE W2 (which
failed the graph containing-region gate at 1.026, DRAM-bound at 99% floor), the
o_proj linear-apply region clears ≥1.03 on both M because o_proj is
memory-latency-bound and the GEMM dominates the region. Promotion still requires
checkpoint-backed TP8/DP8/EP8 acceptance, which is unreachable on this 4-GPU,
weightless host (o_proj is rank-local under DP). The `e2e_candidates` profile
stays default-off and stock remains active. Full evidence:
`glm52-hotspot-goal-runs/tasks/fp8_o_proj_decode_ptx_r1/evidence/03_FINAL_REPORT.md`.
