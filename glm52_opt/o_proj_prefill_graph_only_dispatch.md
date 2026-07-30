# Prefill o_proj: graph-only fixed-N/K dispatch (measured no-replacement)

## What this is

`_E2E_PREFILL["o_proj"]` registers the same fixed-N/K candidate as decode
(DeepGEMM `fp8_gemm_nt`, `compiled_dims="nk"`, N=6144, K=16384) for the locked
promotional prefill buckets M ∈ {2048, 4096}, `graph_only=True`. It reuses the
generic `_graph_only_declines(spec)` guard in `try_dispatch_fp8_gemm` and shares
the `SGLANG_GLM52_O_PROJ_GRAPH_ONLY` toggle with decode (governs both phases;
`=0` forces eager for a diagnostic leaf). `_E2E_DECODE` is unchanged; the two
tables are disjoint, so decode is not affected. Default off (only under
`SGLANG_GLM52_OPT_PROFILE=e2e_candidates`; serving_safe selects stock).

Production prefill is CUDA-graph bound by default on CUDA
(`default_prefill_backend() == Backend.BREAKABLE`), so — like decode — selection
happens at capture and replay runs no Python; the graph leaf is the relevant
production lane.

## Why it is no-replacement (not promoted)

Unlike memory-latency-bound decode (nk wins ~1.39× at M16), prefill at the real
chunked-prefill buckets is compute/tensor-pipe bound. o_proj GEMM arithmetic
intensity ≈ M·2.0 FLOP/B; the DeepGEMM FP8 compute/HBM crossover ≈ 562 FLOP/B ⇒
crossover at M ≈ 281, far below the promotional range. `compiled_dims="nk"` cuts
~12% of executed instructions, which raises throughput only on the
latency-bound kernel — it cannot move the saturated tensor pipe (goal-13 NCU:
91% tensor-pipe, nk 256.86→256.13 µs). So the specialization is device-neutral
at prefill M.

## Verified behaviour (one B200, single GPU uuid, paired 3 series)

`Kernel-Harness serving_native/o_proj_prefill_graph_only_gpu_contract.py`
(`GPU-30b619de…`, packed-UE8M0, JIT precompile disabled):

- Integrity (all pass): eager declines before launch / capture selects; captured
  leaf is one clean GEMM node whose identity differs from stock; capture
  identical on/off; candidate == stock **bit-exact** (eager+graph+region, err 0).
- Win gate (not met → no-replacement): graph leaf/region 3-series all ~1.00× and
  < 1.03 at both Ms — M2048 leaf 1.0000/1.0008/1.0059, region 1.0012/1.0031/
  1.0029; M4096 leaf 1.0019/0.9996/0.9996, region 1.0010/1.0019/1.0002.
- Crossover diagnostic (device leaf): 1.064× @M256, 1.009× @M512, 1.021× @M1024,
  1.00× @M2048/M4096 — nk's edge lives only near/below the crossover, below the
  representative prefill range.
- Decode non-regression (same lease): decode M16 candidate still selects, stays
  bit-exact, still wins (leaf median 1.3869×).

Disposition: **no-replacement**, default off. Retained as an explicit
e2e_candidates ablation; never promoted absent a ≥1.03 graph leaf+region win.
Full evidence: `glm52-hotspot-goal-runs/tasks/fp8_o_proj_prefill_ptx_r1/evidence/`.
