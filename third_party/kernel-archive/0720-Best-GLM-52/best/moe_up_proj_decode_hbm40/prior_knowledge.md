# Prior Knowledge

- Independent up projection despite production w13 fusion.
- Masked grouped FP8: E=8, K=6144, N=2048, expected_m=128.
- Bytes M16/M32: 102,023,168 / 103,358,464.
- Inclusive 40% limits: 31.882240 us / 32.299520 us.
- Stable local baseline: 47.217 / 47.360 us, 27.01% / 27.28%.
- Seed: `2259efd15195f82c9a75e2e66a94c781691342b07a1c4c13c6fc26435ae41822`.

Primary path is the moe_down winner: fused exact per-call UE8M0 pack with
N-row expansion, then `disable_ue8m0_cast=True`. Required lessons: UE8M0
prepack, fused single launch, do-not-handroll FP8 GEMM, single-kernel floor.

If NCU finds main-kernel headroom after pack, use task-local source or an
isolated DeepGEMM-GLM52 grouped/masked fork; never replace the stock oracle.
