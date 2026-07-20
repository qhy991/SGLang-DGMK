# Prior Knowledge

- Contract: masked grouped FP8 GEMM, E=8, K=6144, N=2048, M={16,32},
  `expected_m=128`; stock f32-scale DeepGEMM oracle.
- Bytes: M16 102,023,168; M32 103,358,464.
- Inclusive 40% limits: 31.882240 us / 32.299520 us.
- User prior ~33%; local M32 27.39%; local M16 timing was unstable.
- Seed: `08a8d67477d6beb4a4d0dd164970df1cc6ad9d73cebf7dff9b3aa5e5ee1fa6a8`.

Primary transferable win is `moe_down_proj_decode_hbm40`: exact fused per-call
UE8M0 packing followed by `disable_ue8m0_cast=True`. Gate N=2048 requires
correct N-row expansion. Validate packed layout against DeepGEMM helpers.

Required lessons: UE8M0 prepack, fused single-launch repack,
do-not-handroll-FP8-GEMM, and single-kernel span floor.

If pack is no longer limiting and NCU identifies grouped GEMM headroom, use an
isolated DeepGEMM-GLM52 fork or task-local source; stock reference stays frozen.
