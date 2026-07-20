# Prior Knowledge — MoE Down Projection Decode 40% HBM

## Live baseline (2026-07-18)

Measured on B200 GPU 0 after confirming 0% utilization and 0 MiB memory use.
GPU 0 is also assigned to an older campaign, so this is not proof that future
runs are uncontended.

| M | candidate us | reference us | HBM | target us |
|---:|---:|---:|---:|---:|
| 16 | 46.98 | 46.94 | 27.28% | 32.041 |
| 32 | 47.11 | 47.10 | 27.69% | 32.617 |

Both shapes passed with `calc_diff=0`. Arithmetic intensities 31.4 and 61.7 are
well below the 562.5 FLOP/byte ridge, so both are memory-bound.

## Workload

E=8, K=2048, N=6144 masked grouped FP8 GEMM. M*8 rows are distributed across
experts, while each expert has an `expected_m=128` slab at M=16. The large
expert-weight footprint dominates the frozen byte model.

## Prior-art direction

KernelWiki `kernel-grouped-gemm` (source-reported) points to persistent/CLC tile
scheduling, expert-tail handling, TMA, and shape-specific grouped-GEMM tiles.
For this very small decode M, launch/scale-transform overhead and padded masked
tiles require direct NCU evidence.

The sibling `o_proj_decode_hbm35` campaign has measured that stateless,
per-call lossless scale packing can expose a faster DeepGEMM path. That result
is not assumed to transfer to masked grouped GEMM; it is only a hypothesis to
measure with the complete packing cost included.
