# Prior Knowledge — index_k prefill 70% HBM

## Live baseline (2026-07-18)

Measured on idle B200 GPU 2 with Kernel-Harness commit `7d79e5e`, DeepGEMM
0.1.4, CUDA 13.0. The shared repository was already dirty in two unrelated
decode candidates; `index_k_prefill/candidate.py` was clean.

| label | candidate us | reference us | HBM utilization | bound |
|---|---:|---:|---:|---|
| M=1024 | 99.86 | 99.90 | 54.17% | memory |
| M=2048 | 100.35 | 99.95 | 53.91% | memory |
| M=4096 | 100.06 | 100.13 | 54.07% | memory |

Correctness passed with `calc_diff=0` on every label. The default candidate is
the reference call, so the M=2048 REGRESS classification is measurement noise,
not an algorithmic regression.

## Shape invariant

`index_k` prefill fixes rows to S=65536 for every workload:

- x: `[65536, 6144]` FP8
- w: `[128, 6144]` FP8
- out: `[65536, 128]` BF16

M is only a workload label. All labels use identical inputs, FLOPs
(103,079,215,104), HBM bytes (432,799,936), and arithmetic intensity (238.17).

## Target

At 8 TB/s, 70% HBM means 5.6 TB/s:

`432,799,936 / 5.6e12 = 77.29 us`.

The campaign needs roughly 1.29x versus the live reference. Relevant mechanisms
are DeepGEMM shape specialization, lossless scale-format handling, TMA/global
load efficiency, N=128 tile geometry, occupancy, and BF16 epilogue stores.
