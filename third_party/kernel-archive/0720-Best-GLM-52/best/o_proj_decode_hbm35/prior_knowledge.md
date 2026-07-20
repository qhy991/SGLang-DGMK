# Prior Knowledge — o_proj decode 35% HBM campaign

## Current modified candidate baseline

Measured 2026-07-18 on idle B200 GPU 1 with Kernel-Harness commit `7d79e5e`;
the repository was dirty before this campaign.

| M | candidate us | reference us | speedup | HBM utilization | correctness |
|---:|---:|---:|---:|---:|---|
| 16 | 45.47 | 51.80 | 1.139x | 27.81% | PASS, calc_diff 4.30e-9 |
| 32 | 52.82 | 53.00 | 1.003x | 24.05% | PASS, calc_diff 0 |

The existing candidate uses a Triton split-K implementation for M=16 and
reference fallback for M=32. It passes the Harness gate (one WIN, one neutral)
but misses the user target on both shapes.

## Target conversion

Using Kernel-Harness's frozen `bytes_hbm` and B200 8 TB/s peak:

- M=16: 101,154,816 bytes / (0.35 * 8 TB/s) = 36.13 us.
- M=32: 101,621,760 bytes / (0.35 * 8 TB/s) = 36.29 us.

The required reductions from the current candidate are about 20.5% (M=16) and
31.3% (M=32). Reference fallback cannot meet the target.

## Related previous result

The earlier production-layout decode experiment reached about 41% HBM through
DeepGEMM `compiled_dims="nk"`, but that used packed UE8M0 scales and a different
Harness contract. Treat it as a mechanism hint, not directly comparable proof.
