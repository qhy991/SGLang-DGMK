# Temporal expert-placement research

This directory preserves the offline tools and frozen v5 map produced by the
“larger optimization boundary with local fallback” exploration.

The accepted N6 map is the reference. v5 replaces only 25 of 75 active MoE
layers and locally falls back to N6 for the remaining 50 active layers; the
78-row model map also contains three dense/inactive rows. Seed 1 was used for
selection, seed 0 for holdout, and seed 2 for external exact replay.

The v5 map passed offline constraints and model correctness, and its matched
Nsys comparison showed substantially shorter DeepEP wait tails. Its server
A-B-A was run on a CPU-frequency-degraded host: the relative signal was large,
but absolute P50 and throughput gates failed. Therefore every file here is
**research-only and not promoted**.

Map SHA-256:

```text
570e58026ae890bfd294a34786a095142ca71d14747c87606a536b20c415620a
```

Raw token-route recordings and raw Nsight traces are intentionally not
published. The selection summary keeps their content hashes and logical
artifact IDs so the private evidence can still be audited.
