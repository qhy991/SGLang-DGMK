# Decision ledger

The benchmark target is first-token latency for a saturated 100K
cached-prefill workload, not decode. “Leaf” means an isolated operator or
kernel measurement; it must not be interpreted as server TTFT.

| Candidate | Evidence | Result | Decision |
|---|---|---|---|
| N6: balanced placement + physical-ID router + DeepEP 120 SM | Five fresh-server pairs plus independent holdout; exact tokens | P50 -5.31%, P90 -4.75%, throughput +7.53%; holdout -5.46%/-6.73%/+7.40% | **Accepted for the frozen cell** |
| Temporal placement v5 | Correctness, external replay, degraded-host A-B-A, matched Nsys | 25/75 active MoE layers changed; relative P50 -68.89%, but absolute P50 3119.99 ms and throughput 320683.49 token/s | Research; blocked by unhealthy host anchor |
| FlashMLA 8-shape band | Bit-exact operator output/LSE; two leaf orders | 8.82%–12.49% faster leaf medians | Operator-admitted development candidate; no formal E2E promotion |
| Exact-M10048 FlashMLA N23 | Bit-exact leaf and model correctness | Leaf -8.85%; E2E P50 -0.949%, P90 -10.50%, throughput +3.84% | Rejected at development P50 gate (<1%) |
| DeepEP send16 | Real-shape 8-rank leaf probe and server screen | Slowest-rank dispatch median -2.45%, wall median -1.53%; unstable server gate | Rejected |
| Equal-chunk DP all-gather | CPU admission tests; runtime path audit | Correct fallback, but material server path was effectively a no-op | Rejected |
| Native SBO | Exact model output and E2E | P50 +1.76%, P90 +1.82%, throughput -0.08% | Rejected |
| Native overlap scheduler | Exact model output and E2E | P50 +40.80%, throughput -18.12% | Rejected |
| Synchronous PrefillDelayer | Distributed state test, exact model output, E2E | P90 -6.72%, but P50 +6.25% and above 2 s | Rejected |
| cpuset-safe affinity | Unit tests and process-tree audit on degraded host | Correct isolation primitive; absolute P50/throughput gates failed | Research; not a performance promotion |
| Contiguous SwiGLU + FP8 fusion (N40) | Correct leaf path and E2E | Leaf -59.1%, but E2E P50 +1.56%, P90 +18.61%, throughput -6.01% | Rejected; canonical leaf-win/E2E-loss example |

The accepted N6 percentage is a three-part combination result. The individual
effects interact and must not be separated or added arithmetically.
