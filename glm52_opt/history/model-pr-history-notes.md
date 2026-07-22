# GLM-5.2 serving-native kernel harness: PR-history note

Query date: 2026-07-22

History source queried:

- `/home/qinhaiyan/AI-Infra-Auto-Driven-SKILLS/model-pr-optimization-history/sglang/glm5-glm51/README.en.md`

Relevant prior work:

- SGLang PR #22850 reduced indexer kernel count by fusing `weights_proj` and K-cache-store work. The current DSA indexer has continued in this direction: CUDA uses `wk_weights_proj` plus fused Q/K preparation by default. Consequently, isolated `index_k_proj` and `index_weights_proj` harness tasks are not the default serving call sites.
- SGLang PR #27053 added GLM-5 FP8 TP8 piecewise-CUDA-graph coverage. A replacement must therefore be evaluated under the real TP/DP split and capture/replay path, not only as a standalone eager kernel.
- SGLang PRs #28437, #28448, and #28460 established the GLM-5.2 deployment recipes. The relevant B200 lanes are TP8 low-latency and TP8/DP8/DeepEP balanced or high-throughput; these lanes select different MoE backends and produce different per-rank token shapes.

Implications for this task:

1. Derive operator reachability, backend, shapes, scale representation, and frequency from a traced serving run for each deployment lane.
2. Use the exact production callable and packed-scale ABI as the harness reference.
3. Treat fused indexer, FlashInfer TRT-LLM DSA/MoE, DeepGEMM masked W13/W2, and CUDA-graph behavior as distinct integration contracts.
4. Require exact-callsite and end-to-end validation before promoting a harness result.
