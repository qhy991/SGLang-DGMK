# GLM-5.2 serving-native kernel integration: PR-history note

Latest query date: 2026-07-29

History source read in full:

- `/home/qinhaiyan/AI-Infra-Auto-Driven-SKILLS/model-pr-optimization-history/sglang/glm5-glm51/README.en.md`

Relevant prior work:

- SGLang PR #18521 introduced GLM DSA model support; PR #20062 made the
  DSA/MLA backend choice explicit and tested. The FlashMLA experiment must
  therefore attach to the selected production backend rather than assume that
  every GLM request reaches one attention implementation.
- SGLang PR #22850 reduced indexer kernel count by fusing `weights_proj` and K-cache-store work. The current DSA indexer has continued in this direction: CUDA uses `wk_weights_proj` plus fused Q/K preparation by default. Consequently, isolated `index_k_proj` and `index_weights_proj` harness tasks are not the default serving call sites.
- SGLang PR #27053 added GLM-5 FP8 TP8 piecewise-CUDA-graph coverage. A replacement must therefore be initialized before capture and evaluated under the real TP/DP split and replay path, not only as a standalone eager kernel.
- SGLang PR #28607 added GLM DSA end-to-end coverage. This reinforces the
  requirement to preserve the exact paged-KV metadata, sparse-index, and
  attention-backend contract.
- SGLang PRs #28437, #28448, and #28460 established the GLM-5.2 deployment recipes and B300 validation. The relevant lanes select different MoE/DSA backends and produce different per-rank token shapes; #28448 also reports that FP8 KV was not universally faster on H200, so dtype/backend choices cannot be generalized across devices.

Implications for this task:

1. Derive operator reachability, backend, shapes, scale representation, and frequency from a traced serving run for each deployment lane.
2. Use the exact production callable and packed-scale ABI as the harness reference.
3. Treat fused indexer, FlashInfer TRT-LLM DSA/MoE, DeepGEMM masked W13/W2, and CUDA-graph behavior as distinct integration contracts.
4. Load an external kernel provider only after GPU assignment and before
   warmup/graph capture; reject an invalid provider at startup.
5. Require exact-callsite, containing-region, CUDA Graph, and end-to-end
   validation before promoting a harness result.
