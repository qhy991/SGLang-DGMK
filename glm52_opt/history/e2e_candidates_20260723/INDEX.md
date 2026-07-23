# E2E candidate archive (2026-07-23)

Archived leaf/component winners from GLM-5.2 production goals for
**explicit end-to-end testing**. Default `serving_safe` stays empty.

## Enable for e2e

```bash
# Prefer side-channel so DP workers inherit the same knobs:
export SGLANG_GLM52_ENV_FILE=/path/to/sglang/glm52_opt/runtime.env.e2e_candidates
# or copy that file to SGLANG_GLM52_ENV_FILE / glm52_opt/runtime.env
```

Then restart serve. Worker log should show
`profile=e2e_candidates` and the selected ops.

## Included (wired behind `e2e_candidates`)

| Goal | Op | Phase | Leaf/component | Serving note |
|------|-----|-------|----------------|--------------|
| 10 | `o_proj` | decode M16/32 | graph ~1.08–1.52× fixed-N/K | Eager dispatch often ~1.0× or worse; test CUDA-graph decode |
| 09 | `moe_gate_proj` (fused W13) | prefill contig | PSUM ~1.05× | Needs `expert_start_loc` PSUM layout from `ep_scatter` |
| 08 | `moe_down_proj` (W2) | prefill contig | PSUM ~1.06× | Same PSUM path as 09 |

## Archived but NOT enabled (too risky / regressing)

| Goal | Why skipped for e2e enable |
|------|----------------------------|
| 07 BM16 W2 decode | Real ~1.06–1.09× leaf, but alignment is process-global |
| 13 o_proj prefill | Region ~0.97× regression |
| 14 q_b packed | True kernel slower; event “speedup” was submission artifact |
| 15 indexer wq_b | Fast path numerically wrong; SM budget regresses |
| 16 indexer score | Correct but fails repeated 1.03× gate |

## Promotion rule

Do not move any op into default `serving_safe` until three paired
e2e/graph series clear ~1.03× and TP8/EP8 acceptance exists.
