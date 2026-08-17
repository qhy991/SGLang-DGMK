# GLM-5.2 inference optimization research

This directory collects the reviewed GLM-5.2 runtime path, canonical expert maps, compact evidence, research tooling, and teaching documentation from the 8×B300 optimization campaign.

## Current decision

N6 is the only accepted candidate for the frozen 100K cached-prefill cell. It combines balanced static expert placement, direct physical expert IDs from the fused router, and DeepEP normal dispatch/combine at 120 SM. The archived no-profiler result is P50 TTFT -5.31%, P90 -4.75%, and total-token throughput +7.53%, with a separate correctness probe and independent client-seed holdout.

This statement does not cover decode, arbitrary context lengths, continuous online traffic, other EP sizes, or attention CP8. The archived cell is TP8/DP8/EP8 with `attn_cp_size=1`.

## Beginner learning path

Start with [`b300_100k/docs/LEARNING_PATH_CN.md`](b300_100k/docs/LEARNING_PATH_CN.md). It links a complete Chinese curriculum:

1. CUDA hardware and execution concepts;
2. compute, communication, and control DAGs;
3. attention, KV cache, MoE, DeepEP, TP/DP/EP/CP;
4. leaf versus end-to-end evidence;
5. Nsys versus NCU;
6. the accepted source changes;
7. every N1–N40 candidate and the wider historical experiment campaign;
8. safe reproduction on a new B300 host.

## Sources of truth

| Fact | Canonical source |
|---|---|
| Frozen workload and gates | [`b300_100k/workload_contract.json`](b300_100k/workload_contract.json) |
| Accepted N6 measurements | [`b300_100k/evidence/n6/`](b300_100k/evidence/n6/) |
| v5 research measurements | [`b300_100k/evidence/v5/`](b300_100k/evidence/v5/) |
| Accepted expert map | [`glm52_100k_x11_static_expert_map.json`](glm52_100k_x11_static_expert_map.json) |
| v5 research map and tooling | [`research/temporal_placement/`](research/temporal_placement/) |
| Human-readable decision summary | [`b300_100k/RESULTS.md`](b300_100k/RESULTS.md) |
| Rerun contract | [`b300_100k/REPRODUCTION.md`](b300_100k/REPRODUCTION.md) |
| Full teaching narrative | [`b300_100k/docs/GLM52_B300_SYSTEMATIC_TEACHING_REPORT_20260817_CN.md`](b300_100k/docs/GLM52_B300_SYSTEMATIC_TEACHING_REPORT_20260817_CN.md) |

Markdown explains the evidence; it does not replace raw samples, correctness JSON, maps, or the workload contract.

## Runtime status

The N6 router path is default-off and selected by a narrow admission gate. A benchmark is valid only when the expected selection marker, map SHA, request decomposition, and DeepEP configuration are recorded. A request that completes through an existing fallback path is not an N6 router-fusion measurement.

The source-reviewed integration on current `main` has not yet rerun the formal cell on a new healthy B300 host. Restore the native anchor before interpreting candidate performance.

## Publication boundary

The public repository contains reviewed code, tests, maps, compact evidence, and sanitized teaching/catalog documents. A private retirement archive retains raw Nsight reports, profiler databases, server logs, model/data assets, environment identities, unreviewed worktrees, and the 7,056-file historical compact evidence tree. Catalog entries prefixed with `private-archive:` refer to that recovery source and are not repository links.
