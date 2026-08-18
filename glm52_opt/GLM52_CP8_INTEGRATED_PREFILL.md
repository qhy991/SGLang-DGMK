# GLM-5.2 CP8/EP8 integrated prefill candidate

## Frozen objective

Improve no-profiler request TTFT P50 and P90 for the exact GLM-5.2-FP8 cell:

- 8×B300 SM103;
- TP8, attention CP8, DP1, MoE EP8;
- zigzag/in-sequence prefill CP, eager prefill, overlap scheduling disabled;
- one logical request shape with a 90,000-token prefix, 89,984 page-aligned
  cached tokens, a 10,000-token suffix, and one output token;
- 110 measured requests at concurrency 11;
- FlashMLA-KV, FP8 KV cache, page size 64, DeepEP normal dispatch/combine with
  120 SMs, the frozen balanced expert map, and router static-placement fusion.

The authoritative performance comparison is adjacent fresh-server control /
integrated pairs. Profiler kernel sums and isolated operator timings are not the
admission metric.

## Integration rule

The branch contains the reviewed tree's existing and historical optimization
implementations plus the KDA-derived CP8 migrations. Presence in the branch is
not the same as being armed in this workload. The single runtime entry is
`host_e2e_prefill_cp8_integrated.env`; server-owned invariants are frozen by
`run_cp8_integrated_fresh_abba_host.sh`.

| Optimization family | Code in branch | Armed in this cell | Evidence decision |
|---|---:|---:|---|
| CP8 zigzag sharding and CP-local FlashMLA metadata repair | yes | yes | existing validated baseline |
| balanced expert placement | yes | yes | keep; identity comparison improved P50/P90 |
| router static-placement fusion | yes | yes | exact output and 8-rank hit proof |
| DeepEP normal dispatch/combine, 120 SMs | yes | yes | keep; seed24 was null |
| reviewed eager prefill indexer path | yes | yes | existing validated baseline |
| direct packed MLA-KV multicast into final zigzag rows | yes | yes | operator, stress, fresh pairs, and matched Nsight passed |
| combined CP8 indexer halves at local q_rows=1,252 | yes | yes | independent five-pair win; composition requires fresh pairs |
| packed NCCL MLA-KV exchange | yes | no | serving STOP |
| reusable prefill Top-K workspace | yes | no | null at the target shape |
| fused final-index Top-K materialization | harness retained | no | twice null/negative |
| clustered/paged MQA experiments | yes | switches preserved | CP8 strict gates keep non-matching branches inactive |
| DeepEP seed24 configuration | harness retained | no | sub-1% null |
| decode fixed-N/K, FlashMLA decode, and MoE alignment winners | yes | no | different decode/CUDA-Graph cell |
| masked MoE SwiGLU+quant variants | yes | no | different decode cell and not part of prefill admission |
| MoK prefill candidates | launcher integration retained | no | independent campaign; not composed post hoc |

## Runtime graph rewrite

The active composition rewrites two different critical edges:

1. `BF16 local KV -> packed uint8[M,656] -> NCCL AllGather -> rank-major
   rearrange -> paged cache store` becomes `BF16 local KV -> packed
   uint8[M,656] -> SM103 multimem direct scatter into final zigzag rows -> paged
   cache store`.
2. `early-zigzag MQA+TopK -> late-zigzag MQA+TopK -> concatenate` becomes one
   MQA invocation with per-row visible endpoints followed by one TopK transform.

They share no claimed additive operator number. Only the measured composition
may be reported.

## Invariants and fail-closed boundary

- Direct transport requires SM103, CP size 8, zigzag CP, local M=10,048,
  global M=80,384, contiguous uint8 rows of 656 bytes, the declared symmetric
  memory protocol, and eager prefill.
- Combined indexer requires NVIDIA e4m3fn CUDA, CP size 8, zigzag CP, H32/D128,
  eager prefill, disabled overlap scheduling, batch size one, and exactly 1,252
  local query rows. Warmup and every non-target shape use the reviewed two-call
  path.
- Unsupported optimized configurations must reject; measurement must not hide
  a fallback.
- The native/control path remains unchanged and selectable through the matched
  control environment.

## Correctness and promotion gates

Every fresh arm must prove 110/110 requests, 110 output tokens, the exact input
and sentinel hashes, 8×110 matching prefill lines, cache=89,984, scheduled
rows=10,048, and the same 110-token trajectory. Each pair also runs the frozen
11-input sequential probe; output token IDs must be exact and selected-token
logprob max/mean absolute error must stay below 1e-3/1e-4.

The five-pair screen requires at least four P50 wins, at least four non-negative
P90 pairs, median P50 improvement of at least 1%, and non-negative median P90.
The integrated arm must show direct-path counts on all eight ranks, combined
indexer hits on all eight ranks, balanced static placement, router fusion, and
DeepEP120. Promotion beyond screening additionally requires a matched Systems
profile and the existing delayed-rank direct-transport stress contract.

## Explicit non-goals

This evidence does not generalize to decode, CUDA Graph, online arrival
processes, another KV/cache shape, another CP/EP topology, or another GPU. Code
retention for those cells does not make their gains part of this prefill result.
