# Production W2 decode enable and fallback policy

Date: 2026-07-22

## Active policy

No goal-07 replacement is enabled. `SGLANG_GLM52_OPT=0` is the reference
policy, and the default serving-safe profile without an explicit operator
allowlist performs no implicit W2 swap. Stock SGLang/DeepGEMM is active for
every W2 bucket.

| Operator | Local M | Scheduling hint | Packed ABI | Required topology | Goal-07 replacement | Active implementation |
|---|---:|---:|---|---|---|---|
| W2 / `moe_down_proj` | 16 | plan 4 | exact int32 UE8M0 | TP8/DP8/EP8 | disabled | stock SGLang/DeepGEMM |
| W2 / `moe_down_proj` | 16 | current source 5 | exact int32 UE8M0 | TP8/DP8/EP8 | disabled | stock SGLang/DeepGEMM |
| W2 / `moe_down_proj` | 32 | plan 8 | exact int32 UE8M0 | TP8/DP8/EP8 | disabled | stock SGLang/DeepGEMM |
| W2 / `moe_down_proj` | 32 | current source 9 | exact int32 UE8M0 | TP8/DP8/EP8 | disabled | stock SGLang/DeepGEMM |
| W2 / `moe_down_proj` | any other or unknown | any | any | any | disabled | stock SGLang/DeepGEMM |

Plan hints 4/8 and current-source-derived hints 5/9 are separate evidence
lanes. No result transfers from one hint, local-M bucket, ABI, graph mode, or
topology to another.

## Why the locally faster BM16 configuration is disabled

Alignment 16 passed the local 3% paired-p50 leaf gate on all four named
workloads, fresh correctness, 30-replay leaf CUDA Graph checks, and edge-mask
active-row correctness. It is still not a deployable production replacement:

1. DeepGEMM's alignment selector is process-global, so enabling BM16 for these
   calls would also affect other grouped GEMMs and masks.
2. BM16 can increase logical tile count and B reloads when experts have more
   than 16 rows; the fixed-mask speedup is not a general mask oracle.
3. Dispatch may not read `masked_m` on the host, synchronize the device, or
   mutate process-global tuning state on the serving hot path.
4. Live EP8 masks and the exact eight-rank recipe/stream/overlap configuration
   have not been observed on this four-GPU host.
5. The required TP8 full-region and SGLang end-to-end gates are blocked here.

No production SGLang dispatch or DeepGEMM source path was modified. The SGLang
commit for this goal adds only CPU contract tests. Alignment 16 exists solely
as isolated measurement, profile, graph, and edge evidence, with alignment 128
restored after each experiment.

## Fail-closed contract for any future promotion

A future static oracle may enable only an individually measured
`operator x local-M x expected_m x ABI x topology x graph-mode` bucket that
satisfies all of the following:

1. at least a 1.03x alternating paired-p50 leaf speedup with fresh,
   independently allocated, NaN-poisoned correctness passing;
2. exact production FP8 and packed int32 UE8M0 tensors, with no timed adapter;
3. no host read of `masked_m`, device synchronization, allocation, copy, pack,
   or process-global tuning mutation on the hot path;
4. preserved stream, output-buffer, recipe, signal, SM-allocation,
   return-value, and CUDA Graph semantics;
5. a non-regressing complete TP8/DP8/EP8 DeepEP dispatch -> W13 -> fused
   SwiGLU+quant -> W2 -> DeepEP combine result; and
6. a non-regressing eight-rank SGLang end-to-end decode result.

Every unsupported ABI, shape, topology, recipe, overlap mode, signal contract,
or graph mode must call stock. Recipe-bearing calls and calls carrying
`overlap_args` stay on stock unless a future candidate explicitly implements
and validates those contracts. The existing CPU regression test verifies that
recipe/overlap paths bypass replacement, preserve stock arguments and SM scope,
and return the original stock object.

## Rollback

Rollback requires no migration: stock is already active. The isolated
experiment restores alignment 128, uses the pinned post1 package, and leaves
the shared venv unchanged. The production acceptance threshold is not weakened
by the local leaf result or by any TP4 diagnostic.
