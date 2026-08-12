# Qwen3.5-122B-A10B TP2: Fusion-Space-First campaign

## Objective

Improve TP2 decode throughput on two B200 GPUs by expanding the optimization
domain before choosing local kernel tactics. Direct output, reduce+push, and
similar mechanisms are examples, not prescribed solutions.

The canonical domain is a decoder-layer transition:

```text
input norm/quant -> attention or GDN -> projection -> TP communication
-> residual/post-attention norm -> router/shared+routed MoE -> TP communication
-> next-layer residual/input norm
```

The campaign may temporarily accept a slower intermediate implementation when
it establishes a real producer-to-consumer boundary and exposes a measurable
new bottleneck. The final retained implementation must win end to end.

## Pins and ownership

- Hardware: `verda-b200x4`, exactly two B200 GPUs for the measured server.
- Model: Qwen3.5-122B-A10B FP8, TP size 2.
- Baseline SGLang commit: `2a51dee179cb3ebde8cf54185017eba65217203a`.
- Read-only baseline worktree: `/home/qinhaiyan/sglang-clean-qwen35-tp2-p1`.
- Candidate worktree: `/home/qinhaiyan/ane-qwen35-tp2-p23-fusion-space-first`.
- Correctness/performance harness owner:
  `/home/qinhaiyan/ane-qwen35-tp2-evidence-integration-20260810/omoe`.
- This file owns the campaign scope and gates. Per-attempt records contain only
  observations, commands, artifacts, and a decision.

## Existing boundary and first extension

Stock SGLang already supports the downstream half of the desired domain:
MoE output can be marked for FlashInfer MNNVL all-reduce fusion, and the next
layer can perform all-reduce + residual add + RMSNorm together.

The block-FP8 FlashInfer TRT-LLM MoE path nevertheless materializes its result
in a private output and copies it into a symmetric tensor before that fused
consumer. Attempt P23 removes this intervening ownership transfer by supplying
the symmetric tensor as the MoE kernel's output. The routed result, shared
expert add, TP collective, residual add, and next norm can then operate along
one buffer lifetime.

This is only the first boundary-establishment slice. It is not the final
optimization hypothesis.

## Optimization ladder

1. **Prove the boundary.** Establish direct producer output into the symmetric
   collective input. Confirm pointer identity and removal of the copy kernel.
2. **Profile the expanded domain.** Capture the whole layer transition at
   decode B1/B8/B32/B108. Attribute kernel time, gaps, stream waits, memory
   traffic, launch count, and communication overlap.
3. **Optimize the new bottleneck.** Choose only from evidence. Candidate
   directions include MoE finalization/shared-add fusion, removing dual-stream
   synchronization, router/quant co-scheduling, owner-ready collective launch,
   or a persistent cross-layer schedule.
4. **Expand again when locally saturated.** Move from the MLP tail/next norm to
   the whole decoder layer, then to adjacent-layer pipelining. Do not create a
   second execution core; variants remain policy on one canonical path.

## Evidence required at every rung

### Boundary proof

- The FlashInfer API accepts the supplied output tensor in the pinned runtime.
- Returned/consumed storage is the symmetric allocation; no fallback copy.
- The tensor remains eligible for the existing fused MNNVL all-reduce path.
- Shared-expert accumulation preserves that storage and ordering.

### Correctness gate

Run the frozen seven-case TP2 gate from the harness owner. Record all hashes and
the exact server command. No performance claim survives a single mismatch,
server crash, or unsupported high-batch row.

### Performance gate

- Use baseline/candidate/baseline (A/B/A), same GPUs, environment, model, and
  request sequence; retain all samples.
- Primary metric: decode B1/B8/B32/B108 throughput geometric mean.
- Report each row, TTFT/prefill effects, memory, capture time, and variance.
- An initial range-expansion attempt may be retained for one profiling cycle at
  up to 5% geomean regression only if boundary proof succeeds and the profile
  identifies the next removable cost.
- A production keep requires at least 3% repeatable decode-geomean improvement
  with no protected-row regression, unless the scope is explicitly narrowed.

## Retained production slice

The retained change supplies the existing symmetric collective buffer directly
to the block-FP8 FlashInfer TRT-LLM MoE kernel. Two explicit SGLang custom ops
(standard routing and routed-topk) declare `mutates_args=["output"]`, return
`None`, and reject a FlashInfer result that changes the output pointer, shape,
or stride. Existing allocating wrappers and non-block-FP8 paths are unchanged.

This targets the measured production composition: DeepGEMM owns dense/shared
projections, FlashInfer TRT-LLM owns routed MoE, and FlashInfer MNNVL owns the
TP collective. It does not include the unsafe no-clone fan-out or the later
collective/norm/quant prototypes.

## Qualification evidence

- Static, CPU ownership, Meta `torch.compile(fullgraph=True)`, CUDA eager and
  CUDA Graph contracts pass for both wrappers.
- The frozen TP2 gate passes, including 254/256 teacher matches, 100% graph vs.
  eager agreement, the B108 capacity row, and stable frozen hashes.
- Frozen B32, B108, and 8K full-service Nsys captures preserve output hashes.
- At B32, each rank still executes 90,613 kernels, while all 3,072 D2D copies
  of 196,608 bytes immediately following FlashInfer `finalizeKernel` disappear.
  These copies cost 4.43--4.54 ms/rank in the baseline trace.
- Hash-bound bounded NCU replays select the actual production shapes. B32
  `finalizeKernel` takes 6.56 us at 31.13% achieved occupancy and 69.33% long-
  scoreboard stalls. B108 `finalizeKernelVecLoad` takes 7.90 us at 10.99%
  occupancy, with 58.10% long-scoreboard and 17.88% barrier stalls. The kernels
  are dependency/underfill limited; isolated tactic tuning is not the retained
  follow-up.

## End-to-end result

The final A/P23/A bracket uses three rounds per arm. The candidate is normalized
against the geometric mean of its adjacent baselines.

| Decode row | P23 (tok/s) | Adjacent baseline (tok/s) | Delta |
| --- | ---: | ---: | ---: |
| B1 | 199.96 | 190.56 | +4.93% |
| B8 | 1034.83 | 995.49 | +3.95% |
| B32 | 2030.48 | 2003.88 | +1.33% |
| B108 | 2715.81 | 2686.08 | +1.11% |
| Geometric mean | 1033.54 | 1005.22 | **+2.82%** |

All decode rows improve and all hashes remain stable. Prefill geometric mean is
+5.16%; the protected 32K row is -0.34%. The result is 0.18 percentage points
below the campaign's original universal-default threshold of +3%. It is
therefore admitted to `sglang-dgmk` as a separately reviewable production
candidate with measured evidence, rather than being represented as a universal
default winner. Rollback is the single commit that owns this slice.

## Excluded experiments

- P25 removes an additional clone but changes one frozen B108 output hash.
- P28/P29 expand through collective, residual, RMSNorm, and next quantization,
  but their cross-process correctness/performance admission is not complete.
- P24 modifies a DeepGEMM MoE finalizer that the measured production path does
  not execute.

None of these experiments is present in this branch.
