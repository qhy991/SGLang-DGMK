# TP AllReduce prior-art note

This note records the source material consulted before changing the TP
AllReduce path. It is evidence for experiment selection, not runtime evidence.
Runtime reachability and performance must come from this isolated branch and
the locked four-GPU lane.

## Model-family history

The GLM-5/5.1/5.2 PR history was queried with:

```text
python3 scripts/query.py --framework sglang --model glm5-glm51 --paths-only
```

The returned document was
`sglang/glm5-glm51/README.en.md` in the local
`model-pr-optimization-history` knowledge base. Relevant conclusions:

- GLM-5's `GlmMoeDsaForCausalLM` entered through the DeepSeek-V2 model path in
  PR #18521. Consequently, DeepSeek-V2 communicator and layer-transition code
  is part of the GLM reachability audit; a search limited to `glm4_moe.py`
  would miss material callers.
- PR #25814 enabled FlashInfer AllReduce fusion for an H200 GLM-5 FP8 recipe.
  This is evidence that fused AllReduce-plus-normalization is an established
  serving optimization, but it is not evidence for B200, GLM-5.2, or the
  fixed TP4 shapes.
- PR #27053 added TP8 GLM-5 piecewise-CUDA-graph validation. It reinforces that
  graph replay is a required compatibility lane and that TP4 evidence cannot
  be relabelled as the TP8 production gate.
- PRs #28437 and #28460 added and verified GLM-5.2 deployment recipes on
  eight-GPU systems. Those recipes are the source of the separate TP8
  production acceptance requirement; this goal's four-rank measurements are
  diagnostic only.

## KernelWiki prior art

The local KernelWiki snapshot (captured through 2026-04-27) was queried for
B200/SGLang custom AllReduce, NVLink, synchronization, and small-message
latency. The most relevant entries were:

- SGLang PR #17591: re-enabled AllReduce fusion on SM100 after a B200 TP4
  benchmark found the fused route faster than NCCL even with a known kernel
  limitation.
- SGLang PR #7621: added the TRT-LLM/FlashInfer AllReduce plus RMSNorm/add path
  for B200.
- SGLang PR #8731: extended fused AllReduce plus residual RMSNorm through
  communicator, linear, MoE, and model call sites.
- FlashInfer PR #1096: introduced the TRT-LLM non-MoE custom AllReduce backend.

These records prioritize caller/consumer fusion and routing verification over
inventing a standalone collective kernel. They do not establish that the
fixed direct `GroupCoordinator.all_reduce` workload reaches the fused path.

The required Kernel-Harness warm start was also run as
`python3 testbench/bin/brief.py tp_allreduce_reachability`. There is no frozen
task or prior recorded run under that production-goal name, but its KernelWiki
lookup returned four additional FlashInfer records:

- FlashInfer PR #1507 synchronized the fused AllReduce launch configuration
  with TensorRT-LLM and reported a large result on a different Llama/TP2
  workload. This makes the installed FlashInfer backend a comparison target,
  not a result transferable to GLM-5.2/TP4.
- FlashInfer PR #1265 made the standalone AllReduce output optional when the
  following RMSNorm consumes the reduction internally. This reinforces that
  output allocation/alias semantics must be measured at the actual caller.
- FlashInfer PR #1159 added TensorRT-LLM's finalized MoE-AllReduce fusion. It is
  relevant to the non-A2A MoE caller, while the balanced DeepEP lane remains an
  A2A exclusion.
- FlashInfer PR #1321 concerns multi-node NVLink AllReduce. The fixed lane is
  single-node TP4, so it is recorded but excluded from local experiment choice.

The warm-start warning is explicit: no `testbench/tasks/glm52/` directory or
`run.sh` exists for `tp_allreduce_reachability`. This goal therefore uses the
production-specific `serving_native` lane rather than inventing or modifying a
frozen synthetic task.

## Branch-local and newer-upstream source

The isolated SGLang branch is pinned at `f93f8867b4bc124c9809c9110ec7361ed11b6b4a`.
Its JIT custom-AllReduce v2 path is eligible only for contiguous, 16-byte
aligned payloads no larger than 16 MiB. On SM100 TP4, the fixed 192 KiB and
384 KiB decode payloads select the push algorithm; the 96 MiB prefill payload
falls through to another communicator. These are static predictions to be
checked by runtime tracing.

The locally available `origin/main` contains commit
`132ade55cd65c3679a60adec563c8e6ca81238c5` (PR #31049), which rewrites JIT
custom-AllReduce v2 with decoupled communicator/storage and explicit SM100
configuration tables. That commit is newer than this branch and is therefore
an experimental comparison candidate only. It must not be treated as the
stock baseline, silently rebased into the goal branch, or promoted without
the same paired correctness, graph, stream, and TP8-preservation gates.

## Historical local measurement lead (not goal evidence)

A read-only search found one earlier Codex-session transcript containing eager
TP4 comparisons of the SGLang coordinator against raw in-place c10d/NCCL for
the 192 KiB and 384 KiB decode payloads. It is retained only as motivation to
repeat the c10d comparison: that run bypassed the scheduler wrapper, used the
non-isolated dirty checkouts, persisted no result JSON or raw samples, used a
loose allclose check, did not validate output aliasing, and had no graph,
stream, topology, profiler, or containing-region evidence. A contemporaneous
JIT-cache specialization strongly suggests that its reference used TP4 BF16
one-shot push, but there was no dispatch trace. None of its latency numbers are
accepted as baseline or result evidence for this goal.

The transcript is local prior art at
`/home/qinhaiyan/.codex/sessions/2026/07/22/rollout-2026-07-22T10-22-50-019f8959-7a2c-7863-84c9-44fd163d9a8e.jsonl`.
The new exact runner deliberately rejects raw c10d when its in-place alias
contract differs from the selected reference; a faster but ABI-incompatible
historical result cannot authorize a dispatch change.

## Experiment implications

1. Trace production dispatcher decisions before benchmarking a raw NCCL or
   custom kernel candidate.
2. Measure eager and CUDA-graph replay separately and record output aliasing.
3. Compare already-eligible SGLang backends and static thresholds before
   writing a new kernel.
4. Treat any fused producer/consumer improvement as a separate serving-region
   result; do not attribute it to the direct collective microbenchmark.
5. Keep all TP8 behavior stock until an actual eight-rank run validates it.
