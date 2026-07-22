# TP AllReduce attempt ledger

Status: **PRE-RUNTIME TEMPLATE.** No attempt below has a measured outcome yet.
Never fill a numeric field from recollection, an unlocked run, provider-only
output, or another topology. Link the immutable result/profile artifact.

## Required entry contract

For every experiment, preserve all of these fields even when it fails before
timing:

1. **Attempt identity:** stable ID, date, source SHAs, exact operator, M, dtype,
   message bytes, rank topology, graph/eager mode, and stream.
2. **Reachability basis:** observed caller, selected communicator/backend,
   decision predicates, algorithm, input/output alias and poststate, following
   consumer, and trace artifact. Label source-only predictions explicitly.
3. **Hypothesis and baseline evidence:** the bottleneck being tested and the
   exact three-run/reference-control evidence motivating it.
4. **Exact delta:** config, source, build, candidate, or threshold change. For an
   external library record upstream repository/base, local commit, build command,
   artifact path, and resolved import path.
5. **Expected effect:** expected runtime behavior and, when relevant, expected
   PTX/SASS, memory transactions, synchronization, occupancy, tail, register,
   spill, launch, overlap, or NVLink effect.
6. **Correctness and contracts:** exact reduction values, two alternating input
   variants, destructive-input restoration outside timing, dtype/shape/stride,
   in/out-place alias/poststate, allocator ownership, graph replay, default and
   nondefault streams, full-tensor consumer readiness, and per-rank error status.
7. **Performance:** alternating paired rank-max raw samples; reference and
   candidate p50; paired speedup p50, p10, and p90; noise control; and the >=3%
   decision. Provider scouting is not a gate result.
8. **Profiler delta:** Nsight Systems host gaps, registration/setup, barriers,
   communication/custom kernels, graph replay, overlap, and NVLink evidence.
   Add Nsight Compute plus PTX/SASS/ptxas evidence only for a device-code
   hypothesis.
9. **Risk:** unsupported shapes/topologies, adapter tax, alias or allocator
   change, graph/stream hazards, launch overhead, dynamic dispatch cost, and TP8
   generalization risk.
10. **Decision and rollback:** promote, reject, or blocked; exact enabled bucket
    if any; stock fallback; rollback commit/config; and remaining containing-
    region/end-to-end gates.

Failed and ABI-incompatible attempts are evidence. Keep them; do not rewrite
them as absent or successful.

## Planned attempt A0: stock reachability and reference characterization

- Status: PENDING
- Scope: M16 graph/nondefault; M32 graph/nondefault; prefill eager/nondefault;
  TP4 BF16 direct coordinator diagnostics.
- Hypothesis: runtime tracing and Nsight Systems will identify the actually
  selected stock backend and whether launch/setup/communication/consumer work
  dominates each fixed shape.
- Baseline evidence: PENDING three-run rank-max baselines and reference-control
  noise measurements.
- Exact delta: none; `SGLANG_GLM52_OPT=0`; tracing only in separate ineligible
  reachability runs.
- Expected effect: characterization only; no speedup claim.
- Correctness: PENDING exact semantic matrix and alias/poststate contracts.
- Performance/distribution: PENDING.
- Profiler/NVLink evidence: PENDING; see
  [PROFILE_ANALYSIS.md](PROFILE_ANALYSIS.md).
- Risk: direct harness may not reproduce any real GLM model caller or following
  consumer.
- Decision/rollback: PENDING; stock is already the rollback and remains active.

## Planned attempt A1: raw in-place c10d/NCCL comparison

- Status: PENDING; an ABI rejection is a valid negative result.
- Scope: each fixed TP4 shape, matched graph/eager mode and nondefault stream.
- Hypothesis: raw c10d may characterize NCCL latency, but it is deployable only
  if it matches the trace-proven reference alias/poststate and stream contracts.
- Baseline evidence: PENDING A0.
- Exact delta: `serving_native/candidates/allreduce_torch.py`; the candidate
  performs in-place `torch.distributed.all_reduce` on the timed input.
- Expected effect: NCCL collective with no out-of-place clone; exact kernel and
  synchronization expectations must be confirmed in Nsight Systems.
- Correctness/contracts: PENDING. Do not waive an alias/poststate mismatch even
  if values or latency look favorable.
- Performance/distribution: PENDING if and only if the exact runner persists a
  valid result.
- Profiler delta: PENDING only if ABI-compatible; otherwise inapplicable.
- Risk: destructive input and output alias may differ from a custom out-of-place
  reference; raw c10d is not automatically the production backend.
- Decision/rollback: PENDING; no integration is authorized by this comparison.

## Planned attempt A2: cloned out-of-place c10d/NCCL comparison

- Status: PENDING.
- Scope: each fixed TP4 shape, matched graph/eager mode and nondefault stream.
- Hypothesis: cloning before c10d can preserve an out-of-place reference ABI,
  but the timed clone/allocator/traffic tax may eliminate any collective gain.
- Baseline evidence: PENDING A0.
- Exact delta: `serving_native/candidates/allreduce_torch_outplace.py`; clone and
  c10d are both inside the timed candidate call.
- Expected effect: one extra full-tensor copy/allocation plus NCCL; confirm the
  copy, registration behavior, kernels, and ready-region dependency in profile.
- Correctness/contracts: PENDING exact values, source preservation, output alias,
  graph replay, streams, and consumer readiness.
- Performance/distribution: PENDING.
- Profiler delta: PENDING if ABI-compatible.
- Risk: adapter tax, allocation/graph ownership, and a direct-harness-only win.
- Decision/rollback: PENDING; stock remains active.

## Planned attempt A3: existing SGLang provider/threshold scout

- Status: PENDING; performance-only scouting, never a gate result.
- Scope: upstream TP4 message-size sweep including the fixed 192 KiB, 384 KiB,
  and relevant larger-message neighborhoods.
- Hypothesis: comparing NCCL, AOT custom AllReduce, JIT custom AllReduce v2, and
  FlashInfer will identify whether an already available provider or static
  threshold deserves a production-ABI experiment before any new kernel work.
- Baseline evidence: PENDING trace-selected backend and A0 profiles.
- Exact delta: none to production dispatch; run the committed upstream benchmark.
- Expected effect: provider latency trends only. The scout does not establish
  correctness, aliasing, graph, consumer, or production reachability.
- Correctness/contracts: not provided by the scout; must be rerun through the
  exact serving-native candidate gate before consideration.
- Performance/distribution: PENDING raw scout log; do not copy into the paired
  result table.
- Profiler delta: not applicable unless a provider is promoted to a separate
  exact candidate attempt.
- Risk: provider-specific ABI/setup differences and TP4-to-TP8 non-transfer.
- Decision/rollback: PENDING; no dispatch change from scouting alone.

## Additional attempts

Append one complete section per new source, configuration, threshold, fusion, or
kernel hypothesis. Do not merge multiple deltas into one attempt. No device-code
attempt is currently authorized by evidence; if one is added, include Nsight
Compute, ptxas resources, and PTX/SASS inspection before promotion.
