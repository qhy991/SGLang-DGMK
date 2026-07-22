# TP AllReduce attempt ledger

Status: **RUNTIME EVIDENCE IN PROGRESS.** Infrastructure failures and probes
below are measured outcomes, but no backend performance result is accepted
until the clean campaign and analyzer complete. Never fill a numeric field from
recollection, an unlocked run, provider-only output, or another topology. Link
the immutable result/profile artifact.

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

## Infrastructure attempt P-1: CPU-barrier start alignment

- Status: **REJECTED AND ABORTED BEFORE BASELINES** on 2026-07-22.
- Identity: Harness `4c58357c2d7142f2160dd6674e8dde002ddc64e0`,
  SGLang `e2ce8e099e7a279e30e6d72fd4e2fe056299204a`, TP4 BF16 direct
  coordinator diagnostics on four B200s. Immutable partial artifacts and
  manifest: [runtime/tp_allreduce_reachability_20260722T172535Z_aborted_alignment](runtime/tp_allreduce_reachability_20260722T172535Z_aborted_alignment/).
- Hypothesis: synchronizing each selected stream and then entering a blocking
  TP CPU-group barrier would align the four CUDA start-event records closely
  enough for rank-max collective timing.
- Exact delta: no backend/candidate delta; this tested the then-current timing
  infrastructure with stock `SGLANG_GLM52_OPT=0`.
- Correctness/reachability: the three trace runs and two completed M16 semantic
  runs passed exact values, alias/poststate, source immutability, stream
  readiness, and trace-hook checks. Decode capture selected custom AllReduce V2
  `ONE_SHOT_PUSH`; prefill selected c10d `SUM` because the 96 MiB message was
  outside custom-AR eligibility.
- Failure evidence: 10 of 23 persisted samples exceeded the unchanged 500,000
  ns rank-start-envelope gate. M16 graph trace was 366,165 ns; M32 graph trace
  448,246 ns; prefill eager trace 690,673 ns; M16 eager/default failed 7/10
  (max 874,754 ns); M16 eager/nondefault failed 2/10 (max 616,768 ns).
  Individual host brackets were only 48--108 us, so the excess was inter-rank
  enqueue skew rather than `start.record` call duration. Two later required
  steps have exit 120 because the campaign was deliberately terminated.
- Performance/profiler: ineligible and not cited. No baseline, paired, backend,
  producer, or Nsight result was allowed to proceed.
- Risk/decision/rollback: a 0.55--0.87 ms rank launch skew can dominate decode
  latency and invalidate rank-max comparison. The run was stopped and stock
  remained active. Rollback point is Harness `4c58357`; the replacement timing
  mechanism was required to retain the strict evidence gate.

## Infrastructure attempt P0: common scheduled-start deadline

- Status: **VALIDATED FOR CAMPAIGN USE; NOT A BACKEND PERFORMANCE RESULT** on
  2026-07-22.
- Identity: Harness `46ba02607b54f62cdef5a43c99e558b12b545aa6`, SGLang
  `ccf6d059350afbc56c39f8722df0b90871e8bdd4`, TP4 BF16; raw results and
  manifest: [runtime/tp_allreduce_alignment_probe_20260722T175813Z](runtime/tp_allreduce_alignment_probe_20260722T175813Z/).
- Hypothesis: after restoration and selected-stream synchronization, gathering
  same-host monotonic arrival timestamps, choosing `max(arrivals)+5 ms`, and
  busy-waiting to that common target will remove CPU barrier-return skew while
  keeping all synchronization outside CUDA-event timing.
- Exact delta: Harness commit `46ba026` persists four arrivals, four identical
  targets, and four host record brackets per sample. Analyzer commit `ccf6d059`
  rederives the target, forbids pre-target records, rederives the envelope, and
  retains the 500,000 ns cap. The deliberate wait remains inside the outer
  profiler NVTX range and must be labeled, not attributed to a backend.
- Correctness/contracts: exact correctness passed for 50 M16 graph/nondefault,
  50 M32 graph/nondefault, and 50 prefill eager/nondefault stock samples.
  Every recorded target equaled `max(arrivals)+5,000,000 ns`; no pre-target
  start was observed.
- Alignment distribution: zero of 150 samples exceeded 500,000 ns. Median/max
  envelopes were 46,636/300,981 ns for M16, 54,577/447,884 ns for M32, and
  55,290/445,848 ns for prefill.
- Performance/profiler: reference-only probe latencies are deliberately not a
  baseline or speedup claim; no Nsight profile was collected.
- Risk/decision/rollback: ordinary scheduler preemption can still miss a target,
  so every full-campaign sample remains fail-closed on its actual envelope. The
  mechanism is accepted only as measurement infrastructure for the next clean
  campaign; it does not alter SGLang backend dispatch or stock fallback.

## Infrastructure attempt P1: shutdown hang and long-tail alignment

- Status: **REJECTED PARTIAL CAMPAIGN; FIXES IMPLEMENTED, VALIDATION PENDING**
  on 2026-07-22.
- Identity: Harness `46ba02607b54f62cdef5a43c99e558b12b545aa6`, SGLang
  `49c03b20e861782002eec6f5be7126e7e19c2d28`, TP4 BF16 on four B200s;
  immutable raw files, incident note, and manifest:
  [runtime/tp_allreduce_reachability_20260722T181038Z_aborted_shutdown_alignment](runtime/tp_allreduce_reachability_20260722T181038Z_aborted_shutdown_alignment/).
- Hypothesis: the scheduled-start mechanism would remain within 500 us over a
  full campaign, and raw c10d CUDA-graph execution would exit through the
  generic NCCL shutdown barrier after serializing its result.
- Correctness/reachability: all reachability and semantic steps completed with
  exact values, alias/poststate, source immutability, and stream readiness. The
  M16 c10d payload also completed 100 alternating exact A/B pairs and the final
  CPU-group persistence acknowledgement.
- Failure evidence: `paired/m16_c10d_inplace.json` was written at
  18:24:46 UTC, after which all four workers remained live and futex-waiting
  for more than six minutes after the last logged device-context barrier
  warning. The recorded source flow is consistent with a post-persistence NCCL
  device-group barrier stall, but no worker stacks were captured. No candidate
  `status.tsv` row or after-state receipt exists. Separately, the M16
  reference-control had 14/200 start envelopes above 500,000 ns (maximum
  1,065,874 ns), c10d had 5/200 (maximum 910,102 ns), and baseline runs 2 and 3
  had one violation each.
- Performance/profiler: all timing is ineligible. For audit identity only, the
  partial c10d payload records reference/candidate ready-region p50s of
  0.194192/0.192288 ms and paired p10/p50/p90 ratios of
  0.758696/1.018643/1.265677. The p50 was below 1.03 even before rejection. No
  backend scout, producer ABI, or Nsight profile was reached.
- Exact infrastructure delta for the retry: normal TP AllReduce teardown now
  uses the existing TP CPU group and never injects a final NCCL barrier after a
  captured raw c10d collective. Measurement admission retries the entire
  logical A/B pair, with fixed logical order, solely when rederived host start
  brackets exceed 500 us. The exact input variant alternates for every physical
  pair to retain stale-output detection. The limit is ten total attempts; every
  completed physical attempt is retained in a successful result or bounded
  alignment-failure receipt, and CUDA latency is excluded from admission.
- Risk/decision/rollback: a serialized JSON without a successful outer command
  exit is not a completed result. Whole-pair retry avoids one-sided selection,
  remains bounded and fail-closed, and requires a new clean locked validation.
  Stock dispatch remained active throughout.

## Infrastructure attempt P2: retry ledger observed; teardown hung with live graphs

- Status: **REJECTED TEARDOWN PROBE; GRAPH-RESET FIX IMPLEMENTED, CLEAN
  VALIDATION PENDING** on 2026-07-22.
- Identity: dirty descendants of Harness
  `46ba02607b54f62cdef5a43c99e558b12b545aa6` and SGLang
  `49c03b20e861782002eec6f5be7126e7e19c2d28`; TP4 BF16 M16,
  CUDA Graph/nondefault stream, raw in-place c10d candidate. The payload,
  incident receipt, explicit rejection note, and manifest are under
  [runtime/tp_allreduce_shutdown_retry_probe_20260722T185615Z_aborted_live_graph](runtime/tp_allreduce_shutdown_retry_probe_20260722T185615Z_aborted_live_graph/).
  The payload records the base SHAs and dirty filenames, not the dirty source
  contents; those files were subsequently modified, so exact source
  reproduction is unavailable. The lock-wrapper and exit-124 facts are also
  contemporaneous operator observations because no raw terminal log was
  copied.
- Hypothesis: the TP CPU-group final rendezvous would permit clean shutdown,
  while bounded whole-pair retry would retain 100 eligible logical pairs under
  rare host preemption.
- Correctness/alignment: the serialized payload passed exact correctness and
  accepted 100 logical pairs from 125 physical attempts. All accepted start
  envelopes stayed below 500,000 ns (reference maximum 440,866 ns; candidate
  maximum 457,594 ns); 25 whole physical pairs were retained as rejected
  attempts. Admission did not inspect CUDA latency.
- Failure evidence: despite the complete payload, the four workers did not
  exit before the 180-second outer timeout; the wrapper returned 124 at
  approximately 18:59:23 UTC. The payload also records dirty worktrees and
  `/home/qinhaiyan/miniconda3/envs/sglang/bin/python`, not the required
  Kernel-Harness environment.
- Performance/profiler: ineligible because there is no successful outer exit.
  For audit identity only, the payload records reference/candidate ready-region
  p50s of 0.200032/0.204816 ms and paired p10/p50/p90 ratios of
  0.678812/0.973438/1.248220. They cannot support a backend decision. No profile
  was collected.
- Source diagnosis and follow-up delta: source review found a mechanism
  consistent with the hang: CUDA Graphs retaining captured NCCL work were still
  alive when process groups were destroyed, and NCCL 2.28.9 waits for those
  graph references. The runner now registers each graph before capture,
  synchronizes its execution stream, resets graphs in reverse creation order,
  severs runner references, confirms success over the TP CPU group, and only
  then destroys SGLang subgroups and WORLD. A reset failure skips explicit
  communicator destruction and fails closed.
- Risk/decision/rollback: the serialized retry ledger is internally coherent,
  but its timed result remains rejected and clean validation was still required
  at this point. No backend or threshold change is enabled; stock dispatch
  remains active. Rollback remains Harness `46ba026` plus SGLang `49c03b20`.

## Infrastructure attempt P3: explicit captured-graph release

- Status: **VALIDATED FOR CLEAN CAMPAIGN USE; NOT A BACKEND PERFORMANCE
  RESULT** on 2026-07-22.
- Identity: exact at-probe Kernel-Harness file hashes landed unchanged as
  `bea4a8cd294a84f2d305cd023eb3fed281de4b1e`; SGLang serving-runtime
  source `49c03b20e861782002eec6f5be7126e7e19c2d28` (dirty files were confined
  to offline history/analyzer evidence); TP4 BF16 M16,
  CUDA Graph/nondefault stream, raw in-place c10d candidate. Result, command,
  outer-exit receipt, review note, and manifest:
  [runtime/tp_allreduce_graph_reset_probe_20260722T191541Z_validation](runtime/tp_allreduce_graph_reset_probe_20260722T191541Z_validation/).
- Hypothesis: captured NCCL graph references, not a final rendezvous alone,
  caused communicator destruction to wait indefinitely. Explicit graph reset
  before any process-group teardown should let the same payload exit normally.
- Exact delta: every `CUDAGraph` is registered immediately after creation,
  including partial-capture failures. Teardown synchronizes the execution
  stream, resets graphs in reverse creation order, clears runner graph
  references, confirms reset success over the TP CPU group, and only then
  destroys SGLang subgroups and WORLD. Reset failure skips explicit
  communicator destruction and fails closed.
- Correctness/alignment: exact correctness passed. Twenty logical A/B pairs
  were accepted from 21 physical attempts; the one rejected whole pair remains
  in the ledger. Accepted reference/candidate host envelopes were at most
  413,319/261,727 ns.
- Lifecycle result: the locked command acquired physical GPUs 0--3, persisted
  the result, completed graph and process-group teardown, and returned exit 0
  after 18.632 seconds. This directly distinguishes it from P1/P2, whose JSON
  existed but whose outer commands never completed.
- Performance/profiler: deliberately ineligible because both worktrees were
  dirty and the probe used only 20 repeats. The payload's favorable speedup is
  not cited and no profile was collected.
- Risk/decision/rollback: the lifecycle and alignment infrastructure are
  accepted for a fresh clean campaign. No SGLang backend, threshold, or TP4/TP8
  dispatch policy changed; stock remains active. Rollback is the previous
  committed Harness `46ba026` runner.

## Infrastructure attempt P4: clean campaign rejected by stale analyzer schema

- Status: **REJECTED PARTIAL CAMPAIGN; ANALYZER FIX VALIDATED OFFLINE, FRESH
  CLEAN CAMPAIGN REQUIRED** on 2026-07-22.
- Identity: clean Harness `bea4a8cd294a84f2d305cd023eb3fed281de4b1e`
  and SGLang `a91928c6f713d69722176a8d13781367e9b78dc4`, physical GPUs
  0--3 under the all-GPU lock. Immutable raw files, status ledger, failure note,
  fix receipt, offline replay receipts, and manifest:
  [runtime/tp_allreduce_reachability_20260722T192433Z_aborted_analyzer_contract](runtime/tp_allreduce_reachability_20260722T192433Z_aborted_analyzer_contract/).
- Hypothesis: the now-validated lifecycle and pair-retry runner would complete
  the full clean campaign, and the committed analyzer would resolve the exact
  in-place/out-of-place ABI before later GPU work.
- Completed evidence: environment/topology, all three runtime traces, all nine
  semantic rows, three baselines per shape, three reference controls, and six
  c10d ABI attempts exited as expected. Exact in-place c10d passed for all
  shapes; cloned out-of-place c10d failed its all-rank alias/poststate contract.
- Failure: all three required ABI resolver steps exited 1 because the analyzer
  still required the pre-retry `rank_start_alignment` text. Source inspection
  during fix development found a second stale restriction: current collective
  failure records include four scheduled arrivals and a common target in
  addition to rank/error/bracket. The required phase guard stopped before
  backend scouting, producer controls, profiles, and after-state receipts.
- Exact delta and validation: the analyzer now requires the complete retry
  timing contract and independently validates failure arrivals, the
  `max(arrivals)+5,000,000 ns` target, rank brackets, and cross-rank equality.
  Fifteen CPU tests pass. All three resolvers replayed successfully against the
  untouched archive, selecting `inplace` and rejecting `outplace`.
- Performance/profiler: ineligible and deliberately not summarized; the
  campaign has no profiler or after-state evidence and the corrected analyzer
  is a new source revision.
- Risk/decision/rollback: offline replay validates the fix but cannot resume a
  source-frozen campaign. Rerun from a fresh path after committing the analyzer
  and archive. No backend or threshold is enabled; all TP4/TP8 behavior remains
  stock.

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
