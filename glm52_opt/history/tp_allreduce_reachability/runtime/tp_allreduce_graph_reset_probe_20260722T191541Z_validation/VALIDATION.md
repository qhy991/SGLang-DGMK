# Captured-graph teardown validation

This locked TP4 diagnostic validates the runner lifecycle fix. It is not clean
performance evidence and does not satisfy any TP8 production gate.

The exact three Kernel-Harness files present during the probe were unchanged
through commit `bea4a8cd294a84f2d305cd023eb3fed281de4b1e`; their SHA-256 values
are recorded in `EXIT_RECEIPT.json`. The SGLang runtime source was committed at
`49c03b20e861782002eec6f5be7126e7e19c2d28`; its dirty files were confined to
this goal's offline history/analyzer evidence and did not alter serving code.

- The command acquired the four scheduler locks and exposed physical GPUs
  0,1,2,3. It ran `tp4_allreduce_decode_m16`, BF16 `[16,6144]` per rank,
  reference custom AllReduce plus raw in-place c10d, CUDA Graph replay, and a
  nondefault stream with `SGLANG_GLM52_OPT=0`.
- `result.json` was serialized at 2026-07-22 19:16:05.212837594 UTC. Unlike the
  two preceding teardown failures, all workers then returned, the wrapper
  released its locks, and the complete command exited 0 after 18.632 seconds.
- The executed source path synchronizes the selected stream, calls `reset()`
  on the candidate and reference graph in reverse creation order, confirms
  reset success through the TP CPU group, and only then destroys SGLang
  subgroups and WORLD. Successful outer exit therefore validates the complete
  persistence-through-teardown path, not just JSON serialization.
- Exact correctness passed. Twenty logical A/B pairs were accepted from 21
  physical attempts; one whole pair was rejected by the fixed host-only
  envelope rule. Accepted reference and candidate envelopes were at most
  413,319 ns and 261,727 ns respectively.
- The payload's candidate speedup and `performance_eligible` flag are not used.
  Both worktrees were deliberately dirty with the lifecycle implementation and
  analyzer changes, and only 20 repeats were requested. This is an
  infrastructure validation; the clean committed campaign remains the sole
  performance evidence source.

The exact command and the observed outer receipt are in `EXIT_RECEIPT.json`.
The stock SGLang path remains active.
