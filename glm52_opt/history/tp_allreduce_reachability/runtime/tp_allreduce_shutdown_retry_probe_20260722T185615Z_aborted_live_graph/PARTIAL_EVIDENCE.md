# Partial teardown probe with live captured graphs

This operator-observed locked TP4 probe is retained as negative infrastructure
evidence. Its terminal lock receipt was not copied into the archive, so the
wrapper/exit observation is explicitly qualified in `INCIDENT_RECEIPT.json`.
It is not a baseline, candidate result, or acceptance result.

- The probe exercised `tp4_allreduce_decode_m16`, BF16 `[16, 6144]` per rank,
  CUDA Graph replay on a nondefault stream, with the raw in-place c10d
  candidate and `SGLANG_GLM52_OPT=0`.
- `result.json` was serialized at 2026-07-22 18:56:39.726939113 UTC, but the
  wrapped command did not return before its 180-second outer timeout and
  exited 124 at approximately 18:59:23 UTC. There is no successful outer
  command receipt.
- The completed payload is internally self-consistent: it records 100 logical
  A/B pairs accepted from 125 physical attempts, with 25 physical pairs
  rejected solely by the fixed 500,000 ns host start-envelope rule. Every
  accepted side stayed below that limit (reference maximum 440,866 ns;
  candidate maximum 457,594 ns), and exact correctness passed.
- Those timings remain ineligible. The command hung after persistence, both
  worktrees were dirty by construction, and the result records the external
  conda Python instead of the required Kernel-Harness environment. Its
  observed 0.973438x paired-p50 speedup must not be cited as performance
  evidence.
- Source review after the probe identified a mechanism consistent with the
  hang: changing the final rendezvous to Gloo was insufficient while Python
  still retained CUDA Graphs containing NCCL work. NCCL v2.28.9-1
  `src/enqueue.cc` (persistent graph references) and `src/init.cc`
  (`ncclCommDestroy` wait loop) document the matching mechanism. The follow-up
  runner registers every graph before capture, synchronizes the execution
  stream, resets graphs in reverse creation order, and only then enters
  process-group teardown.
- Exact dirty-source reproduction is unavailable. The payload records the two
  base SHAs and dirty filenames, but not the dirty file contents, and those
  files were modified further after this probe. The result is useful only for
  the internally rederivable ledger and failure chronology.

`INCIDENT_RECEIPT.json` separates direct filesystem facts from the live
operator observation and source-derived diagnosis. A new clean, locked probe
and campaign are required before accepting any result.
