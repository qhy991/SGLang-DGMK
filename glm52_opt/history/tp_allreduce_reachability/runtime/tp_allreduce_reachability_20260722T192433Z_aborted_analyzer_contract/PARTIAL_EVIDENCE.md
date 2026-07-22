# Clean campaign stopped by stale analyzer contracts

This campaign is preserved as rejected infrastructure evidence. It is not a
performance result and no number from its timed payloads is eligible for a
backend or deployment decision.

- The campaign acquired physical GPUs 0,1,2,3 and recorded clean source heads
  Kernel-Harness `bea4a8cd294a84f2d305cd023eb3fed281de4b1e` and SGLang
  `a91928c6f713d69722176a8d13781367e9b78dc4`.
- Environment/topology checks, all three reachability traces, the full nine-row
  semantic matrix, three baselines per shape, three reference controls, and
  all six c10d ABI attempts completed. In-place c10d passed the exact reference
  ABI for every shape; cloned out-of-place c10d failed the expected all-rank
  alias/poststate contract.
- All three required `*_c10d_abi_resolution` steps then exited 1. The committed
  analyzer still expected the pre-retry `rank_start_alignment` prose. Source
  inspection during fix development also found that it accepted an older
  collective-failure record schema, while the committed runner emitted the
  expanded retry contract and scheduled-start failure metadata. The
  campaign stopped before backend scouting, producer controls, profiling, or
  after-state receipts, exactly as its required-phase guard specifies.
- The fix validates the complete retry timing contract and the failure record's
  four arrivals, derived common target, and per-rank start bracket. Fifteen CPU
  tests pass. Replaying the fixed analyzer against this untouched archive
  returns `valid: true` for M16, M32, and prefill, selecting `inplace` and
  canonically rejecting `outplace`; those receipts are under
  `offline_fixed_analyzer_replay/`.
- Offline replay proves the analyzer fix, not campaign completion. Because the
  analyzer source contents changed and later required phases never ran, a fresh
  clean locked campaign from a new committed revision is mandatory. Stock
  SGLang remains active for every TP4 and TP8 bucket.

`ANALYZER_FIX_RECEIPT.json` records exact hashes and the failure/replay
classification. `status.tsv` is the automatic outer-step ledger.
