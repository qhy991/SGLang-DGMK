# TP AllReduce results

Status: **PRE-RUNTIME TEMPLATE — no performance result or final disposition is
recorded yet.** Populate this file only from the immutable locked-campaign
artifacts after `analyze_tp4_campaign.py` reports `"valid": true`.

## Scope and evidence boundary

- Local measurements are four-rank TP4 diagnostics on the fixed BF16
  `[16,6144]`, `[32,6144]`, and `[8192,6144]` per-rank shapes.
- A direct `GroupCoordinator.all_reduce` hit proves dispatcher/backend behavior
  for that direct harness call. It does not prove that a GLM-5.2 model region
  reaches the same caller, ABI, graph state, or backend.
- The direct harness's full-tensor negation and position gather are exact
  same-stream consumers for dependency validation. They are not a substitute
  for the real producer-AllReduce-consumer region.
- TP4 evidence must never be relabeled as TP8, DP8, or EP8 production evidence.
  The external acceptance blocker and preserved gate are in
  [EXTERNAL_TP8_BLOCKER.md](EXTERNAL_TP8_BLOCKER.md).

## Source identity

| Repository | Analysis base | Landed implementation head | Campaign start/end SHA |
|---|---|---|---|
| Kernel-Harness | `bcd005409e65786af82c86f621507ebef12b2766` | `799765caad984ac2a010f762adaa873a7374018d` | PENDING |
| SGLang | `f93f8867b4bc124c9809c9110ec7361ed11b6b4a` | trace/hot-path head `0b51106d8138e950add18b5ec9cf6915ab9d321e`; later evidence-only hardening must be captured by the campaign SHA | PENDING |

The campaign must also record the Python executable, package/import paths,
CUDA, PyTorch, NCCL, device UUIDs, clocks, power, topology, P2P capability, and
NVLink state. Any source change between campaign start and finish invalidates
the measurement.

## Runtime reachability

Do not replace `PENDING` with a source prediction. Copy only trace-observed
values and link the corresponding JSONL record.

| Shape | Mode / stream | Caller chain | Group / world | Bytes | Selected backend and communicator | Decision predicates / algorithm | Input/output alias | Following consumer | Trace artifact |
|---|---|---|---|---:|---|---|---|---|---|
| M16 `[16,6144]` | graph / nondefault | PENDING | PENDING | 196608 | PENDING | PENDING | PENDING | direct-harness exact consumer only | PENDING |
| M32 `[32,6144]` | graph / nondefault | PENDING | PENDING | 393216 | PENDING | PENDING | PENDING | direct-harness exact consumer only | PENDING |
| prefill `[8192,6144]` | eager / nondefault | PENDING | PENDING | 100663296 | PENDING | PENDING | PENDING | direct-harness exact consumer only | PENDING |

Runtime reachability for every material model caller remains a separate item.
Record the loaded checkpoint/config, fully resolved launch arguments, caller
frequency, graph/replay state, stream dependencies, and real following consumer
when an eight-rank model host is available. Absence of a Python hook during
graph replay is not proof of absence; correlate capture records with stable
kernel names in Nsight Systems.

## Exact semantics

Every row must pass on all four ranks. Link the corresponding result JSON rather
than copying terminal prose.

| Shape | eager/default | eager/nondefault | graph/nondefault | Two alternating exact inputs | Destructive-input restoration | Reference alias/poststate | Full-tensor consumer readiness |
|---|---|---|---|---|---|---|---|
| M16 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| M32 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| prefill | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

CUDA Graph/default-stream is intentionally fail-closed and is not an omitted
test configuration.

## Baseline and paired results

Distributed latency is the maximum across ranks. Report three uncontended stock
run medians, their spread, the paired reference-control noise floor, and the
alternating reference/candidate distribution. Do not convert a profiler run or
provider-scout number into a gate result.

| Bucket | Stock run p50s (ms) | Stock median / spread | Reference-control paired p50, p10, p90 | Candidate / exact delta | Candidate paired p50 speedup | p10 / p90 | Correct | Profiler delta | Decision |
|---|---|---|---|---|---:|---|---|---|---|
| M16 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| M32 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| prefill | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

A candidate needs a paired rank-max p50 gain of at least 3% unless the committed
evidence establishes a tighter local noise floor. No enabled bucket may regress.
Each attempt's hypothesis, exact delta, expected effect, correctness,
distribution, profiler evidence, risk, decision, and rollback point belongs in
[attempt_ledger.md](attempt_ledger.md).

## Containing-region and end-to-end acceptance

| Gate | TP4 diagnostic | TP8 production | Status |
|---|---|---|---|
| Single-GPU production packed-ABI O projection | advisory producer evidence only | insufficient alone | PENDING |
| Real producer -> AllReduce -> real consumer | direct synthetic consumer is insufficient | required | BLOCKED EXTERNALLY |
| Resolved model caller frequency and graph replay | direct harness is insufficient | required | BLOCKED EXTERNALLY |
| Three identical server runs and end-to-end improvement | not production acceptance | required | BLOCKED EXTERNALLY |

## Exact enable and fallback policy

1. The current production policy is stock SGLang for every TP4 and TP8 bucket.
   The campaign exports `SGLANG_GLM52_OPT=0`; no branch-local dispatch oracle or
   backend threshold change is enabled.
2. `SGLANG_ALL_REDUCE_TRACE` is diagnostic-only and must remain unset in normal
   performance or production runs.
3. A TP4 candidate may be described as a diagnostic result only after exact
   correctness, alias/poststate, graph, stream, paired-latency, and profiler
   checks pass for that exact operator x M x ABI x topology bucket.
4. Any losing, unsupported, untraced, or untested bucket stays on the stock
   implementation without a device-to-host dispatch read, host synchronization,
   or per-call environment mutation.
5. TP8/DP8/EP8 remains stock until the external trace, three stock baselines,
   paired candidate gate, real containing-region validation, and three identical
   end-to-end runs all pass. TP4 evidence cannot enable a TP8 branch.
6. If no deployable bucket passes every applicable gate, the final disposition
   is **No replacement** and all stock behavior remains active.

## Final disposition

**PENDING.** Select exactly one COMMON_RULES disposition after the evidence is
complete:

- **Production win** only if an externally validated production bucket passes
  correctness, >=3% paired p50, graph/overlap, containing-region, and end-to-end
  gates with stock fallback elsewhere.
- **No replacement** if local characterization and justified configuration or
  source attempts show no deployable gain, or if production acceptance remains
  unavailable. State the binding limit and every external blocker.

## Artifact index

Populate after all measurements finish and immutable artifacts are copied from
the fresh `/tmp/tp_allreduce_reachability_*` campaign root:

- Campaign status and analyzer summary: PENDING
- Environment/topology/P2P/NVLink logs: PENDING
- Per-rank reachability JSONL and result JSON: PENDING
- Semantics, baseline, reference-control, and candidate result JSON: PENDING
- Backend-scout and producer-ABI logs/results: PENDING
- Nsight Systems reports and exported stats: PENDING
- Final reviewed attempt ledger and profile analysis: PENDING
