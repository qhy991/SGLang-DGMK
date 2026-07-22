# External TP8 production acceptance blocker

Status: **BLOCKED ON AN EXTERNAL EIGHT-GPU MODEL HOST.** This is a validation
boundary, not permission to weaken or relabel the production gate.

## Locally established constraints

- The user-provided host contract exposes physical GPUs 0-3 only. The local
  scheduler wrapper therefore supports a four-rank diagnostic lane, not a TP8,
  DP8, or EP8 production run.
- `/mnt/OS-oKqEXySb/models/GLM-5.2-NVFP4` exists but is empty on this host. There
  is no local checkpoint/config/tokenizer from which to resolve the actual model
  architecture revision, launch arguments, caller frequency, or end-to-end
  behavior.
- The direct TP4 serving-native runner can validate the coordinator, selected
  backend, exact reduction, alias/poststate, graph, stream, and synthetic
  full-tensor consumer contracts. It cannot prove that a GLM model call site
  reaches that ABI or replace the real containing-region/server gate.

Do not download or substitute a different checkpoint, reduce TP/DP/EP from eight
to four, divide decode M by data-parallel size, or cite TP4 evidence as TP8.

## Requirements blocked locally

- [ ] Load and record the exact GLM-5.2 checkpoint/config/tokenizer identities.
- [ ] Record fully resolved TP8/DP8/EP8 launch flags, environment, package/import
  paths, B200 UUIDs/clocks, full-NVLink topology, and P2P capability.
- [ ] Trace every material AllReduce caller with exact phase/bucket frequency,
  group ranks, dtype, shape/stride/bytes, graph/replay state, stream, selected
  communicator/backend/algorithm, decision predicates, alias/poststate, and real
  following consumer.
- [ ] Distinguish AllGather, DeepEP/A2A, and successful fused paths from actual
  `GroupCoordinator.all_reduce` fallback hits.
- [ ] Add separately named TP8 serving-native workloads only for trace-proven
  ABIs; never overwrite the TP4 diagnostics.
- [ ] Capture three uncontended TP8 stock baselines using rank-max latency.
- [ ] Run exact alternating paired candidate comparisons for every proposed
  operator x M x ABI x topology bucket; require >=3% p50 and no enabled
  regression.
- [ ] Validate graph replay, streams, allocator/alias semantics, and the real
  producer-AllReduce-consumer containing region.
- [ ] Run three identical candidate servers and show containing-region and
  end-to-end improvement.

## Preserved gate

The exact server and workload-generator commands are preserved under
`Preserved external TP8 production gate` in [commands.md](commands.md). They
retain:

- `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`;
- `--tp-size 8 --dp-size 8 --ep-size 8`;
- `--enable-dp-attention` and the saved DeepEP/A2A deployment contract;
- `SGLANG_GLM52_OPT=0` for the stock reference;
- distinct prefill, decode-M16, and decode-M32 requests;
- trace-first creation of any TP8 microbenchmark; and
- three stock baselines, exact paired >=3% gate, containing-region validation,
  and three identical end-to-end candidate runs before promotion.

The external operator must run those commands under that host's exclusive
eight-GPU scheduler and retain all raw outputs, source SHAs, trace files, and
profiles. The local four-GPU wrapper is not an eight-GPU scheduler.

## Fallback while blocked

Stock SGLang remains active for every TP8/DP8/EP8 bucket. No TP8 workload,
message-size oracle, threshold change, or candidate dispatch may be enabled from
local TP4 evidence. `SGLANG_ALL_REDUCE_TRACE` remains unset outside bounded
diagnostic runs. If local TP4 work finds no deployable result, or if external
acceptance cannot be completed, the honest disposition is **No replacement**.
