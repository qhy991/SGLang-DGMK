# TP AllReduce profiler analysis

Status: **PRE-RUNTIME TEMPLATE — no Nsight result has been interpreted.** This
file must be completed from the locked campaign's `.nsys-rep`, exported stats,
result JSON, and topology/NVLink logs. A nonempty report alone is not evidence;
the expected measured-only NVTX range and rank work must be present.

## Artifact and range checks

| Bucket / implementation | Expected NVTX range | Result JSON | `.nsys-rep` | stats export | Range present | Rank work mapped |
|---|---|---|---|---|---|---|
| M16 stock | `serving_native/tp4_allreduce_decode_m16/cuda_graph/nondefault/reference` | PENDING | PENDING | PENDING | PENDING | PENDING |
| M32 stock | `serving_native/tp4_allreduce_decode_m32/cuda_graph/nondefault/reference` | PENDING | PENDING | PENDING | PENDING | PENDING |
| prefill stock | `serving_native/tp4_allreduce_prefill/eager/nondefault/reference` | PENDING | PENDING | PENDING | PENDING | PENDING |
| M16 ABI-compatible c10d, if any | `serving_native/tp4_allreduce_decode_m16/cuda_graph/nondefault/paired` | PENDING | PENDING | PENDING | PENDING | PENDING |
| M32 ABI-compatible c10d, if any | `serving_native/tp4_allreduce_decode_m32/cuda_graph/nondefault/paired` | PENDING | PENDING | PENDING | PENDING | PENDING |
| prefill ABI-compatible c10d, if any | `serving_native/tp4_allreduce_prefill/eager/nondefault/paired` | PENDING | PENDING | PENDING | PENDING | PENDING |

Rank 0 owns the process-tree NVTX range, with CPU-group barriers bracketing all
ranks. Confirm that the report contains child-rank CUDA/NCCL/custom work; do not
interpret a launcher-only range as a collective profile. For graph buckets,
pair the Python capture-time trace with stable replay kernel names because graph
replay does not re-enter the Python hook.

## Per-bucket attribution checklist

Complete every field separately for M16, M32, and prefill:

- [ ] Total measured NVTX duration and number of measured repeats agree with
  the result JSON.
- [ ] Host launch gaps and CUDA API time are separated from device execution.
- [ ] Communicator initialization, IPC registration, workspace allocation, JIT,
  and graph capture are identified as inside or outside the measured range.
- [ ] CPU-group and device/NCCL barriers or waits are identified and attributed.
- [ ] The trace-selected communicator/backend/algorithm is mapped to exact
  kernel names; fallback kernels are not mislabeled as the predicted route.
- [ ] Collective-only and full-consumer ready-region durations are distinguished.
- [ ] Producer/consumer stream waits and overlap are shown; lack of a Python
  replay event is not treated as absence.
- [ ] CUDA Graph replay count and stable kernel sequence are established for
  M16/M32.
- [ ] Kernel tail/straggler and rank-max behavior are described across all ranks.
- [ ] NVLink topology, P2P capability, and before/after counters are correlated
  without turning cumulative counters into per-call bandwidth claims.
- [ ] Any clone/copy/adapter allocation and traffic are charged to the candidate.
- [ ] Binding limit and remaining optimization headroom are stated.

## Stock characterization

| Bucket | Host gaps / APIs | Registration/setup | Barriers/waits | Communication/custom kernels | Consumer work | Graph replay / overlap | NVLink evidence | Binding limit |
|---|---|---|---|---|---|---|---|---|
| M16 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| M32 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| prefill | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

## Candidate delta

Only add a candidate row when the exact ABI gate persisted a correct paired
result. A failed in-place/out-of-place contract is a negative attempt, not a
profile omission.

| Bucket / attempt | Added/removed APIs and kernels | Registration or allocation delta | Barrier/wait delta | Communication duration delta | Consumer/overlap delta | NVLink delta | Paired distribution link | Interpretation |
|---|---|---|---|---|---|---|---|---|
| PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

## Device-code conditional lane

There is no device-code hypothesis in the pre-campaign branch. Nsight Compute,
ptxas resource reporting, and PTX/SASS inspection are therefore not claimed.
If a later attempt changes CUDA/PTX/Triton/device code or depends on memory
transactions, synchronization, occupancy, tail behavior, registers, spills,
instruction mix, vectorization, or tensor-core scheduling, add a dedicated NCU
report before promotion and record:

- exact kernel regex/name, invocation, shape, rank, report path, and source SHA;
- DRAM/L2 transactions and throughput, NVLink-relevant traffic where available;
- achieved occupancy, registers, shared memory, spills, warp stalls, and tail;
- synchronization/atomic/instruction mix and generated PTX/SASS mapping;
- ptxas resource output and a before/after explanation tied to the hypothesis.

## Conclusion

PENDING. State whether each shape is host/launch-, synchronization-,
communication-, memory-, or consumer-bound; whether an eligible existing backend
improves the complete ready region; and why the profile supports promotion or an
evidence-backed no-replacement decision.
