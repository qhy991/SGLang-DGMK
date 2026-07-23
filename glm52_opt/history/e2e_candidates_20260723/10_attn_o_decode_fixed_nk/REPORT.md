# GLM-5.2 attention O-projection decode baseline

**Disposition: NO REPLACEMENT; stock fallback remains active for M16 and M32.**

Fixed-N/K DeepGEMM is a real device-kernel and CUDA-graph win on the production
packed ABI, but the existing SGLang integration loses in all six eager paired
sessions. The complete GLM-5.2 server and TP8/DP8/EP8 acceptance gates are
unavailable because this host has no model weights and only four GPUs. A
graph-only microbenchmark win cannot waive those gates, so no production
allowlist or SGLang source is changed.

## Exact enable/fallback policy

| Bucket / condition | Candidate enabled | Active path |
|---|---:|---|
| decode M16, packed int32 UE8M0 | no | stock `w8a8_block_fp8_matmul_deepgemm` |
| decode M32, packed int32 UE8M0 | no | stock `w8a8_block_fp8_matmul_deepgemm` |
| eager, graph capture miss, other M/ABI/dtype/topology | no | stock SGLang |

The default `serving_safe` decode registry stays empty without an explicit
operator allowlist. There is no host readback, adapter, pack/copy kernel, or
timer change.

## Production reachability

The current path is:

```text
Glm4MoeAttention.forward_core
  -> RowParallelLinear(o_proj)
  -> Fp8LinearMethod.apply
  -> dynamic FP8 activation quantization with packed int32 UE8M0 scales
  -> w8a8_block_fp8_matmul_deepgemm
  -> deep_gemm.fp8_gemm_nt
```

The local shapes are `[M,16384] x [6144,16384]^T` for independently fixed
M=16 and M=32. Output is `[M,6144]` BF16. The exact call trace, source anchors,
scale shapes/strides, and hit-counter evidence are in
[`analysis/source_reachability.md`](analysis/source_reachability.md).

## Production versus frozen synthetic

The protocols differ, so absolute latency is shown for provenance and only
within-row paired speedup is interpreted.

| Lane / protocol | M16 candidate / reference | M32 candidate / reference | Meaning |
|---|---:|---:|---|
| frozen f32-scale task, CUPTI cold-L2 kernel | 33.160 / 52.729 us, 1.590x | 34.128 / 54.017 us, 1.583x | historical packed candidate versus slower synthetic reference |
| exact packed ABI, eager paired CUDA event | 46.624 / 67.536 us, 1.428x | 53.056 / 67.840 us, 1.305x | direct fixed-N/K library call |
| integrated SGLang dispatch, eager paired | 72.656 / 71.712 us, 0.987x | 75.504 / 73.200 us, 0.957x | dispatch overhead vetoes promotion |

The frozen result passed correctness but is only mismatch evidence. Its two
`result.json` audits are `PROVISIONAL`: the environment contained this
uncommitted evidence tree; the archived packed candidate itself was clean.

## Three paired production-ABI sessions

Each session has five alternating warmups and 30 alternating reference/candidate
pairs. The reference is always OPT0.

| Candidate | Bucket | Session paired medians | Sessions >=1.03 | Worst session p10 |
|---|---:|---|---:|---:|
| identity | M16 | 0.999x / 0.980x / 0.989x | 0/3 | 0.780x |
| identity | M32 | 1.001x / 0.999x / 1.009x | 0/3 | 0.886x |
| direct fixed N/K | M16 | 1.425x / 1.428x / 1.524x | 3/3 | 1.140x |
| direct fixed N/K | M32 | 1.305x / 1.262x / 1.307x | 3/3 | 1.122x |
| integrated dispatch | M16 | 1.012x / 0.987x / 0.886x | 0/3 | 0.624x |
| integrated dispatch | M32 | 0.957x / 0.915x / 0.976x | 0/3 | 0.742x |

All comparisons are internally paired on one wrapper-selected physical GPU.
No unpaired latency from different GPUs is used as a speedup. The raw JSON and
logs are under [`benchmarks/`](benchmarks/), with a regenerated concise table in
[`analysis/paired_summary.csv`](analysis/paired_summary.csv).

## Components and containing layer

| Component median | M16 | M32 |
|---|---:|---:|
| stock low-level total | 63.456 us | 64.384 us |
| direct default DeepGEMM total | 50.320 us | 50.880 us |
| direct fixed-N/K total | 40.784 us | 47.920 us |
| preallocated stock GEMM | 46.720 us | 46.064 us |
| preallocated fixed-N/K GEMM | 36.720 us | 43.248 us |
| integrated GLM-5.2 dispatch | 61.856 us | 70.704 us |
| stock quant + O projection | 116.896 us | 118.736 us |
| candidate quant + O projection | 114.960 us | 120.224 us |

Scale-layout passthrough and same-dtype output view produce no device kernel;
their event timings are measurement-floor/host overhead, not adapter work.
Graph-node traces prove the candidate layer has exactly the same quant kernel
plus one specialized GEMM.

Native CUDA Graph capture and replay pass with max absolute error zero:

| Bucket | Session 1 stock -> candidate | Session 2 stock -> candidate | Speedups |
|---|---:|---:|---:|
| M16 | 35.824 -> 26.064 us | 37.056 -> 27.536 us | 1.374x / 1.346x |
| M32 | 36.784 -> 33.936 us | 36.512 -> 31.120 us | 1.084x / 1.173x |

This is real `Fp8LinearMethod.apply` with an `o_proj` prefix, including dynamic
activation quantization. It is not relabeled as full attention or server e2e.

## Profiler conclusion

Nsight Systems sees one 148-block x 256-thread DeepGEMM launch per low-level
call. Stock kernel medians are 28.208/28.448 us at M16/M32; fixed N/K records
19.104/25.760 us. Captured layer replay contains quantization followed by GEMM
on one stream, with no adapter/copy.

NCU shows about 102 MB read in every case, one wave, one block/SM, 12.5%
theoretical occupancy, and only ~0.078 eligible warps/scheduler/cycle. Long
scoreboard and barrier waits dominate. Fixed N/K does not reduce bytes; it
reduces executed instructions 15.20%/15.72% and active cycles 19.89%/14.45%,
raising effective DRAM-read bandwidth.

Decoded SASS falls from 2,200 to 1,072 instructions and registers from 37 to
36. Exact configurations, cubins, disassembly, NCU metrics, and the full
six-dimension diagnosis are in
[`analysis/kernel_config_and_sass.md`](analysis/kernel_config_and_sass.md) and
[`analysis/profiler_diagnosis.md`](analysis/profiler_diagnosis.md).

## Experiment decision

The smallest candidate changes only DeepGEMM's compile specialization to
`compiled_dims="nk"` and accepts the caller's packed scales directly. It clears
the direct and graph gates with exact output equality. The already-present
SGLang dispatch correctly reaches it without device adapters, but its Python
lookup/allocation/hit-accounting cost is paid by eager execution and causes
repeatable integration losses. Enabling it would therefore broaden a
graph-replay win into unvalidated eager and capture-miss regressions.

The complete hypothesis, expected code effect, paired distributions, risk,
decision, and rollback point are in
[`analysis/experiment_ledger.md`](analysis/experiment_ledger.md). The external
candidate diff is
[`analysis/candidate_source.patch`](analysis/candidate_source.patch); SGLang has
no source diff.

## External acceptance boundary

The known GLM-5.2 model directory is empty, and the partial Hugging Face cache
contains only configuration files. This host exposes four B200s while
production acceptance is TP8/DP8/EP8. O projection is rank-local under DP
attention and has no independent four-rank collective, so duplicating the GEMM
on four ranks would add no contract and is not substituted for the eight-rank
server gate.

See [`analysis/external_validation_blockers.md`](analysis/external_validation_blockers.md).
Stock remains active until complete-server alternating decode proves the
execution-mode mix and true topology.

## Validation and artifact map

Completed checks:

- Kernel-Harness structural selftest: 24 tasks, 0 problems.
- `serving_native` selftest: 39 fixed workloads.
- Nine SGLang registry plain-assert tests passed in the required venv (pytest is
  not installed there).
- Production packed-ABI eager and graph correctness passed for M16/M32.
- Nsys and NCU raw reports are complete.
- Frozen synthetic results were audited and their provisional status is
  disclosed.

The final GPU-aware environment and harness verifier passed after
documentation/knowledge generation: 24 task contracts, 13 knowledge entries,
task projection sync, two valid provisional-result audits, and all pointer
checks. Its exact output is
[`analysis/final_gpu_validation.txt`](analysis/final_gpu_validation.txt). The
complete validation matrix is
[`analysis/validation_matrix.md`](analysis/validation_matrix.md).

Key artifacts:

- raw paired/component/graph results: [`benchmarks/`](benchmarks/);
- raw `.nsys-rep`/`.ncu-rep`: [`reports/`](reports/);
- selected generated DeepGEMM sources/cubins: [`configs/`](configs/);
- machine-readable summaries: [`analysis/summary.json`](analysis/summary.json);
- exact dependencies and hashes:
  [`analysis/dependency_identity.txt`](analysis/dependency_identity.txt);
- GPU allocation record:
  [`analysis/wrapper_allocations.txt`](analysis/wrapper_allocations.txt).

Only reproducible JIT cache expansions and derived Nsys SQLite exports were
removed; exact selected cubins, raw profiler reports, and exported summaries are
preserved. No frozen oracle/generated task file, existing knowledge entry,
remote branch, or installed package was modified.
