# GLM-5.2 B200 indexer `wq_b` decode — final report

## Disposition

**No replacement.** The strongest standalone kernel was faster at M16 but
changed the post-RoPE/quant FP8 query and sparse top-k. The numerically exact
stock-DeepGEMM SM-budget experiment then failed every long source-integrated
full-region performance series. SGLang is restored to its stock path and no
bucket is enabled.

This satisfies the shared-rule no-replacement outcome: reachability, three
stock baselines per bucket, Nsys/NCU/PTX/SASS characterization, native packed
candidate attempts, exact consumer checks, CUDA Graph/full-region tests, and
an explicit fallback policy are preserved.

## Reachability and production ABI

| Property | Evidence |
|---|---|
| SGLang owner | `Indexer._fused_q_prepare_and_store` |
| Checkpoint prefix | `model.layers.3.self_attn.indexer.wq_b`; source constructs it with `add_prefix("wq_b", prefix)` |
| Projection | local M16 or M32 × K2048, weight N4096 × K2048, BF16 output before query preparation |
| Quantized inputs | activation and weight `float8_e4m3fn` |
| Scale ABI | column-major/TMA-aligned packed `int32` UE8M0; activation `[M,4]`, weight `[4096,4]` |
| Stock callable | `w8a8_block_fp8_matmul_deepgemm` through the FP8 `ReplicatedLinear` method |
| Stream contract | `wq_b` is issued on `alt_stream` while BF16 `wk_weights_proj` runs on the current stream; fused Q and fused K/cache-store form the second overlap stage |
| Graph contract | both fixed decode buckets are captured and replayed as CUDA Graphs; dispatch may not perform device-to-host work |
| Topology | projection is rank-local/replicated; production acceptance remains TP8/DP8/EP8 |

The production-shaped workload added in this goal constructs the real
`Fp8Config`/`ReplicatedLinear`, packed scales, `LayerNorm`, RoPE wrapper,
page-64 index-K cache accessor, alternate CUDA stream, and invokes the exact
SGLang method. It checks the query, head weights, written K plane, and written
scale plane with alternating cache poison.

## Frozen stock reference

All stock measurements used `SGLANG_GLM52_OPT=0` on physical GPU 0,
UUID `GPU-30b619de-87f2-1862-0d07-a595da8fe417`, B200/SM100, Torch
2.11.0+cu130, CUDA 13.0, Kernel-Harness base `bcd005409e65786`, and SGLang base
`f93f8867b4bc`.

| Bucket | eager stock median, run 1 | run 2 | run 3 | median of run medians |
|---|---:|---:|---:|---:|
| M16 | 0.046048 ms | 0.045360 ms | 0.046176 ms | 0.046048 ms |
| M32 | 0.054512 ms | 0.044992 ms | 0.046496 ms | 0.046496 ms |

The first M32 series is visibly noisy and is retained rather than discarded.
Performance decisions use same-process alternating pairs, not these independent
stock runs.

## Characterization

The stock DeepGEMM kernel is a one-wave, 148-CTA, 256-thread SM100 kernel with
37 registers/thread, 231,212 bytes shared memory/CTA, a 128×128×128 tile,
12 stages, and two-CTA cooperative `UTCQMMA`/TMEM plus `UTMALDG` TMA traffic.
NCU reports zero local or shared spilling.

| NCU metric | M16 | M32 |
|---|---:|---:|
| profiled kernel duration | 10.656 us | 10.496 us |
| DRAM throughput | 800.144 GB/s | 815.488 GB/s |
| DRAM peak percentage | 10.466% | 10.688% |
| SM peak percentage | 3.456% | 6.601% |
| grid waves/SM | 1.0 | 1.0 |

NCU replay overhead makes its duration larger than the approximately
6.4–6.5 us Nsight Systems duration, so the two clocks are not mixed in a
speedup claim. The low DRAM and SM percentages, one-wave grid, and barrier/long
scoreboard samples identify a tiny launch/synchronization-dominated GEMM rather
than a bandwidth-saturated one. Its optimization leverage is primarily fusion
or reliable overlap, not more arithmetic throughput.

## Attempts

### Native packed Triton

The strongest existing skinny-M idea was adapted to the actual N4096/K2048
shape and consumed packed UE8M0 directly, with no scale adapter.

| Bucket | graph reference | graph candidate | paired p10 | paired p50 | paired p90 |
|---|---:|---:|---:|---:|---:|
| M16 | 0.010672 ms | 0.008800 ms | 1.1020× | 1.2018× | 1.3569× |
| M32 | 0.011296 ms | 0.011360 ms | 0.9189× | 0.9858× | 1.0839× |

At M16, NCU measured 8.416 us, 1.009 TB/s, and 13.305% SM peak for a
256-CTA, 64-thread, 48-register, no-spill classic-HMMA kernel. M32 measured
9.984 us. This is a real standalone M16 improvement, but it is not deployable:

- BF16 GEMM output differed in 4/65,536 elements (maximum absolute difference
  1.0).
- After fused RoPE/quant, query FP8 differed in 6/65,536 elements (maximum
  absolute difference 16.0) and failed the strict region gate.
- Real `deep_gemm.fp8_mqa_logits` over context 4096 differed in 2,044/65,536
  logits (maximum absolute difference 18.97161865234375).
- One of 16 rows changed three members of top-k 2048.

All block-N/warp/stage variants reproduced the same query failure. Even the
apparently fastest invalid region sample is rejected.

### Stock DeepGEMM graph SM budget

To preserve math, the second source experiment kept DeepGEMM and changed only
its graph-captured persistent grid: 48 SMs at M16 and 32 at M32. A short
50-pair portfolio suggested gains, so the exact fail-closed SGLang source hook
was implemented and then tested with three independent 200-pair series.

| Bucket | series | reference | candidate | paired p10 | paired p50 | paired p90 |
|---|---:|---:|---:|---:|---:|---:|
| M16 | 1 | 0.017152 ms | 0.017344 ms | 0.9226× | 0.9907× | 1.0433× |
| M16 | 2 | 0.016928 ms | 0.017488 ms | 0.8983× | 0.9761× | 1.0704× |
| M16 | 3 | 0.017824 ms | 0.018016 ms | 0.8956× | 0.9913× | 1.0836× |
| M32 | 1 | 0.017360 ms | 0.017680 ms | 0.9152× | 0.9855× | 1.0316× |
| M32 | 2 | 0.017328 ms | 0.017600 ms | 0.9340× | 0.9858× | 1.0466× |
| M32 | 3 | 0.017216 ms | 0.017536 ms | 0.9134× | 0.9845× | 1.0424× |

Every series recorded the exact source dispatch hit and passed eager plus graph
replay correctness with cache poison. None reached 1.0×, much less the required
1.03×.

Nsight Systems confirmed that the candidate graph captured the intended 48-CTA
M16 and 32-CTA M32 DeepGEMM launches. It also explained the failure:

- M16 graph-node span grew from approximately 17.44 us stock to 21.44 us
  candidate; the BF16 GEMM began only after a roughly 4.13 us post-`wq_b` gap
  in the candidate trace.
- M32 graph-node span grew from approximately 19.10 us stock to 20.34 us
  candidate.

Reducing the persistent grid did not reliably cause graph-node concurrency.
The exploratory short-run gain was graph-scheduler noise and is superseded by
the long source-integrated series.

## Containing region, consumers, and end to end

The production-shaped containing region covers the two GEMMs, both stream
waits, fused Q RoPE/quant, fused K norm/RoPE/cache store, page-64 cache layout,
and poisoned-output checks. The final exact experiment passed those semantics
but regressed both buckets. Because it failed the full-region gate, it was not
eligible for score/top-k/attention or complete-server promotion.

The rejected Triton path was nevertheless driven through the real score/top-k
consumer, which caught the sparse-index change described above. The exact
DeepGEMM experiment returns the same query, weights, and K/cache bytes, so it
cannot change those downstream values; it was rejected solely on latency.

An exclusive four-B200, four-rank diagnostic then ran the exact source hook as
independent rank-local replicas and reduced every sample with the maximum
rank latency. Eager and graph outputs were bitwise equal on all ranks.

| Bucket | rank-max reference | rank-max candidate | paired p10 | paired p50 | paired p90 |
|---|---:|---:|---:|---:|---:|
| M16 | 0.021168 ms | 0.021136 ms | 0.8749× | 1.0168× | 1.1589× |
| M32 | 0.021008 ms | 0.022944 ms | 0.8170× | 0.9236× | 1.0412× |

The M16 median remains below the required 1.03× and its lower tail regresses;
M32 regresses decisively. This diagnostic therefore independently supports
the no-replacement decision, but it is not TP8 acceptance.

A complete SGLang checkpoint/server end-to-end run was unavailable: this host
has no complete GLM-5.2 weight shards, and only four B200s are exposed. No
NVFP4/BF16 model was substituted for the fixed packed-FP8 target, and no
four-rank diagnostic is relabeled as the required TP8/DP8/EP8 acceptance.
These limitations do not leave a candidate awaiting promotion: all candidates
already failed an earlier mandatory gate.

## Final source and fallback

The source-integrated experiment is archived as
`../indexer-wq-b-sms-integrated-20260723/source/rejected_sglang_sms_budget.patch`.
It is not installed. The final SGLang worktree is stock at the goal’s base plus
no goal-specific commit.

The Kernel-Harness production workload, candidates, scripts, raw reports, and
this analysis are retained and committed locally. Production selection remains
stock for M16, M32, eager, graph, and all unsupported shapes/ABIs/topologies.

## Final validation

After restoring SGLang to its exact base, the structural task selftest reported
24 tasks and no problems, the serving-native selftest reported 41 workloads,
and knowledge lint/index/distill checks passed. The GPU-aware harness verifier
confirmed all 24 generated task directories are in sync and reported no
invalid or provisional frozen results; its normal pointer audit retained the
pre-existing advisory that `runs/index.jsonl` is absent.

Fresh M16 and M32 stock production-region smoke runs completed on the restored
tree. The archived Triton region candidate also compiled and ran without its
former SGLang source module, then failed at the expected strict
`output.q_fp8` check with maximum absolute difference 16.0. This proves the
negative candidate is reproducible while the live SGLang tree remains stock.

## Evidence index

- Stock runs, Nsys, NCU, JIT CUDA/PTX/cubin/SASS:
  `../indexer-wq-b-stock-v4-20260723/`
- Native packed Triton runs, Nsys, NCU, TTIR/TTGIR/LLVM/PTX/cubin/SASS:
  `../indexer-wq-b-packed-v2-20260723/` and
  `../indexer-wq-b-packed-graph-v2-20260723/`
- Full-region correctness, top-k consumer, tile and SM portfolios:
  `../indexer-wq-b-region-correctness-20260723/`
- Rejected Triton source integration:
  `../indexer-wq-b-sglang-v2-20260723/` and
  `../indexer-wq-b-production-graph-v3-20260723/` through `v5`
- Final source-integrated 200-pair series, hit counters, and Nsys:
  `../indexer-wq-b-sms-integrated-20260723/`
- Four-rank diagnostic (separately labeled):
  `../indexer-wq-b-sms-tp4-diagnostic-20260723/`
- Machine-readable paired summary and attempt ledger:
  `paired_summary.json`, `attempt_ledger.json`
- Exact fallback and external-gate status:
  `FALLBACK_POLICY.md`, `EXTERNAL_VALIDATION_BLOCKER.md`
- Final verifier and post-restore smoke logs:
  `validation/`
