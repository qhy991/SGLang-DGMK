# GLM-5.2 indexer score decode optimization report

Date: 2026-07-23

## Outcome

**No replacement.** Keep the stock SGLang DeepGEMM paged-MQA score backend for
both reachable decode buckets, M16 and M32.

The shipped SM100 CuTe-DSL alternative is numerically and graph correct, but it
does not clear the required `1.03x` threshold in every one of three paired
series at any decision-bearing scope:

| Scope | M16 median / minimum series | M32 median / minimum series |
|---|---:|---:|
| score + top-k, CUDA graph | 1.011231x / 1.003703x | 1.040749x / 0.977756x |
| complete indexer | 1.031918x / 1.014806x | 1.028806x / 1.015328x |
| indexer + selected DSA | 1.011807x / 1.011058x | 1.020334x / 1.019186x |
| four-rank diagnostic | 1.004793x / 1.002506x | 1.023810x / 1.017586x |

Every row fails the all-series gate. No candidate is integrated into SGLang,
and stock is the only enabled path.

The real-checkpoint SGLang end-to-end and TP8/DP8/EP8 gates remain externally
blocked by missing model files and a four-GPU host. They are not weakened or
relabeled. This does not conceal a candidate win: the alternative already
fails the necessary repeated rank-local and four-rank containing-region gates.

## Reached production path

The pinned config is
`nvidia/GLM-5.2-NVFP4@aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa`,
architecture `GlmMoeDsaForCausalLM`. Current SGLang source plus the runtime
fixture resolve normal CUDA decode as:

```text
DeepseekMLAForwardMixin
  -> Indexer.forward_cuda
  -> Indexer._fused_q_prepare_and_store
  -> Indexer._get_topk_paged
  -> deepgemm_paged_mqa_logits_split
  -> deep_gemm.fp8_paged_mqa_logits(clean_logits=False)
  -> topk_transform_512_v2
  -> FlashInfer TRT-LLM trtllm-gen DSA decode
```

`DSAPagedMQALogitsBackend.resolve("auto")` selects DeepGEMM on CUDA. Normal
decode has `next_n=1`, so it uses the split wrapper, not the target-verification
native path. Explicit `cutedsl` is the reached configuration alternative
tested here.

The fixed rank-local ABI is:

| Tensor/config | Contract |
|---|---|
| Q | FP8 E4M3 `[M,32,128]` |
| index K cache | fused uint8 `[M*128,64,1,132]` |
| gates | FP32 `[M,32]` |
| compact page table | int32 `[M,128]` |
| context / page | 8192 / 64 |
| scheduler metadata | int32 `[149,2]` |
| logits | FP32 `[M,8192]` |
| top-k | 2048 physical token slots per row |
| graph / SMs | decode CUDA graph / all 148 SMs at PP1 |

The configured init/local masks are zero, so masking is a no-op.
`clean_logits=False` is intentional because top-k-v2 consumes each row only
through its sequence length and maps logical positions through the compact
page table.

Normal decode never enters the short-extend logits skip. Model-level index
sharing is distinct: 21 of 78 producer layers run the indexer (layers 0, 1, 2,
then 6 through 74 every four layers), while 57 layers reuse carried top-k
indices.

## Exact workloads and correctness

The session added six named single-rank workloads:

- `indexer_score_decode_m16/m32`;
- `indexer_complete_decode_m16/m32`;
- `indexer_dsa_decode_m16/m32`.

It also added explicit `tp4_indexer_dsa_decode_m16/m32` diagnostics whose
contract says TP4/DP4 diagnostic only, not TP8/DP8 acceptance.

The complete-indexer fixture invokes the real unbound SGLang fused preparation
method with packed-int32 UE8M0 `wq_b`, BF16 `wk_weights_proj`, dual streams,
LayerNorm, interleaved RoPE, Q quantization, and current-token page-64 K-cache
store. The DSA fixture passes its exact physical top-k set to
`trtllm_batch_decode_with_kv_cache_mla`.

For both score buckets:

- CUDA-graph replay passes;
- maximum absolute logits difference is
  `2.384185791015625e-7`;
- every row has exact 2048-element physical-slot set equality;
- zero rows have a different set.

Top-k order may differ for tied or near-tied scores; DSA consumes the set, and
set equality is exact. All paired results additionally pass pre-timing,
post-timing, bidirectional reference-graph, and candidate-graph checks.
Containing-region checks cover current-token K-cache mutation and DSA output.

## Same-lease paired results

Each ratio is the median of same-process, alternating per-pair
`reference_ms / candidate_ms` values. The complete series and profiler
collection stayed in one wrapper lease on physical GPU 1
(`GPU-5b9be10b-5bfc-b658-9b31-f7ae8516dc54`).

### Score plus top-k

| Mode | Bucket | Three CuTe-DSL series | Decision |
|---|---:|---|---|
| graph | M16 | 1.019197x, 1.003703x, 1.011231x | reject |
| graph | M32 | 0.977756x, 1.040790x, 1.040749x | reject |
| eager | M16 | 0.819856x, 0.805730x, 0.802370x | reject |
| eager | M32 | 0.798817x, 0.780203x, 0.791694x | reject |

The one favorable M32 graph median reverses in its first series. Shape-only
dispatch cannot safely enable a bucket whose repeated result crosses both
sides of 1.0.

### Complete indexer and selected DSA

| Scope | Bucket | Three CuTe-DSL series | Decision |
|---|---:|---|---|
| complete indexer | M16 | 1.014806x, 1.031918x, 1.056648x | reject |
| complete indexer | M32 | 1.028806x, 1.032633x, 1.015328x | reject |
| selected DSA | M16 | 1.011807x, 1.014385x, 1.011058x | reject |
| selected DSA | M32 | 1.021669x, 1.019186x, 1.020334x | reject |

These are 60-pair bidirectional CUDA-graph series. The selected-DSA result is
the complete locally reproducible target region, not merely the score leaf.

### Four-rank diagnostic

The all-GPU wrapper held physical GPUs 0-3 for all 12 results. Latency is the
maximum CUDA-event time across ranks:

| Bucket | Three CuTe-DSL series | Decision |
|---|---|---|
| M16 | 1.004793x, 1.002506x, 1.018696x | reject |
| M32 | 1.023810x, 1.030381x, 1.017586x | reject |

This diagnostic is useful topology evidence but remains explicitly separate
from TP8 acceptance.

## Profiler diagnosis

NCU reports that DeepGEMM launches 148 CTAs of 384 threads, one CTA per B200
SM, using 168 registers/thread and 216.5 KiB shared memory/CTA. At M16/M32 it
reaches only 9.79/15.77% SM peak, 21.80/33.32% DRAM read peak, and
0.125/0.146 eligible warps per cycle. Long scoreboard accounts for 253/393
and 393/543 PC samples. There are no local-load/store spill instructions.

CuTe-DSL lowers resources to 80 registers and 112.625 KiB, but its grid is
still 148 CTAs, so it cannot use the theoretical second resident block. It
remains underfilled at 6.88/11.50% SM and 22.26/37.28% DRAM read peak.

One-iteration Nsys kernel sums show the whole score-to-top-k device sequence:

| Backend | M16 | M32 |
|---|---:|---:|
| DeepGEMM fill + score + top-k | 12.288 us | 13.696 us |
| CuTe-DSL fill + score + top-k | 12.032 us | 13.696 us |

At M16, score changes from 5.120 to 4.928 us while top-k alone is 5.888 us.
At M32, score changes from 6.464 to 6.240 us but total device time is
unchanged. NCU independently shows top-k as a 16/32-CTA, 9.760/10.112 us
launch with only 0.78/1.44% DRAM read peak.

The binding limit is therefore a small, one-wave paged gather plus separate
fill/top-k launches, not an HBM or tensor roof. The complete indexer adds
packed-FP8 `wq_b`, BF16 projection/reduction, Q/K preparation, and cache store.
Selected DSA adds about 16.1 us at M16 and 21.7 us at M32 in the instrumented
capture. The leaf saving is too small to survive those regions.

Source counters map the generated kernels only to `?:0`; the measured
resource, spill, pipe, and stall facts do not support inventing a
line-specific PTX/SASS rewrite. The full six-dimension analysis is in the
profile `REPORT.md`.

## Source and fallback state

- Kernel-Harness source base:
  `bcd005409e65786af82c86f621507ebef12b2766`
- decision-bearing score source:
  `6c86a8de3138dfdf883a5c47924f4fc1d0862abb`
- decision-bearing region source:
  `63f1bcf5a937b9e2aeeeff09d39934947c370882`
- decision-bearing TP4 source:
  `fbf2410c06135a8a29d7de2a5506b7806299a088`
- SGLang base/final:
  `f93f8867b4bc124c9809c9110ec7361ed11b6b4a`
- SGLang net diff: empty

No installed package was overwritten. Candidate compilation/setup is outside
timed replay. The source candidates are external dispatch adapters and remain
available for reproduction, but no SGLang code imports them.

The exact final policy is simple: AUTO continues to select DeepGEMM for M16,
M32, every graph/eager mode, and every topology. There is no candidate
allowlist. Unsupported cases remain stock by construction.

## External validation boundary

The configured model directory has no files. The pinned Hugging Face snapshot
has only a 15,517-byte config and 215-byte generation config, with no
tokenizer or weight shards. The host has four B200s, while production requires
eight ranks. The isolated venv also has no production DeepEP package.

Consequently a real-checkpoint SGLang decode baseline, complete 78-layer
end-to-end metric, and TP8/DP8/EP8 rank-max gate are unperformed and unpassed.
The exact blocker and required external lane are in
`external_validation_blocker.md`.

## Validation and artifact integrity

- corrected score campaign: `failures=0`, 24 paired results, four Nsys
  reports, twelve NCU reports;
- complete-indexer/DSA campaign: `failures=0`, 24 paired results and eight
  Nsys reports;
- TP4 diagnostic: `failures=0`, 12 paired results;
- all four nested artifact manifests verify;
- structural serving-native and harness tests pass;
- the append-only knowledge entry lints and generated indexes are current;
- SGLang remains clean at its recorded SHA.

The raw final check commands, output, and return codes are persisted in
`final_checks.json`. `verify_harness.py --json` returns `ok=true`; its normal
non-strict pointer audit retains the advisory that `runs/index.jsonl` is
missing because this serving-native campaign did not create frozen-task runs.

Serving-native results are not frozen-task `result.json` records, so
`audit_result.py` is inapplicable. Their runner/candidate hashes,
pre/post correctness, repository status, raw samples, and SHA-256 manifests
provide the serving-native audit trail.

The first campaign is retained but explicitly superseded because its profiler
commands failed and its graph capture order was not balanced.

## Deliverables

- reachability and ABI: `reachability.md`, `backend_validation.json`
- exact workload source: `serving_native/indexer_region.py`,
  `serving_native/{runner,workloads,selftest}.py`
- candidates:
  `serving_native/candidates/indexer_{score,region}_cutedsl.py`
- score evidence: `runs/20260723T113910Z/`
- complete region evidence: `region_runs/20260723T120153Z/`
- TP4 diagnostic: `tp4_runs/20260723T121417Z/`
- profiler artifacts:
  `profile/indexer-score-decode-20260723T113910Z/`
- paired table, attempt and fallback:
  `paired_results_summary.md`, `attempt_ledger.md`, `fallback_policy.md`
- provenance and validation:
  `source_provenance.md`, `validation_matrix.md`,
  `external_validation_blocker.md`
- append-only recipe:
  `testbench/knowledge/entries/glm52-production--indexer-score-decode--b200--20260723a.json`

No branch was pushed and no remote state was modified.
