# GLM-5.2 CP8 direct packed MLA-KV communication

Status: direct-only candidate passed transport, operator, exact x1,
three-repeat x11, five adjacent fresh-server AB/BA pairs, delayed-rank/slot-
reuse stress, and matched causal Nsight. It is ready for isolated branch
promotion, but remains default-off and unmerged while the broader migration
matrix closes. Composition with combined-indexer passed a sequential screen
but still needs fresh pairs due one P90 outlier.

## Boundary

```text
control:
  local BF16[10048,576]
  -> NCCL rank-major AllGather
  -> zigzag split/cat rerange
  -> global row quantize/pack
  -> page-store uint8[80384,656]

packed-NCCL predecessor:
  local quantize/pack uint8[10048,656]
  -> NCCL rank-major AllGather
  -> zigzag split/cat rerange
  -> page store

direct candidate:
  local quantize/pack uint8[10048,656]
  -> SM103 multimem.st directly into final zigzag-global rows
  -> page store
```

Every rank owns two blocks of the one-sequence zigzag layout. Rank `r` writes
block `r` and block `15-r` directly into a single pre-rendezvoused symmetric
buffer. Thirty-two CTAs perform 128-bit multicast stores. Entry and exit
release/acquire barriers protect the one-slot lifetime; the page-store consumer
runs on the same stream before the next layer reaches the next entry barrier.

Primary SubCUDA classification: `Inline PTX`. The dataflow rewrite determines
the final row mapping, while the performance-carrying transport is the existing
Triton inline-PTX `multimem.st.relaxed.sys.global.v4.f32` plus system-scope
release/acquire atomics. It is not Direct PTX and introduces no authored PTX
module or ptxas configuration.

## Strict admission

`SGLANG_GLM52_CP8_DIRECT_PACKED_KV_COMM=1` raises, rather than falling back,
unless all of the following hold:

- NVIDIA SM103 and non-graph eager ordinary extend;
- GLM hidden/head/Q-LoRA/KV-LoRA dimensions exactly match;
- DSA + FlashMLA-KV + FP8 cache;
- attention CP8, DCP off;
- metadata batch size 1 with 16 zigzag blocks and exactly two blocks owned by
  the current rank;
- aligned contiguous uint8 `[M,656]` packed rows;
- final global rows fit the frozen 80,384-row symmetric allocation;
- `out_cache_loc` rows equal local M × 8;
- the torch symmetric-memory handle reports a non-zero multicast pointer.

The older packed-NCCL selector and the direct selector are mutually exclusive.

## Evidence to date

Transport precondition, eight B300 ranks:

- multicast pointer non-zero on every rank;
- 16 mutated-input iterations;
- safe/unsafe output and entry-sync on/off combinations byte-exact;
- no-clone multimem median speedup about 1.29–1.36× over NCCL for the transport
  precondition shape.

Real-shape direct scatter, two independent 30-pair replicates:

| replicate | packed NCCL+rerrange | direct | median speedup | p10 | correctness |
|---|---:|---:|---:|---:|---|
| 1 | 1.116896 ms | 1.001904 ms | 1.1054× | 1.0905× | all-rank byte-exact |
| 2 | 1.095984 ms | 0.988928 ms | 1.1085× | 1.0750× | all-rank byte-exact |

Exact x1 passed twice. The accepted diagnostic returned the same ten output
IDs as control and recorded 8×78 direct hits at local M10,048/global M80,384.

Three-repeat sequential x11 screen:

| arm | P50 repeats (ms) | P90 repeats (ms) | cross-repeat median |
|---|---|---|---|
| control | 3768.749 / 3742.176 / 3724.065 | 3818.241 / 3753.764 / 3748.716 | P50 3742.176 / P90 3753.764 |
| direct | 3517.275 / 3538.934 / 3555.957 | 3640.155 / 3557.350 / 3563.783 | P50 3538.934 / P90 3563.783 |

The observed latency reductions are 5.4311% at P50 and 5.0611% at P90. Every
arm has 110/110 success/output, matching input/sentinel hashes, the same
110-token sequence, and 8×110 cache/shape lines. This screen is positive but
sequential; fresh-server AB/BA remains the promotion authority.

Final runtime source contract:

```text
/mnt/b300-shared/home/qinhaiyan/wwxq/bench_results/
sglang_kda_to_glm52_cp8_migration_20260818/
runtime_source_contract_cp8_direct_packed_kv_final.json
sha256 0697adb967e7eba0bdf0c02f09eead414d1474fad5b368bd0ee1e1ef75852f93
```

The final contract differs from the five-pair runtime only by importing the
reviewed `q_rows==1252` combined-indexer guard while its selector is disabled;
the direct-only execution path is unchanged. It also explicitly includes the
tracked `triton_symm_mem_ag.py` Inline-PTX helper dependency.

Sequential serving receipt:

```text
cp8_direct_packed_mla_kv_serving_ab_exact_x11.json
sha256 ef68758e7f5f3a76d05071c8ad6fb9687b69a7bb173c74ed0eac01190a547313
```

## Promotion boundary

1. at least four of five fresh-server pairs win P50 — PASS, 5/5;
2. median paired P50 reduction at least 1% — PASS, 5.1071%;
3. at least four of five P90 pairs do not regress and median P90 does not
   regress — PASS, 5/5 and 5.4701%;
4. exact inputs, outputs, cache/shape, and 8×78 direct hits — PASS;
5. delayed-rank and one-slot reuse stress completes without deadlock or stale
   data — PASS, 24 epochs and five delayed-rank events;
6. matched Nsight attributes the improvement to removal of rank-major NCCL and
   rerange without a new critical-path stall — PASS; 6240 direct kernels,
   equal quant/store/sparse/MQA/TopK/DeepEP coverage, lower NCCL family;
7. selector stays default-off until all prior gates pass.
