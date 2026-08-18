# GLM-5.2 CP8 packed MLA-KV communication

Status: **STOP; keep default-off.** The two-replicate 8-GPU operator gate and
the exact-input serving correctness gate pass, but the matched concurrency-11
TTFT screen does not meet the replacement boundary.

## Objective

For the B300 GLM-5.2 prefill cell below, replace one per-layer communication
boundary without changing the cache ABI or attention math:

- TP8 / DP1 / attention CP8 / effective attention TP1 / MoE EP8;
- 90,000 logical prefix tokens, of which 89,984 are page-aligned cache hits;
- 10,000 logical suffix tokens, 10,016 actually uncached tokens, and 10,048
  scheduled rows per attention-CP rank after runtime padding;
- 10,048 local rows and 80,384 global rows in each measured layer wave;
- FlashMLA-KV with FP8 MLA KV cache;
- eager ordinary prefill, DCP disabled.

## Invariants

The candidate preserves the exact row-wise cache representation:

```text
512 FP8 latent bytes + 4 FP32 scales (16 bytes) + 64 BF16 RoPE values (128 bytes)
= 656 bytes/token
```

Quantization is row-local, so it commutes with rank-major AllGather and the CP
rerange. The candidate must produce byte-identical packed rows, write the same
`out_cache_loc`, and make the attention backend skip its duplicate store.
Unsupported shapes, devices, modes, backends, cache layouts, or parallel
topologies return to the unchanged BF16 path before mutation.

## Dataflow

Baseline:

```text
local BF16[10048,576]
  -> BF16 AllGather+rerrange [80384,576]
  -> global quantize/pack
  -> paged FP8 cache write
```

Candidate:

```text
local BF16[10048,576]
  -> local quantize/pack [10048,656] uint8
  -> uint8 AllGather+rerrange [80384,656]
  -> direct paged FP8 cache write
```

Per-rank communicated input falls from 11,575,296 to 6,591,488 bytes
(`-43.0556%`). The producer, collective, and cache consumer remain a single
admission unit; no isolated quant or collective number may promote the change.

## Admission gate

`SGLANG_GLM52_CP8_PACKED_KV_COMM=1` is necessary but not sufficient. Runtime
admission also requires:

- NVIDIA SM103;
- GLM-5.2 dimensions H6144, H64, Q-LoRA2048, KV-LoRA512, QK 192+64, V256;
- CP size 8 and DCP disabled;
- ordinary non-speculative extend outside CUDA-graph capture;
- `flashmla_kv` and a DSA FP8 cache exposing the packed writer;
- BF16 K-nope/K-RoPE tensors with 512/64 last dimensions;
- global `out_cache_loc` rows equal to `local_M * 8`.

The environment default is false.

## Operator evidence

Two independent B300 8-rank runs used balanced BC/CB ordering, eight warmups,
30 recorded pairs, rank-max CUDA-event time, and bytewise correctness:

| replicate | BF16 baseline | packed candidate | median speedup | paired p10 |
|---|---:|---:|---:|---:|
| 1 | 0.514016 ms | 0.259904 ms | 1.9671x | 1.6495x |
| 2 | 0.515808 ms | 0.263296 ms | 1.9629x | 1.9066x |

The sibling Indexer-K experiment reduced payload by 48.4375% but regressed the
complete boundary to 0.8744x and 0.8550x in two replicates. It is deliberately
absent from the integration patch.

Artifacts on B300-M2:

```text
/mnt/b300-shared/home/qinhaiyan/wwxq/bench_results/
sglang_kda_to_glm52_cp8_migration_20260818/
```

The final candidate runtime contract SHA256 is
`ea0ba5deebebb3240cf07447c1b6eb5105b1cc1b8b3e51af87f8bcc34dec25fe`.

## Serving result and replacement boundary

The exact-input x1 diagnostic passed with 10/10 requests and outputs. Every
rank recorded ten `cached=89984/new=10048` lines, all eight ranks selected the
packed path at `local_M=10048`, and the ten output token IDs exactly matched
the control.

The final no-profiler x11 screen used 110 requests per repeat and three repeats
per arm. Every repeat had 110/110 responses and output tokens, 8x110 exact
cache/shape evidence, the same input/sentinel hashes, and the same 110-token
output sequence across control and candidate:

| arm | repeat | P50 TTFT | P90 TTFT |
|---|---:|---:|---:|
| control | 1 | 3768.749 ms | 3818.241 ms |
| control | 2 | 3742.176 ms | 3753.764 ms |
| control | 3 | 3724.065 ms | 3748.716 ms |
| candidate | 1 | 3747.452 ms | 3833.407 ms |
| candidate | 2 | 3727.892 ms | 3820.917 ms |
| candidate | 3 | 3709.438 ms | 3720.459 ms |

The cross-repeat medians are:

| metric | control | candidate | candidate change |
|---|---:|---:|---:|
| P50 | 3742.176 ms | 3727.892 ms | -14.284 ms (-0.3817%) |
| P90 | 3753.764 ms | 3820.917 ms | +67.154 ms (+1.7890%) |

The screen required at least 1% P50 improvement and no P90 regression. The
candidate fails both conditions, so the selector remains false by default and
the unchanged BF16 path remains the production path. This STOP is also robust
to a looser 2% P90 tolerance because the P50 gain remains below 1%.

The arms share source, input, container, and server-argument contracts, but
were run sequentially rather than as adjacent AB/BA fresh-server pairs. That
limits precision of the small observed deltas and precludes a positive claim;
it does not justify promoting a result that is below the P50 gate and unstable
at P90. Any future candidate that passes this screen still needs AB/BA pairs.

The fail-closed summary is stored on B300-M2 as:

```text
/mnt/b300-shared/home/qinhaiyan/wwxq/bench_results/
sglang_kda_to_glm52_cp8_migration_20260818/
cp8_packed_mla_kv_serving_ab_exact_x11.json
sha256 ef536a43e1e17b9016fa360638d408b1fec4601f9b7cf4ee59987962e58e7f9f
```

It verifies all of the following rather than accepting timing alone:

1. all eight ranks log candidate selection at the real local M;
2. exact-input x1 and x11 output tokens match the repaired control;
3. no fallback, CUDA/NCCL error, cache-shape error, or worker divergence;
4. every x11 repeat contains 110 TTFT samples and 8x110 cache/shape lines;
5. all candidate workers select `local_M=10048` and emit no rejection line.

Because the no-profiler screen failed, no candidate Nsight trace was used to
rescue the result after the fact. A future candidate would need a stronger
graph rewrite, such as eliminating the remaining packed concatenation/rerange/
store materializations, and must restart at the operator and serving gates.
