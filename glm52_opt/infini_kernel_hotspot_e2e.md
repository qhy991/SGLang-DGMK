# GLM-5.2 `infini_kernel` hotspot provider

## Outcome and safety boundary

This branch registers three decode hotspots at their real SGLang production
call sites. W13 and W2 now have concrete, source-scoped DeepGEMM
implementations; FlashMLA remains an external provider interface:

| User-facing operator | Registry op | Implementation | Exact local ABI | Nsys NVTX identity |
|---|---|---|---|---|
| FlashMLA sparse decode | `dsa_decode_attn` | external API-v1 provider | M=16/32, Q `[M,1,64,576]` BF16, FP8 KV page 64×656, top-k 2048 | `infini_kernel_glm52_flashmla_sparse_decode_fp8_topk2048` |
| MoE gate+up (fused W13) | `moe_gate_proj` | built-in post1 DeepGEMM BM32/2SM per-call config | E32, slab 1024, K6144, N4096, packed int32 UE8M0 | `infini_kernel_glm52_moe_w13_decode` |
| MoE down (W2) | `moe_down_proj` | built-in post1 DeepGEMM scoped BM16 per-call override | E32, slab 1024, K2048, N6144, packed int32 UE8M0 | `infini_kernel_glm52_moe_w2_decode` |

The profile is default-off and requires exactly one explicit
`SGLANG_GLM52_OPT_OPS` value per fresh process. W13/W2 are hypotheses for
end-to-end validation, not promoted production winners. Their candidate and
authoritative fallback are built from the same immutable DeepGEMM
`v0.1.4.post1` commit. An optimized CUDA/CuTe, CUTLASS, Triton, or inline-PTX
FlashMLA extension is still supplied as a Python provider module.

“FlashMLA sparse decode” is the semantic workload name here: the production
hook is `flash_mla_with_kvcache` with sparse physical indices and paged FP8 KV.
It is deliberately not the separate `flash_mla_sparse_fwd` ABI.

Selection is fail-closed:

- unrelated phases, speculative modes, unregistered M/expected-M buckets, and
  explicitly unsupported recipe/overlap paths return to the unmodified stock
  path before a candidate launch;
- after a supported no-overlap W13/W2 decode bucket reaches its target route,
  shape-family, dtype, stride, scale-layout, device, call-site, or layer-binding
  drift is fatal instead of silently measuring stock;
- a selected provider launches exactly once;
- selected artifact/provider initialization, launch, or return-contract errors
  are fatal and cannot fall through to stock;
- missing, multiple, or unknown `SGLANG_GLM52_OPT_OPS` values are fatal instead
  of silently running stock after a spelling error;
- a candidate run therefore cannot report a false hit after silently executing
  the baseline.

## Why these three

The default short-prompt/decode-heavy OPT0 Nsight Systems capture reports:

| Category | Summed GPU kernel time |
|---|---:|
| NCCL communication | 31.3% |
| DeepEP communication | 21.2% |
| DeepGEMM | 20.3% |
| MoE activation/quantization | 7.2% |
| DSA/FlashMLA | 5.1% |

Communication is 52.5% of summed kernel time, so compute-only microbenchmark
shares must not be read as full-server Amdahl shares. In the separate
communication-free GLM-5.2 layer benchmark, decode M16/M32 shows:

- FlashMLA DSA: about 13.6–13.7%;
- fused W13 plus W2: about 40% in total;
- attention O projection: about 13.3–14.3%.

The user-observed FlashMLA 15.5% / ~226 us and MoE ~17% / 120–220 us are
plausible for a different NVTX or layer window, but are not substituted for
the full-capture percentages above. The ranking is consistent: FlashMLA and
the two MoE GEMMs are the largest compute-side decode targets remaining after
communication, and have more end-to-end leverage than another small O-proj
round.

The exact production-ABI FlashMLA call was also profiled separately. Nsys saw
M16 main/combine/overlap/chain spans of about
17.5/12.5/4.1/25.9 us and M32 spans of about
25.1/9.8/4.0/30.9 us. NCU showed the main kernel at roughly 22.3/30.4 us,
168 registers/thread, about 232.7 KB launch shared memory, one CTA per SM,
only 0.21/0.29 eligible warps per scheduler-cycle, low L2 reuse, and dominant
long-scoreboard/barrier stalls. Thus a ~226 us range is not the same boundary
as one exact FlashMLA provider call and must not be used as its leaf baseline.

For 64k incremental prefill, the available whole-server capture is instead
dominated by DeepEP dispatch (77.9% DeepEP, including low-latency busy-wait),
with DeepGEMM 12.2% and DSA 2.8%. Prefill work must therefore measure wall time
and overlap, not optimize solely by summed-kernel percentage.

Local evidence:

- [short-decode whole-server Nsys categories](e2e_gpu_kern_categories.md);
- [64k incremental-prefill Nsys summary](nsys_prefill_64k_SUMMARY.md);
- [W2 BM16 leaf/profile history](history/e2e_candidates_20260723/07_moe_w2_decode_bm16/FINAL_REPORT.md);
- [why leaf gains can disappear under CUDA Graph replay](sglang_integration_cuda_graph_replay_analysis_zh.md).

## FlashMLA provider API

Only FlashMLA needs an out-of-tree provider. Set
`INFINI_KERNEL_API_VERSION = 1` and implement:

```python
INFINI_KERNEL_API_VERSION = 1
PROVIDER_INFO = {
    "name": "my-sm100-kernels",
    "git_commit": "<candidate-source-sha>",
    "build_id": "<cubin-or-wheel-sha256>",
}


def initialize(*, gpu_id: int | None) -> None:
    # Optional. Load the DSO, bind JIT caches, and warm exact variants here.
    # This runs after CUDA device assignment and before graph capture.
    ...


def flashmla_sparse_decode(
    *,
    q,
    k_cache,
    cache_seqlens,
    head_dim_v,
    tile_scheduler_metadata,
    num_splits,
    softmax_scale,
    indices,
    block_table,
    is_fp8_kvcache,
):
    # Same public contract as sgl_kernel.flash_mla.flash_mla_with_kvcache.
    # Return exactly (output, lse).
    ...
```

The module reference can be an importable module name or an absolute `.py`
path. A compiled extension can be imported by that provider and keep its
kernel implementation out of the SGLang tree.

Every callback must enqueue on PyTorch's current CUDA stream and be safe during
CUDA Graph capture. It must not synchronize the device, switch streams without
an explicit dependency, JIT-compile, or create hidden persistent allocations
on the hot path. Put module loading, cubin selection, and exact-shape warmup in
`initialize()`.

## Run one operator at a time

FlashMLA:

```bash
export SGLANG_GLM52_OPT=1
export SGLANG_GLM52_OPT_PROFILE=hotspot_candidates
export SGLANG_GLM52_OPT_OPS=flashmla_sparse_decode
export SGLANG_GLM52_OPT_M_BUCKETS='dsa_decode_attn:16|32'
export SGLANG_GLM52_HOTSPOT_MODULE=/absolute/path/provider.py
export SGLANG_GLM52_INFINI_KERNEL_NVTX=0
python -m sglang.launch_server \
  <unchanged GLM-5.2 arguments> \
  --dsa-decode-backend flashmla_kv
```

Fused W13 first materializes one post1 stock and one post1 candidate:

```bash
export DEEPGEMM_W13_BASE_REPO=/path/to/sgl-deep-gemm
export PYTHON_BIN=/path/to/sglang/python
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" \
  third_party/deepgemm_w13/build_variants.py --force

export SGLANG_GLM52_OPT=1
export SGLANG_GLM52_OPT_PROFILE=hotspot_candidates
export SGLANG_GLM52_OPT_OPS=moe_w13
export SGLANG_GLM52_OPT_M_BUCKETS='moe_gate_proj:16|32'
export SGLANG_GLM52_INFINI_KERNEL_NVTX=0

third_party/deepgemm_w13/run_with_stock.sh \
  python -m sglang.launch_server <unchanged GLM-5.2 arguments>
```

W2 first materializes its post1 stock/candidate overlay:

```bash
export DEEPGEMM_W2_BM16_BASE_REPO=/path/to/sgl-deep-gemm
export HARNESS_PYTHON=/path/to/sglang/python
third_party/deepgemm_w2_bm16/build_overlay.sh

export SGLANG_GLM52_OPT=1
export SGLANG_GLM52_OPT_PROFILE=hotspot_candidates
export SGLANG_GLM52_OPT_OPS=moe_w2
export SGLANG_GLM52_OPT_M_BUCKETS='moe_down_proj:16|32'
export SGLANG_GLM52_INFINI_KERNEL_NVTX=0

third_party/deepgemm_w2_bm16/run_with_exact_post1_stock.sh \
  python -m sglang.launch_server <unchanged GLM-5.2 arguments>
```

Before a checkpoint-backed server run, exercise the same registered entrypoint
and one CUDA Graph capture/replay with:

```bash
# Keep the W13 environment above.
third_party/deepgemm_w13/run_with_stock.sh \
  python scripts/glm52_moe_registration_smoke.py --op w13

# Or, in a fresh process, keep the W2 environment above.
third_party/deepgemm_w2_bm16/run_with_exact_post1_stock.sh \
  python scripts/glm52_moe_registration_smoke.py --op w2
```

This zero-input smoke is not a performance or full numerical benchmark. It
proves that SGLang binds the manifest stock, reaches the distinct candidate
through the production wrapper, captures it, replays without Python re-entry,
and matches stock on active rows. Both modes passed on a 148-SM B200 on
2026-07-29.

Aliases are normalized as follows:

```text
flashmla_sparse_decode -> dsa_decode_attn
moe_w13 / moe_gate_up  -> moe_gate_proj
moe_w2                 -> moe_down_proj
```

Use a fresh server process for each stock/candidate arm. For the W13 and W2
stock arm, use the same operator-specific launcher but unset
`SGLANG_GLM52_OPT`, `SGLANG_GLM52_OPT_PROFILE`, `SGLANG_GLM52_OPT_OPS`, and
`SGLANG_GLM52_OPT_M_BUCKETS`. This keeps the exact manifest stock, JIT/cache
layout, server command, and workload unchanged. Do not enable W13 and W2 in
one process: their separately attested stock artifacts are intentionally
tested as independent single-op ablations.

## Exact guards

### FlashMLA sparse decode

The provider is selected only for:

- exact `ForwardMode.DECODE`, local M 16 or 32;
- Q BF16 contiguous `[M,1,64,576]`;
- FP8 E4M3 contiguous paged KV `[num_pages,64,1,656]`;
- cache lengths int32 `[M]`;
- sparse physical indices int32 `[M,1,2048]`;
- scheduler metadata int32 `[148,8]` and cumulative splits int32 `[M+1]`;
- empty int32 block table `[M,0]`;
- value dimension 512, FP8 KV enabled, and softmax scale 0.0625;
- all tensors on one CUDA device.

The returned output must be contiguous BF16 `[M,1,64,512]` on the same device.
Speculative modes and the separate `flashmla_sparse_fwd` layout do not select
this entry.

### W13 and W2

Both masked grouped GEMMs require:

- exact `ForwardMode.DECODE`, local M 16/32;
- expected-M 4/5 for M16 or 8/9 for M32;
- E32 and a fixed 1024-row expert slab;
- contiguous FP8 E4M3 activations and weights;
- TMA-aligned packed int32 UE8M0 scales with exact column-major strides;
- contiguous BF16 output and int32 E32 mask;
- no FP4/MXFP8 recipe and no DeepEP/TBO overlap argument.

W13 is one fused N4096 call; there is no separate gate and up launch in the
production path. Measuring two historical N2048 GEMMs is an interface
mismatch.

Once a supported no-overlap M16/M32 decode bucket reaches the target W13 shape
family or the rebound W2 runner, target ABI drift aborts that candidate
process. Explicitly unsupported recipe/overlap calls remain stock-compatible,
so an end-to-end A/B must still require the candidate NVTX range and graph
node; startup readiness alone is not a hit.

## Nsys and hit verification

Enable NVTX only for profiler collection:

```bash
export SGLANG_GLM52_INFINI_KERNEL_NVTX=1
nsys profile \
  --trace=cuda,nvtx \
  --cuda-graph-trace=node \
  --output=glm52-hotspot \
  python -m sglang.launch_server <unchanged arguments>
```

For W13/W2, place the corresponding `run_with_stock.sh` or
`run_with_exact_post1_stock.sh` before `nsys` so the profiled process still
uses the manifest stock denominator.

Expected ranges include:

```text
infini_kernel_glm52_flashmla_sparse_decode_fp8_topk2048[M=16]
infini_kernel_glm52_moe_w13_decode[M=16,N=4096,K=6144]
infini_kernel_glm52_moe_w2_decode[M=16,N=6144,K=2048]
```

For FlashMLA, also require the existing first-hit counter for the exact op/M
bucket. For the built-in W13/W2 direct paths, require the exact NVTX range and
inspect CUDA Graph nodes/kernel identities; a startup-ready record alone is
not evidence that the candidate ran. Turn NVTX back off for authoritative
latency measurements.

For W13/W2, the candidate-only JIT source also renames the compiled
`__global__` template to `infini_kernel_glm52_moe_w13_decode` or
`infini_kernel_glm52_moe_w2_decode`. The prefix therefore remains searchable
on CUDA Graph replay nodes even though replay does not re-enter the Python
NVTX context. The FlashMLA name above is only an NVTX range until its external
provider gives the compiled symbol an `infini_kernel_...` prefix; Python
cannot rename a cubin symbol after compilation.

## Fair promotion gate

For each M/expected-M bucket:

1. Compare against the same manifest's post1 stock symbol with identical input bytes,
   packed scale layout, stream, PDL/SM budget, scheduler metadata, and output
   ownership.
2. Warm JIT and provider initialization outside timing.
3. Run fresh correctness before timing and after timing; test masks with empty,
   boundary, skewed, and changed experts.
4. Measure leaf eager and independently captured stock/candidate CUDA Graphs.
5. Measure the containing region:
   - FlashMLA main plus combine;
   - W13 → stock SwiGLU/packed quant → stock W2;
   - stock W13 → stock SwiGLU/packed quant → W2.
6. Alternate AB and BA order across at least three independent series. Report
   pooled, order-balanced, AB-median, BA-median, p10/p50/p90, and raw pairs.
7. Reject graph copy/adapter/fallback nodes and require candidate/stock graphs
   to differ only at the target kernel node.
8. Finish with checkpoint-backed TP8/DP8/EP8 serving TTFT, TPOT, throughput,
   output-correctness, and rank-max latency.

Historical evidence explains why every stage is required:

- FlashMLA MAX_SPLITS 160→32 improved eager by roughly 4% at M16, but graph
  replay was approximately 1.00×;
- W13 BM32/2-SM improved leaf graph by roughly 4.3% and the graph containing
  region pooled by roughly 3.66%, but one mandatory BA estimator was only
  1.028125×;
- W2 BM16 reduced isolated device work, but older process-global tuning and
  incomplete topology/region evidence were not deployable.

These are useful optimization leads, not production wins. The stock path
remains the default until the complete gate passes.

## Optimization priority

1. **W13 BM32/2-SM fixed-shape tuning**: closest to a real region win. Explore
   BM/BN, stages, 1-SM versus 2-SM, TMA/barrier waits, and epilogue surface.
   The measured two-SM candidate changed NCU duration from 136.58 to 128.32 us
   and output writes from 31.45 to 10.08 MB, but a required graph-region BA
   estimator was only 1.028125x. Those measurements used v0.1.4; the registered
   implementation reapplies the same per-call delta to post1 and therefore
   must establish fresh leaf, graph, and containing-region evidence.
2. **W2 fixed expected-M tiling**: reduce padded output/TMEM stores without
   process-global selectors; bind configuration per call. Historical BM16
   reduced the production leaf from 75.5–76.3 to 68.6–70.1 us and cut output
   stores sharply, but the old selector was process-global. A later stage
   experiment was invalid because stock was actually BM128/stage-8 rather than
   the assumed stage-12; future work must verify generated template constants
   before timing.
3. **FlashMLA main kernel first, combine second**: NCU shows a bandwidth and
   synchronization-latency problem. Investigate vectorized sparse gathers,
   cache policy, split scheduling, barrier lifetime, and register pressure;
   pure host-wrapper or combine-only changes are unlikely to survive replay.
   The measured split-count and combine specialization attempts already
   reduced combine work but lost or stayed flat after full-chain graph replay.
4. **W13/SwiGLU/W2 fusion boundary**: potentially higher leverage than another
   leaf-only tile round, but must preserve packed UE8M0 and graph semantics.
   A standalone SwiGLU+quant kernel reached about 1.51–1.64x in leaf graph
   tests, yet the complete W13→activation→W2 region improved only about
   1.7–2.0%; further standalone tuning is therefore lower priority than a
   boundary/fusion change.

The Blackwell implementation reference points used for this ranking are:

- KernelWiki `kernel-flashmla` (`wiki/kernels/flashmla.md`);
- `kernel-sparse-mla` (`wiki/kernels/sparse-mla.md`);
- `kernel-grouped-gemm` (`wiki/kernels/grouped-gemm.md`);
- `kernel-fused-moe` (`wiki/kernels/fused-moe.md`);
- `pattern-memory-bound` and `pattern-pipeline-stalls`;
- FlashInfer PR 2836 for sparse-MLA kernel selection;
- vLLM PR 19566 for M-dependent SM100 FP8 tile/cluster dispatch.

These pages are source-reported or upstream-code evidence. They guide the
search space; only measurements on the exact GLM-5.2 ABI can establish a win.
