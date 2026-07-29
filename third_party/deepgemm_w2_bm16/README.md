# GLM-5.2 W2 decode BM16 exact-post1 overlay

This directory carries the reproducible, source-scoped W2 decode candidate. Both
stock and candidate start at the exact `sgl-deep-gemm` post1 commit:

- version: `v0.1.4.post1` (normalized distribution version `0.1.4.post1`)
- commit: `edcf77b276965de8f03cdc47c23f01b08bf7c7ab`
- CUTLASS: `f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`
- fmt: `553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28`

`source.patch` adds the optional per-call keyword
`masked_block_m_override=0` to the ordinary masked API across Python, FFI, and
C++. SGLang passes `16` only after its W2 contract is prepared. The override is
placed in `GemmDesc` before `get_best_config`, so storage, pipeline, launch
configuration, generated code, and JIT identity are all derived from BM16. The
four candidate JIT identities are:

```text
sm100_m_grouped_fp8_fp4_gemm_masked_1d1d_glm52_w2_bm16_v2_em4
sm100_m_grouped_fp8_fp4_gemm_masked_1d1d_glm52_w2_bm16_v2_em5
sm100_m_grouped_fp8_fp4_gemm_masked_1d1d_glm52_w2_bm16_v2_em8
sm100_m_grouped_fp8_fp4_gemm_masked_1d1d_glm52_w2_bm16_v2_em9
```

The candidate-only generated CUDA source names its compiled template
`infini_kernel_glm52_moe_w2_decode`, so the actual Nsys/NCU kernel node keeps
an `infini_kernel` identity during CUDA Graph replay. This is independent of
the per-expected-M JIT cache identities above and changes no instructions or
launch geometry.

The nonzero override is accepted only for the packed-int32 SM100 contract
E32/slab1024/K2048/N6144, PDL enabled, 148 SMs, no recipe, and no overlap.
Ordinary calls retain the stock heuristic. No process-global alignment setter
exists in the patch.

`build_tool.patch` is separate from the runtime delta. It only makes the
CPU-only, fixed-SM100, no-wheel build reproducible. Build both artifacts with:

```bash
export DEEPGEMM_W2_BM16_BASE_REPO=/path/to/sgl-deep-gemm
export HARNESS_PYTHON=/path/to/sglang/python
third_party/deepgemm_w2_bm16/build_overlay.sh
```

The default cache root is
`build/deepgemm-w2-bm16-cache`. It can be moved without changing the source
identity:

```bash
export DEEPGEMM_W2_BM16_CACHE_ROOT=/fast/local/cache/glm52-w2
```

The builder records the resolved four cache paths in the generated manifest.
The launcher reads and exports those paths; no user-specific absolute path is
compiled into the tracked source contract.

The generated manifest records both fresh source statuses, exact binary diff
identities, base/submodule commits, package `__init__.py` and `VERSION` hashes,
shared-object hashes, core-source hashes, toolchain, and runtime/cache contract.
Revalidate an existing build without recompiling:

```bash
third_party/deepgemm_w2_bm16/verify_source_reproducibility.sh
```

The normal SGLang stock/fallback import must resolve to the staged exact-post1
artifact before Python imports SGLang. Start every candidate/reference process
through:

```bash
third_party/deepgemm_w2_bm16/run_with_exact_post1_stock.sh <command> [args...]
```

That launcher verifies the generated manifest and then prepends the stock site
directory in the new process environment. Runtime readiness only verifies the
already-bound module; it never mutates `sys.path` or replaces a loaded module.

For the candidate arm:

```bash
export SGLANG_GLM52_OPT=1
export SGLANG_GLM52_OPT_PROFILE=hotspot_candidates
export SGLANG_GLM52_OPT_OPS=moe_w2
export SGLANG_GLM52_OPT_M_BUCKETS='moe_down_proj:16|32'

third_party/deepgemm_w2_bm16/run_with_exact_post1_stock.sh \
  python -m sglang.launch_server <unchanged arguments>
```

With that candidate environment still set, validate the rebound W2 runner and
one CUDA Graph capture/replay:

```bash
third_party/deepgemm_w2_bm16/run_with_exact_post1_stock.sh \
  python scripts/glm52_moe_registration_smoke.py --op w2
```

The smoke uses zero inputs and is not performance evidence. Runtime/layer ABI
preparation or exact selected-bucket drift is fatal, so it cannot pass by
silently executing stock.

For the stock arm, unset the GLM52 optimization profile/op variables but keep
the same launcher, manifest, server arguments, and workload.
