# Task26 `em8_bm16_stage11` v3 exact-post1 overlay

This directory defines a separately versioned Task26 candidate. It is not a
retry, continuation, or reinterpretation of the consumed stage12 run
`20260728T182705Z`.

Both stock and candidate start from the exact authoritative post1 source:

- version: `v0.1.4.post1` (normalized distribution version `0.1.4.post1`)
- commit: `edcf77b276965de8f03cdc47c23f01b08bf7c7ab`
- CUTLASS: `f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`
- fmt: `553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28`

`source.patch` adds two optional per-call arguments to the ordinary masked
Python/FFI/C++ API:

```text
masked_block_m_override=0
masked_num_stages_override=0
```

The only non-stock pair accepted is `(16, 11)`, and only for exact
M32/`expected_m=8`, E32/slab1024/K2048/N6144, packed-int32 UE8M0 scales,
SM100, PDL enabled, 148 SMs, no recipe, and no overlap. Both overrides enter
`GemmDesc` before `get_best_config`. Layout, storage, the complete pipeline
configuration and its shared-memory size, launch configuration, generated
code, and the JIT key therefore agree on BM16/stage11.
The `(0, 0)` path retains the stock max-stage selection and does not execute
the stage11-only bounds or memory assertions.

The exact generated-kernel identity is:

```text
sm100_m_grouped_fp8_fp4_gemm_masked_1d1d_glm52_w2_em8_bm16_stage11_v3
```

The build identity additionally names `expected-m8`, `bm16`, `stages11`,
packed UE8M0, PDL, 148 SMs, and the no-recipe/no-overlap contract. Stage10 is
only predeclared in provenance as `em8_bm16_stage10`; it is not implemented,
eligible, dispatched, built, or benchmarked unless stage11 is first falsified
and a later audit explicitly releases it.

For this exact layout, the derived pipeline storage is 18,432 bytes per
stage plus 9,004 fixed bytes: stock stage12 is 230,188 bytes and candidate
stage11 is 211,756 bytes. The reduction does not enable two CTAs per SM; it is
only a falsifiable reduced-pipeline-pressure hypothesis.

`build_tool.patch` remains separate from the runtime delta and only supports a
CPU-only fixed-SM100, no-wheel build. No build is authorized by the source
commit itself. After independent source review, build both side-by-side
artifacts with:

```bash
third_party/deepgemm_w2_em8_bm16_stage11/build_overlay.sh
```

The stage11 build and launch environment must use only these fresh paths:

```text
DG_JIT_CACHE_DIR=/home/qinhaiyan/glm52-v2-goal-runs/cache/26-moe_w2_decode_scoped_bm16/em8_bm16_stage11_v3/deepgemm
SGLANG_DG_CACHE_DIR=/home/qinhaiyan/glm52-v2-goal-runs/cache/26-moe_w2_decode_scoped_bm16/em8_bm16_stage11_v3/deepgemm
TRITON_CACHE_DIR=/home/qinhaiyan/glm52-v2-goal-runs/cache/26-moe_w2_decode_scoped_bm16/em8_bm16_stage11_v3/triton
TORCH_EXTENSIONS_DIR=/home/qinhaiyan/glm52-v2-goal-runs/cache/26-moe_w2_decode_scoped_bm16/em8_bm16_stage11_v3/torch_extensions
```

`base_lock.json`, the source/core hashes, and `overlay_manifest.py` bind the
variant name/version, exact base/submodules, binary-safe source diff, API
overrides, JIT identity, package hashes, runtime contract, and cache paths.
After a released build, record `build_provenance.json` and re-run:

```bash
third_party/deepgemm_w2_em8_bm16_stage11/verify_source_reproducibility.sh
```

The normal SGLang stock/fallback import must resolve to the new task-local
exact-post1 stock artifact before any SGLang import. Every stage11 process
must start through:

```bash
third_party/deepgemm_w2_em8_bm16_stage11/run_with_exact_post1_stock.sh \
  <command> [args...]
```

The launcher verifies tracked provenance before prepending the exact stock
site. Runtime readiness only verifies already-bound modules and never performs
a late module replacement.
