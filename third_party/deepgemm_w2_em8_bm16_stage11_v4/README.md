# Task26 `em8_bm16_stage11` v4 exact-post1 READY bundle

This is a fresh v4 experiment identity. The immutable v3 attempt remains a
terminal `no-replacement`: it consumed its sentinel before discovering that
its overlay was absent and also omitted the containing-region CUDA Graph
lane. Nothing here retries, edits, deletes, or reinterprets v3 evidence.

The v4 kernel semantics are deliberately unchanged from v3. Both stock and
candidate start from exact `sgl-deep-gemm` `v0.1.4.post1`:

- commit `edcf77b276965de8f03cdc47c23f01b08bf7c7ab`
- CUTLASS `f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`
- fmt `553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28`

`source.patch` adds the optional per-call arguments
`masked_block_m_override=0` and `masked_num_stages_override=0`. The only
non-stock pair accepted is `(16, 11)`, scoped to exact M32/expected-M8,
E32/slab1024/K2048/N6144, packed-int32 UE8M0 scales, SM100, PDL, 148 SMs,
no recipe, and no overlap. The `(0, 0)` path retains stock behavior. No
process-global alignment setter is used.

The fresh JIT identity is:

```text
sm100_m_grouped_fp8_fp4_gemm_masked_1d1d_glm52_w2_em8_bm16_stage11_v4
```

The shared-memory model remains 18,432 bytes per stage plus 9,004 fixed
bytes: stock stage12 is 230,188 bytes and candidate stage11 is 211,756 bytes.
This does not enable two CTAs per SM; reduced pipeline pressure remains a
falsifiable performance hypothesis.

## Two-phase release

Building and readiness publication are CPU-only release operations. They must
run before entering the flexible-GPU wrapper, with no inherited GPU lock.
Export the exact v4 caches and explicitly hide devices:

```bash
export CUDA_VISIBLE_DEVICES=''
export DG_JIT_CACHE_DIR=/home/qinhaiyan/glm52-v2-goal-runs/cache/26-moe_w2_decode_scoped_bm16/em8_bm16_stage11_v4/deepgemm
export SGLANG_DG_CACHE_DIR="$DG_JIT_CACHE_DIR"
export TRITON_CACHE_DIR=/home/qinhaiyan/glm52-v2-goal-runs/cache/26-moe_w2_decode_scoped_bm16/em8_bm16_stage11_v4/triton
export TORCH_EXTENSIONS_DIR=/home/qinhaiyan/glm52-v2-goal-runs/cache/26-moe_w2_decode_scoped_bm16/em8_bm16_stage11_v4/torch_extensions
third_party/deepgemm_w2_em8_bm16_stage11_v4/build_overlay.sh
```

The build removes mutable bytecode caches, makes both staged package trees
read-only, and creates an exact manifest, a fresh source replay, and tracked
`build_provenance.json`. Manifest schema v6 and provenance schema v5 bind every
regular package file and directory by relative path and permission mode, every
file by byte count and SHA-256, and forbid symlinks, hardlinks, and special
files. This covers the Python/JIT sources and the staged DeepGEMM/CUTLASS
headers as well as `_C.so`. It moves the external bundle to:

```text
build/deepgemm-w2-em8-bm16-stage11-v4-ready-bundles/<content-sha256>/
```

It intentionally does not write `READY`. Review and commit the generated
provenance while keeping both SGLang and Kernel-Harness repositories clean,
then publish:

```bash
third_party/deepgemm_w2_em8_bm16_stage11_v4/publish_ready.sh
```

`publish_ready.sh` rechecks the manifest, tracked provenance, both package
hash sets, source replay, source inputs, content-addressed directory, cache
contract, and both clean exact repository heads. It then atomically writes the
immutable `READY` record. Any changed input invalidates verification.

## GPU phase

The production driver and stock launcher only consume an already-published
bundle. Neither contains a build path. The driver verifies the complete READY
contract before checking the lease sentinel or inherited GPU lock, before
creating a run root, before consuming the one-attempt sentinel, and before
querying `nvidia-smi`.

The exact portfolio is:

1. W2 leaf eager
2. W2 leaf CUDA Graph
3. W13 + SwiGLU/packed-quant + W2 containing-region eager
4. the same containing region under CUDA Graph

For containing-region evidence, only the one W2 graph node may differ; the
ordered non-W2 nodes and their multiset must remain exact. Every lane uses the
same three-series alternating protocol and requires finite pooled,
order-balanced, AB-median, and BA-median estimates of at least 1.03x.

Every runtime process starts through:

```bash
third_party/deepgemm_w2_em8_bm16_stage11_v4/run_with_exact_post1_stock.sh \
  <command> [args...]
```

That launcher re-verifies READY and binds the bundle’s exact-post1 stock
package before any SGLang import. Runtime preparation verifies READY again
before its first `torch.cuda` query. Missing or corrupt readiness is terminal
and must not consume a GPU attempt.
