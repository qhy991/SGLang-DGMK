# GLM-5.2 decode W13 DeepGEMM experiment

This directory contains the source-scoped, default-off W13 experiment.
`build_variants.py` materializes stock and candidate from the same immutable
SGL DeepGEMM `v0.1.4.post1` commit
`edcf77b276965de8f03cdc47c23f01b08bf7c7ab`. This is the same source
denominator used by the W2 registration and current SGLang integration. The
candidate differs only by `patches/0001-explicit-w13-config.patch`.

The patch adds an optional per-call tuple:

`(block_m, block_n, block_k, num_stages, cluster_n)`

It is accepted only by the SM100 masked grouped FP8 path at the exact W13 ABI
(`E=32`, slab `M=1024`, `N=4096`, `K=6144`, expected-M 4/5/8/9, packed int32
scales). It never mutates DeepGEMM's process-global alignment. Omitting the
tuple preserves upstream behavior.

The candidate-only generated CUDA source names its compiled template
`infini_kernel_glm52_moe_w13_decode`, so Nsys/NCU can identify the actual
kernel node, including CUDA Graph replay; this is only a symbol rename and
does not change generated instructions or launch geometry.

The bounded configurations are:

- stock: upstream heuristic (`BM128/BN128/BK128`, 8 stages, two-CTA)
- historical anchor: `(32, 128, 128, 11, 2)`
- genuine one-CTA comparison: `(32, 128, 128, 10, 1)`

Generated modules and JIT caches default to
`build/deepgemm-w13-variants`; they are not installed into the active
environment. Override the source repository or output with `--upstream` and
`--output`, or with `DEEPGEMM_W13_BASE_REPO` and `DEEPGEMM_W13_OUTPUT`.

CPU-only reproducibility audit (does not import Torch, compile, JIT, or query
CUDA):

```bash
python3 third_party/deepgemm_w13/build_variants.py \
  --upstream /path/to/sgl-deep-gemm \
  --audit-materialization
```

The script reconstructs stock twice and candidate twice from pinned `git
archive` inputs, applies only the tracked patch to candidate, and rejects any
tree or patch hash drift. It never runs the copied upstream submodule-update
step; archive trees intentionally have no `.git` directory.

After the P0 CPU contract is closed and `df -BG /` shows at least 8 GiB free,
build both modules with the same compiler function, flags, dependency commits,
include template, and linker template:

```bash
CUDA_VISIBLE_DEVICES='' \
  /path/to/sglang/python third_party/deepgemm_w13/build_variants.py \
  --upstream /path/to/sgl-deep-gemm \
  --force
```

The builder fixes `MAX_JOBS=1` internally and records it in the runtime
attestation, so a caller's shell cannot change the build schedule.

The schema-2 manifest records the exact source trees, base blob hashes,
complete patch hash, build environment and command templates, package and DSO
paths/hashes, compiler-binary hashes, the complete generated Ninja files, one
common path-normalized Ninja-plan hash, plus distinct stock/candidate JIT
roots. JIT is fixed to NVCC rather than NVRTC. This host-only extension build
must not be used as performance evidence.

At runtime, both production and the serving-native harness defer W13 DSO load,
CUDA queries, runtime configuration, and JIT until after the worker/leased GPU
is assigned. Startup deterministically binds and warms stock before importing
SGLang `compile_utils`, disables broad masked-GEMM precompile, then binds and
warms candidate. Each module is set and read back independently at
`PDL=true`, `num_sms=148`, and `tc_util=100`; both mutation directions are
tested and restored. A poison cache-path probe proves that the two lazy
compilers remain bound to their own cache roots. The caller's cache
environment is restored before serving.

Always start both the stock and candidate arms through the same-source stock
launcher. It binds the manifest stock package as the authoritative
`deep_gemm` import before SGLang starts:

```bash
third_party/deepgemm_w13/run_with_stock.sh \
  python -m sglang.launch_server <unchanged arguments>
```

For the candidate arm, set:

```bash
export SGLANG_GLM52_OPT=1
export SGLANG_GLM52_OPT_PROFILE=hotspot_candidates
export SGLANG_GLM52_OPT_OPS=moe_w13
export SGLANG_GLM52_OPT_M_BUCKETS='moe_gate_proj:16|32'
```

For the stock arm, unset `SGLANG_GLM52_OPT` and the profile/op variables but
keep the same launcher, manifest, server arguments, cache state, and workload.
Startup aborts if the normal SGLang fallback is not the exact manifest stock.
With the candidate environment still set, validate the real wrapper and CUDA
Graph route before launching a checkpoint:

```bash
third_party/deepgemm_w13/run_with_stock.sh \
  python scripts/glm52_moe_registration_smoke.py --op w13
```

The smoke uses zero inputs and is not performance evidence. It verifies the
distinct module identities, eager selection, capture/replay, no replay-time
Python re-entry, and active-row agreement with stock.

The historical `(32,128,128,11,2)` result was measured on the preceding
v0.1.4 source. The per-call patch applies cleanly to post1, but the post1 build
is deliberately treated as an end-to-end candidate that must be revalidated;
the old leaf number is not automatically promoted.
