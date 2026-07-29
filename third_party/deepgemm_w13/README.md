# GLM-5.2 decode W13 DeepGEMM experiment

This directory contains the source-scoped, default-off Goal 24 experiment.
`build_variants.py` materializes both stock and candidate modules from the
same immutable SGL DeepGEMM v0.1.4 commit. The candidate differs only by
`patches/0001-explicit-w13-config.patch`.

The patch adds an optional per-call tuple:

`(block_m, block_n, block_k, num_stages, cluster_n)`

It is accepted only by the SM100 masked grouped FP8 path at the exact W13 ABI
(`E=32`, slab `M=1024`, `N=4096`, `K=6144`, expected-M 4/5/8/9, packed int32
scales). It never mutates DeepGEMM's process-global alignment. Omitting the
tuple preserves upstream behavior.

The bounded configurations are:

- stock: upstream heuristic (`BM128/BN128/BK128`, 8 stages, two-CTA)
- historical anchor: `(32, 128, 128, 11, 2)`
- genuine one-CTA comparison: `(32, 128, 128, 10, 1)`

Generated modules and JIT caches live under the task-local cache and are not
installed into the active environment.

CPU-only reproducibility audit (does not import Torch, compile, JIT, or query
CUDA):

```bash
python3 third_party/deepgemm_w13/build_variants.py --audit-materialization
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
  /path/to/task/python third_party/deepgemm_w13/build_variants.py --force
```

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
