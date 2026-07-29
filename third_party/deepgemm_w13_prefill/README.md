# GLM-5.2 prefill W13 PSUM experiment

This directory owns Task 28's default-off, source-scoped DeepGEMM builds.
`build_variants.py` reconstructs three independent modules from immutable SGL
DeepGEMM `v0.1.4.post1` commit
`edcf77b276965de8f03cdc47c23f01b08bf7c7ab`:

- `stock`: unmodified source, used with the stock row-map ABI;
- `psum`: byte-identical source in a distinct DSO and JIT cache, used with the
  existing endpoint/PSUM ABI;
- `xor`: the same base plus the one tracked UTCCP transpose patch, used with
  the same endpoint/PSUM ABI.

The XOR patch is compile-time restricted to
`MGroupedContiguousWithPsumLayout`. It replaces the 128-bit shared-memory
transpose store with four XOR-permuted scalar-store phases. Each lane writes
the same four packed scale words to the same final addresses, while lanes that
share a starting bank select distinct columns per phase. It does not change
the MMA, TMEM, TMA, mbarrier, cluster, output, or scheduler code.

CPU-only source reconstruction:

```bash
python3 third_party/deepgemm_w13_prefill/build_variants.py \
  --audit-materialization
```

The actual build is a CUDA command and therefore must run through the campaign
GPU wrapper with the task-local caches exported:

```bash
/home/qinhaiyan/glm52-goal-runs/with_flexible_gpu.sh \
  env KERNEL_HARNESS_PYTHON=/path/to/kernel-harness/.venv/bin/python \
  SGLANG_DG_CACHE_DIR=/path/to/task-cache/deepgemm \
  DG_JIT_CACHE_DIR=/path/to/task-cache/deepgemm \
  TRITON_CACHE_DIR=/path/to/task-cache/triton \
  TORCH_EXTENSIONS_DIR=/path/to/task-cache/torch_extensions \
  CUDA_HOME=/usr/local/cuda MAX_JOBS=1 \
  bash -lc '"$KERNEL_HARNESS_PYTHON" \
    third_party/deepgemm_w13_prefill/build_variants.py --force'
```

The schema-3 manifest records the complete immutable source identity,
normalized compiler plan, compilers, three distinct DSOs, and three distinct
JIT cache roots. Neither module is installed into the active environment.
