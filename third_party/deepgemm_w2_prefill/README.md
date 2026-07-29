# Task 30 W2 prefill DeepGEMM variants

This directory reconstructs three independent, side-by-side DeepGEMM modules
from the exact post1-compatible commit
`edcf77b276965de8f03cdc47c23f01b08bf7c7ab`:

- `stock`: stock source, row-map denominator;
- `psum`: byte-identical stock source, endpoint/PSUM stage-8 control;
- `stage7`: the single tracked source change, capped to seven stages only for
  E32/M35200/N6144/K2048 PSUM, expected-M 1024, compiled `nk`, no gap zeroing.

The build always reconstructs the base, CUTLASS, and fmt from pinned git
archives. It never modifies or builds the divergent DeepGEMM root checkout.

```bash
python third_party/deepgemm_w2_prefill/build_variants.py \
  --audit-materialization

python third_party/deepgemm_w2_prefill/build_variants.py --force
```

Every CUDA-capable invocation in this campaign is made through the task GPU
lease wrapper. The output defaults to the Task 30 cache and records the
complete source, compiler, DSO, build-plan, and JIT-cache identities.
