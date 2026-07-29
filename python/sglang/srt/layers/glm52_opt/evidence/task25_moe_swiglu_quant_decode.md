# GLM-5.2 Task 25 disposition

Terminal disposition: **no-replacement**.

This branch contains a default-off fused masked SwiGLU plus direct packed
UE8M0 portfolio:

- Triton `row2048_w8`
- Triton `split1024_w4`
- Triton `group512_w1`
- bounded CUDA `cuda_valid_cta`

All variants reproduce the exact production FP8 E4M3 output and
`int32 [32,1024,4]` packed-scale view, pass eager and independent CUDA Graph
replay, retain inactive-row poison, and feed unchanged stock W2. Automatic
ineligibility selects stock before candidate launch; a selected candidate
error propagates without stock retry.

The optimized graph leaf is substantially faster, but none of the mandatory
`stock W13 -> activation/quant -> stock W2` eager or graph lanes passed the
per-series four-estimator `1.03x` rule. Across the best Triton checks and the
complete CUDA matrix, `0/10` candidate containing-region lanes passed.
Therefore this code is retained as research evidence and must not become a
production default.

Default/rollback state:

```bash
export SGLANG_GLM52_OPT=0
unset SGLANG_GLM52_OPT_PROFILE
unset SGLANG_GLM52_OPT_OPS
unset SGLANG_GLM52_SWIGLU_QUANT_VARIANT
```

Pinned provenance:

- SGLang exact base:
  `1c671bf3a30360100e7947c87e0c873a387ad0be`
- SGLang candidate source:
  `03b41db52f955d9816a62f393233936449e42553`
- Kernel-Harness exact base:
  `d432ea821494b591d6f5bbbf2adb4301c4ce6579`
- Kernel-Harness candidate source:
  `f48bba19ac5017ac82c51d7c9d22bf9ab4acca81`
- Kernel-Harness evidence commit:
  `d042a0c`

The complete raw corpus, audits, correctness matrix, profile/code-generation
package, commands, and final report live in:

```text
kernel-harness/serving_native/evidence/25_moe_swiglu_quant_decode/
kernel-harness/profile/task25-swiglu-triton-row2048-m16-fallback-decision/
```

The evidence corpus contains 34 official results and 102 timing series. All
34 results passed correctness and standalone audit with clean source
provenance. No winner side channel was written, no production default was
enabled, and no branch was pushed.
