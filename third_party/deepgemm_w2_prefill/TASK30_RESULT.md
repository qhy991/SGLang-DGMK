# Task 30 W2 prefill result

Terminal disposition: **`no-replacement`**. The Task 30 selector remains
production-default-off.

The retained stock-source PSUM stage-8 specialization is correct and wins its
same-base W2 leaf gate, but it does not clear the mandatory unchanged
containing-region gate:

`stock W13 -> stock activation/packed quant -> stock-or-PSUM W2`

Authoritative eager aggregate estimates:

| Boundary | Pooled | Balanced | AB median | BA median | Required gate |
| --- | ---: | ---: | ---: | ---: | --- |
| W2 leaf | 1.057661x | 1.058283x | 1.058459x | 1.058107x | pass; every estimator in every series passes |
| W13/activation/W2 region | 1.011264x | 1.011238x | 1.011198x | 1.011278x | **fail; every series is below 1.03** |
| Stage 8 / stage 7 source effect | 0.969134x | 0.969041x | 0.969039x | 0.969043x | fail the 1.00 non-regression rule |

The stage-7 patch is preserved as rejected evidence. It is scoped only to
E32/M35200/N6144/K2048 endpoint-PSUM, expected-M1024, compiled-NK, no gap
zeroing, and the two-CTA cluster layout. It removes one 25,600-byte pipeline
stage but loses latency hiding and regresses every source-effect series.

Production validation proves:

- the Task 28 endpoint device object is consumed directly;
- activation input is FP8 `[35200,2048]` with packed-int32 UE8M0 scale
  `[35200,4]` stride `[1,35200]`;
- W13 and Task 29 activation candidates stay disabled and stock;
- selected W2 is exactly one GemmType-5 launch with compiled-NK,
  expected-M1024, and no zero padding;
- unsupported calls fall back before launch, while exact-target ABI or recipe
  errors abort;
- a selected kernel error propagates without a stock-W2 retry; and
- valid-row production output is bit-identical to same-base stock.

Final measured-cubin evidence proves PTX `tcgen05.mma.cta_group::2` and SASS
`UTCQMMA.2CTA` (16 each), an actual `[2,1,1]` cluster with 148 CTAs / 74
clusters, TMEM/TMA/mbarrier/cluster synchronization, 50 registers/thread,
214,828 total shared bytes, and zero stack/local/LDL/STL spills.

Pinned build:

- DeepGEMM `edcf77b276965de8f03cdc47c23f01b08bf7c7ab`;
- CUTLASS `f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`;
- fmt `553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28`;
- manifest SHA256
  `9bee8aac4df9213cf64f9d92ca611785d0da4b4ecb239377ff7da0bb1cc75891`;
- selected DSO SHA256
  `f8cca6b3d7859e0d4b9a3d34e24d2bc6edd7dfec95a4802065da184924fd2de7`;
- measured cubin SHA256
  `d80be5e594ec23caa3beeb943eec751ce16793c312f22c1ebf6fbb37566663d5`.

The complete raw evidence, formal audit, attempt ledger, graph semantics,
external TP8 commands, and terminal report are committed in Kernel-Harness
commit `bf9aae6` on branch `goal/glm52-v2-30-moe-w2-prefill-psum`.

The local host has four B200s and no target checkpoint. Exact TP8/DP8/EP8
checkpoint acceptance was not run, but the local required region failure
already makes this revision ineligible for external acceptance. Do not enable
the production selector for this result.
