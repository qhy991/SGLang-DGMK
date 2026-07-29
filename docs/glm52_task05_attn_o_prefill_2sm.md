# GLM-5.2 Task 05: attention O-projection prefill two-SM

Date: 2026-07-29

Disposition: **no replacement** for rank-local attention `o_proj` prefill at
`M=4096, N=6144, K=16384` on B200.

This task uses SGLang runtime base
`0a723222cf653758dcf5ad677453b226f1981444`. Production remains on that
unmodified stock path. This branch adds only this report: no candidate kernel,
selector, registry entry, environment flag, quantizer, stream, graph node,
model source, or default is installed or enabled.

## Production contract

The current call was reached through
`model.layers.0.self_attn.o_proj`, a `RowParallelLinear` with
`input_is_parallel=True`, attention TP size one, and
`reduce_results=False`. It consumes:

- BF16 caller input `[4096, 16384]`;
- FP8 E4M3 activation `[4096, 16384]`;
- FP8 E4M3 weight `[6144, 16384]`;
- packed int32 UE8M0 K128 activation scales `[4096, 32]`, stride
  `[1, 4096]`;
- packed int32 UE8M0 K128 weight scales `[6144, 32]`, stride `[1, 6144]`;
- BF16 output `[4096, 6144]`.

One runtime call reaches the group-128 production quantizer,
`Fp8LinearMethod.apply`, the packed DeepGEMM dispatcher, registered custom op,
and `deep_gemm.fp8_gemm_nt`. It makes no Triton fallback or row-parallel
all-reduce. All hops stay on the same non-default stream and the returned
tensor aliases the wrapper-prepared BF16 output buffer.

Normal TP8 and TP8/DP8 prefill resolve to eager execution. A separately
registered breakable CUDA Graph configuration was used only to prove semantic
liveness. Graph performance is not a production promotion lane for this
bucket.

The current stock JIT recipe is swap-AB `240x128x128`, cluster `1x2`, six
stages, PDL=true, and 148 SMs. Its final binary uses 42 registers, zero
stack/local/spills, and 90 `UTCQMMA.2CTA` instructions.

## Bounded attempts

The older compiled-dimension, PDL-off, process-global SM-count, 148-to-144,
and six-to-five-stage experiments were accepted as conclusive negative
evidence and not repeated.

The current-source DeepGEMM M256 adjacent tile is compile-ineligible: its
aligned TMEM demand is 524 columns, above the 512-column limit. A per-kernel
146-CTA useful-cluster candidate is correct and preserves the cooperative
two-SM datapath, but its three full eager leaf series have minimum estimators
0.980459, 0.993494, and 0.986920. It fails the uniform 1.03 gate.

The independent CUTLASS 4.2.1 portfolio consumes the existing packed scales
inside CUTLASS's software-blockwise scale-producer warp. It uses no external
scale expansion, helper launch, transpose/copy, workspace, allocation, global
state, or fallback. Each final candidate proves
`tcgen05.mma.cta_group::2`, `UTCQMMA.2CTA`, actual cluster `2x1x1`, two-CTA
TMEM, CLC/TMA activity, and one graph kernel node.

| CUTLASS tile | minimum eager estimator |
|---|---:|
| direct 128x256 | 0.204535 |
| direct 256x128 | 0.210688 |
| transpose-equivalent 256x128 | 0.312185 |

All three are more than ten percent slower than stock at the plan's first
leaf screen. No DeepGEMM or CUTLASS refinement slot became eligible, and no
concrete NCU question remained.

## Enable and rollback

There is no Task 05 enable route. The diagnostic candidates live only in the
Kernel-Harness evidence branch and must not be copied into production.

Rollback is a no-op: retain stock SGLang and keep `SGLANG_GLM52_OPT=0`. A
future attempt must begin from a new current-source mechanism and
independently pass all leaf, apply, containing RowParallel, graph semantic,
dispatch, checkpoint, and eight-B200 topology gates before any default can
change.
