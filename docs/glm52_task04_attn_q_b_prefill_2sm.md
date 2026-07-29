# GLM-5.2 Task 04: attention q_b prefill two-SM

Date: 2026-07-29

Disposition: **no replacement** for attention `q_b_proj` prefill at
`M=4096, N=16384, K=2048` on B200.

This task used SGLang base
`7ab48410199c87abd78bc6761d495bcc835069c9`. Production remains on the
unmodified stock SGLang path. No candidate kernel, selector, registry entry,
environment flag, graph node, quantizer, stream, or model source is installed
or enabled by this branch.

## Production contract

Two fresh wrapper-leased processes reached attention `q_b_proj`, not the DSA
indexer's `wq_b`. The current production call consumed:

- FP8 E4M3 activation `[4096, 2048]`;
- FP8 E4M3 weight `[16384, 2048]`;
- packed int32 UE8M0 K128 activation scales with shape `[4096, 4]` and stride
  `[1, 4096]`;
- packed int32 UE8M0 K128 weight scales with shape `[16384, 4]` and stride
  `[1, 16384]`;
- BF16 output `[4096, 16384]`.

The containing query-preparation graph retained q-a RMSNorm, group-128
quantization, q_b projection, the 64-head `[192, 64]` q-nope/q-pe split, and
reached RoPE on the attention stream. Its four kernel nodes were unchanged
except when an explicitly selected diagnostic q_b candidate replaced the q_b
node. Graph mutation, output-poison, deterministic replay, non-default-stream,
and forbidden-node checks passed.

The selected stock JIT source uses tile `128x224x128`, stage count 6, and a
two-CTA cluster. Its cubin has 90 `UTCQMMA.2CTA` and zero
`UTCQMMA.1CTA`; the runtime graph reports cluster `2x1x1`.

## Bounded attempts

Current-source DeepGEMM per-call `compiled_dims` variants were compared with
dynamic, N/K, and exact M/N/K compilation. Direct same-process
specialization-to-specialization comparisons did not clear 1.03. The exact
M/N/K candidate passed eager apply and containing-region screens, but failed
all production-active CUDA Graph performance gates:

| Boundary | all required estimators |
|---|---:|
| leaf graph | 1.016756–1.023797 |
| apply graph | 1.018327–1.025796 |
| containing region graph | 1.012810–1.019586 |

The mandatory CUTLASS 4.2.1 B1 portfolio directly consumed the packed scales.
It used no scale expansion, physical transpose, helper kernel, external
workspace, output copy, or process-global mutation. All three variants were
bitwise equal to stock on the retained adversarial cases and proved PTX
`tcgen05.mma.cta_group::2`, SASS `UTCQMMA.2CTA`, zero
`UTCQMMA.1CTA`, and an actual `2x1x1` cluster:

| Variant | required estimator range | device-profile speedup |
|---|---:|---:|
| direct 128x128x128 | 0.273497–0.533480 | 0.183965 |
| direct 128x256x128 | 0.571031–0.815971 | 0.416843 |
| transpose-equivalent 128x128x128 | 0.268189–0.310887 | 0.183982 |

Every B1 series was more than 10 percent slower than stock, including the
strongest estimator. No candidate was within the plan's 10 percent
refinement-entry window and the existing device profiles exposed no concrete
tractable limiter. Therefore zero B2 refinements were run and NCU was not
invoked.

## Enable and rollback

There is no Task 04 enable route. Keep `SGLANG_GLM52_OPT=0` for the explicit
stock reference and do not select the archived `q_b_prefill.py`; that artifact
packs or caches scales and mutates global SM state, so its historical result is
inadmissible here.

Rollback is a no-op: retain stock SGLang. A future attempt must start from a
new current-source hypothesis and independently satisfy leaf, apply,
containing-region, graph, dispatch, and checkpoint-backed acceptance gates.
