# GLM-5.2 Task 03 q_b decode one-SM disposition

Date: 2026-07-29

Task 03 ends as **no-replacement** for both local decode buckets:
M16 and M32 at N16384 K2048. No SGLang runtime or kernel source is changed,
and no q_b bucket is enabled.

The task required the exact current stock denominator to be one-SM before
timing a one-SM CUTLASS replacement. Runtime-selected JIT source and cubin
evidence on SGLang base
`0a723222cf653758dcf5ad677453b226f1981444` shows the opposite in both buckets:

| M | DeepGEMM template | TMEM | SASS | launch cluster |
|---:|---|---|---|---|
| 16 | `kNumMulticast=2` | `Allocator2Sm` | 90 `UTCQMMA.2CTA`, 0 `UTCQMMA.1CTA` | 2x1x1 |
| 32 | `kNumMulticast=2` | `Allocator2Sm` | 90 `UTCQMMA.2CTA`, 0 `UTCQMMA.1CTA` | 2x1x1 |

The production route was separately reached through
`Fp8LinearMethod.apply` with the q_b operation context, BF16 caller input,
stock per-token/group-128 quantization, FP8 E4M3 operands, packed-int32 UE8M0
activation/weight scales, and BF16 output. Independent graph captures contain
the quantizer and the exact selected DeepGEMM node, with reproducible outputs.
The global GLM52 experimental registry remained disabled and untouched.

The target is specifically `self_attn.q_b_proj` (N16384), not the DSA
indexer's `wq_b`/index up-projection (N4096). Current GLM source constructs an
alternate stream at the model level but does not pass it into
`DeepseekV2AttentionMLA`; therefore the generic q_b/indexer overlap branch is
not reached on this checkout. The existing attention-stream q_a RMSNorm,
64-head q view, 192/64 split, and RoPE route remain unchanged.

Per the task's mandatory stop rule, no CUTLASS B1 tile, native block-scaled B2
route, profile-gated refinement, or optional DeepGEMM control was built or
timed. The archived FP32-scale denominator and the rejected Task 14 packed-warp
variants were not reused.

Production policy is unchanged:

- stock remains the only q_b path;
- no new registration or allowlist entry exists;
- `SGLANG_GLM52_OPT=0` remains the explicit reference setting;
- rollback requires no action beyond retaining this source base.

Complete raw evidence, disassembly, audits, attempt ledger, and report are in
the paired Kernel-Harness task branch under
`serving_native/evidence/glm52_prod_03_attn_q_b_decode_1sm/`.
