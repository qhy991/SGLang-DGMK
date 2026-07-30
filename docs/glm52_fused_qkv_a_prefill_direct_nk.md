# GLM-5.2 fused-QKV-A prefill direct-N/K specialization

This branch contains a default-off, layer-private DeepGEMM specialization for
the exact GLM-5.2 fused-QKV-A prefill projection at local
`M=4096, N=2624, K=6144`.

## Status

The implementation is an `external-acceptance-candidate`, not a production
default. The measured implementation revision is
`811ed57af0929bebc48a1655e7e666bac8cd75cd`; later commits in this branch are
evidence/documentation-only unless their diff says otherwise.

Local B200 measurements passed three independent same-process 50-pair AB/BA
series at every required eager boundary:

| Boundary | Pooled speedup | Minimum estimator across its series |
|---|---:|---:|
| packed GEMM leaf | 1.173894x | 1.131806x |
| BF16 quantize + packed GEMM apply | 1.091133x | 1.078588x |
| projection + split + two RMSNorm | 1.114767x | 1.083918x |

Independent CUDA Graph captures passed semantic liveness and exact
correctness. Graph performance is diagnostic because runtime tracing
established that this production prefill route is eager. Exact
checkpoint-backed TP8/DP8/EP8 acceptance remains external.

The complete result, raw hashes, attempt ledger and external commands are in
the paired Kernel-Harness branch under
`serving_native/evidence/glm52_prod_02_attn_fused_qkv_a_prefill/`.

## Feature flag

The feature is disabled by default:

```bash
SGLANG_OPT_GLM52_FUSED_QKV_A_PREFILL_DIRECT_NK=0
```

It may be enabled only for controlled acceptance:

```bash
export SGLANG_OPT_GLM52_FUSED_QKV_A_PREFILL_DIRECT_NK=1
```

Do not simultaneously enable the legacy `SGLANG_GLM52_OPT` registry entry for
`fused_qkv_a_proj/prefill/M4096`. The model constructor now rejects that
ambiguous double registration instead of silently choosing one route.

Unset the variable or set it to `0` for immediate rollback. Do not make it
default-on until the external acceptance procedure passes.

## Exact dispatch scope

The layer configurator binds a private runner only when all static production
facts match:

- architecture `GlmMoeDsaForCausalLM`, model type `glm_moe_dsa`, and the
  exact GLM-5.2 attention fingerprint;
- hidden size 6144, q-LoRA rank 2048, KV-LoRA rank 512, rope head dimension
  64, and fused output width 2624;
- replicated attention projection with attention TP size 1;
- checkpoint-serialized FP8 E4M3 weight `[2624,6144]`;
- packed int32 UE8M0 K128 weight scale `[2624,12]`, stride `[1,2624]`;
- no LoRA and no unsupported target-verify/decode path.

The runtime context additionally requires `ForwardMode.EXTEND` and local
`M=4096`. The packed GEMM helper requires contiguous FP8 activation
`[4096,6144]`, packed int32 scale `[4096,12]` with stride `[1,4096]`, block
recipe `[128,128]`, and BF16 output.

Unsupported model, layer, shape, phase, dtype, device, scale representation,
stride, recipe, bias, LoRA or topology chooses stock before candidate launch.
An error after candidate invocation propagates; there is no candidate-to-stock
retry.

## Kernel delta and preserved work

The candidate changes only the DeepGEMM template key:

```python
deep_gemm.fp8_gemm_nt(
    (q_input, x_scale),
    (weight, weight_scale),
    output,
    compiled_dims="nk",
)
```

Stock uses the same call with dynamic compiled dimensions. The production
BF16 per-token/group-128 quantizer, packed activation and weight scales, BF16
output, allocation policy, stream, split views, q/k RMSNorm kernels and all
non-target nodes remain unchanged. PDL is not modified.

The selected B200 kernel is a genuine two-SM tcgen05 implementation: PTX
contains `tcgen05.mma.cta_group::2`, SASS contains `UTCQMMA.2CTA`, and the live
launch uses cluster `[2,1,1]`.

## Validation

The focused test is:

```bash
PYTHONPATH=python \
  python -m pytest -q \
    test/registered/unit/layers/quantization/test_glm52_fused_qkv_a_direct_nk.py
```

It covers the default-off environment contract, exact model/layer
configuration, decode and prefill isolation, context lifetime, static and
runtime ABI rejection, explicit-failure behavior, and the packed prefill
runner. Checkpoint/eight-GPU validation must follow the paired
Kernel-Harness `EXTERNAL_ACCEPTANCE.md`.
