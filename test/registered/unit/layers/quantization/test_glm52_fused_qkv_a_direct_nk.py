"""CPU-only contracts for GLM-5.2 fused-QKV-A fixed-N/K dispatch."""

from __future__ import annotations

import inspect
import os
import sys
import unittest
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.kernels.ops.quantization import fp8_kernel
from sglang.srt.environ import envs
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.deep_gemm_wrapper import compile_utils
from sglang.srt.layers.deep_gemm_wrapper import entrypoint as deep_gemm_entrypoint
from sglang.srt.layers.glm52_opt.context import (
    fused_qkv_a_direct_nk_context,
    get_fused_qkv_a_direct_nk_context,
)
from sglang.srt.layers.quantization import fp8 as fp8_module
from sglang.srt.layers.quantization import fp8_utils
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import deepseek_v2 as deepseek_v2_module


class _FakeTensor:
    def __init__(
        self,
        shape,
        dtype,
        *,
        stride=None,
        device="cuda:0",
        is_cuda=True,
        contiguous=True,
    ):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.dtype = dtype
        self.device = device
        self.is_cuda = is_cuda
        self._stride = (
            tuple(stride)
            if stride is not None
            else (
                self.shape[-1],
                1,
            )
        )
        self._contiguous = contiguous
        self._version = 0

    def is_contiguous(self):
        return self._contiguous

    def stride(self):
        return self._stride


def _packed_tensors(m=16):
    return (
        _FakeTensor((m, 6144), fp8_utils.fp8_dtype),
        _FakeTensor((2624, 6144), fp8_utils.fp8_dtype),
        _FakeTensor(
            (m, 12),
            torch.int32,
            stride=(1, m),
            contiguous=False,
        ),
        _FakeTensor(
            (2624, 12),
            torch.int32,
            stride=(1, 2624),
            contiguous=False,
        ),
    )


def _bf16_input(m=16):
    return _FakeTensor((m, 6144), torch.bfloat16)


def _method(runner=fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback):
    method = object.__new__(fp8_module.Fp8LinearMethod)
    method.use_marlin = False
    method.use_mxfp8 = False
    method.block_quant = True
    method.w8a8_block_fp8_linear = runner
    method.weight_block_size = [128, 128]
    return method


def _support_patches():
    return (
        patch.object(fp8_utils, "_is_sm100_supported", True),
        patch.object(deep_gemm_wrapper, "ENABLE_JIT_DEEPGEMM", True),
        patch.object(deep_gemm_wrapper, "DEEPGEMM_SCALE_UE8M0", True),
        patch.object(deep_gemm_wrapper, "DEEPGEMM_SUPPORTS_COMPILED_DIMS", True),
    )


def _target_config():
    return SimpleNamespace(
        architectures=["GlmMoeDsaForCausalLM"],
        model_type="glm_moe_dsa",
        head_dim=192,
        max_position_embeddings=1048576,
        index_share_for_mtp_iteration=True,
        rms_norm_eps=1e-6,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=2048,
        index_topk_freq=1,
        indexer_rope_interleave=False,
        rope_interleave=True,
    )


def _fingerprint(config=None, **updates):
    kwargs = dict(
        hidden_size=6144,
        q_lora_rank=2048,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        use_dsa=True,
        is_nextn=False,
        attn_tp_size=1,
        projection_input_size=6144,
        projection_output_size=2624,
    )
    kwargs.update(updates)
    return deepseek_v2_module._is_glm52_dsa_target_fused_qkv_a(
        config or _target_config(),
        **kwargs,
    )


class _FakeLinear:
    def __init__(self, *args, **kwargs):
        tp_size = int(kwargs.get("tp_size") or 1)
        self.prefix = kwargs.get("prefix")
        self.input_size_per_partition = int(args[0]) // tp_size
        self.output_size_per_partition = int(args[1])
        self.quant_method = _method()
        self.weight = _FakeTensor(
            (int(args[1]), int(args[0])),
            fp8_utils.fp8_dtype,
        )
        self.weight_scale_inv = object()


class _FakeRadixAttention:
    def __init__(self, *args, **kwargs):
        self.kv_b_proj = object()


def _construct_attention(
    config,
    *,
    feature_enabled=True,
    prefill_feature_enabled=False,
    is_nextn=False,
    attn_tp_size=1,
):
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "sglang.srt.layers.glm52_opt.config.is_enabled",
                return_value=False,
            )
        )
        stack.enter_context(
            patch.object(
                deepseek_v2_module,
                "get_parallel",
                return_value=SimpleNamespace(
                    attn_tp_rank=0,
                    attn_tp_size=attn_tp_size,
                    dcp_enabled=False,
                ),
            )
        )
        stack.enter_context(
            patch.object(
                deepseek_v2_module,
                "get_server_args",
                return_value=SimpleNamespace(
                    kv_cache_dtype="auto",
                    device="cpu",
                ),
            )
        )
        for name in (
            "ReplicatedLinear",
            "ColumnParallelLinear",
            "RowParallelLinear",
        ):
            stack.enter_context(patch.object(deepseek_v2_module, name, _FakeLinear))
        stack.enter_context(
            patch.object(
                deepseek_v2_module,
                "RMSNorm",
                lambda *args, **kwargs: object(),
            )
        )
        stack.enter_context(
            patch.object(
                deepseek_v2_module,
                "Indexer",
                lambda *args, **kwargs: object(),
            )
        )
        stack.enter_context(
            patch.object(
                deepseek_v2_module,
                "RadixAttention",
                _FakeRadixAttention,
            )
        )
        for name in (
            "init_mha_forward",
            "init_mla_forward",
            "init_mla_fused_rope_rocm_forward",
            "init_mla_fused_rope_cpu_forward",
        ):
            stack.enter_context(
                patch.object(
                    deepseek_v2_module.DeepseekV2AttentionMLA,
                    name,
                )
            )
        stack.enter_context(
            patch.object(
                envs.SGLANG_OPT_GLM52_FUSED_QKV_A_DECODE_DIRECT_NK,
                "get",
                return_value=feature_enabled,
            )
        )
        stack.enter_context(
            patch.object(
                envs.SGLANG_OPT_GLM52_FUSED_QKV_A_PREFILL_DIRECT_NK,
                "get",
                return_value=prefill_feature_enabled,
            )
        )
        return deepseek_v2_module.DeepseekV2AttentionMLA(
            config,
            hidden_size=6144,
            num_heads=64,
            qk_nope_head_dim=192,
            qk_rope_head_dim=64,
            v_head_dim=256,
            q_lora_rank=2048,
            kv_lora_rank=512,
            layer_id=0,
            prefix="model.layers.0.self_attn",
            skip_rope=True,
            is_nextn=is_nextn,
        )


class TestDecodeContext(unittest.TestCase):
    def test_only_exact_decode_buckets_publish_context(self):
        for m in (16, 32):
            with (
                self.subTest(m=m),
                fused_qkv_a_direct_nk_context(ForwardMode.DECODE, m),
            ):
                self.assertEqual(
                    get_fused_qkv_a_direct_nk_context(),
                    (ForwardMode.DECODE, m),
                )
            self.assertIsNone(get_fused_qkv_a_direct_nk_context())

        unsupported = (
            (ForwardMode.EXTEND, 16),
            (ForwardMode.TARGET_VERIFY, 16),
            (ForwardMode.DRAFT_EXTEND_V2, 32),
            (ForwardMode.DECODE, 1),
            (ForwardMode.DECODE, 64),
            (ForwardMode.DECODE, True),
            (ForwardMode.DECODE, 16.0),
            (object(), 16),
            (1, 16),
            (None, 16),
        )
        for mode, m in unsupported:
            with self.subTest(mode=mode, m=m):
                with fused_qkv_a_direct_nk_context(mode, m):
                    self.assertIsNone(get_fused_qkv_a_direct_nk_context())

    def test_context_nesting_and_exception_restore(self):
        with fused_qkv_a_direct_nk_context(ForwardMode.DECODE, 32):
            outer = get_fused_qkv_a_direct_nk_context()
            with fused_qkv_a_direct_nk_context(ForwardMode.EXTEND, 16):
                self.assertEqual(get_fused_qkv_a_direct_nk_context(), outer)
            with self.assertRaisesRegex(RuntimeError, "sentinel"):
                with fused_qkv_a_direct_nk_context(ForwardMode.DECODE, 16):
                    self.assertEqual(
                        get_fused_qkv_a_direct_nk_context(),
                        (ForwardMode.DECODE, 16),
                    )
                    raise RuntimeError("sentinel")
            self.assertEqual(get_fused_qkv_a_direct_nk_context(), outer)
        self.assertIsNone(get_fused_qkv_a_direct_nk_context())

    def test_only_exact_explicit_prefill_bucket_publishes_context(self):
        with fused_qkv_a_direct_nk_context(
            ForwardMode.EXTEND,
            4096,
            allow_decode=False,
            allow_prefill=True,
        ):
            self.assertEqual(
                get_fused_qkv_a_direct_nk_context(),
                (ForwardMode.EXTEND, 4096),
            )
        self.assertIsNone(get_fused_qkv_a_direct_nk_context())

        unsupported = (
            (ForwardMode.EXTEND, 16),
            (ForwardMode.EXTEND, 4095),
            (ForwardMode.MIXED, 4096),
            (ForwardMode.TARGET_VERIFY, 4096),
            (ForwardMode.SPLIT_PREFILL, 4096),
            (ForwardMode.DLLM_EXTEND, 4096),
            (ForwardMode.DECODE, 4096),
        )
        for mode, m in unsupported:
            with (
                self.subTest(mode=mode, m=m),
                fused_qkv_a_direct_nk_context(
                    mode,
                    m,
                    allow_decode=False,
                    allow_prefill=True,
                ),
            ):
                self.assertIsNone(get_fused_qkv_a_direct_nk_context())


class TestModelIsolation(unittest.TestCase):
    def test_feature_is_default_off(self):
        names = (
            "SGLANG_OPT_GLM52_FUSED_QKV_A_DECODE_DIRECT_NK",
            "SGLANG_OPT_GLM52_FUSED_QKV_A_PREFILL_DIRECT_NK",
        )
        old = {name: os.environ.pop(name, None) for name in names}
        try:
            self.assertFalse(envs.SGLANG_OPT_GLM52_FUSED_QKV_A_DECODE_DIRECT_NK.get())
            self.assertFalse(
                envs.SGLANG_OPT_GLM52_FUSED_QKV_A_PREFILL_DIRECT_NK.get()
            )
        finally:
            for name, value in old.items():
                if value is not None:
                    os.environ[name] = value

    def test_exact_fingerprint_and_negative_matrix(self):
        self.assertTrue(_fingerprint())
        cases = (
            ("glm51_arch", {"architectures": ["Glm4MoeForCausalLM"]}, {}),
            ("model_type", {"model_type": "glm4_moe"}, {}),
            ("head_dim", {"head_dim": 64}, {}),
            ("max_position", {"max_position_embeddings": 202752}, {}),
            ("mtp_marker", {"index_share_for_mtp_iteration": False}, {}),
            ("hidden", {}, {"hidden_size": 4096}),
            ("q_lora", {}, {"q_lora_rank": 1536}),
            ("kv_lora", {}, {"kv_lora_rank": 448}),
            ("rope", {}, {"qk_rope_head_dim": 128}),
            ("non_dsa", {}, {"use_dsa": False}),
            ("nextn", {}, {"is_nextn": True}),
            ("attn_tp", {}, {"attn_tp_size": 2}),
            ("projection_k", {}, {"projection_input_size": 3072}),
            ("projection_n", {}, {"projection_output_size": 2560}),
        )
        for name, config_updates, arg_updates in cases:
            with self.subTest(name=name):
                config = _target_config()
                for key, value in config_updates.items():
                    setattr(config, key, value)
                self.assertFalse(_fingerprint(config, **arg_updates))

    def test_legacy_registry_and_direct_route_are_mutually_exclusive(self):
        with (
            patch(
                "sglang.srt.layers.glm52_opt.config.is_enabled",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.glm52_opt.registry.lookup",
                side_effect=lambda op, phase, m: (
                    object()
                    if (op, phase, m)
                    == ("fused_qkv_a_proj", "prefill", 4096)
                    else None
                ),
            ),
            self.assertRaisesRegex(ValueError, "enable exactly one"),
        ):
            deepseek_v2_module._reject_glm52_fused_qkv_a_route_conflict(
                enable_decode_direct_nk=False,
                enable_prefill_direct_nk=True,
            )

        with (
            patch(
                "sglang.srt.layers.glm52_opt.config.is_enabled",
                return_value=False,
            ),
            patch(
                "sglang.srt.layers.glm52_opt.registry.lookup",
            ) as lookup,
        ):
            deepseek_v2_module._reject_glm52_fused_qkv_a_route_conflict(
                enable_decode_direct_nk=True,
                enable_prefill_direct_nk=True,
            )
        lookup.assert_not_called()

    def test_real_attention_constructor_marks_only_exact_target(self):
        attention = _construct_attention(_target_config())
        projection = attention.fused_qkv_a_proj_with_mqa
        self.assertEqual(
            projection.prefix,
            "model.layers.0.self_attn.fused_qkv_a_proj_with_mqa",
        )
        self.assertTrue(projection._glm52_fused_qkv_a_decode_direct_nk)
        self.assertIs(
            projection.quant_method.w8a8_block_fp8_linear,
            fp8_utils.deepgemm_w8a8_block_fp8_linear_fused_qkv_a_decode_direct_nk_dispatch,
        )

        for name, kwargs in (
            ("feature_off", {"feature_enabled": False}),
            ("nextn", {"is_nextn": True}),
            ("attn_tp2", {"attn_tp_size": 2}),
        ):
            with self.subTest(name=name):
                projection = _construct_attention(
                    _target_config(),
                    **kwargs,
                ).fused_qkv_a_proj_with_mqa
                self.assertFalse(
                    hasattr(
                        projection,
                        "_glm52_fused_qkv_a_decode_direct_nk",
                    )
                )

        prefill_projection = _construct_attention(
            _target_config(),
            feature_enabled=False,
            prefill_feature_enabled=True,
        ).fused_qkv_a_proj_with_mqa
        self.assertTrue(prefill_projection._glm52_fused_qkv_a_prefill_direct_nk)
        self.assertFalse(
            hasattr(
                prefill_projection,
                "_glm52_fused_qkv_a_decode_direct_nk",
            )
        )
        self.assertIs(
            prefill_projection.quant_method.w8a8_block_fp8_linear,
            fp8_utils.deepgemm_w8a8_block_fp8_linear_fused_qkv_a_decode_direct_nk_dispatch,
        )

    def test_scope_is_private_to_non_lora_projection_call(self):
        source = inspect.getsource(
            deepseek_v2_module.DeepseekV2AttentionMLA.prepare_qkv_latent
        )
        context_pos = source.index("fused_qkv_a_direct_nk_context")
        projection_pos = source.index(
            "self.fused_qkv_a_proj_with_mqa(hidden_states)",
            context_pos,
        )
        self.assertGreater(projection_pos, context_pos)
        self.assertIn("not lora_active", source)
        self.assertIn("isinstance(", source)
        self.assertNotIn(
            "fused_qkv_a_direct_nk_context",
            inspect.getsource(deepseek_v2_module.DeepseekV2AttentionMLA.forward_core),
        )

    def test_prepare_qkv_latent_scopes_only_exact_non_lora_decode(self):
        cases = (
            (ForwardMode.DECODE, 16, False, (ForwardMode.DECODE, 16)),
            (ForwardMode.DECODE, 32, False, (ForwardMode.DECODE, 32)),
            (ForwardMode.TARGET_VERIFY, 16, False, None),
            (ForwardMode.EXTEND, 16, False, None),
            (ForwardMode.DECODE, 64, False, None),
            (ForwardMode.DECODE, 16, True, None),
        )

        class Projection:
            _glm52_fused_qkv_a_decode_direct_nk = True

            def __init__(self, expected, lora_active):
                self.expected = expected
                self.set_lora = lora_active
                self.calls = 0

            def __call__(self, hidden_states):
                self.calls += 1
                if get_fused_qkv_a_direct_nk_context() != self.expected:
                    raise AssertionError("projection observed wrong task context")
                return (
                    hidden_states.new_empty((hidden_states.shape[0], 2624)),
                    None,
                )

        for mode, m, lora_active, expected in cases:
            with self.subTest(
                mode=mode.name,
                m=m,
                lora_active=lora_active,
            ):
                projection = Projection(expected, lora_active)
                module = SimpleNamespace(
                    q_lora_rank=2048,
                    use_min_latency_fused_a_gemm=False,
                    fused_qkv_a_proj_with_mqa=projection,
                )
                with patch.object(
                    deepseek_v2_module,
                    "get_bf16_gemm_backend",
                    return_value=SimpleNamespace(
                        is_cutedsl=lambda: False,
                    ),
                ):
                    output = (
                        deepseek_v2_module.DeepseekV2AttentionMLA.prepare_qkv_latent(
                            module,
                            torch.empty((m, 6144), dtype=torch.bfloat16),
                            SimpleNamespace(forward_mode=mode),
                        )
                    )
                self.assertEqual(tuple(output.shape), (m, 2624))
                self.assertEqual(projection.calls, 1)
                self.assertIsNone(get_fused_qkv_a_direct_nk_context())

    def test_prepare_qkv_latent_scopes_only_exact_non_lora_prefill(self):
        cases = (
            (ForwardMode.EXTEND, 4096, False, (ForwardMode.EXTEND, 4096)),
            (ForwardMode.EXTEND, 4095, False, None),
            (ForwardMode.MIXED, 4096, False, None),
            (ForwardMode.TARGET_VERIFY, 4096, False, None),
            (ForwardMode.DECODE, 4096, False, None),
            (ForwardMode.EXTEND, 4096, True, None),
        )

        class Projection:
            _glm52_fused_qkv_a_prefill_direct_nk = True

            def __init__(self, expected, lora_active):
                self.expected = expected
                self.set_lora = lora_active
                self.calls = 0

            def __call__(self, hidden_states):
                self.calls += 1
                if get_fused_qkv_a_direct_nk_context() != self.expected:
                    raise AssertionError("projection observed wrong task context")
                return (
                    hidden_states.new_empty((hidden_states.shape[0], 2624)),
                    None,
                )

        for mode, m, lora_active, expected in cases:
            with self.subTest(mode=mode.name, m=m, lora_active=lora_active):
                projection = Projection(expected, lora_active)
                module = SimpleNamespace(
                    q_lora_rank=2048,
                    use_min_latency_fused_a_gemm=False,
                    fused_qkv_a_proj_with_mqa=projection,
                )
                with patch.object(
                    deepseek_v2_module,
                    "get_bf16_gemm_backend",
                    return_value=SimpleNamespace(is_cutedsl=lambda: False),
                ):
                    output = (
                        deepseek_v2_module.DeepseekV2AttentionMLA.prepare_qkv_latent(
                            module,
                            torch.empty((m, 6144), dtype=torch.bfloat16),
                            SimpleNamespace(forward_mode=mode),
                        )
                    )
                self.assertEqual(tuple(output.shape), (m, 2624))
                self.assertEqual(projection.calls, 1)
                self.assertIsNone(get_fused_qkv_a_direct_nk_context())


class TestPackedAbiAndRunner(unittest.TestCase):
    def test_exact_packed_abi(self):
        q_input, weight, x_scale, weight_scale = _packed_tensors()
        patches = _support_patches()
        with patches[0], patches[1], patches[2], patches[3]:
            self.assertTrue(
                fp8_utils._is_glm52_fused_qkv_a_decode_direct_nk_packed_weight_abi(
                    weight,
                    weight_scale,
                    [128, 128],
                )
            )
            self.assertTrue(
                fp8_utils._is_glm52_fused_qkv_a_decode_direct_nk_activation_abi(
                    q_input,
                    x_scale,
                    m=16,
                    device=weight.device,
                )
            )

            weight_scale._stride = (12, 1)
            self.assertFalse(
                fp8_utils._is_glm52_fused_qkv_a_decode_direct_nk_packed_weight_abi(
                    weight,
                    weight_scale,
                    [128, 128],
                )
            )
            weight_scale._stride = (1, 2624)
            x_scale._stride = (12, 1)
            self.assertFalse(
                fp8_utils._is_glm52_fused_qkv_a_decode_direct_nk_activation_abi(
                    q_input,
                    x_scale,
                    m=16,
                    device=weight.device,
                )
            )

    def test_bound_runner_uses_direct_path_for_all_exact_buckets(self):
        for m, mode, context_kwargs in (
            (16, ForwardMode.DECODE, {}),
            (32, ForwardMode.DECODE, {}),
            (
                4096,
                ForwardMode.EXTEND,
                {"allow_decode": False, "allow_prefill": True},
            ),
        ):
            q_input, weight, x_scale, weight_scale = _packed_tensors(m)
            output = object()
            patches = _support_patches()
            with (
                self.subTest(m=m),
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patch.object(
                    fp8_utils,
                    "sglang_per_token_group_quant_fp8",
                    return_value=(q_input, x_scale),
                ) as quantize,
                patch.object(
                    fp8_utils,
                    "w8a8_block_fp8_matmul_deepgemm_fused_qkv_a_compiled_nk",
                    return_value=output,
                ) as direct,
                patch.object(
                    fp8_utils,
                    "_glm52_fused_qkv_a_prefill_direct_nk_packed_gemm",
                    return_value=output,
                ) as prefill_direct,
                patch.object(
                    fp8_utils,
                    "deepgemm_w8a8_block_fp8_linear_with_fallback",
                ) as stock,
            ):
                runner = fp8_utils.bind_glm52_fused_qkv_a_decode_direct_nk_runner(
                    weight,
                    weight_scale,
                    [128, 128],
                )
                self.assertIsInstance(
                    runner,
                    fp8_utils.Glm52FusedQkvADecodeDirectNkRunner,
                )
                with fused_qkv_a_direct_nk_context(mode, m, **context_kwargs):
                    result = runner(
                        _bf16_input(m),
                        weight,
                        [128, 128],
                        weight_scale,
                    )

            self.assertIs(result, output)
            quantize.assert_called_once()
            if m == 4096:
                prefill_direct.assert_called_once_with(
                    q_input,
                    weight,
                    x_scale,
                    weight_scale,
                    [128, 128],
                    torch.bfloat16,
                    abi_prevalidated=True,
                )
                direct.assert_not_called()
            else:
                direct.assert_called_once()
                prefill_direct.assert_not_called()
            stock.assert_not_called()

    def test_postquant_abi_failure_propagates_without_stock(self):
        q_input, weight, x_scale, weight_scale = _packed_tensors()
        x_scale._stride = (12, 1)
        patches = _support_patches()
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patch.object(
                fp8_utils,
                "sglang_per_token_group_quant_fp8",
                return_value=(q_input, x_scale),
            ),
            patch.object(
                fp8_utils,
                "w8a8_block_fp8_matmul_deepgemm_fused_qkv_a_compiled_nk",
            ) as direct,
            patch.object(
                fp8_utils,
                "deepgemm_w8a8_block_fp8_linear_with_fallback",
            ) as stock,
        ):
            runner = fp8_utils.Glm52FusedQkvADecodeDirectNkRunner(
                weight,
                weight_scale,
            )
            with (
                fused_qkv_a_direct_nk_context(ForwardMode.DECODE, 16),
                self.assertRaisesRegex(RuntimeError, "activation quantizer"),
            ):
                runner(
                    _bf16_input(),
                    weight,
                    [128, 128],
                    weight_scale,
                )

        direct.assert_not_called()
        stock.assert_not_called()

    def test_identity_or_version_drift_falls_back_before_quantization(self):
        _, weight, _, weight_scale = _packed_tensors()
        runner = fp8_utils.Glm52FusedQkvADecodeDirectNkRunner(
            weight,
            weight_scale,
        )
        cases = (
            ("weight_identity", _FakeTensor((2624, 6144), fp8_utils.fp8_dtype)),
            ("version", weight),
        )
        for name, call_weight in cases:
            with self.subTest(name=name):
                if name == "version":
                    weight._version += 1
                sentinel = object()
                with (
                    fused_qkv_a_direct_nk_context(ForwardMode.DECODE, 16),
                    patch.object(
                        fp8_utils,
                        "sglang_per_token_group_quant_fp8",
                    ) as quantize,
                    patch.object(
                        fp8_utils,
                        "deepgemm_w8a8_block_fp8_linear_with_fallback",
                        return_value=sentinel,
                    ) as stock,
                ):
                    result = runner(
                        _bf16_input(),
                        call_weight,
                        [128, 128],
                        weight_scale,
                    )
                self.assertIs(result, sentinel)
                quantize.assert_not_called()
                stock.assert_called_once()

    def test_dynamic_metadata_misses_fall_back_before_quantization(self):
        _, weight, _, weight_scale = _packed_tensors()
        cases = (
            ("no_context", nullcontext(), _bf16_input(), None, None, [128, 128]),
            (
                "target_verify",
                fused_qkv_a_direct_nk_context(
                    ForwardMode.TARGET_VERIFY,
                    16,
                ),
                _bf16_input(),
                None,
                None,
                [128, 128],
            ),
            (
                "m_mismatch",
                fused_qkv_a_direct_nk_context(ForwardMode.DECODE, 16),
                _bf16_input(32),
                None,
                None,
                [128, 128],
            ),
            (
                "wrong_k",
                fused_qkv_a_direct_nk_context(ForwardMode.DECODE, 16),
                _FakeTensor((16, 3072), torch.bfloat16),
                None,
                None,
                [128, 128],
            ),
            (
                "wrong_dtype",
                fused_qkv_a_direct_nk_context(ForwardMode.DECODE, 16),
                _FakeTensor((16, 6144), torch.float16),
                None,
                None,
                [128, 128],
            ),
            (
                "noncuda",
                fused_qkv_a_direct_nk_context(ForwardMode.DECODE, 16),
                _FakeTensor(
                    (16, 6144),
                    torch.bfloat16,
                    device="cpu",
                    is_cuda=False,
                ),
                None,
                None,
                [128, 128],
            ),
            (
                "noncontiguous",
                fused_qkv_a_direct_nk_context(ForwardMode.DECODE, 16),
                _FakeTensor(
                    (16, 6144),
                    torch.bfloat16,
                    contiguous=False,
                ),
                None,
                None,
                [128, 128],
            ),
            (
                "input_scale",
                fused_qkv_a_direct_nk_context(ForwardMode.DECODE, 16),
                _bf16_input(),
                object(),
                None,
                [128, 128],
            ),
            (
                "bias",
                fused_qkv_a_direct_nk_context(ForwardMode.DECODE, 16),
                _bf16_input(),
                None,
                object(),
                [128, 128],
            ),
            (
                "block_recipe",
                fused_qkv_a_direct_nk_context(ForwardMode.DECODE, 16),
                _bf16_input(),
                None,
                None,
                [64, 128],
            ),
        )
        for name, context, input_tensor, input_scale, bias, block_size in cases:
            with self.subTest(name=name):
                runner = fp8_utils.Glm52FusedQkvADecodeDirectNkRunner(
                    weight,
                    weight_scale,
                )
                sentinel = object()
                with (
                    context,
                    patch.object(
                        fp8_utils,
                        "sglang_per_token_group_quant_fp8",
                    ) as quantize,
                    patch.object(
                        fp8_utils,
                        "deepgemm_w8a8_block_fp8_linear_with_fallback",
                        return_value=sentinel,
                    ) as stock,
                ):
                    result = runner(
                        input_tensor,
                        weight,
                        block_size,
                        weight_scale,
                        input_scale=input_scale,
                        bias=bias,
                    )
                self.assertIs(result, sentinel)
                quantize.assert_not_called()
                stock.assert_called_once()

    def test_configure_is_exact_and_lifecycle_rebind_is_source_proved(self):
        _, weight, _, weight_scale = _packed_tensors()
        layer = SimpleNamespace(
            quant_method=_method(),
            weight=weight,
            weight_scale_inv=weight_scale,
            input_size_per_partition=6144,
            output_size_per_partition=2624,
        )
        patches = _support_patches()
        with patches[0], patches[1], patches[2], patches[3]:
            self.assertTrue(
                fp8_module.configure_glm52_fused_qkv_a_decode_direct_nk(layer)
            )
        self.assertTrue(layer._glm52_fused_qkv_a_decode_direct_nk)
        self.assertIsInstance(
            layer.quant_method.w8a8_block_fp8_linear,
            fp8_utils.Glm52FusedQkvADecodeDirectNkRunner,
        )
        self.assertTrue(
            fp8_module.configure_glm52_fused_qkv_a_prefill_direct_nk(layer)
        )
        self.assertTrue(layer._glm52_fused_qkv_a_prefill_direct_nk)

        layer.output_size_per_partition = 2625
        layer.quant_method = _method()
        self.assertFalse(fp8_module.configure_glm52_fused_qkv_a_decode_direct_nk(layer))
        lifecycle = inspect.getsource(
            fp8_module.Fp8LinearMethod.process_weights_after_loading_block_quant
        )
        self.assertIn(
            "bind_glm52_fused_qkv_a_decode_direct_nk_runner",
            lifecycle,
        )
        self.assertIn("or use_glm52_fused_qkv_a_direct_nk_runner", lifecycle)


class TestCompiledNkLeaf(unittest.TestCase):
    def test_kernel_wrapper_has_exact_bucket_and_custom_op(self):
        q_input, weight, x_scale, weight_scale = _packed_tensors()
        output = _FakeTensor((16, 2624), torch.bfloat16)
        with (
            patch.object(
                fp8_kernel,
                "prepare_block_fp8_matmul_inputs",
                return_value=(16, 2624, 6144, output),
            ),
            patch.object(
                fp8_kernel.deep_gemm_wrapper,
                "ENABLE_JIT_DEEPGEMM",
                True,
            ),
            patch.object(
                fp8_kernel,
                "deep_gemm_fp8_fp8_bf16_nt_fused_qkv_a_compiled_nk",
            ) as custom_op,
        ):
            result = fp8_kernel.w8a8_block_fp8_matmul_deepgemm_fused_qkv_a_compiled_nk(
                q_input,
                weight,
                x_scale,
                weight_scale,
                [128, 128],
                torch.bfloat16,
            )
        self.assertIs(result, output)
        custom_op.assert_called_once_with(
            q_input,
            x_scale,
            weight,
            weight_scale,
            output,
        )

    def test_prefill_leaf_uses_direct_compiled_nk_call(self):
        q_input, weight, x_scale, weight_scale = _packed_tensors(4096)
        output = _FakeTensor((4096, 2624), torch.bfloat16)
        q_input.new_empty = Mock(return_value=output)
        deep_gemm = SimpleNamespace(fp8_gemm_nt=Mock())
        patches = _support_patches()
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
        ):
            result = fp8_utils._glm52_fused_qkv_a_prefill_direct_nk_packed_gemm(
                q_input,
                weight,
                x_scale,
                weight_scale,
                [128, 128],
                torch.bfloat16,
            )
        self.assertIs(result, output)
        q_input.new_empty.assert_called_once_with(
            (4096, 2624),
            dtype=torch.bfloat16,
        )
        deep_gemm.fp8_gemm_nt.assert_called_once_with(
            (q_input, x_scale),
            (weight, weight_scale),
            output,
            compiled_dims="nk",
        )

    def test_entrypoint_calls_compiled_dims_nk(self):
        lhs = (
            _FakeTensor((16, 6144), fp8_utils.fp8_dtype),
            _FakeTensor((16, 12), torch.int32, stride=(1, 16)),
        )
        rhs = (
            _FakeTensor((2624, 6144), fp8_utils.fp8_dtype),
            _FakeTensor((2624, 12), torch.int32, stride=(1, 2624)),
        )
        out = _FakeTensor((16, 2624), torch.bfloat16)
        deep_gemm = SimpleNamespace(fp8_gemm_nt=Mock())
        with (
            patch.object(
                deep_gemm_entrypoint,
                "DEEPGEMM_SUPPORTS_COMPILED_DIMS",
                True,
            ),
            patch.object(
                deep_gemm_entrypoint,
                "deep_gemm",
                deep_gemm,
                create=True,
            ),
            patch.object(deep_gemm_entrypoint, "_sanity_check_input"),
            patch.object(
                compile_utils,
                "deep_gemm_execution_hook",
                return_value=nullcontext(),
            ),
        ):
            deep_gemm_entrypoint.gemm_nt_f8f8bf16_fused_qkv_a_compiled_nk(
                lhs,
                rhs,
                out,
            )
        deep_gemm.fp8_gemm_nt.assert_called_once_with(
            lhs,
            rhs,
            out,
            compiled_dims="nk",
        )

    def test_compile_identity_is_distinct_and_m_list_is_bounded(self):
        kernel_type = (
            compile_utils.DeepGemmKernelType.GEMM_NT_F8F8BF16_FUSED_QKV_A_COMPILED_NK
        )
        self.assertNotEqual(
            kernel_type,
            compile_utils.DeepGemmKernelType.GEMM_NT_F8F8BF16,
        )
        source = inspect.getsource(compile_utils._maybe_compile_deep_gemm_one_type_all)
        self.assertIn("[16, 32]", source)
        executor_source = inspect.getsource(
            compile_utils._FusedQkvACompiledNkWarmupExecutor
        )
        self.assertIn("n == 2624", executor_source)
        self.assertIn("k == 6144", executor_source)
        self.assertIn('compiled_dims="nk"', executor_source)

    def test_candidate_code_does_not_use_legacy_registry_or_global_knob(self):
        source = inspect.getsource(
            fp8_utils._glm52_fused_qkv_a_decode_direct_nk_from_bf16
        )
        self.assertNotIn("try_dispatch_fp8_gemm", source)
        self.assertNotIn("set_num_sms", source)
        self.assertNotIn("deepgemm_w8a8_block_fp8_linear_with_fallback", source)


if __name__ == "__main__":
    unittest.main()
