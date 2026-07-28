"""CPU-only contract tests for GLM-5.2 attention O decode fixed-N/K dispatch."""

from __future__ import annotations

import inspect
import os
import threading
import unittest
from contextlib import ExitStack, contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from sglang.kernels.ops.quantization.fp8_kernel import (
    deep_gemm_fp8_fp8_bf16_nt_compiled_nk,
)
from sglang.srt.environ import envs
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.deep_gemm_wrapper import compile_utils
from sglang.srt.layers.deep_gemm_wrapper import entrypoint as deep_gemm_entrypoint
from sglang.srt.layers.glm52_opt import dispatch as glm52_dispatch
from sglang.srt.layers.quantization import fp8 as fp8_module
from sglang.srt.layers.quantization import fp8_utils
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from sglang.srt.model_executor.runner import (
    decode_cuda_graph_runner as decode_cuda_graph_runner_module,
)
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
    FullCudaGraphBackend,
)
from sglang.srt.models import deepseek_v2 as deepseek_v2_module
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla import (
    DeepseekMLAForwardMixin,
)
from sglang.srt.models.deepseek_v2 import (
    DeepseekV2AttentionMLA,
    DeepseekV2ForCausalLM,
)
from sglang.srt.models.glm4_moe import (
    Glm4MoeAttention,
    GlmMoeDsaForCausalLM,
)
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode


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
        self._stride = tuple(stride) if stride is not None else (self.shape[-1], 1)
        self._contiguous = contiguous

    def is_contiguous(self):
        return self._contiguous

    def stride(self):
        return self._stride

    def view(self, *shape):
        return self


class _FakeOutput:
    def __init__(self):
        self.to_dtype = None
        self.view_shape = None

    def to(self, *, dtype):
        self.to_dtype = dtype
        return self

    def view(self, *shape):
        self.view_shape = shape
        return self


def _method(runner=fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback):
    method = object.__new__(fp8_module.Fp8LinearMethod)
    method.use_marlin = False
    method.use_mxfp8 = False
    method.block_quant = True
    method.w8a8_block_fp8_linear = runner
    method.weight_block_size = [128, 128]
    return method


def _layer(*, marked=True, prefix="model.layers.0.self_attn.o_proj"):
    layer = SimpleNamespace(
        prefix=prefix,
        weight=object(),
        weight_scale_inv=object(),
    )
    if marked is not None:
        layer._glm52_attn_o_decode_direct_nk_graph = marked
    return layer


def _configured_layer():
    layer = _layer(marked=None)
    layer.quant_method = _method()
    if not fp8_module.configure_glm52_attn_o_decode_direct_nk(layer):
        raise AssertionError("test fixture did not mark the compatible layer")
    return layer.quant_method, layer


def _model(*layers):
    return SimpleNamespace(modules=lambda: iter(layers))


class _FakeLinear:
    def __init__(self, *args, **kwargs):
        self.prefix = kwargs.get("prefix")
        self.tp_size = kwargs.get("tp_size", 1)
        self.input_size_per_partition = int(args[0]) // self.tp_size
        self.quant_method = _method()
        self.weight = SimpleNamespace(dtype=torch.bfloat16, shape=(16, 256))


class _FakeRadixAttention:
    def __init__(self, *args, **kwargs):
        self.kv_b_proj = object()


def _dsa_config(
    architecture="GlmMoeDsaForCausalLM",
    *,
    model_type="glm_moe_dsa",
):
    return SimpleNamespace(
        architectures=[architecture],
        model_type=model_type,
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


def _glm51_config():
    config = _dsa_config()
    config.head_dim = 64
    config.max_position_embeddings = 202752
    del config.index_share_for_mtp_iteration
    return config


def _construct_dsa_attention(
    config,
    *,
    is_nextn=False,
    attn_tp_size=1,
    feature_enabled=True,
):
    with ExitStack() as stack:
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
                return_value=SimpleNamespace(kv_cache_dtype="auto", device="cpu"),
            )
        )
        for name in ("ReplicatedLinear", "ColumnParallelLinear", "RowParallelLinear"):
            stack.enter_context(patch.object(deepseek_v2_module, name, _FakeLinear))
        stack.enter_context(
            patch.object(
                deepseek_v2_module, "RMSNorm", lambda *args, **kwargs: object()
            )
        )
        stack.enter_context(
            patch.object(
                deepseek_v2_module, "Indexer", lambda *args, **kwargs: object()
            )
        )
        stack.enter_context(
            patch.object(deepseek_v2_module, "RadixAttention", _FakeRadixAttention)
        )
        for name in (
            "init_mha_forward",
            "init_mla_forward",
            "init_mla_fused_rope_rocm_forward",
            "init_mla_fused_rope_cpu_forward",
        ):
            stack.enter_context(patch.object(DeepseekV2AttentionMLA, name))
        stack.enter_context(
            patch.object(
                envs.SGLANG_OPT_GLM52_ATTN_O_DECODE_DIRECT_NK,
                "get",
                return_value=feature_enabled,
            )
        )
        return DeepseekV2AttentionMLA(
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


class TestModelReachability(unittest.TestCase):
    def test_glm52_dsa_target_reaches_deepseek_attention_o_projection(self):
        self.assertTrue(
            issubclass(GlmMoeDsaForCausalLM, DeepseekV2ForCausalLM),
        )
        attention = _construct_dsa_attention(_dsa_config())
        self.assertEqual(
            attention.o_proj.prefix,
            "model.layers.0.self_attn.o_proj",
        )
        self.assertTrue(attention.o_proj._glm52_attn_o_decode_direct_nk_graph)
        self.assertEqual(attention.o_proj.input_size_per_partition, 16384)
        self.assertIs(
            attention.o_proj.quant_method.w8a8_block_fp8_linear,
            fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback,
        )

    def test_constructor_weight_processing_produces_real_packed_scale(self):
        attention = _construct_dsa_attention(_dsa_config())
        method = attention.o_proj.quant_method
        method.quant_config = SimpleNamespace(weight_block_size=[128, 128])
        method.is_checkpoint_fp8_serialized = True
        method.convert_mxfp8_to_block = False

        attention.o_proj.orig_dtype = torch.bfloat16
        attention.o_proj.weight = torch.nn.Parameter(
            torch.empty((6144, 16384), dtype=fp8_utils.fp8_dtype),
            requires_grad=False,
        )
        attention.o_proj.weight_scale_inv = torch.nn.Parameter(
            torch.ones((48, 128), dtype=torch.float32),
            requires_grad=False,
        )
        self.assertEqual(
            tuple(attention.o_proj.weight_scale_inv.shape),
            (48, 128),
        )
        self.assertEqual(
            attention.o_proj.weight_scale_inv.dtype,
            torch.float32,
        )

        def cpu_reference_requant(weight, unpacked_scale, block_size):
            self.assertEqual(block_size, [128, 128])
            packed_scale = fp8_utils.transform_scale_ue8m0(
                unpacked_scale,
                mn=int(weight.shape[0]),
                use_torch_impl=True,
            )
            return weight.detach(), packed_scale

        with (
            patch.object(fp8_module, "_is_cpu", False),
            patch.object(fp8_module, "_is_fp8_fnuz", False),
            patch.object(deep_gemm_wrapper, "ENABLE_JIT_DEEPGEMM", True),
            patch.object(deep_gemm_wrapper, "DEEPGEMM_SCALE_UE8M0", True),
            patch.object(
                fp8_utils,
                "requant_weight_ue8m0",
                side_effect=cpu_reference_requant,
            ) as requant,
        ):
            method.process_weights_after_loading(attention.o_proj)

        requant.assert_called_once()
        packed_scale = attention.o_proj.weight_scale_inv
        self.assertEqual(tuple(packed_scale.shape), (6144, 32))
        self.assertEqual(tuple(packed_scale.stride()), (1, 6144))
        self.assertEqual(packed_scale.dtype, torch.int32)
        self.assertTrue(packed_scale.format_ue8m0)
        self.assertIs(
            method.w8a8_block_fp8_linear,
            fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback,
        )

    def test_non_deepgemm_weight_processing_identity_remains_false(self):
        method = _method(lambda **kwargs: None)
        method.quant_config = SimpleNamespace(weight_block_size=[128, 128])
        method.is_checkpoint_fp8_serialized = True
        method.convert_mxfp8_to_block = False
        layer = SimpleNamespace(
            orig_dtype=torch.bfloat16,
            weight=torch.nn.Parameter(
                torch.empty((64, 128), dtype=fp8_utils.fp8_dtype),
                requires_grad=False,
            ),
            weight_scale_inv=torch.nn.Parameter(
                torch.ones((1, 1), dtype=torch.float32),
                requires_grad=False,
            ),
        )
        with (
            patch.object(fp8_module, "_is_cpu", False),
            patch.object(fp8_module, "_is_fp8_fnuz", False),
            patch.object(
                fp8_module,
                "requant_block_scale_ue8m0_for_deepgemm",
                return_value=False,
            ) as requant,
        ):
            method.process_weights_after_loading(layer)

        self.assertFalse(requant.call_args.kwargs["use_deepgemm_runner"])

    def test_non_target_nextn_and_disabled_construction_stay_unmarked(self):
        non_target = _construct_dsa_attention(
            _dsa_config("DeepseekV3ForCausalLM"),
        )
        nextn = _construct_dsa_attention(_dsa_config(), is_nextn=True)
        disabled = _construct_dsa_attention(
            _dsa_config(),
            feature_enabled=False,
        )
        marker = "_glm52_attn_o_decode_direct_nk_graph"
        self.assertFalse(hasattr(non_target.o_proj, marker))
        self.assertFalse(hasattr(nextn.o_proj, marker))
        self.assertFalse(hasattr(disabled.o_proj, marker))
        for attention in (non_target, nextn, disabled):
            self.assertIs(
                attention.o_proj.quant_method.w8a8_block_fp8_linear,
                fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback,
            )

    def test_glm51_fingerprint_negative_constructor(self):
        glm51 = _construct_dsa_attention(_glm51_config())
        self.assertFalse(hasattr(glm51.o_proj, "_glm52_attn_o_decode_direct_nk_graph"))
        self.assertIs(
            glm51.o_proj.quant_method.w8a8_block_fp8_linear,
            fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback,
        )

    def test_attention_tp_topology_isolation(self):
        tp8 = _construct_dsa_attention(_dsa_config(), attn_tp_size=8)
        self.assertEqual(tp8.o_proj.input_size_per_partition, 2048)
        self.assertFalse(hasattr(tp8.o_proj, "_glm52_attn_o_decode_direct_nk_graph"))
        self.assertIs(
            tp8.o_proj.quant_method.w8a8_block_fp8_linear,
            fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback,
        )

    def test_exact_model_fingerprint_topology_and_old_path_guards(self):
        predicate = deepseek_v2_module._is_glm52_dsa_target_attn_o
        kwargs = {
            "hidden_size": 6144,
            "num_heads": 64,
            "qk_nope_head_dim": 192,
            "v_head_dim": 256,
            "use_dsa": True,
            "is_nextn": False,
            "attn_tp_size": 1,
            "o_proj_input_size_per_partition": 16384,
        }
        self.assertTrue(predicate(_dsa_config(), **kwargs))
        for name, config, updates in (
            (
                "architecture",
                _dsa_config("DeepseekV3ForCausalLM"),
                {},
            ),
            (
                "model_type",
                _dsa_config(model_type="deepseek_v3"),
                {},
            ),
            ("glm51", _glm51_config(), {}),
            ("hidden_size", _dsa_config(), {"hidden_size": 4096}),
            ("qk_nope", _dsa_config(), {"qk_nope_head_dim": 128}),
            ("projection_k", _dsa_config(), {"v_head_dim": 128}),
            ("not_dsa", _dsa_config(), {"use_dsa": False}),
            ("nextn", _dsa_config(), {"is_nextn": True}),
            ("attn_tp8", _dsa_config(), {"attn_tp_size": 8}),
            (
                "local_k_2048",
                _dsa_config(),
                {"o_proj_input_size_per_partition": 2048},
            ),
        ):
            with self.subTest(name=name):
                self.assertFalse(predicate(config, **(kwargs | updates)))
        self.assertNotIn(
            "_glm52_attn_o_decode_direct_nk_graph",
            inspect.getsource(Glm4MoeAttention.__init__),
        )


class TestGraphCaptureLifecycle(unittest.TestCase):
    def test_feature_is_default_off(self):
        name = "SGLANG_OPT_GLM52_ATTN_O_DECODE_DIRECT_NK"
        old = os.environ.pop(name, None)
        try:
            self.assertFalse(envs.SGLANG_OPT_GLM52_ATTN_O_DECODE_DIRECT_NK.get())
        finally:
            if old is not None:
                os.environ[name] = old

    def test_configure_is_marker_only_and_eager_runner_stays_exact_stock(self):
        method, layer = _configured_layer()

        self.assertTrue(layer._glm52_attn_o_decode_direct_nk_graph)
        self.assertFalse(hasattr(layer, "_glm52_attn_o_decode_direct_nk"))
        self.assertIs(
            method.w8a8_block_fp8_linear,
            fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback,
        )
        apply_source = inspect.getsource(fp8_module.Fp8LinearMethod.apply)
        self.assertNotIn("_glm52_attn_o_decode_direct_nk_graph", apply_source)
        self.assertNotIn(
            "deepgemm_w8a8_block_fp8_linear_attn_o_decode_direct_nk",
            apply_source,
        )

        layer.weight = _FakeTensor((6144, 16384), fp8_utils.fp8_dtype)
        output = _FakeOutput()
        with (
            patch.object(
                fp8_module,
                "use_intel_amx_backend",
                return_value=False,
            ),
            patch.object(
                fp8_utils,
                "sglang_per_token_group_quant_fp8",
                return_value=(object(), object()),
            ) as quant,
            patch.object(
                glm52_dispatch,
                "try_dispatch_fp8_gemm",
                return_value=output,
            ) as registry,
        ):
            result = method.apply(
                layer,
                _FakeTensor((16, 16384), torch.bfloat16),
            )

        self.assertIs(result, output)
        quant.assert_called_once()
        registry.assert_called_once()
        self.assertIs(
            method.w8a8_block_fp8_linear,
            fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback,
        )

    def test_arm_temporarily_swaps_and_restores_exact_stock_runner(self):
        method, layer = _configured_layer()
        stock = fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback
        direct = fp8_utils.deepgemm_w8a8_block_fp8_linear_attn_o_decode_direct_nk

        with fp8_module.arm_glm52_attn_o_decode_direct_nk_for_cuda_graph(
            _model(layer)
        ) as armed:
            self.assertEqual(armed, 1)
            self.assertIs(method.w8a8_block_fp8_linear, direct)

        self.assertIs(method.w8a8_block_fp8_linear, stock)

    def test_nested_arming_is_idempotent_and_restores_at_outer_exit(self):
        method, first = _configured_layer()
        second = _layer(marked=True)
        second.quant_method = method
        model = _model(first, second)
        stock = fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback
        direct = fp8_utils.deepgemm_w8a8_block_fp8_linear_attn_o_decode_direct_nk

        with fp8_module.arm_glm52_attn_o_decode_direct_nk_for_cuda_graph(
            model
        ) as outer_armed:
            self.assertEqual(outer_armed, 1)
            self.assertIs(method.w8a8_block_fp8_linear, direct)
            with fp8_module.arm_glm52_attn_o_decode_direct_nk_for_cuda_graph(
                model
            ) as inner_armed:
                self.assertEqual(inner_armed, 0)
                self.assertIs(method.w8a8_block_fp8_linear, direct)
            self.assertIs(method.w8a8_block_fp8_linear, direct)

        self.assertIs(method.w8a8_block_fp8_linear, stock)

    def test_parallel_arming_is_serialized_for_the_full_capture_scope(self):
        method, layer = _configured_layer()
        model = _model(layer)
        stock = fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback
        direct = fp8_utils.deepgemm_w8a8_block_fp8_linear_attn_o_decode_direct_nk
        owner_entered = threading.Event()
        contender_attempting = threading.Event()
        release_owner = threading.Event()
        contender_entered = threading.Event()
        failures = []

        def owner():
            try:
                with fp8_module.arm_glm52_attn_o_decode_direct_nk_for_cuda_graph(
                    model
                ) as armed:
                    self.assertEqual(armed, 1)
                    self.assertIs(method.w8a8_block_fp8_linear, direct)
                    owner_entered.set()
                    if not release_owner.wait(timeout=2):
                        raise AssertionError("test did not release the owner")
            # Transfer any thread failure to the unittest-owning thread.
            except BaseException as error:  # noqa: BLE001
                failures.append(error)

        def contender():
            try:
                if not owner_entered.wait(timeout=2):
                    raise AssertionError("owner did not enter the arming scope")
                contender_attempting.set()
                with fp8_module.arm_glm52_attn_o_decode_direct_nk_for_cuda_graph(
                    model
                ) as armed:
                    self.assertEqual(armed, 1)
                    self.assertIs(method.w8a8_block_fp8_linear, direct)
                    contender_entered.set()
            # Transfer any thread failure to the unittest-owning thread.
            except BaseException as error:  # noqa: BLE001
                failures.append(error)

        owner_thread = threading.Thread(target=owner)
        contender_thread = threading.Thread(target=contender)
        owner_thread.start()
        contender_thread.start()
        self.assertTrue(contender_attempting.wait(timeout=2))
        self.assertFalse(contender_entered.wait(timeout=0.1))
        release_owner.set()
        owner_thread.join(timeout=2)
        contender_thread.join(timeout=2)

        self.assertFalse(owner_thread.is_alive())
        self.assertFalse(contender_thread.is_alive())
        self.assertEqual(failures, [])
        self.assertTrue(contender_entered.is_set())
        self.assertIs(method.w8a8_block_fp8_linear, stock)

    def test_foreign_or_nonstock_runner_fails_closed(self):
        method, layer = _configured_layer()
        direct = fp8_utils.deepgemm_w8a8_block_fp8_linear_attn_o_decode_direct_nk
        method.w8a8_block_fp8_linear = direct
        try:
            with (
                self.assertRaisesRegex(
                    RuntimeError,
                    "without the exact stock runner",
                ),
                fp8_module.arm_glm52_attn_o_decode_direct_nk_for_cuda_graph(
                    _model(layer)
                ),
            ):
                self.fail("foreign runner must not enter the capture scope")
        finally:
            method.w8a8_block_fp8_linear = (
                fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback
            )

    def test_body_exception_restores_exact_stock_runner(self):
        method, layer = _configured_layer()
        stock = fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback

        with (
            self.assertRaisesRegex(ValueError, "capture failed"),
            fp8_module.arm_glm52_attn_o_decode_direct_nk_for_cuda_graph(_model(layer)),
        ):
            raise ValueError("capture failed")

        self.assertIs(method.w8a8_block_fp8_linear, stock)

    def test_unexpected_mutation_fails_loudly_after_restoring_stock(self):
        method, layer = _configured_layer()
        stock = fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback

        with (
            self.assertRaisesRegex(
                RuntimeError,
                "unexpected runner mutation",
            ),
            fp8_module.arm_glm52_attn_o_decode_direct_nk_for_cuda_graph(_model(layer)),
        ):
            method.w8a8_block_fp8_linear = Mock()

        self.assertIs(method.w8a8_block_fp8_linear, stock)

    def test_stock_function_has_no_candidate_branch(self):
        source = inspect.getsource(
            fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback,
        )
        self.assertNotIn("attn_o_decode_direct_nk", source)
        self.assertLess(
            source.index("sglang_per_token_group_quant_fp8"),
            source.index("try_dispatch_fp8_gemm"),
        )
        self.assertLess(
            source.index("try_dispatch_fp8_gemm"),
            source.index("w8a8_block_fp8_matmul_deepgemm"),
        )

    def test_unmarked_and_incompatible_layers_never_arm(self):
        stock = fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback

        unmarked = _layer(marked=None)
        unmarked.quant_method = _method()

        marlin = _layer(marked=True)
        marlin.quant_method = _method()
        marlin.quant_method.use_marlin = True

        mxfp8 = _layer(marked=True)
        mxfp8.quant_method = _method()
        mxfp8.quant_method.use_mxfp8 = True

        non_block = _layer(marked=True)
        non_block.quant_method = _method()
        non_block.quant_method.block_quant = False

        non_fp8 = _layer(marked=True)
        non_fp8.quant_method = SimpleNamespace()

        for name, layer in (
            ("unmarked", unmarked),
            ("marlin", marlin),
            ("mxfp8", mxfp8),
            ("non_block", non_block),
            ("non_fp8", non_fp8),
        ):
            with self.subTest(name=name):
                original = getattr(
                    layer.quant_method,
                    "w8a8_block_fp8_linear",
                    None,
                )
                with fp8_module.arm_glm52_attn_o_decode_direct_nk_for_cuda_graph(
                    _model(layer)
                ) as armed:
                    self.assertEqual(armed, 0)
                    self.assertIs(
                        getattr(
                            layer.quant_method,
                            "w8a8_block_fp8_linear",
                            None,
                        ),
                        original,
                    )
                self.assertIs(
                    getattr(
                        layer.quant_method,
                        "w8a8_block_fp8_linear",
                        None,
                    ),
                    original,
                )

        self.assertIs(unmarked.quant_method.w8a8_block_fp8_linear, stock)

    def test_incompatible_quant_methods_are_not_marked(self):
        marlin = _method()
        marlin.use_marlin = True
        mxfp8 = _method()
        mxfp8.use_mxfp8 = True
        non_block = _method()
        non_block.block_quant = False
        incompatible = (
            ("non_fp8", SimpleNamespace()),
            ("non_stock_runner", _method(Mock())),
            ("marlin", marlin),
            ("mxfp8", mxfp8),
            ("non_block", non_block),
        )
        for name, method in incompatible:
            with self.subTest(name=name):
                layer = _layer(marked=None)
                layer.quant_method = method
                self.assertFalse(
                    fp8_module.configure_glm52_attn_o_decode_direct_nk(layer)
                )
                self.assertFalse(hasattr(layer, "_glm52_attn_o_decode_direct_nk_graph"))

    def test_forward_mla_has_no_task_branch(self):
        source = inspect.getsource(DeepseekMLAForwardMixin.forward_absorb_core)
        self.assertEqual(source.count("self.o_proj(attn_bmm_output)"), 1)
        for stale_branch in (
            "_glm52_attn_o_decode_direct_nk",
            "attn_o_direct_nk_context",
            "deepgemm_w8a8_block_fp8_linear_attn_o_decode_direct_nk",
        ):
            self.assertNotIn(stale_branch, source)

    def test_decode_runner_arms_only_inside_eligible_capture(self):
        model = object()
        for config_name, config in (
            ("legacy_default", None),
            (
                "explicit_full",
                SimpleNamespace(
                    decode=SimpleNamespace(backend=Backend.FULL),
                ),
            ),
        ):
            with self.subTest(config=config_name):
                events = []

                @contextmanager
                def model_capture_context(_events=events):
                    _events.append("model:enter")
                    try:
                        yield
                    finally:
                        _events.append("model:exit")

                @contextmanager
                def arm_context(armed_model, _events=events):
                    self.assertIs(armed_model, model)
                    _events.append("arm:enter")
                    try:
                        yield 1
                    finally:
                        _events.append("arm:exit")

                runner = object.__new__(DecodeCudaGraphRunner)
                runner.model_runner = SimpleNamespace(
                    server_args=SimpleNamespace(cuda_graph_config=config),
                    model=model,
                )
                runner.capture_forward_mode = ForwardMode.DECODE
                runner.num_tokens_per_req = 1
                runner.backend = object.__new__(FullCudaGraphBackend)
                runner.capture = Mock(
                    side_effect=lambda _events=events: _events.append("capture"),
                )

                with (
                    patch.object(
                        envs.SGLANG_OPT_GLM52_ATTN_O_DECODE_DIRECT_NK,
                        "get",
                        return_value=True,
                    ),
                    patch.object(
                        decode_cuda_graph_runner_module,
                        "model_capture_mode",
                        model_capture_context,
                    ),
                    patch.object(
                        fp8_module,
                        "arm_glm52_attn_o_decode_direct_nk_for_cuda_graph",
                        side_effect=arm_context,
                    ) as arm,
                ):
                    runner._capture_with_model_mode()

                arm.assert_called_once_with(model)
                runner.capture.assert_called_once_with()
                self.assertEqual(
                    events,
                    [
                        "model:enter",
                        "arm:enter",
                        "capture",
                        "arm:exit",
                        "model:exit",
                    ],
                )

    def test_decode_runner_rejects_every_non_exact_capture_shape(self):
        explicit_full = SimpleNamespace(
            decode=SimpleNamespace(backend=Backend.FULL),
        )
        breakable = SimpleNamespace(
            decode=SimpleNamespace(backend=Backend.BREAKABLE),
        )

        class FullBackendSubclass(FullCudaGraphBackend):
            pass

        cases = (
            (
                "feature_off",
                False,
                ForwardMode.DECODE,
                1,
                explicit_full,
                object.__new__(FullCudaGraphBackend),
            ),
            (
                "target_verify",
                True,
                ForwardMode.TARGET_VERIFY,
                1,
                explicit_full,
                object.__new__(FullCudaGraphBackend),
            ),
            (
                "multi_token",
                True,
                ForwardMode.DECODE,
                2,
                explicit_full,
                object.__new__(FullCudaGraphBackend),
            ),
            (
                "configured_breakable",
                True,
                ForwardMode.DECODE,
                1,
                breakable,
                object.__new__(FullCudaGraphBackend),
            ),
            (
                "runtime_not_full",
                True,
                ForwardMode.DECODE,
                1,
                explicit_full,
                object(),
            ),
            (
                "runtime_full_subclass",
                True,
                ForwardMode.DECODE,
                1,
                explicit_full,
                object.__new__(FullBackendSubclass),
            ),
        )
        for name, feature, mode, tokens, config, backend in cases:
            with self.subTest(name=name):
                runner = object.__new__(DecodeCudaGraphRunner)
                runner.model_runner = SimpleNamespace(
                    server_args=SimpleNamespace(cuda_graph_config=config),
                    model=object(),
                )
                runner.capture_forward_mode = mode
                runner.num_tokens_per_req = tokens
                runner.backend = backend
                runner.capture = Mock()

                with (
                    patch.object(
                        envs.SGLANG_OPT_GLM52_ATTN_O_DECODE_DIRECT_NK,
                        "get",
                        return_value=feature,
                    ),
                    patch.object(
                        decode_cuda_graph_runner_module,
                        "model_capture_mode",
                        return_value=nullcontext(),
                    ),
                    patch.object(
                        fp8_module,
                        "arm_glm52_attn_o_decode_direct_nk_for_cuda_graph",
                        return_value=nullcontext(),
                    ) as arm,
                ):
                    runner._capture_with_model_mode()

                arm.assert_not_called()
                runner.capture.assert_called_once_with()

    def test_initial_capture_and_recapture_share_lifecycle_hook(self):
        init_source = inspect.getsource(DecodeCudaGraphRunner.__init__)
        self.assertIn("self._capture_with_model_mode()", init_source)
        self.assertNotIn("self.capture()", init_source)

        runner = object.__new__(DecodeCudaGraphRunner)
        runner.capture_hidden_mode = CaptureHiddenMode.NULL
        runner.enable_return_hidden_states = False
        runner.backend = SimpleNamespace(cleanup=Mock())
        runner._capture_with_model_mode = Mock()
        forward_batch = SimpleNamespace(
            capture_hidden_mode=CaptureHiddenMode.FULL,
            spec_info=None,
        )

        runner.recapture_if_needed(forward_batch)

        self.assertEqual(runner.capture_hidden_mode, CaptureHiddenMode.FULL)
        runner.backend.cleanup.assert_called_once_with()
        runner._capture_with_model_mode.assert_called_once_with(
            enter_model_capture_mode=False
        )

        runner.recapture_if_needed(forward_batch)
        runner.backend.cleanup.assert_called_once_with()
        runner._capture_with_model_mode.assert_called_once_with(
            enter_model_capture_mode=False
        )

    def test_recapture_preserves_stock_absence_of_model_capture_mode(self):
        runner = object.__new__(DecodeCudaGraphRunner)
        runner.model_runner = SimpleNamespace(
            server_args=SimpleNamespace(cuda_graph_config=None),
            model=object(),
        )
        runner.capture_forward_mode = ForwardMode.DECODE
        runner.num_tokens_per_req = 1
        runner.backend = object.__new__(FullCudaGraphBackend)
        runner.capture = Mock()

        with (
            patch.object(
                envs.SGLANG_OPT_GLM52_ATTN_O_DECODE_DIRECT_NK,
                "get",
                return_value=False,
            ),
            patch.object(
                decode_cuda_graph_runner_module,
                "model_capture_mode",
            ) as model_mode,
        ):
            runner._capture_with_model_mode(enter_model_capture_mode=False)

        model_mode.assert_not_called()
        runner.capture.assert_called_once_with()

    def test_feature_on_recapture_arms_without_entering_model_capture_mode(self):
        model = object()
        events = []

        @contextmanager
        def arm_context(armed_model):
            self.assertIs(armed_model, model)
            events.append("arm:enter")
            try:
                yield 1
            finally:
                events.append("arm:exit")

        runner = object.__new__(DecodeCudaGraphRunner)
        runner.model_runner = SimpleNamespace(
            server_args=SimpleNamespace(cuda_graph_config=None),
            model=model,
        )
        runner.capture_forward_mode = ForwardMode.DECODE
        runner.num_tokens_per_req = 1
        runner.backend = object.__new__(FullCudaGraphBackend)
        runner.capture = Mock(side_effect=lambda: events.append("capture"))

        with (
            patch.object(
                envs.SGLANG_OPT_GLM52_ATTN_O_DECODE_DIRECT_NK,
                "get",
                return_value=True,
            ),
            patch.object(
                decode_cuda_graph_runner_module,
                "model_capture_mode",
            ) as model_mode,
            patch.object(
                fp8_module,
                "arm_glm52_attn_o_decode_direct_nk_for_cuda_graph",
                side_effect=arm_context,
            ) as arm,
        ):
            runner._capture_with_model_mode(enter_model_capture_mode=False)

        model_mode.assert_not_called()
        arm.assert_called_once_with(model)
        runner.capture.assert_called_once_with()
        self.assertEqual(events, ["arm:enter", "capture", "arm:exit"])


class TestPackedAbi(unittest.TestCase):
    def _tensors(self, m=16):
        packed_k = 32
        return (
            _FakeTensor((m, 16384), fp8_utils.fp8_dtype),
            _FakeTensor((6144, 16384), fp8_utils.fp8_dtype),
            _FakeTensor(
                (m, packed_k),
                torch.int32,
                stride=(1, m),
                contiguous=False,
            ),
            _FakeTensor(
                (6144, packed_k),
                torch.int32,
                stride=(1, 6144),
                contiguous=False,
            ),
        )

    def _support_patches(self):
        return (
            patch.object(fp8_utils, "_is_sm100_supported", True),
            patch.object(deep_gemm_wrapper, "ENABLE_JIT_DEEPGEMM", True),
            patch.object(deep_gemm_wrapper, "DEEPGEMM_SCALE_UE8M0", True),
            patch.object(
                deep_gemm_wrapper,
                "DEEPGEMM_SUPPORTS_COMPILED_DIMS",
                True,
            ),
        )

    def test_exact_packed_abi_accepts_both_buckets(self):
        patches = self._support_patches()
        with patches[0], patches[1], patches[2], patches[3]:
            for m in (16, 32):
                with self.subTest(m=m):
                    self.assertTrue(
                        fp8_utils._is_glm52_attn_o_decode_direct_nk_abi(
                            *self._tensors(m),
                            [128, 128],
                            torch.bfloat16,
                        )
                    )

    def test_unsupported_abi_and_build_fail_closed(self):
        q, weight, q_scale, weight_scale = self._tensors()
        patches = self._support_patches()
        with patches[0], patches[1], patches[2], patches[3]:
            bad_scale = _FakeTensor(
                q_scale.shape,
                torch.float32,
                stride=q_scale.stride(),
                contiguous=False,
            )
            bad_stride = _FakeTensor(
                q_scale.shape,
                torch.int32,
                stride=(32, 1),
                contiguous=True,
            )
            for name, args in (
                (
                    "activation_scale_dtype",
                    (q, weight, bad_scale, weight_scale, [128, 128]),
                ),
                (
                    "activation_scale_layout",
                    (q, weight, bad_stride, weight_scale, [128, 128]),
                ),
                (
                    "weight_shape",
                    (
                        q,
                        _FakeTensor((4096, 16384), fp8_utils.fp8_dtype),
                        q_scale,
                        weight_scale,
                        [128, 128],
                    ),
                ),
                (
                    "block_size",
                    (q, weight, q_scale, weight_scale, [64, 128]),
                ),
            ):
                with self.subTest(name=name):
                    self.assertFalse(
                        fp8_utils._is_glm52_attn_o_decode_direct_nk_abi(
                            *args,
                            torch.bfloat16,
                        )
                    )

        with (
            patch.object(fp8_utils, "_is_sm100_supported", True),
            patch.object(
                deep_gemm_wrapper,
                "ENABLE_JIT_DEEPGEMM",
                True,
            ),
            patch.object(
                deep_gemm_wrapper,
                "DEEPGEMM_SCALE_UE8M0",
                True,
            ),
            patch.object(
                deep_gemm_wrapper,
                "DEEPGEMM_SUPPORTS_COMPILED_DIMS",
                False,
            ),
        ):
            self.assertFalse(
                fp8_utils._is_glm52_attn_o_decode_direct_nk_abi(
                    q,
                    weight,
                    q_scale,
                    weight_scale,
                    [128, 128],
                    torch.bfloat16,
                )
            )

    def test_supported_call_quantizes_once_and_selects_compiled_nk(self):
        input_bf16 = _FakeTensor((16, 16384), torch.bfloat16)
        q, weight, q_scale, weight_scale = self._tensors()
        output = _FakeOutput()
        patches = self._support_patches()
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patch.object(
                fp8_utils,
                "sglang_per_token_group_quant_fp8",
                return_value=(q, q_scale),
            ) as quant,
            patch.object(
                fp8_utils,
                "w8a8_block_fp8_matmul_deepgemm_compiled_nk",
                return_value=output,
            ) as compiled,
            patch.object(
                fp8_utils,
                "deepgemm_w8a8_block_fp8_linear_with_fallback",
            ) as high_level_fallback,
        ):
            result = fp8_utils.deepgemm_w8a8_block_fp8_linear_attn_o_decode_direct_nk(
                input_bf16,
                weight,
                [128, 128],
                weight_scale,
            )

        self.assertIs(result, output)
        self.assertEqual(output.to_dtype, torch.bfloat16)
        self.assertEqual(output.view_shape, (16, 6144))
        quant.assert_called_once()
        compiled.assert_called_once()
        high_level_fallback.assert_not_called()

    def test_unsupported_high_level_call_falls_back_before_quantization(self):
        good_input = _FakeTensor((16, 16384), torch.bfloat16)
        _, good_weight, _, good_weight_scale = self._tensors()
        cases = (
            ("bias", good_input, good_weight, good_weight_scale, object()),
            (
                "weight_scale_shape",
                good_input,
                good_weight,
                _FakeTensor(
                    (6144, 31),
                    torch.int32,
                    stride=(1, 6144),
                    contiguous=False,
                ),
                None,
            ),
            (
                "weight_scale_stride",
                good_input,
                good_weight,
                _FakeTensor(
                    (6144, 32),
                    torch.int32,
                    stride=(32, 1),
                    contiguous=True,
                ),
                None,
            ),
            (
                "weight_scale_device",
                good_input,
                good_weight,
                _FakeTensor(
                    (6144, 32),
                    torch.int32,
                    stride=(1, 6144),
                    device="cuda:1",
                    contiguous=False,
                ),
                None,
            ),
            (
                "weight_device",
                good_input,
                _FakeTensor(
                    (6144, 16384),
                    fp8_utils.fp8_dtype,
                    device="cuda:1",
                ),
                good_weight_scale,
                None,
            ),
            (
                "input_device",
                _FakeTensor(
                    (16, 16384),
                    torch.bfloat16,
                    device="cpu",
                    is_cuda=False,
                ),
                good_weight,
                good_weight_scale,
                None,
            ),
        )
        patches = self._support_patches()
        with patches[0], patches[1], patches[2], patches[3]:
            for name, input_bf16, weight, weight_scale, bias in cases:
                sentinel = object()
                with (
                    self.subTest(name=name),
                    patch.object(
                        fp8_utils,
                        "deepgemm_w8a8_block_fp8_linear_with_fallback",
                        return_value=sentinel,
                    ) as fallback,
                    patch.object(
                        fp8_utils,
                        "sglang_per_token_group_quant_fp8",
                    ) as quant,
                ):
                    result = fp8_utils.deepgemm_w8a8_block_fp8_linear_attn_o_decode_direct_nk(
                        input_bf16,
                        weight,
                        [128, 128],
                        weight_scale,
                        bias=bias,
                    )

                self.assertIs(result, sentinel)
                fallback.assert_called_once()
                quant.assert_not_called()

    def test_post_quant_layout_miss_uses_registry_first_without_requantizing(self):
        input_bf16 = _FakeTensor((16, 16384), torch.bfloat16)
        q, weight, q_scale, weight_scale = self._tensors()
        output = _FakeOutput()
        patches = self._support_patches()
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patch.object(
                fp8_utils,
                "sglang_per_token_group_quant_fp8",
                return_value=(q, q_scale),
            ) as quant,
            patch.object(
                fp8_utils,
                "_is_glm52_attn_o_decode_direct_nk_abi",
                return_value=False,
            ),
            patch.object(
                fp8_utils,
                "w8a8_block_fp8_matmul_deepgemm_compiled_nk",
            ) as compiled,
            patch.object(
                glm52_dispatch,
                "try_dispatch_fp8_gemm",
                return_value=output,
            ) as registry,
            patch.object(
                fp8_utils,
                "w8a8_block_fp8_matmul_deepgemm",
            ) as stock_gemm,
        ):
            result = fp8_utils.deepgemm_w8a8_block_fp8_linear_attn_o_decode_direct_nk(
                input_bf16,
                weight,
                [128, 128],
                weight_scale,
            )

        self.assertIs(result, output)
        quant.assert_called_once()
        compiled.assert_not_called()
        registry.assert_called_once_with(
            q,
            weight,
            q_scale,
            weight_scale,
            [128, 128],
            torch.bfloat16,
            bias=None,
        )
        stock_gemm.assert_not_called()

    def test_post_quant_registry_miss_uses_stock_gemm_without_requantizing(self):
        input_bf16 = _FakeTensor((32, 16384), torch.bfloat16)
        q, weight, q_scale, weight_scale = self._tensors(m=32)
        output = _FakeOutput()
        patches = self._support_patches()
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patch.object(
                fp8_utils,
                "sglang_per_token_group_quant_fp8",
                return_value=(q, q_scale),
            ) as quant,
            patch.object(
                fp8_utils,
                "_is_glm52_attn_o_decode_direct_nk_abi",
                return_value=False,
            ),
            patch.object(
                glm52_dispatch,
                "try_dispatch_fp8_gemm",
                return_value=None,
            ) as registry,
            patch.object(
                fp8_utils,
                "w8a8_block_fp8_matmul_deepgemm",
                return_value=output,
            ) as stock_gemm,
            patch.object(
                fp8_utils,
                "w8a8_block_fp8_matmul_deepgemm_compiled_nk",
            ) as compiled,
        ):
            result = fp8_utils.deepgemm_w8a8_block_fp8_linear_attn_o_decode_direct_nk(
                input_bf16,
                weight,
                [128, 128],
                weight_scale,
            )

        self.assertIs(result, output)
        quant.assert_called_once()
        registry.assert_called_once()
        stock_gemm.assert_called_once_with(
            q,
            weight,
            q_scale,
            weight_scale,
            [128, 128],
            output_dtype=torch.bfloat16,
        )
        compiled.assert_not_called()


def _custom_op_tensors(device):
    return (
        torch.empty((2, 4), device=device, dtype=torch.float8_e4m3fn),
        torch.empty((2, 1), device=device, dtype=torch.int32),
        torch.empty((3, 4), device=device, dtype=torch.float8_e4m3fn),
        torch.empty((3, 1), device=device, dtype=torch.int32),
        torch.empty((2, 3), device=device, dtype=torch.bfloat16),
    )


class TestCustomOpCompilation(unittest.TestCase):
    def test_meta_and_fake_tensor_implementations(self):
        self.assertIsNone(
            deep_gemm_fp8_fp8_bf16_nt_compiled_nk(
                *_custom_op_tensors("meta"),
            )
        )
        with FakeTensorMode():
            tensors = _custom_op_tensors("cpu")
            self.assertIsInstance(tensors[0], FakeTensor)
            self.assertIsNone(deep_gemm_fp8_fp8_bf16_nt_compiled_nk(*tensors))

    def test_torch_compile_captures_mutating_custom_op(self):
        captured = {}

        def backend(graph_module, example_inputs):
            captured["graph_module"] = graph_module

            def run(*args):
                return (args[-1],)

            return run

        def invoke(a, a_scale, b, b_scale, output):
            deep_gemm_fp8_fp8_bf16_nt_compiled_nk(
                a,
                a_scale,
                b,
                b_scale,
                output,
            )
            return output

        compiled = torch.compile(invoke, backend=backend, fullgraph=True)
        tensors = _custom_op_tensors("cpu")
        result = compiled(*tensors)
        self.assertIs(result, tensors[-1])
        targets = {
            node.target
            for node in captured["graph_module"].graph.nodes
            if node.op == "call_function"
        }
        self.assertIn(
            torch.ops.sglang.deep_gemm_fp8_fp8_bf16_nt_compiled_nk,
            targets,
        )


class TestDeepGemmEntrypoint(unittest.TestCase):
    def test_compiled_nk_is_the_only_api_delta(self):
        lhs = (_FakeTensor((16, 16384), fp8_utils.fp8_dtype), object())
        rhs = (_FakeTensor((6144, 16384), fp8_utils.fp8_dtype), object())
        out = object()
        call = Mock()
        hook = Mock(return_value=nullcontext())
        with (
            patch.object(
                deep_gemm_entrypoint,
                "deep_gemm",
                SimpleNamespace(fp8_gemm_nt=call),
                create=True,
            ),
            patch.object(
                deep_gemm_entrypoint,
                "_sanity_check_input",
            ),
            patch.object(
                deep_gemm_entrypoint.compile_utils,
                "deep_gemm_execution_hook",
                hook,
            ),
        ):
            deep_gemm_entrypoint.gemm_nt_f8f8bf16_compiled_nk(lhs, rhs, out)

        call.assert_called_once_with(lhs, rhs, out, compiled_dims="nk")
        hook.assert_called_once_with(
            16,
            6144,
            16384,
            1,
            compile_utils.DeepGemmKernelType.GEMM_NT_F8F8BF16_COMPILED_NK,
        )

    def test_fixed_nk_precompile_has_distinct_two_bucket_identity(self):
        special = compile_utils.DeepGemmKernelType.GEMM_NT_F8F8BF16_COMPILED_NK
        stock = compile_utils.DeepGemmKernelType.GEMM_NT_F8F8BF16
        self.assertNotEqual(special, stock)

        with (
            patch.object(
                compile_utils,
                "_ENABLE_JIT_DEEPGEMM_PRECOMPILE",
                True,
            ),
            patch.object(compile_utils, "_DO_COMPILE_ALL", True),
            patch.object(compile_utils, "_IS_FIRST_RANK_ON_NODE", False),
            patch.object(compile_utils, "_IN_PRECOMPILE_STAGE", True),
            patch.object(compile_utils, "_INITIALIZATION_DICT", {}),
            patch.object(compile_utils, "_BUILTIN_M_LIST", [1, 2, 3]),
            patch.object(
                compile_utils,
                "_compile_deep_gemm_one_type_all",
            ) as compile_all,
        ):
            compile_utils._maybe_compile_deep_gemm_one_type_all(
                special,
                6144,
                16384,
                1,
            )
            compile_utils._maybe_compile_deep_gemm_one_type_all(
                special,
                6144,
                16384,
                1,
            )
            compile_utils._maybe_compile_deep_gemm_one_type_all(
                stock,
                6144,
                16384,
                1,
            )

        self.assertEqual(compile_all.call_count, 2)
        self.assertEqual(compile_all.call_args_list[0].kwargs["m_list"], [16, 32])
        self.assertEqual(compile_all.call_args_list[1].kwargs["m_list"], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
