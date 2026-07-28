"""CPU-only contracts for Task26 em8/BM16/stage11."""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import subprocess
import sys
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from sglang.srt.layers.deep_gemm_wrapper import entrypoint
from sglang.srt.layers.glm52_opt import config, dispatch
from sglang.srt.layers.glm52_opt import (
    experimental_deepgemm_em8_bm16_stage11 as stage11,
)
from sglang.srt.layers.glm52_opt.context import op_context
from sglang.srt.model_executor.forward_batch_info import ForwardMode


def _tensor(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    stride: tuple[int, ...],
    *,
    contiguous: bool,
) -> Mock:
    tensor = Mock(spec=torch.Tensor)
    tensor.shape = shape
    tensor.dtype = dtype
    tensor.layout = torch.strided
    tensor.device = torch.device("cuda:0")
    tensor.stride.return_value = stride
    tensor.is_contiguous.return_value = contiguous
    return tensor


def _inputs():
    lhs = (
        _tensor(
            (32, 1024, 2048),
            torch.float8_e4m3fn,
            (2097152, 2048, 1),
            contiguous=True,
        ),
        _tensor(
            (32, 1024, 4),
            torch.int32,
            (4096, 1, 1024),
            contiguous=False,
        ),
    )
    rhs = (
        _tensor(
            (32, 6144, 2048),
            torch.float8_e4m3fn,
            (12582912, 2048, 1),
            contiguous=True,
        ),
        _tensor(
            (32, 6144, 4),
            torch.int32,
            (24576, 1, 6144),
            contiguous=False,
        ),
    )
    rhs[1].format_ue8m0 = True
    out = _tensor(
        (32, 1024, 6144),
        torch.bfloat16,
        (6291456, 6144, 1),
        contiguous=True,
    )
    masked_m = _tensor((32,), torch.int32, (1,), contiguous=True)
    return lhs, rhs, out, masked_m


def _dispatch(
    *,
    mode: ForwardMode = ForwardMode.DECODE,
    local_m: int = 32,
    expected_m: int = 8,
    op: str = "moe_down_proj",
    eligible: bool = True,
    launch=None,
):
    launch = launch or Mock(return_value=None)
    contract = SimpleNamespace(
        launch=launch,
        current_forward_state=lambda: (mode, local_m),
    )
    inputs = _inputs()
    with op_context(op):
        result = dispatch.try_dispatch_moe_w2_em8_bm16_stage11(
            contract,
            *inputs,
            expected_m=expected_m,
            callsite_eligible=eligible,
        )
    return result, launch, inputs


def test_exact_dispatch_passes_joint_per_call_overrides():
    result, launch, inputs = _dispatch()
    assert result is True
    launch.assert_called_once_with(
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[3],
        8,
        masked_block_m_override=16,
        masked_num_stages_override=11,
    )
    inputs[3].cpu.assert_not_called()
    inputs[3].item.assert_not_called()
    inputs[3].tolist.assert_not_called()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": ForwardMode.EXTEND},
        {"mode": ForwardMode.TARGET_VERIFY},
        {"local_m": 16},
        {"expected_m": 9},
    ],
)
def test_dispatch_declines_every_non_exact_state_before_launch(kwargs):
    result, launch, _ = _dispatch(**kwargs)
    assert result is False
    launch.assert_not_called()


@pytest.mark.parametrize("kwargs", [{"op": "moe_gate_proj"}, {"eligible": False}])
def test_exact_dispatch_admission_failure_propagates(kwargs):
    with pytest.raises(RuntimeError, match="admission failed"):
        _dispatch(**kwargs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("local_m", True),
        ("local_m", 32.0),
        ("expected_m", True),
        ("expected_m", 8.0),
    ],
)
def test_dispatch_rejects_bool_or_non_integral_bucket_metadata(field, value):
    with pytest.raises(TypeError):
        _dispatch(**{field: value})


@pytest.mark.parametrize("local_m", [True, 32.0, "32"])
def test_forward_context_rejects_non_integral_local_m(local_m):
    with (
        pytest.raises(TypeError),
        stage11.forward_context(ForwardMode.DECODE, local_m),
    ):
        pass


@pytest.mark.parametrize(
    "launch",
    [
        Mock(side_effect=RuntimeError("candidate failed")),
        Mock(return_value=object()),
    ],
)
def test_post_invoke_failure_propagates_without_stock(launch):
    inputs = _inputs()
    runtime_contract = SimpleNamespace(
        launch=launch,
        current_forward_state=lambda: (ForwardMode.DECODE, 32),
    )
    layer_contract = SimpleNamespace(
        callsite_checked=True,
        callsite_eligible=True,
    )
    callsite_prepare = Mock(return_value=True)
    stock = Mock()
    armed = partial(
        entrypoint._grouped_gemm_nt_f8f8bf16_masked_w2_bm16,
        layer_contract,
        runtime_contract,
        callsite_prepare,
        dispatch.try_dispatch_moe_w2_em8_bm16_stage11,
        strict_profile=True,
    )
    with (
        patch.object(entrypoint, "deep_gemm", stock, create=True),
        patch.object(entrypoint, "_ensure_cuda", side_effect=lambda value: value),
        patch.object(entrypoint, "_sanity_check_input"),
        patch.object(
            entrypoint.compile_utils,
            "deep_gemm_execution_hook",
            return_value=nullcontext(),
        ),
        op_context("moe_down_proj"),
        pytest.raises(RuntimeError),
    ):
        armed(*inputs, expected_m=8)
    callsite_prepare.assert_called_once()
    stock.fp8_m_grouped_gemm_nt_masked.assert_not_called()


def test_exact_recipe_or_overlap_failure_propagates_without_stock():
    inputs = _inputs()
    candidate_dispatch = Mock(side_effect=AssertionError("candidate launched"))
    stock = SimpleNamespace(fp8_m_grouped_gemm_nt_masked=Mock())
    callsite_prepare = Mock(side_effect=RuntimeError("unsupported exact route"))
    armed = partial(
        entrypoint._grouped_gemm_nt_f8f8bf16_masked_w2_bm16,
        SimpleNamespace(callsite_checked=True, callsite_eligible=True),
        SimpleNamespace(),
        callsite_prepare,
        candidate_dispatch,
        strict_profile=True,
    )
    overlap = SimpleNamespace(num_sms=116, signal=object())
    with (
        patch.object(entrypoint, "deep_gemm", stock, create=True),
        patch.object(entrypoint, "_ensure_cuda", side_effect=lambda value: value),
        patch.object(entrypoint, "_sanity_check_input"),
        patch.object(
            entrypoint.compile_utils,
            "deep_gemm_execution_hook",
            return_value=nullcontext(),
        ),
        patch.object(
            entrypoint,
            "configure_deep_gemm_num_sms",
            return_value=nullcontext(),
        ),
        pytest.raises(RuntimeError, match="unsupported exact route"),
    ):
        armed(*inputs, expected_m=8, overlap_args=overlap)
    callsite_prepare.assert_called_once()
    candidate_dispatch.assert_not_called()
    stock.fp8_m_grouped_gemm_nt_masked.assert_not_called()


def test_layer_and_callsite_contract_latch_exact_packed_ue8m0_only():
    lhs, rhs, out, masked_m = _inputs()
    runtime = SimpleNamespace(
        device_index=0,
        current_forward_state=stage11.current_forward_state,
    )
    with (
        patch.object(stage11, "_REQUESTED", True),
        patch.object(stage11, "_PREPARED", runtime),
    ):
        contract = stage11.create_layer_contract(
            w2_weight=rhs[0],
            w2_scale=rhs[1],
            block_shape=[128, 128],
            deep_gemm_backend=True,
            is_fp4_experts=False,
            use_mxfp8=False,
        )
        assert contract is not None and contract.static_eligible
        with stage11.forward_context(ForwardMode.DECODE, 32):
            assert stage11.prepare_callsite_contract(
                contract,
                lhs,
                rhs,
                out,
                masked_m,
                expected_m=8,
                recipe_a=None,
                recipe_b=None,
                overlap_args=None,
            )
        assert contract.callsite_checked and contract.callsite_eligible
        masked_m.cpu.assert_not_called()
        masked_m.item.assert_not_called()
        masked_m.tolist.assert_not_called()

        lhs[0].stride.side_effect = AssertionError("hot-path ABI rescan")
        with stage11.forward_context(ForwardMode.DECODE, 32):
            assert stage11.prepare_callsite_contract(
                contract,
                lhs,
                rhs,
                out,
                masked_m,
                expected_m=8,
                recipe_a=None,
                recipe_b=None,
                overlap_args=None,
            )


@pytest.mark.parametrize(
    "mode,local_m,expected_m,recipe_a,overlap",
    [
        (ForwardMode.TARGET_VERIFY, 32, 8, None, None),
        (ForwardMode.DECODE, 16, 8, None, None),
        (ForwardMode.DECODE, 32, 9, None, None),
    ],
)
def test_ineligible_warmup_does_not_latch_before_exact_decode(
    mode,
    local_m,
    expected_m,
    recipe_a,
    overlap,
):
    lhs, rhs, out, masked_m = _inputs()
    runtime = SimpleNamespace(
        device_index=0,
        current_forward_state=stage11.current_forward_state,
    )
    with (
        patch.object(stage11, "_REQUESTED", True),
        patch.object(stage11, "_PREPARED", runtime),
    ):
        contract = stage11.create_layer_contract(
            w2_weight=rhs[0],
            w2_scale=rhs[1],
            block_shape=[128, 128],
            deep_gemm_backend=True,
            is_fp4_experts=False,
            use_mxfp8=False,
        )
        with stage11.forward_context(mode, local_m):
            assert not stage11.prepare_callsite_contract(
                contract,
                lhs,
                rhs,
                out,
                masked_m,
                expected_m=expected_m,
                recipe_a=recipe_a,
                recipe_b=None,
                overlap_args=overlap,
            )
        assert contract.callsite_checked is False
        with stage11.forward_context(ForwardMode.DECODE, 32):
            assert stage11.prepare_callsite_contract(
                contract,
                lhs,
                rhs,
                out,
                masked_m,
                expected_m=8,
                recipe_a=None,
                recipe_b=None,
                overlap_args=None,
            )


@pytest.mark.parametrize(
    "recipe_a,overlap",
    [
        ((1, 128), None),
        (None, SimpleNamespace(num_sms=116)),
    ],
)
def test_exact_callsite_unsupported_route_raises(recipe_a, overlap):
    lhs, rhs, out, masked_m = _inputs()
    runtime = SimpleNamespace(
        device_index=0,
        current_forward_state=stage11.current_forward_state,
    )
    with (
        patch.object(stage11, "_REQUESTED", True),
        patch.object(stage11, "_PREPARED", runtime),
    ):
        contract = stage11.create_layer_contract(
            w2_weight=rhs[0],
            w2_scale=rhs[1],
            block_shape=[128, 128],
            deep_gemm_backend=True,
            is_fp4_experts=False,
            use_mxfp8=False,
        )
        with (
            stage11.forward_context(ForwardMode.DECODE, 32),
            pytest.raises(RuntimeError, match="does not support"),
        ):
            stage11.prepare_callsite_contract(
                contract,
                lhs,
                rhs,
                out,
                masked_m,
                expected_m=8,
                recipe_a=recipe_a,
                recipe_b=None,
                overlap_args=overlap,
            )
        assert contract.callsite_checked is False


def _fake_runtime(*, pdl_settable: bool = True):
    state = {"num_sms": 148, "tc_util": 87, "pdl": False}

    def launch(
        *args,
        masked_block_m_override=0,
        masked_num_stages_override=0,
    ):
        return None

    module = SimpleNamespace(
        __file__="/fake/deep_gemm/__init__.py",
        _C=object(),
        get_num_sms=lambda: state["num_sms"],
        set_num_sms=lambda value: state.__setitem__("num_sms", int(value)),
        get_tc_util=lambda: state["tc_util"],
        set_tc_util=lambda value: state.__setitem__("tc_util", int(value)),
        get_pdl=lambda: state["pdl"],
        set_pdl=lambda value: state.__setitem__("pdl", value) if pdl_settable else None,
        fp8_m_grouped_gemm_nt_masked=launch,
    )
    return module, state


def _cache_env(root: str) -> dict[str, str]:
    return {
        "DG_JIT_CACHE_DIR": f"{root}/deepgemm",
        "SGLANG_DG_CACHE_DIR": f"{root}/deepgemm",
        "TRITON_CACHE_DIR": f"{root}/triton",
        "TORCH_EXTENSIONS_DIR": f"{root}/torch_extensions",
    }


def _minimal_manifest(cache_env: dict[str, str]) -> dict:
    return {
        "variant": {
            "name": "em8_bm16_stage11",
            "version": 3,
            "predeclared_fallback": "em8_bm16_stage10",
            "fallback_eligible": False,
        },
        "candidate_api": {"jit_identity": stage11.JIT_IDENTITY},
        "base": {
            "commit": stage11.BASE_COMMIT,
            "version": "0.1.4.post1",
            "submodules": {
                "third-party/cutlass": ("f3fde58372d33e9a5650ba7b80fc48b3b49d40c8"),
                "third-party/fmt": ("553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28"),
            },
        },
        "stock": {"extension_sha256": "1" * 64},
        "candidate": {"extension_sha256": "2" * 64},
        "runtime_contract": {"cache_paths": cache_env},
    }


def test_prepare_establishes_and_proves_independent_runtime_state(tmp_path):
    cache_env = _cache_env(
        "/home/qinhaiyan/glm52-v2-goal-runs/cache/"
        "26-moe_w2_decode_scoped_bm16/em8_bm16_stage11_v3"
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_minimal_manifest(cache_env)))
    stock, stock_state = _fake_runtime()
    candidate, candidate_state = _fake_runtime()
    candidate.__file__ = "/fake/candidate/__init__.py"
    with (
        patch.dict(
            os.environ,
            cache_env,
            clear=False,
        ),
        patch.object(stage11, "_verify_manifest", return_value=manifest_path),
        patch.object(stage11, "_ensure_stock", return_value=stock),
        patch.object(stage11, "_load_candidate", return_value=candidate),
        patch.object(stage11, "_PREPARED", None),
        patch.object(stage11, "_PREPARE_ERROR", None),
        patch.object(torch.cuda, "current_device", return_value=0),
        patch.object(torch.cuda, "get_device_capability", return_value=(10, 0)),
        patch.object(
            torch.cuda,
            "get_device_properties",
            return_value=SimpleNamespace(multi_processor_count=148),
        ),
    ):
        contract = stage11.prepare_deep_gemm(0)
    assert stock_state == {"num_sms": 148, "tc_util": 87, "pdl": True}
    assert candidate_state == {"num_sms": 148, "tc_util": 87, "pdl": True}
    evidence = contract.evidence()
    assert evidence["variant_name"] == "em8_bm16_stage11"
    assert evidence["masked_num_stages_override"] == 11
    assert evidence["candidate_jit_identity"] == stage11.JIT_IDENTITY
    assert evidence["fallback_eligible"] is False
    assert contract.candidate_module is candidate
    assert contract.launch is candidate.fp8_m_grouped_gemm_nt_masked


def test_prepare_fails_closed_when_candidate_pdl_cannot_be_synchronized(
    tmp_path,
    monkeypatch,
):
    cache_env = _cache_env("/task")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_minimal_manifest(cache_env)))
    stock, _ = _fake_runtime()
    candidate, _ = _fake_runtime(pdl_settable=False)
    monkeypatch.setattr(stage11, "_PREPARED", None)
    monkeypatch.setattr(stage11, "_PREPARE_ERROR", None)
    with (
        patch.dict(
            os.environ,
            cache_env,
            clear=False,
        ),
        patch.object(stage11, "_verify_manifest", return_value=manifest_path),
        patch.object(stage11, "_ensure_stock", return_value=stock),
        patch.object(stage11, "_load_candidate", return_value=candidate),
        patch.object(torch.cuda, "current_device", return_value=0),
        patch.object(torch.cuda, "get_device_capability", return_value=(10, 0)),
        patch.object(
            torch.cuda,
            "get_device_properties",
            return_value=SimpleNamespace(multi_processor_count=148),
        ),
        pytest.raises(RuntimeError),
    ):
        stage11.prepare_deep_gemm(0)
    assert stage11._PREPARED is None
    assert "RuntimeError" in str(stage11._PREPARE_ERROR)


def test_profile_is_explicit_and_down_projection_only(monkeypatch):
    monkeypatch.setenv("SGLANG_GLM52_OPT", "1")
    monkeypatch.setenv("SGLANG_GLM52_OPT_PROFILE", "serving_safe")
    monkeypatch.delenv("SGLANG_GLM52_OPT_OPS", raising=False)
    assert config.w2_em8_bm16_stage11_enabled() is False
    monkeypatch.setenv(
        "SGLANG_GLM52_OPT_PROFILE",
        config.W2_EM8_BM16_STAGE11_PROFILE,
    )
    assert config.w2_em8_bm16_stage11_enabled() is True
    assert config.w2_bm16_enabled() is False
    monkeypatch.setenv("SGLANG_GLM52_OPT_OPS", "moe_gate_proj")
    assert config.w2_em8_bm16_stage11_enabled() is False
    monkeypatch.setenv("SGLANG_GLM52_OPT_OPS", "moe_down_proj")
    assert config.w2_em8_bm16_stage11_enabled() is True


def test_worker_setup_selects_only_stage11_contract_and_callables():
    contract = SimpleNamespace(
        forward_context=Mock(),
        evidence=lambda: {"variant_name": "em8_bm16_stage11"},
    )
    with (
        patch.object(entrypoint, "deep_gemm", SimpleNamespace(), create=True),
        patch.object(
            entrypoint.envs.SGLANG_DEEPGEMM_PDL,
            "get",
            return_value=False,
        ),
        patch.object(entrypoint.compile_utils, "update_deep_gemm_config"),
        patch.object(config, "w2_em8_bm16_stage11_enabled", return_value=True),
        patch.object(config, "w2_bm16_enabled", return_value=False),
        patch.object(stage11, "prepare_deep_gemm", return_value=contract),
        patch.object(entrypoint, "_W2_BM16_PROFILE_REQUESTED", False),
        patch.object(entrypoint, "_W2_BM16_PREPARED_CONTRACT", None),
        patch.object(entrypoint, "_W2_BM16_DISPATCH", None),
        patch.object(entrypoint, "_W2_BM16_CALLSITE_PREPARE", None),
        patch.object(entrypoint, "_W2_BM16_LAYER_PREPARE", None),
        patch.object(entrypoint, "_W2_BM16_STRICT_PROFILE", False),
    ):
        context = entrypoint.update_deep_gemm_config(
            0,
            SimpleNamespace(),
        )
        assert context is contract.forward_context
        assert entrypoint._W2_BM16_PROFILE_REQUESTED is True
        assert entrypoint._W2_BM16_PREPARED_CONTRACT is contract
        assert (
            entrypoint._W2_BM16_DISPATCH
            is dispatch.try_dispatch_moe_w2_em8_bm16_stage11
        )
        assert entrypoint._W2_BM16_CALLSITE_PREPARE is stage11.prepare_callsite_contract
        assert entrypoint._W2_BM16_LAYER_PREPARE is stage11.create_layer_contract
        assert entrypoint._W2_BM16_STRICT_PROFILE is True


def test_explicit_worker_setup_prepare_failure_propagates():
    with (
        patch.object(entrypoint, "deep_gemm", SimpleNamespace(), create=True),
        patch.object(
            entrypoint.envs.SGLANG_DEEPGEMM_PDL,
            "get",
            return_value=False,
        ),
        patch.object(entrypoint.compile_utils, "update_deep_gemm_config"),
        patch.object(config, "w2_em8_bm16_stage11_enabled", return_value=True),
        patch.object(
            stage11,
            "prepare_deep_gemm",
            side_effect=RuntimeError("prepare rejected"),
        ),
        pytest.raises(RuntimeError, match="prepare rejected"),
    ):
        entrypoint.update_deep_gemm_config(0, SimpleNamespace())


def test_explicit_layer_prepare_failure_propagates_without_installing_wrapper():
    lhs, rhs, _, _ = _inputs()
    runner_core = SimpleNamespace(set_masked_down_gemm=Mock())
    with (
        patch.object(entrypoint, "_W2_BM16_PROFILE_REQUESTED", True),
        patch.object(entrypoint, "_W2_BM16_STRICT_PROFILE", True),
        patch.object(entrypoint, "_W2_BM16_PREPARED_CONTRACT", object()),
        patch.object(entrypoint, "_W2_BM16_CALLSITE_PREPARE", Mock()),
        patch.object(entrypoint, "_W2_BM16_DISPATCH", Mock()),
        patch.object(
            entrypoint,
            "_W2_BM16_LAYER_PREPARE",
            Mock(side_effect=RuntimeError("layer rejected")),
        ),
        pytest.raises(RuntimeError, match="layer rejected"),
    ):
        entrypoint.configure_w2_bm16_masked_down_gemm(
            runner_core,
            w2_weight=rhs[0],
            w2_scale=rhs[1],
            block_shape=[128, 128],
            deep_gemm_backend=True,
            is_fp4_experts=False,
            use_mxfp8=False,
        )
    runner_core.set_masked_down_gemm.assert_not_called()
    lhs[0].cpu.assert_not_called()


def test_explicit_exact_callsite_abi_failure_propagates():
    lhs, rhs, out, masked_m = _inputs()
    runtime = SimpleNamespace(
        device_index=0,
        current_forward_state=stage11.current_forward_state,
    )
    with (
        patch.object(stage11, "_REQUESTED", True),
        patch.object(stage11, "_PREPARED", runtime),
    ):
        contract = stage11.create_layer_contract(
            w2_weight=rhs[0],
            w2_scale=rhs[1],
            block_shape=[128, 128],
            deep_gemm_backend=True,
            is_fp4_experts=False,
            use_mxfp8=False,
        )
        lhs[0].shape = (32, 1024, 1024)
        with (
            stage11.forward_context(ForwardMode.DECODE, 32),
            pytest.raises(RuntimeError, match="callsite ABI rejected"),
        ):
            stage11.prepare_callsite_contract(
                contract,
                lhs,
                rhs,
                out,
                masked_m,
                expected_m=8,
                recipe_a=None,
                recipe_b=None,
                overlap_args=None,
            )


def test_explicit_layer_static_abi_failure_propagates():
    _, rhs, _, _ = _inputs()
    rhs[1].format_ue8m0 = False
    with (
        patch.object(stage11, "_REQUESTED", True),
        patch.object(stage11, "_PREPARED", object()),
        pytest.raises(RuntimeError, match="layer/runtime ABI"),
    ):
        stage11.create_layer_contract(
            w2_weight=rhs[0],
            w2_scale=rhs[1],
            block_shape=[128, 128],
            deep_gemm_backend=True,
            is_fp4_experts=False,
            use_mxfp8=False,
        )


def test_source_patch_and_manifest_encode_exact_stage11_identity():
    repo = Path(__file__).resolve().parents[3]
    overlay = repo / "third_party" / "deepgemm_w2_em8_bm16_stage11"
    source = (overlay / "source.patch").read_text()
    manifest_tool = (overlay / "overlay_manifest.py").read_text()
    lock = json.loads((overlay / "base_lock.json").read_text())
    assert "masked_block_m_override = 0" in source
    assert "masked_num_stages_override = 0" in source
    assert ".masked_num_stages_override = masked_num_stages_override" in source
    assert "int num_stages = max_num_stages" in source
    assert "num_stages = desc.masked_num_stages_override" in source
    assert (
        "masked_block_m_override == 0 and masked_num_stages_override == 0"
    ) in source
    assert (
        "masked_block_m_override == 16 and masked_num_stages_override == 11"
    ) in source
    for forbidden in (
        "masked_block_m_override == 16 and masked_num_stages_override == 0",
        "masked_block_m_override == 16 and masked_num_stages_override == 10",
        "masked_block_m_override == 16 and masked_num_stages_override == 12",
        "masked_block_m_override == 128 and masked_num_stages_override == 11",
    ):
        assert forbidden not in source
    assert "get_expected_m() == 8" in source
    assert "masked_num_stages_override >= 1" in source
    assert "masked_num_stages_override <= max_num_stages" in source
    assert "smem_per_stage == 18432" in source
    assert "smem_extra == 9004" in source
    assert "max_num_stages == 12" in source
    assert "num_stages == 11" in source
    assert "== 211756" in source
    assert "expected_m == 8" in source
    assert "expected_m == 4" not in source
    assert "glm52_w2_em8_bm16_stage11_v3" in source
    assert "glm52_w2_em8_bm16_stage11_v1" not in source
    assert "stages11" in source
    assert "SGLANG_GLM52_W2_BM16_JIT_CACHE" not in source
    assert "set_mk_alignment_for_contiguous_layout(" not in source
    assert lock["variant"] == {
        "fallback_eligible": False,
        "name": "em8_bm16_stage11",
        "predeclared_fallback": "em8_bm16_stage10",
        "version": 3,
    }
    assert "masked_num_stages_override" in manifest_tool
    assert "fallback_eligible" in manifest_tool
    assert "/em8_bm16_stage11_v3" in manifest_tool
    assert "expected_build = BUILD_TOOL_PATCH" not in manifest_tool
    assert "_build_tool_diff_record(source, role=role)" in manifest_tool


def _load_overlay_manifest_tool():
    repo = Path(__file__).resolve().parents[3]
    path = repo / "third_party" / "deepgemm_w2_em8_bm16_stage11" / "overlay_manifest.py"
    spec = importlib.util.spec_from_file_location("task26_stage11_manifest", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stock_and_candidate_build_tool_diff_reject_tampering(tmp_path):
    tool = _load_overlay_manifest_tool()
    source = tmp_path / "source"
    source.mkdir()
    base_repo = Path("/home/qinhaiyan/DeepGEMM-GLM52")
    base_script = source / "build_sgl_deep_gemm.sh"
    base_script.write_bytes(
        subprocess.check_output(
            [
                "git",
                "-C",
                str(base_repo),
                "show",
                f"{tool.BASE_COMMIT}:build_sgl_deep_gemm.sh",
            ]
        )
    )
    base_script.chmod(0o755)
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", base_script.name], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Task26",
            "-c",
            "user.email=task26@example.invalid",
            "commit",
            "-qm",
            "base",
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(source), "apply", str(tool.BUILD_TOOL_PATCH)],
        check=True,
    )
    expected_sha = tool._sha256(tool.BUILD_TOOL_PATCH)
    for role in ("stock", "candidate"):
        record = tool._build_tool_diff_record(source, role=role)
        assert record["sha256"] == expected_sha

    base_script.write_text(base_script.read_text() + "\n# tampered\n")
    for role in ("stock", "candidate"):
        with pytest.raises(RuntimeError, match="pinned build-tool patch"):
            tool._build_tool_diff_record(source, role=role)


def test_default_off_imports_neither_experimental_candidate_module():
    repo = Path(__file__).resolve().parents[3]
    code = """
import sys
import sglang.srt.layers.quantization.fp8
import sglang.srt.model_executor.runner.decode_cuda_graph_runner
for name in (
    "sglang.srt.layers.glm52_opt.experimental_deepgemm",
    "sglang.srt.layers.glm52_opt.experimental_deepgemm_em8_bm16_stage11",
):
    assert name not in sys.modules, name
"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["SGLANG_GLM52_OPT"] = "0"
    env.pop("SGLANG_GLM52_OPT_PROFILE", None)
    env["PYTHONPATH"] = os.pathsep.join([str(repo / "python"), str(repo), *sys.path])
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_api_and_build_identity_are_jointly_versioned():
    source = inspect.getsource(stage11)
    assert stage11.VARIANT_NAME == "em8_bm16_stage11"
    assert stage11.PREDECLARED_FALLBACK == "em8_bm16_stage10"
    assert stage11.VARIANT_VERSION == 3
    assert "stage11-v3" in stage11.BUILD_ID
    assert stage11.JIT_IDENTITY.endswith("stage11_v3")
    assert "stage11" in stage11.BUILD_ID
    assert "expected-m8" in stage11.BUILD_ID
    assert "masked_num_stages_override" in source
    assert stage11.FORWARD_BUCKET == (32, 8)
