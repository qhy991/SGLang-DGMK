"""CPU-only tests for the post-assignment W13 candidate contract."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from sglang.srt.layers.glm52_opt import w13_decode
from sglang.srt.layers.glm52_opt.w13_context import (
    get_w13_decode_forward_marker,
    w13_decode_forward_scope,
)


def _fake_tensor(shape, stride, dtype, device="cuda:0"):
    return SimpleNamespace(
        is_cuda=True,
        dtype=dtype,
        shape=shape,
        stride=lambda: stride,
        storage_offset=lambda: 0,
        device=torch.device(device),
    )


def _exact_inputs():
    lhs = (
        _fake_tensor(w13_decode._A_SHAPE, w13_decode._A_STRIDE, torch.float8_e4m3fn),
        _fake_tensor(w13_decode._AS_SHAPE, w13_decode._AS_STRIDE, torch.int32),
    )
    rhs = (
        _fake_tensor(w13_decode._B_SHAPE, w13_decode._B_STRIDE, torch.float8_e4m3fn),
        _fake_tensor(w13_decode._BS_SHAPE, w13_decode._BS_STRIDE, torch.int32),
    )
    out = _fake_tensor(w13_decode._OUT_SHAPE, w13_decode._OUT_STRIDE, torch.bfloat16)
    mask = _fake_tensor(w13_decode._MASK_SHAPE, w13_decode._MASK_STRIDE, torch.int32)
    return lhs, rhs, out, mask


def _matches(
    *, expected_m=4, overlap=None, max_block_n=256, recipe_a=None, recipe_b=None
):
    lhs, rhs, out, mask = _exact_inputs()
    return w13_decode._contract_matches(
        lhs,
        rhs,
        out,
        mask,
        expected_m,
        overlap,
        max_block_n,
        recipe_a,
        recipe_b,
    )


def test_default_is_off():
    assert not w13_decode.dispatch_state()["enabled"]


def test_hotspot_registry_selects_only_w13_and_defaults_to_bm32_2sm():
    with (
        patch(
            "sglang.srt.layers.glm52_opt.config.is_enabled",
            return_value=True,
        ),
        patch(
            "sglang.srt.layers.glm52_opt.config.profile_name",
            return_value="hotspot_candidates",
        ),
        patch(
            "sglang.srt.layers.glm52_opt.config.hotspot_candidate_ops",
            return_value=frozenset({"moe_gate_proj"}),
        ),
        patch.dict(
            os.environ,
            {"SGLANG_GLM52_W13_DECODE_VARIANT": ""},
            clear=False,
        ),
    ):
        assert w13_decode.requested_variant() == "bm32_2sm"
        assert w13_decode.initialization_requested()

    with (
        patch(
            "sglang.srt.layers.glm52_opt.config.is_enabled",
            return_value=True,
        ),
        patch(
            "sglang.srt.layers.glm52_opt.config.profile_name",
            return_value="hotspot_candidates",
        ),
        patch(
            "sglang.srt.layers.glm52_opt.config.hotspot_candidate_ops",
            return_value=frozenset({"moe_down_proj"}),
        ),
        patch.dict(
            os.environ,
            {"SGLANG_GLM52_W13_DECODE_VARIANT": "bm32_2sm"},
            clear=False,
        ),
    ):
        assert w13_decode.requested_variant() == ""
        assert not w13_decode.initialization_requested()


def test_import_performs_no_cuda_query_dso_load_or_cache_mutation():
    script = r"""
import os
import sys
import torch

def forbidden(*args, **kwargs):
    raise AssertionError("CUDA queried during W13 module import")

torch.cuda.current_device = forbidden
torch.cuda.get_device_capability = forbidden
before = {
    name: os.environ.get(name)
    for name in (
        "DG_JIT_CACHE_DIR",
        "SGLANG_DG_CACHE_DIR",
        "DG_JIT_USE_NVRTC",
        "SGLANG_DG_USE_NVRTC",
    )
}
import sglang.srt.layers.glm52_opt.w13_decode as module
assert module.dispatch_state()["reason"] == "not_initialized"
assert before == {
    name: os.environ.get(name)
    for name in (
        "DG_JIT_CACHE_DIR",
        "SGLANG_DG_CACHE_DIR",
        "DG_JIT_USE_NVRTC",
        "SGLANG_DG_USE_NVRTC",
    )
}
assert not any(name.startswith("deep_gemm_w13_") for name in sys.modules)
"""
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        env=environment,
    )


def test_explicit_invalid_or_unmaterialized_variant_aborts_before_cuda():
    old_state = w13_decode._STATE
    try:
        with patch.dict(
            os.environ,
            {"SGLANG_GLM52_W13_DECODE_VARIANT": "not-a-variant"},
            clear=False,
        ):
            with unittest.TestCase().assertRaisesRegex(
                RuntimeError,
                "unsupported requested W13 variant",
            ):
                w13_decode.initialize_w13_decode_after_assignment(
                    0,
                    object(),
                    compile_utils_loader=lambda: None,
                )
        with patch.dict(
            os.environ,
            {
                "SGLANG_GLM52_W13_DECODE_VARIANT": "bm32_1sm",
                "SGLANG_GLM52_W13_DECODE_MANIFEST": "",
            },
            clear=False,
        ):
            with unittest.TestCase().assertRaisesRegex(
                RuntimeError,
                "exact build manifest",
            ):
                w13_decode.initialize_w13_decode_after_assignment(
                    0,
                    object(),
                    compile_utils_loader=lambda: None,
                )
    finally:
        w13_decode._STATE = old_state


def test_exact_four_expected_m_contracts_match():
    assert all(_matches(expected_m=value) for value in (4, 5, 8, 9))


def test_overlap_recipe_and_unrelated_expected_m_fail_closed():
    assert not _matches(overlap=object())
    assert not _matches(recipe_a=(1, 128))
    assert not _matches(recipe_b=(1, 128))
    assert not _matches(expected_m=6)
    assert not _matches(max_block_n=160)


def test_shape_stride_dtype_offset_and_device_fail_closed():
    lhs, rhs, out, mask = _exact_inputs()
    cases = [
        (lhs[0], "shape", (32, 1023, 6144)),
        (lhs[1], "dtype", torch.float32),
        (rhs[0], "is_cuda", False),
        (rhs[1], "stride", lambda: (1, 1, 1)),
        (out, "storage_offset", lambda: 1),
        (mask, "device", torch.device("cuda:1")),
    ]
    for tensor, field, value in cases:
        old = getattr(tensor, field)
        setattr(tensor, field, value)
        assert not w13_decode._contract_matches(
            lhs, rhs, out, mask, 4, None, 256, None, None
        )
        setattr(tensor, field, old)


def test_disabled_dispatch_does_not_touch_contract_or_launch():
    lhs, rhs, out, mask = _exact_inputs()
    with patch.object(
        w13_decode, "_contract_matches", side_effect=AssertionError("must not run")
    ):
        assert not w13_decode.try_dispatch_w13_decode(
            lhs,
            rhs,
            out,
            mask,
            4,
            overlap_args=None,
            max_block_n=256,
            recipe_a=None,
            recipe_b=None,
        )


class _Mode:
    def __init__(self, decode: bool):
        self.decode = decode

    def is_decode(self):
        return self.decode


def _forward_batch(decode=True):
    return SimpleNamespace(forward_mode=_Mode(decode))


def test_private_forward_scope_covers_all_four_buckets_and_resets():
    assert get_w13_decode_forward_marker() is None
    cases = ((16, 4), (16, 5), (32, 8), (32, 9))
    for token_bucket, expected_m in cases:
        with w13_decode_forward_scope(
            _forward_batch(), token_bucket, graph_capture=True
        ):
            marker = get_w13_decode_forward_marker()
            assert marker is not None
            assert marker.token_bucket == token_bucket
            assert marker.graph_capture is True
            assert w13_decode._marker_matches(marker, expected_m)
        assert get_w13_decode_forward_marker() is None


def test_private_forward_scope_rejects_non_decode_and_resets_on_error():
    with w13_decode_forward_scope(
        _forward_batch(decode=False), 16, graph_capture=False
    ):
        assert get_w13_decode_forward_marker() is None
    try:
        with w13_decode_forward_scope(_forward_batch(), 32, graph_capture=False):
            assert get_w13_decode_forward_marker() is not None
            raise RuntimeError("scope exit")
    except RuntimeError:
        pass
    assert get_w13_decode_forward_marker() is None


def test_real_eager_and_graph_forward_callsites_enter_private_scope():
    root = Path(__file__).resolve().parents[3]
    graph_source = (
        root / "python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py"
    ).read_text()
    eager_source = (
        root / "python/sglang/srt/model_executor/runner/eager_runner.py"
    ).read_text()
    assert "with w13_decode_forward_scope(" in graph_source
    assert "graph_capture=True" in graph_source
    assert "w13_decode_forward_scope(" in eager_source
    assert "graph_capture=False" in eager_source


class _FakeModule:
    def __init__(self, name):
        self.name = name
        self.pdl = False
        self.num_sms = 0
        self.tc_util = 0
        self.calls = []

    def set_pdl(self, value):
        self.pdl = bool(value)

    def get_pdl(self):
        return self.pdl

    def set_num_sms(self, value):
        self.num_sms = int(value)

    def get_num_sms(self):
        return self.num_sms

    def set_tc_util(self, value):
        self.tc_util = int(value)

    def get_tc_util(self):
        return self.tc_util

    def fp8_m_grouped_gemm_nt_masked(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return None


def test_dispatch_requires_private_marker_and_bucket_expected_m_pair():
    old_state = w13_decode._STATE
    candidate = _FakeModule("candidate")
    w13_decode._STATE = w13_decode._DispatchState(
        True,
        "ready",
        variant="bm32_1sm",
        config=w13_decode.VARIANT_CONFIGS["bm32_1sm"],
        gpu_id=0,
        candidate_module=candidate,
    )
    lhs, rhs, out, mask = _exact_inputs()
    kwargs = dict(
        overlap_args=None,
        max_block_n=256,
        recipe_a=None,
        recipe_b=None,
    )
    try:
        assert not w13_decode.try_dispatch_w13_decode(lhs, rhs, out, mask, 4, **kwargs)
        with w13_decode_forward_scope(_forward_batch(), 16, graph_capture=False):
            assert w13_decode.try_dispatch_w13_decode(lhs, rhs, out, mask, 4, **kwargs)
            with pytest.raises(RuntimeError, match="unexpected expected_m"):
                w13_decode.try_dispatch_w13_decode(
                    lhs, rhs, out, mask, 8, **kwargs
                )
        assert len(candidate.calls) == 1
    finally:
        w13_decode._STATE = old_state


def test_selected_w13_bucket_abi_drift_fails_without_candidate_launch():
    old_state = w13_decode._STATE
    candidate = _FakeModule("candidate")
    w13_decode._STATE = w13_decode._DispatchState(
        True,
        "ready",
        variant="bm32_2sm",
        config=w13_decode.VARIANT_CONFIGS["bm32_2sm"],
        gpu_id=0,
        candidate_module=candidate,
    )
    lhs, rhs, out, mask = _exact_inputs()
    try:
        with (
            w13_decode_forward_scope(
                _forward_batch(),
                16,
                graph_capture=False,
            ),
            pytest.raises(RuntimeError, match="no longer matches"),
        ):
            w13_decode.try_dispatch_w13_decode(
                lhs,
                rhs,
                out,
                mask,
                4,
                overlap_args=object(),
                max_block_n=256,
                recipe_a=None,
                recipe_b=None,
            )
        assert candidate.calls == []
    finally:
        w13_decode._STATE = old_state


def test_post_assignment_initializer_binds_stock_before_compile_utils_and_restores_env():
    old_state = w13_decode._STATE
    stock = _FakeModule("stock")
    candidate = _FakeModule("candidate")
    events = []
    compile_utils = SimpleNamespace(
        _ENABLE_JIT_DEEPGEMM_PRECOMPILE=True,
        update_deep_gemm_config=lambda gpu_id, server_args: events.append(
            ("compile_utils_update", gpu_id, server_args)
        ),
    )
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = root / "manifest.json"
        manifest.write_text("{}")
        records = {}
        for name in ("stock", "candidate"):
            package = root / name / "package"
            cache = root / name / "jit"
            package.mkdir(parents=True)
            cache.mkdir(parents=True)
            (package / "__init__.py").write_text("")
            (package / "_C.so").write_bytes(name.encode())
            records[name] = {
                "package": str(package),
                "package_init_sha256": w13_decode._sha256(package / "__init__.py"),
                "shared_object": str(package / "_C.so"),
                "shared_object_sha256": w13_decode._sha256(package / "_C.so"),
                "jit_cache": str(cache),
            }
        stock.__file__ = str(Path(records["stock"]["package"]) / "__init__.py")
        candidate.__file__ = str(
            Path(records["candidate"]["package"]) / "__init__.py"
        )

        def load_variant(_manifest, name, **_kwargs):
            events.append(("load", name, os.environ["DG_JIT_CACHE_DIR"]))
            assert name == "candidate"
            return candidate, records[name], {}

        def import_stock(name):
            assert name == "deep_gemm"
            events.append(("load", "stock", os.environ["DG_JIT_CACHE_DIR"]))
            return stock

        def launch(module, _tensors, expected_m, config):
            events.append(("launch", module.name, expected_m, config))

        snapshots = {
            str(Path(records["stock"]["jit_cache"])): {"stock.cubin": "a"},
            str(Path(records["candidate"]["jit_cache"])): {"candidate.cubin": "b"},
        }

        original_dg = os.environ.get("DG_JIT_CACHE_DIR")
        original_sglang = os.environ.get("SGLANG_DG_CACHE_DIR")
        original_nvrtc = os.environ.get("DG_JIT_USE_NVRTC")
        original_sgl_nvrtc = os.environ.get("SGLANG_DG_USE_NVRTC")
        with (
            patch.dict(
                os.environ,
                {
                    "SGLANG_GLM52_W13_DECODE_VARIANT": "bm32_1sm",
                    "SGLANG_GLM52_W13_DECODE_MANIFEST": str(manifest),
                    "DG_JIT_CACHE_DIR": "before-dg",
                    "SGLANG_DG_CACHE_DIR": "before-sglang",
                    "DG_JIT_USE_NVRTC": "before-nvrtc",
                    "SGLANG_DG_USE_NVRTC": "before-sgl-nvrtc",
                },
                clear=False,
            ),
            patch.object(
                w13_decode,
                "_variant_record",
                side_effect=lambda _path, name: (records[name], {}),
            ),
            patch.object(
                w13_decode.importlib,
                "import_module",
                side_effect=import_stock,
            ),
            patch.object(w13_decode, "load_variant", side_effect=load_variant),
            patch.object(w13_decode, "_allocate_warm_inputs", return_value={}),
            patch.object(w13_decode, "_launch_named_config", side_effect=launch),
            patch.object(
                w13_decode,
                "_cache_snapshot",
                side_effect=lambda path: snapshots[str(path)],
            ),
            patch.object(torch.cuda, "current_device", return_value=0),
            patch.object(torch.cuda, "get_device_capability", return_value=(10, 0)),
            patch.object(torch.cuda, "synchronize"),
        ):
            configured = w13_decode.initialize_w13_decode_after_assignment(
                0,
                "server-args",
                compile_utils_loader=lambda: (
                    events.append(
                        ("compile_utils_import", os.environ["DG_JIT_CACHE_DIR"])
                    )
                    or compile_utils
                ),
            )
            assert configured
            assert w13_decode.dispatch_state()["enabled"]
            assert os.environ["DG_JIT_CACHE_DIR"] == "before-dg"
            assert os.environ["SGLANG_DG_CACHE_DIR"] == "before-sglang"
            assert os.environ["DG_JIT_USE_NVRTC"] == "before-nvrtc"
            assert os.environ["SGLANG_DG_USE_NVRTC"] == "before-sgl-nvrtc"

        load_stock_index = next(
            index
            for index, event in enumerate(events)
            if event[:2] == ("load", "stock")
        )
        compile_import_index = next(
            index
            for index, event in enumerate(events)
            if event[0] == "compile_utils_import"
        )
        load_candidate_index = next(
            index
            for index, event in enumerate(events)
            if event[:2] == ("load", "candidate")
        )
        assert load_stock_index < compile_import_index < load_candidate_index
        assert events[load_stock_index][2] == records["stock"]["jit_cache"]
        assert events[load_candidate_index][2] == records["candidate"]["jit_cache"]
        assert compile_utils._ENABLE_JIT_DEEPGEMM_PRECOMPILE is False
        assert w13_decode.dispatch_state()["jit_use_nvrtc"] is False
        assert stock.get_pdl() and candidate.get_pdl()
        assert stock.get_num_sms() == candidate.get_num_sms() == 148
        assert stock.get_tc_util() == candidate.get_tc_util() == 100

        if original_dg is None:
            os.environ.pop("DG_JIT_CACHE_DIR", None)
        else:
            os.environ["DG_JIT_CACHE_DIR"] = original_dg
        if original_sglang is None:
            os.environ.pop("SGLANG_DG_CACHE_DIR", None)
        else:
            os.environ["SGLANG_DG_CACHE_DIR"] = original_sglang
        if original_nvrtc is None:
            os.environ.pop("DG_JIT_USE_NVRTC", None)
        else:
            os.environ["DG_JIT_USE_NVRTC"] = original_nvrtc
        if original_sgl_nvrtc is None:
            os.environ.pop("SGLANG_DG_USE_NVRTC", None)
        else:
            os.environ["SGLANG_DG_USE_NVRTC"] = original_sgl_nvrtc
    w13_decode._STATE = old_state
