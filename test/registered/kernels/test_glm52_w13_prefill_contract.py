"""CPU-only tests for the default-off prefill W13 PSUM route."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.glm52_opt import w13_prefill


def _fake_tensor(shape, stride, dtype, device="cuda:0"):
    return SimpleNamespace(
        is_cuda=True,
        dtype=dtype,
        shape=shape,
        stride=lambda: stride,
        storage_offset=lambda: 0,
        device=torch.device(device),
    )


def _w13_inputs():
    lhs = (
        _fake_tensor(
            w13_prefill._A_SHAPE,
            w13_prefill._A_STRIDE,
            torch.float8_e4m3fn,
        ),
        _fake_tensor(
            w13_prefill._AS_SHAPE,
            w13_prefill._AS_STRIDE,
            torch.int32,
        ),
    )
    rhs = (
        _fake_tensor(
            w13_prefill._W13_B_SHAPE,
            w13_prefill._W13_B_STRIDE,
            torch.float8_e4m3fn,
        ),
        _fake_tensor(
            w13_prefill._W13_BS_SHAPE,
            w13_prefill._W13_BS_STRIDE,
            torch.int32,
        ),
    )
    out = _fake_tensor(
        w13_prefill._W13_OUT_SHAPE,
        w13_prefill._W13_OUT_STRIDE,
        torch.bfloat16,
    )
    rowmap = _fake_tensor(w13_prefill._LAYOUT_SHAPE, (1,), torch.int32)
    endpoint = _fake_tensor(w13_prefill._ENDPOINT_SHAPE, (1,), torch.int32)
    return lhs, rhs, out, rowmap, endpoint


def _w2_inputs():
    lhs = (
        _fake_tensor(
            w13_prefill._W2_A_SHAPE,
            w13_prefill._W2_A_STRIDE,
            torch.float8_e4m3fn,
        ),
        _fake_tensor(
            w13_prefill._W2_AS_SHAPE,
            w13_prefill._W2_AS_STRIDE,
            torch.int32,
        ),
    )
    rhs = (
        _fake_tensor(
            w13_prefill._W2_B_SHAPE,
            w13_prefill._W2_B_STRIDE,
            torch.float8_e4m3fn,
        ),
        _fake_tensor(
            w13_prefill._W2_BS_SHAPE,
            w13_prefill._W2_BS_STRIDE,
            torch.int32,
        ),
    )
    out = _fake_tensor(
        w13_prefill._W2_OUT_SHAPE,
        w13_prefill._W2_OUT_STRIDE,
        torch.bfloat16,
    )
    rowmap = _fake_tensor(w13_prefill._LAYOUT_SHAPE, (1,), torch.int32)
    return lhs, rhs, out, rowmap


class _Module:
    def __init__(self):
        self.calls = []

    def m_grouped_fp8_fp4_gemm_nt_contiguous(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return None


def test_default_is_off_and_fixed_endpoint_fixture_is_exact():
    assert not w13_prefill.dispatch_state()["enabled"]
    rowmap, endpoints = w13_prefill._layout_values()
    assert len(rowmap) == 35200
    assert len(endpoints) == 32
    assert sum(w13_prefill.RAW_COUNTS) == 32982
    assert sum(w13_prefill.ALIGNED_COUNTS) == 35200
    start = 0
    for expert, (raw, aligned, endpoint) in enumerate(
        zip(
            w13_prefill.RAW_COUNTS,
            w13_prefill.ALIGNED_COUNTS,
            endpoints,
        )
    ):
        assert endpoint == start + raw
        assert rowmap[start : start + aligned] == [expert] * aligned
        start += aligned


def test_import_performs_no_cuda_query_dso_load_or_cache_mutation():
    script = r"""
import os
import sys
import torch

def forbidden(*args, **kwargs):
    raise AssertionError("CUDA queried during prefill W13 import")

torch.cuda.current_device = forbidden
torch.cuda.get_device_capability = forbidden
names = (
    "DG_JIT_CACHE_DIR",
    "SGLANG_DG_CACHE_DIR",
    "DG_JIT_USE_NVRTC",
    "SGL_DG_USE_NVRTC",
)
before = {name: os.environ.get(name) for name in names}
import sglang.srt.layers.glm52_opt.w13_prefill as module
assert module.dispatch_state()["reason"] == "not_initialized"
assert before == {name: os.environ.get(name) for name in names}
assert not any(name.startswith("deep_gemm_w13_prefill_") for name in sys.modules)
"""
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    subprocess.run([sys.executable, "-c", script], check=True, env=environment)


def test_disabled_dispatch_does_not_touch_contract_or_launch():
    lhs, rhs, out, rowmap, endpoint = _w13_inputs()
    with patch.object(
        w13_prefill,
        "_w13_shape_target",
        side_effect=AssertionError("disabled route must return immediately"),
    ):
        assert not w13_prefill.try_dispatch_w13_prefill(
            lhs,
            rhs,
            out,
            rowmap,
            endpoint,
            recipe_a=None,
            recipe_b=None,
        )


def test_selected_startup_requires_exact_production_topology():
    exact = SimpleNamespace(**w13_prefill.REQUIRED_TOPOLOGY)
    assert (
        w13_prefill._validate_startup_topology(exact)
        == w13_prefill.REQUIRED_TOPOLOGY
    )
    wrong = dict(w13_prefill.REQUIRED_TOPOLOGY)
    wrong["ep_size"] = 4
    with unittest.TestCase().assertRaisesRegex(
        RuntimeError, "requires exact TP8/DP8/EP8"
    ):
        w13_prefill._validate_startup_topology(SimpleNamespace(**wrong))


def test_enabled_w13_uses_endpoint_psum_once_and_w2_uses_stock_rowmap_once():
    old_state = w13_prefill._STATE
    stock = _Module()
    candidate = _Module()
    w13_prefill._STATE = w13_prefill._DispatchState(
        True,
        "ready",
        variant="psum",
        gpu_id=0,
        stock_module=stock,
        candidate_module=candidate,
    )
    try:
        lhs, rhs, out, rowmap, endpoint = _w13_inputs()
        assert w13_prefill.try_dispatch_w13_prefill(
            lhs,
            rhs,
            out,
            rowmap,
            endpoint,
            recipe_a=None,
            recipe_b=None,
        )
        assert len(candidate.calls) == 1
        args, kwargs = candidate.calls[0]
        assert args[3] is endpoint
        assert kwargs == {
            "compiled_dims": "nk",
            "disable_ue8m0_cast": True,
            "use_psum_layout": True,
            "ensure_zero_padding": False,
            "expected_m_for_psum_layout": 1024,
        }
        assert not stock.calls

        w2_lhs, w2_rhs, w2_out, w2_rowmap = _w2_inputs()
        assert w13_prefill.try_dispatch_w2_prefill_stock(
            w2_lhs,
            w2_rhs,
            w2_out,
            w2_rowmap,
            recipe_a=None,
            recipe_b=None,
        )
        assert len(stock.calls) == 1
        args, kwargs = stock.calls[0]
        assert args[3] is w2_rowmap
        assert kwargs == {
            "compiled_dims": "nk",
            "disable_ue8m0_cast": True,
        }
    finally:
        w13_prefill._STATE = old_state


def test_selected_target_aborts_on_missing_endpoint_recipe_or_packed_abi_drift():
    old_state = w13_prefill._STATE
    w13_prefill._STATE = w13_prefill._DispatchState(
        True,
        "ready",
        variant="psum",
        gpu_id=0,
        stock_module=_Module(),
        candidate_module=_Module(),
    )
    try:
        lhs, rhs, out, rowmap, endpoint = _w13_inputs()
        with unittest.TestCase().assertRaisesRegex(
            RuntimeError, "no scatter endpoint"
        ):
            w13_prefill.try_dispatch_w13_prefill(
                lhs,
                rhs,
                out,
                rowmap,
                None,
                recipe_a=None,
                recipe_b=None,
            )
        with unittest.TestCase().assertRaisesRegex(
            RuntimeError, "changed the FP8 recipe"
        ):
            w13_prefill.try_dispatch_w13_prefill(
                lhs,
                rhs,
                out,
                rowmap,
                endpoint,
                recipe_a=(1, 128),
                recipe_b=None,
            )
        lhs[1].stride = lambda: (12, 1)
        with unittest.TestCase().assertRaisesRegex(
            RuntimeError, "violates the packed ABI"
        ):
            w13_prefill.try_dispatch_w13_prefill(
                lhs,
                rhs,
                out,
                rowmap,
                endpoint,
                recipe_a=None,
                recipe_b=None,
            )
    finally:
        w13_prefill._STATE = old_state


def test_runner_has_no_profile_counter_and_keeps_w2_on_rowmap():
    source = (
        Path(__file__).parents[3]
        / "python/sglang/srt/layers/moe/moe_runner/deep_gemm.py"
    ).read_text()
    contiguous = source[source.index("    def _run_contiguous_gemm(") :]
    contiguous = contiguous[: contiguous.index("    def _run_bf16_contiguous_gemm(")]
    assert "record_psum_hit" not in contiguous
    assert "contig_psum_kwargs" not in contiguous
    assert "try_dispatch_w13_prefill(" in contiguous
    assert "try_dispatch_w2_prefill_stock(" in contiguous
    assert "runner_input.expert_start_loc" in contiguous
    assert "m_indices," in contiguous
