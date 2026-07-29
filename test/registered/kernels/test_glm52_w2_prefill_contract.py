"""CPU-only tests for the default-off Task 30 W2 PSUM route."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.layers.glm52_opt import w2_prefill


def _fake_tensor(shape, stride, dtype, device="cuda:0"):
    return SimpleNamespace(
        is_cuda=True,
        dtype=dtype,
        shape=shape,
        stride=lambda: stride,
        storage_offset=lambda: 0,
        device=torch.device(device),
    )


def _inputs():
    lhs = (
        _fake_tensor(
            w2_prefill._A_SHAPE,
            w2_prefill._A_STRIDE,
            torch.float8_e4m3fn,
        ),
        _fake_tensor(
            w2_prefill._AS_SHAPE,
            w2_prefill._AS_STRIDE,
            torch.int32,
        ),
    )
    rhs = (
        _fake_tensor(
            w2_prefill._B_SHAPE,
            w2_prefill._B_STRIDE,
            torch.float8_e4m3fn,
        ),
        _fake_tensor(
            w2_prefill._BS_SHAPE,
            w2_prefill._BS_STRIDE,
            torch.int32,
        ),
    )
    out = _fake_tensor(
        w2_prefill._OUT_SHAPE,
        w2_prefill._OUT_STRIDE,
        torch.bfloat16,
    )
    rowmap = _fake_tensor(w2_prefill._ROWMAP_SHAPE, (1,), torch.int32)
    endpoint = _fake_tensor(w2_prefill._ENDPOINT_SHAPE, (1,), torch.int32)
    return lhs, rhs, out, rowmap, endpoint


class _Module:
    def __init__(self, error: Exception | None = None):
        self.calls = []
        self.error = error

    def m_grouped_fp8_fp4_gemm_nt_contiguous(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.error is not None:
            raise self.error


def test_default_off_and_import_is_cpu_only():
    assert not w2_prefill.dispatch_state()["enabled"]
    script = r"""
import torch
def forbidden(*args, **kwargs):
    raise AssertionError("CUDA queried during W2 prefill import")
torch.cuda.current_device = forbidden
torch.cuda.get_device_capability = forbidden
import sglang.srt.layers.glm52_opt.w2_prefill
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).parents[3] / "python")
    subprocess.run([sys.executable, "-c", script], check=True, env=env)


def test_exact_dispatch_uses_task28_endpoint_once_and_no_fallback():
    old_state = w2_prefill._STATE
    module = _Module()
    w2_prefill._STATE = w2_prefill._DispatchState(
        True,
        "ready",
        variant="stage7",
        gpu_id=0,
        candidate_module=module,
    )
    try:
        lhs, rhs, out, rowmap, endpoint = _inputs()
        assert w2_prefill.try_dispatch_w2_prefill(
            lhs,
            rhs,
            out,
            rowmap,
            endpoint,
            recipe_a=None,
            recipe_b=None,
        )
        assert len(module.calls) == 1
        args, kwargs = module.calls[0]
        assert args[3] is endpoint
        assert rowmap not in args
        assert kwargs == {
            "compiled_dims": "nk",
            "disable_ue8m0_cast": True,
            "use_psum_layout": True,
            "ensure_zero_padding": False,
            "expected_m_for_psum_layout": 1024,
        }
    finally:
        w2_prefill._STATE = old_state


def test_non_target_falls_back_but_malformed_target_and_candidate_error_abort():
    old_state = w2_prefill._STATE
    module = _Module()
    w2_prefill._STATE = w2_prefill._DispatchState(
        True,
        "ready",
        variant="psum",
        gpu_id=0,
        candidate_module=module,
    )
    try:
        lhs, rhs, out, rowmap, endpoint = _inputs()
        wrong_out = _fake_tensor((1, 1), (1, 1), torch.bfloat16)
        assert not w2_prefill.try_dispatch_w2_prefill(
            lhs,
            rhs,
            wrong_out,
            rowmap,
            endpoint,
            recipe_a=None,
            recipe_b=None,
        )
        bad_scale = _fake_tensor(
            w2_prefill._AS_SHAPE,
            (4, 1),
            torch.int32,
        )
        with pytest.raises(RuntimeError, match="packed ABI"):
            w2_prefill.try_dispatch_w2_prefill(
                (lhs[0], bad_scale),
                rhs,
                out,
                rowmap,
                endpoint,
                recipe_a=None,
                recipe_b=None,
            )
        with pytest.raises(RuntimeError, match="FP8 recipe"):
            w2_prefill.try_dispatch_w2_prefill(
                lhs,
                rhs,
                out,
                rowmap,
                endpoint,
                recipe_a=(1, 128),
                recipe_b=None,
            )
        w2_prefill._STATE = w2_prefill._DispatchState(
            True,
            "ready",
            variant="stage7",
            gpu_id=0,
            candidate_module=_Module(RuntimeError("candidate failed")),
        )
        with pytest.raises(RuntimeError, match="candidate failed"):
            w2_prefill.try_dispatch_w2_prefill(
                lhs,
                rhs,
                out,
                rowmap,
                endpoint,
                recipe_a=None,
                recipe_b=None,
            )
    finally:
        w2_prefill._STATE = old_state


def test_production_route_keeps_w13_and_activation_stock_and_reuses_endpoint():
    source = (
        Path(__file__).parents[3]
        / "python/sglang/srt/layers/moe/moe_runner/deep_gemm.py"
    ).read_text()
    contiguous = source[source.index("    def _run_contiguous_gemm(") :]
    contiguous = contiguous[: contiguous.index("    def _run_bf16_contiguous_gemm(")]
    assert contiguous.count("try_dispatch_w2_prefill(") == 1
    assert "runner_input.expert_start_loc" in contiguous
    assert "maybe_silu_mul_quant_packed(" in contiguous
    assert "try_dispatch_w13_prefill(" in contiguous
    assert "try_dispatch_w2_prefill_stock(" in contiguous
    assert "record_" not in contiguous


def test_stage7_patch_is_one_exact_heuristic_change():
    patch = (
        Path(__file__).parents[3] / "third_party/deepgemm_w2_prefill/patches/"
        "0001-exact-w2-psum-stage7.patch"
    ).read_text()
    assert patch.count("diff --git ") == 1
    for value in (
        "MGroupedContiguousWithPsumLayout",
        'desc.compiled_dims == "nk"',
        "not desc.ensure_zero_padding",
        "desc.m == 35200",
        "desc.n == 6144",
        "desc.k == 2048",
        "desc.expected_m == 1024",
        "layout.cluster_m == 1 and layout.cluster_n == 2",
        "DG_HOST_ASSERT(num_stages == 8)",
        "num_stages = 7",
    ):
        assert value in patch
