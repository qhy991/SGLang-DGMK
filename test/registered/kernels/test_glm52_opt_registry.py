"""Unit tests for GLM-5.2 opt registry (no GPU required)."""

from __future__ import annotations

import os
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("SGLANG_GLM52_OPT", "1")
os.environ.setdefault("SGLANG_GLM52_OPT_PROFILE", "full")

from sglang.srt.layers.deep_gemm_wrapper import entrypoint as deep_gemm_entrypoint
from sglang.srt.layers.deep_gemm_wrapper.entrypoint import (
    _glm52_moe_dispatch_compatible,
)
from sglang.srt.layers.glm52_opt.context import (
    get_forward_m,
    prefix_to_op_name,
    set_forward_mode,
)
from sglang.srt.layers.glm52_opt.phase import infer_glm52_phase
from sglang.srt.layers.glm52_opt.registry import lookup


def test_prefix_mapping():
    assert prefix_to_op_name("model.layers.0.self_attn.q_b_proj") == "q_b_proj"
    assert (
        prefix_to_op_name("model.layers.0.self_attn.indexer.wq_b")
        == "index_q_upproj"
    )
    assert (
        prefix_to_op_name("model.layers.0.self_attn.indexer.wk_weights_proj")
        == "index_wk_weights_proj"
    )
    assert lookup("index_wk_weights_proj", "decode") is None
    assert prefix_to_op_name("model.layers.0.mlp.gate_proj") == "moe_gate_proj"


def test_prefix_mapping_does_not_use_substrings():
    assert prefix_to_op_name("model.layers.0.not_q_b_proj_adapter") is None
    assert prefix_to_op_name("model.layers.0.indexer.wk_weights_proj_extra") is None


def test_prefill_full_profile():
    assert lookup("index_q_upproj", "prefill") is not None
    assert lookup("moe_gate_proj", "prefill") is None


def test_decode_registry():
    assert lookup("dsa_decode_attn", "decode") is not None


def test_serving_safe_requires_explicit_allowlist():
    old_profile = os.environ.get("SGLANG_GLM52_OPT_PROFILE")
    old_ops = os.environ.get("SGLANG_GLM52_OPT_OPS")
    try:
        os.environ["SGLANG_GLM52_OPT_PROFILE"] = "serving_safe"
        os.environ.pop("SGLANG_GLM52_OPT_OPS", None)
        assert lookup("q_b_proj", "decode") is None
        assert lookup("fused_qkv_a_proj", "decode") is None

        os.environ["SGLANG_GLM52_OPT_OPS"] = "q_b_proj"
        assert lookup("q_b_proj", "decode") is not None
        assert lookup("fused_qkv_a_proj", "decode") is None

        os.environ["SGLANG_GLM52_OPT_PROFILE"] = "typo"
        assert lookup("q_b_proj", "decode") is None
    finally:
        if old_profile is None:
            os.environ.pop("SGLANG_GLM52_OPT_PROFILE", None)
        else:
            os.environ["SGLANG_GLM52_OPT_PROFILE"] = old_profile
        if old_ops is None:
            os.environ.pop("SGLANG_GLM52_OPT_OPS", None)
        else:
            os.environ["SGLANG_GLM52_OPT_OPS"] = old_ops


def test_e2e_candidates_defaults_to_archived_leaf_winners():
    from sglang.srt.layers.glm52_opt.config import contig_psum_kwargs

    old_profile = os.environ.get("SGLANG_GLM52_OPT_PROFILE")
    old_ops = os.environ.get("SGLANG_GLM52_OPT_OPS")
    old_opt = os.environ.get("SGLANG_GLM52_OPT")
    try:
        os.environ["SGLANG_GLM52_OPT"] = "1"
        os.environ["SGLANG_GLM52_OPT_PROFILE"] = "e2e_candidates"
        os.environ.pop("SGLANG_GLM52_OPT_OPS", None)

        assert lookup("o_proj", "decode", m=16) is not None
        assert lookup("o_proj", "decode", m=32) is not None
        # Historical archive swaps stay off unless listed in the e2e set.
        assert lookup("q_b_proj", "decode") is None
        assert lookup("fused_qkv_a_proj", "decode") is None

        os.environ["SGLANG_GLM52_OPT_OPS"] = "o_proj"
        assert lookup("o_proj", "decode") is not None
        assert contig_psum_kwargs("moe_gate_proj") == {}
        assert contig_psum_kwargs("moe_down_proj") == {}

        os.environ.pop("SGLANG_GLM52_OPT_OPS", None)
        w13 = contig_psum_kwargs("moe_gate_proj")
        w2 = contig_psum_kwargs("moe_down_proj")
        assert w13["use_psum_layout"] is True
        assert w2["expected_m_for_psum_layout"] == 1024
        assert contig_psum_kwargs("o_proj") == {}
    finally:
        for key, old_value in (
            ("SGLANG_GLM52_OPT_PROFILE", old_profile),
            ("SGLANG_GLM52_OPT_OPS", old_ops),
            ("SGLANG_GLM52_OPT", old_opt),
        ):
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def test_m_bucket_selectively_falls_back():
    old_profile = os.environ.get("SGLANG_GLM52_OPT_PROFILE")
    old_ops = os.environ.get("SGLANG_GLM52_OPT_OPS")
    old_buckets = os.environ.get("SGLANG_GLM52_OPT_M_BUCKETS")
    try:
        os.environ["SGLANG_GLM52_OPT_PROFILE"] = "serving_safe"
        os.environ["SGLANG_GLM52_OPT_OPS"] = "q_b_proj"
        os.environ["SGLANG_GLM52_OPT_M_BUCKETS"] = "q_b_proj:16"

        assert lookup("q_b_proj", "decode", m=16) is not None
        assert lookup("q_b_proj", "decode", m=32) is None
        assert lookup("q_b_proj", "decode") is None
        # A bucket entry restricts only its own op.
        os.environ["SGLANG_GLM52_OPT_OPS"] = "q_b_proj,o_proj"
        assert lookup("o_proj", "decode", m=32) is not None
    finally:
        for key, old_value in (
            ("SGLANG_GLM52_OPT_PROFILE", old_profile),
            ("SGLANG_GLM52_OPT_OPS", old_ops),
            ("SGLANG_GLM52_OPT_M_BUCKETS", old_buckets),
        ):
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def test_forward_m_context_tracks_local_bucket():
    try:
        set_forward_mode(None, 16)
        assert get_forward_m() == 16
        set_forward_mode(None, 32)
        assert get_forward_m() == 32
    finally:
        set_forward_mode(None)


def test_moe_swap_preserves_production_overlap_contract():
    assert _glm52_moe_dispatch_compatible(None, None, None)
    assert not _glm52_moe_dispatch_compatible(object(), None, None)
    assert not _glm52_moe_dispatch_compatible(None, (1, 128), None)
    assert not _glm52_moe_dispatch_compatible(None, None, (1, 128))


def test_contiguous_grouped_gemm_forwards_opt_in_controls_only_when_requested():
    class FakeTensor:
        def __init__(self, shape):
            self.shape = shape

    lhs = (FakeTensor((35200, 2048)), FakeTensor((35200, 4)))
    rhs = (FakeTensor((32, 6144, 2048)), FakeTensor((32, 6144, 4)))
    out = FakeTensor((35200, 6144))
    row_layout = FakeTensor((35200,))
    psum_layout = FakeTensor((32,))
    calls = []

    def fake_grouped(lhs_arg, rhs_arg, out_arg, layout_arg, **kwargs):
        calls.append((lhs_arg, rhs_arg, out_arg, layout_arg, kwargs))

    with patch.object(
        deep_gemm_entrypoint,
        "deep_gemm",
        SimpleNamespace(m_grouped_fp8_gemm_nt_contiguous=fake_grouped),
        create=True,
    ), patch.object(
        deep_gemm_entrypoint.compile_utils,
        "deep_gemm_execution_hook",
        lambda *args: nullcontext(),
    ):
        deep_gemm_entrypoint.grouped_gemm_nt_f8f8bf16_contig(
            lhs, rhs, out, row_layout
        )
        assert calls[-1][3] is row_layout
        assert calls[-1][4] == {}

        deep_gemm_entrypoint.grouped_gemm_nt_f8f8bf16_contig(
            lhs,
            rhs,
            out,
            psum_layout,
            compiled_dims="mnk",
            use_psum_layout=True,
            ensure_zero_padding=False,
            expected_m_for_psum_layout=1024,
        )
        assert calls[-1][3] is psum_layout
        assert calls[-1][4] == {
            "compiled_dims": "mnk",
            "use_psum_layout": True,
            "ensure_zero_padding": False,
            "expected_m_for_psum_layout": 1024,
        }


def test_phase_defaults():
    assert infer_glm52_phase(None, 32) == "decode"
    assert infer_glm52_phase(None, 1024) == "prefill"
