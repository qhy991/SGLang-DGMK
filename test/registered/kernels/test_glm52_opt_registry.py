"""Unit tests for GLM-5.2 opt registry (no GPU required)."""

from __future__ import annotations

import os
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

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
from sglang.srt.layers.glm52_opt.dispatch import (
    _fixed_nk_forward_mode_matches,
    _nvtx_range,
    _profiler_range_name,
)
from sglang.srt.layers.glm52_opt.fp8_gemm import run_fp8_gemm
from sglang.srt.layers.glm52_opt.phase import infer_glm52_phase
from sglang.srt.layers.glm52_opt.registry import lookup
from sglang.srt.model_executor.forward_batch_info import ForwardMode


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
    old_buckets = os.environ.get("SGLANG_GLM52_OPT_M_BUCKETS")
    try:
        os.environ["SGLANG_GLM52_OPT"] = "1"
        os.environ["SGLANG_GLM52_OPT_PROFILE"] = "e2e_candidates"
        os.environ.pop("SGLANG_GLM52_OPT_OPS", None)
        os.environ.pop("SGLANG_GLM52_OPT_M_BUCKETS", None)

        o_proj = lookup("o_proj", "decode", m=16)
        assert o_proj is not None
        assert o_proj.implementation == "fixed_nk"
        assert o_proj.profiler_name == "infini_kernel_glm52_attn_o_decode_nk"
        assert lookup("o_proj", "decode", m=32) is not None
        assert lookup("o_proj", "decode", m=64) is None
        # Historical archive swaps stay off unless listed in the e2e set.
        assert lookup("q_b_proj", "decode") is None
        assert lookup("fused_qkv_a_proj", "decode") is None
        # The default MoE names arm only their prefill PSUM hook; they must not
        # accidentally select the archived decode MoE kernels.
        assert lookup("moe_gate_proj", "decode", m=16) is None
        assert lookup("moe_down_proj", "decode", m=16) is None

        os.environ["SGLANG_GLM52_OPT_OPS"] = "o_proj"
        assert lookup("o_proj", "decode", m=16) is not None
        assert contig_psum_kwargs("moe_gate_proj") == {}
        assert contig_psum_kwargs("moe_down_proj") == {}

        os.environ["SGLANG_GLM52_OPT_OPS"] = "index_q_upproj"
        indexer = lookup("index_q_upproj", "decode", m=16)
        assert indexer is not None
        assert indexer.implementation == "fixed_nk"
        assert indexer.n == 4096 and indexer.k == 2048
        assert lookup("index_q_upproj", "decode", m=32) is not None
        assert lookup("index_q_upproj", "decode", m=8) is None
        assert lookup("index_q_upproj", "prefill", m=16) is None

        os.environ["SGLANG_GLM52_OPT_OPS"] = "fused_qkv_a_proj"
        qkv = lookup("fused_qkv_a_proj", "prefill", m=4096)
        assert qkv is not None
        assert qkv.profiler_name == "infini_kernel_glm52_fused_qkv_a_prefill_nk"
        assert qkv.n == 2624 and qkv.k == 6144
        assert lookup("fused_qkv_a_proj", "prefill", m=2048) is None
        # Decode fused_qkv_a_proj is a distinct graph-only fixed-N/K candidate
        # (same N/K, weight shared with prefill; only M changes to {16,32}).
        qkv_dec = lookup("fused_qkv_a_proj", "decode", m=16)
        assert qkv_dec is not None
        assert qkv_dec.implementation == "fixed_nk"
        assert qkv_dec.graph_only is True
        assert qkv_dec.n == 2624 and qkv_dec.k == 6144
        assert qkv_dec.profiler_name == "infini_kernel_glm52_fused_qkv_a_decode_nk"
        assert lookup("fused_qkv_a_proj", "decode", m=32) is not None
        assert lookup("fused_qkv_a_proj", "decode", m=8) is None

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
            ("SGLANG_GLM52_OPT_M_BUCKETS", old_buckets),
        ):
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def test_fixed_nk_uses_packed_abi_without_fallback():
    x = torch.empty((16, 2048), dtype=torch.float8_e4m3fn)
    weight = torch.empty((4096, 2048), dtype=torch.float8_e4m3fn)
    x_scale = torch.empty((16, 4), dtype=torch.int32)
    weight_scale = torch.empty((4096, 4), dtype=torch.int32)
    out = torch.empty((16, 4096), dtype=torch.bfloat16)

    with patch(
        "sglang.srt.layers.glm52_opt.fp8_gemm.deep_gemm.fp8_gemm_nt"
    ) as gemm:
        ok, path = run_fp8_gemm(
            "index_q_upproj",
            x,
            weight,
            x_scale,
            weight_scale,
            out,
            [128, 128],
            archive_ref="must_not_load",
            phase="decode",
            implementation="fixed_nk",
        )
    assert ok and path == "fixed_nk"
    gemm.assert_called_once_with(
        (x, x_scale),
        (weight, weight_scale),
        out,
        compiled_dims="nk",
    )

    with patch(
        "sglang.srt.layers.glm52_opt.fp8_gemm.deep_gemm.fp8_gemm_nt"
    ) as gemm:
        ok, path = run_fp8_gemm(
            "index_q_upproj",
            x,
            weight,
            x_scale.float(),
            weight_scale,
            out,
            [128, 128],
            archive_ref="must_not_load",
            phase="decode",
            implementation="fixed_nk",
        )
    assert not ok and path == "fixed_nk_requires_packed_ue8m0"
    gemm.assert_not_called()


def test_fixed_nk_requires_exact_forward_mode():
    old_profile = os.environ.get("SGLANG_GLM52_OPT_PROFILE")
    old_ops = os.environ.get("SGLANG_GLM52_OPT_OPS")
    try:
        os.environ["SGLANG_GLM52_OPT_PROFILE"] = "e2e_candidates"

        os.environ["SGLANG_GLM52_OPT_OPS"] = "index_q_upproj"
        decode = lookup("index_q_upproj", "decode", m=16)
        set_forward_mode(ForwardMode.DECODE, 16)
        assert _fixed_nk_forward_mode_matches(decode)
        for mode in (
            ForwardMode.EXTEND,
            ForwardMode.MIXED,
            ForwardMode.TARGET_VERIFY,
        ):
            set_forward_mode(mode, 16)
            assert not _fixed_nk_forward_mode_matches(decode)

        os.environ["SGLANG_GLM52_OPT_OPS"] = "fused_qkv_a_proj"
        prefill = lookup("fused_qkv_a_proj", "prefill", m=4096)
        set_forward_mode(ForwardMode.EXTEND, 4096)
        assert _fixed_nk_forward_mode_matches(prefill)
        set_forward_mode(ForwardMode.SPLIT_PREFILL, 4096)
        assert not _fixed_nk_forward_mode_matches(prefill)
    finally:
        set_forward_mode(None)
        for key, old_value in (
            ("SGLANG_GLM52_OPT_PROFILE", old_profile),
            ("SGLANG_GLM52_OPT_OPS", old_ops),
        ):
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def test_infini_kernel_profiler_range_is_exact_and_default_off():
    old_profile = os.environ.get("SGLANG_GLM52_OPT_PROFILE")
    old_ops = os.environ.get("SGLANG_GLM52_OPT_OPS")
    try:
        os.environ["SGLANG_GLM52_OPT_PROFILE"] = "e2e_candidates"
        os.environ["SGLANG_GLM52_OPT_OPS"] = "index_q_upproj"
        spec = lookup("index_q_upproj", "decode", m=16)
        assert (
            _profiler_range_name(spec, 16)
            == "infini_kernel_glm52_index_q_upproj_decode_nk"
            "[M=16,N=4096,K=2048]"
        )

        with (
            patch(
                "sglang.srt.layers.glm52_opt.dispatch."
                "config.emit_infini_kernel_nvtx",
                return_value=False,
            ),
            patch.object(torch.cuda.nvtx, "range_push") as push,
            patch.object(torch.cuda.nvtx, "range_pop") as pop,
            _nvtx_range(_profiler_range_name(spec, 16)),
        ):
            pass
        push.assert_not_called()
        pop.assert_not_called()

        with (
            patch(
                "sglang.srt.layers.glm52_opt.dispatch."
                "config.emit_infini_kernel_nvtx",
                return_value=True,
            ),
            patch.object(torch.cuda.nvtx, "range_push") as push,
            patch.object(torch.cuda.nvtx, "range_pop") as pop,
            _nvtx_range(_profiler_range_name(spec, 16)),
        ):
            pass
        push.assert_called_once_with(_profiler_range_name(spec, 16))
        pop.assert_called_once_with()
    finally:
        for key, old_value in (
            ("SGLANG_GLM52_OPT_PROFILE", old_profile),
            ("SGLANG_GLM52_OPT_OPS", old_ops),
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
