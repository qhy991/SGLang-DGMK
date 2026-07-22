"""Unit tests for GLM-5.2 opt registry (no GPU required)."""

from __future__ import annotations

import os

os.environ.setdefault("SGLANG_GLM52_OPT", "1")
os.environ.setdefault("SGLANG_GLM52_OPT_PROFILE", "full")

from sglang.srt.layers.glm52_opt.context import (
    get_forward_m,
    prefix_to_op_name,
    set_forward_mode,
)
from sglang.srt.layers.glm52_opt.config import explicit_shape_trial_buckets
from sglang.srt.layers.glm52_opt.dsa_prefill import (
    select_trtllm_prefill_enable_pdl,
)
from sglang.srt.layers.glm52_opt.phase import infer_glm52_phase
from sglang.srt.layers.glm52_opt.registry import lookup
from sglang.srt.layers.deep_gemm_wrapper.entrypoint import (
    _glm52_moe_dispatch_compatible,
)


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


def test_explicit_shape_trial_buckets_fail_closed():
    old = {
        key: os.environ.get(key)
        for key in (
            "SGLANG_GLM52_OPT",
            "SGLANG_GLM52_OPT_PROFILE",
            "SGLANG_GLM52_OPT_OPS",
            "SGLANG_GLM52_OPT_M_BUCKETS",
        )
    }
    try:
        os.environ["SGLANG_GLM52_OPT"] = "0"
        os.environ["SGLANG_GLM52_OPT_PROFILE"] = "serving_safe"
        os.environ["SGLANG_GLM52_OPT_OPS"] = "dsa_prefill_attn"
        os.environ["SGLANG_GLM52_OPT_M_BUCKETS"] = "dsa_prefill_attn:4096"
        assert not explicit_shape_trial_buckets("dsa_prefill_attn")

        os.environ["SGLANG_GLM52_OPT"] = "1"
        os.environ["SGLANG_GLM52_OPT_PROFILE"] = "full"
        assert not explicit_shape_trial_buckets("dsa_prefill_attn")

        os.environ["SGLANG_GLM52_OPT_PROFILE"] = "serving_safe"
        os.environ["SGLANG_GLM52_OPT_OPS"] = "q_b_proj"
        assert not explicit_shape_trial_buckets("dsa_prefill_attn")

        os.environ["SGLANG_GLM52_OPT_OPS"] = "dsa_prefill_attn"
        os.environ.pop("SGLANG_GLM52_OPT_M_BUCKETS")
        assert not explicit_shape_trial_buckets("dsa_prefill_attn")

        os.environ["SGLANG_GLM52_OPT_M_BUCKETS"] = "dsa_prefill_attn:4096"
        assert explicit_shape_trial_buckets("dsa_prefill_attn") == frozenset({4096})
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_dsa_prefill_pdl_trial_exact_abi_only():
    exact = dict(
        allowed_m=frozenset({4096}),
        is_plain_extend=True,
        is_prefill=True,
        q_shape=(4096, 1, 64, 576),
        kv_shape=(512, 1, 64, 576),
        block_tables_shape=(4096, 1, 2048),
        seq_lens_shape=(4096,),
        max_seq_len=32768,
        q_is_fp8_e4m3=True,
        kv_is_fp8_e4m3=True,
    )
    assert select_trtllm_prefill_enable_pdl(**exact) is False
    for key, value in (
        ("allowed_m", frozenset()),
        ("is_plain_extend", False),
        ("is_prefill", False),
        ("q_shape", (2048, 1, 64, 576)),
        ("kv_shape", (512, 1, 64, 512)),
        ("block_tables_shape", (4096, 1, 1024)),
        ("seq_lens_shape", (1,)),
        ("max_seq_len", 16384),
        ("q_is_fp8_e4m3", False),
        ("kv_is_fp8_e4m3", False),
    ):
        mismatch = dict(exact)
        mismatch[key] = value
        assert select_trtllm_prefill_enable_pdl(**mismatch) is None


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


def test_phase_defaults():
    assert infer_glm52_phase(None, 32) == "decode"
    assert infer_glm52_phase(None, 1024) == "prefill"
