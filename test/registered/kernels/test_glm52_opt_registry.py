"""Unit tests for GLM-5.2 opt registry (no GPU required)."""

from __future__ import annotations

import os

os.environ.setdefault("SGLANG_GLM52_OPT", "1")
os.environ.setdefault("SGLANG_GLM52_OPT_PROFILE", "full")

from sglang.srt.layers.glm52_opt.context import prefix_to_op_name
from sglang.srt.layers.glm52_opt.phase import infer_glm52_phase
from sglang.srt.layers.glm52_opt.registry import lookup


def test_prefix_mapping():
    assert prefix_to_op_name("model.layers.0.self_attn.q_b_proj") == "q_b_proj"
    assert prefix_to_op_name("model.layers.0.mlp.gate_proj") == "moe_gate_proj"


def test_prefill_full_profile():
    assert lookup("index_q_upproj", "prefill") is not None
    assert lookup("moe_gate_proj", "prefill") is None


def test_decode_registry():
    assert lookup("dsa_decode_attn", "decode") is not None


def test_phase_defaults():
    assert infer_glm52_phase(None, 32) == "decode"
    assert infer_glm52_phase(None, 1024) == "prefill"
