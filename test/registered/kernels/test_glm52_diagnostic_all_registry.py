"""CPU contracts for the exhaustive GLM-5.2 diagnostic registration profile."""

from __future__ import annotations

import os
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from sglang.srt.layers.glm52_opt import config, hotspot_provider
from sglang.srt.layers.glm52_opt.context import prefix_to_op_name
from sglang.srt.layers.glm52_opt.registry import list_enabled, lookup


@contextmanager
def _diagnostic_env(ops: str | None):
    names = (
        "SGLANG_GLM52_OPT",
        "SGLANG_GLM52_OPT_PROFILE",
        "SGLANG_GLM52_OPT_OPS",
    )
    saved = {name: os.environ.get(name) for name in names}
    try:
        os.environ["SGLANG_GLM52_OPT"] = "1"
        os.environ["SGLANG_GLM52_OPT_PROFILE"] = "diagnostic_all"
        if ops is None:
            os.environ.pop("SGLANG_GLM52_OPT_OPS", None)
        else:
            os.environ["SGLANG_GLM52_OPT_OPS"] = ops
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_bare_diagnostic_profile_is_fail_closed():
    with _diagnostic_env(None):
        assert config.diagnostic_candidate_ops() == frozenset()
        assert list_enabled("decode") == []
        assert list_enabled("prefill") == []


def test_all_explicitly_registers_every_interceptable_operator():
    with _diagnostic_env("all"):
        selected = config.diagnostic_candidate_ops()
        assert len(selected) == 15
        assert {spec.op for spec in list_enabled("decode")} == selected
        # Sparse attention is decode-only; all other logical registrations
        # have an exact M4096 prefill boundary.
        assert {spec.op for spec in list_enabled("prefill")} == selected - {
            "dsa_decode_attn"
        }


def test_fixed_shape_matrix_is_exact():
    cases = (
        ("fused_qkv_a_proj", 2624, 6144),
        ("q_b_proj", 16384, 2048),
        ("o_proj", 6144, 16384),
        ("dense_gate_up_proj", 4096, 6144),
        ("dense_down_proj", 6144, 2048),
        ("index_q_upproj", 4096, 2048),
        ("index_k_proj", 128, 6144),
    )
    with _diagnostic_env(",".join(op for op, _, _ in cases)):
        for op, n, k in cases:
            decode = lookup(op, "decode", m=16)
            assert decode is not None
            assert decode.implementation == "fixed_nk"
            assert (decode.n, decode.k) == (n, k)
            assert lookup(op, "decode", m=32) is not None
            assert lookup(op, "decode", m=8) is None

            prefill = lookup(op, "prefill", m=4096)
            assert prefill is not None
            assert prefill.implementation == "fixed_nk"
            assert (prefill.n, prefill.k) == (n, k)
            assert lookup(op, "prefill", m=2048) is None


def test_non_win_boundaries_remain_explicit_and_attributable():
    with _diagnostic_env(
        "indexer_weights,indexer_wk_weights,swiglu_quant,"
        "router_gemm,router_topk,moe_w13,moe_w2"
    ):
        assert lookup(
            "index_weights_proj", "decode", m=16
        ).implementation == "graph_replay"
        assert lookup(
            "index_wk_weights_proj", "prefill", m=4096
        ).implementation == "diagnostic_plugin"
        assert lookup(
            "moe_swiglu_quant", "decode", m=32
        ).implementation == "diagnostic_plugin"
        assert lookup(
            "router_logit_gemm", "prefill", m=4096
        ).implementation == "diagnostic_plugin"
        assert lookup(
            "router_sigmoid_topk", "decode", m=16
        ).implementation == "diagnostic_plugin"
        assert lookup(
            "moe_gate_proj", "prefill", m=4096
        ).implementation == "contig_psum"
        assert lookup(
            "moe_down_proj", "prefill", m=4096
        ).implementation == "contig_psum"

        assert config.contig_psum_kwargs("moe_gate_proj")["use_psum_layout"]
        assert config.contig_psum_kwargs("moe_down_proj")["compiled_dims"] == "nk"


def test_shared_expert_prefixes_do_not_alias_grouped_moe():
    assert (
        prefix_to_op_name(
            "model.layers.3.mlp.shared_experts.gate_up_proj"
        )
        == "dense_gate_up_proj"
    )
    assert (
        prefix_to_op_name("model.layers.3.mlp.shared_experts.down_proj")
        == "dense_down_proj"
    )
    assert (
        prefix_to_op_name("model.layers.3.mlp.experts.gate_up_proj")
        == "moe_gate_proj"
    )
    assert (
        prefix_to_op_name("model.layers.3.mlp.experts.down_proj")
        == "moe_down_proj"
    )


def test_alias_validation_and_all_exclusivity():
    with _diagnostic_env("attention_o,router_topk"):
        assert config.diagnostic_candidate_ops() == {
            "o_proj",
            "router_sigmoid_topk",
        }
    with (
        _diagnostic_env("all,o_proj"),
        TestCase().assertRaisesRegex(ValueError, "cannot be combined"),
    ):
        config.diagnostic_candidate_ops()
    with (
        _diagnostic_env("not_a_glm52_op"),
        TestCase().assertRaisesRegex(ValueError, "not_a_glm52_op"),
    ):
        config.diagnostic_candidate_ops()


def test_provider_is_required_only_for_selected_plugin_ops():
    hotspot_provider._reset_hotspot_provider_for_tests()
    with (
        patch.object(config, "is_enabled", return_value=True),
        patch.object(config, "profile_name", return_value="diagnostic_all"),
        patch.object(
            config,
            "diagnostic_candidate_ops",
            return_value=frozenset({"o_proj"}),
        ),
    ):
        assert not hotspot_provider.initialize_hotspot_provider(gpu_id=0)
        assert hotspot_provider.provider_state()["reason"] == "no_provider_ops"

    hotspot_provider._reset_hotspot_provider_for_tests()
    with (
        patch.object(config, "is_enabled", return_value=True),
        patch.object(config, "profile_name", return_value="diagnostic_all"),
        patch.object(
            config,
            "diagnostic_candidate_ops",
            return_value=frozenset({"moe_gate_proj"}),
        ),
        patch.object(
            config,
            "opt_m_buckets",
            return_value={"moe_gate_proj": frozenset({4096})},
        ),
    ):
        # Prefill W13 PSUM is built in. A prefill-only M allowlist must not
        # require the unrelated decode W13 provider callback at worker startup.
        assert not hotspot_provider.initialize_hotspot_provider(gpu_id=0)
        assert hotspot_provider.provider_state()["reason"] == "no_provider_ops"

    provider = SimpleNamespace(
        __name__="test_diagnostic_provider",
        INFINI_KERNEL_API_VERSION=1,
        router_logit_gemm=Mock(),
        router_sigmoid_topk=Mock(),
        PROVIDER_INFO={"scope": "diagnostic"},
    )
    hotspot_provider._reset_hotspot_provider_for_tests()
    with (
        patch.object(config, "is_enabled", return_value=True),
        patch.object(config, "profile_name", return_value="diagnostic_all"),
        patch.object(
            config,
            "diagnostic_candidate_ops",
            return_value=frozenset(
                {"o_proj", "router_logit_gemm", "router_sigmoid_topk"}
            ),
        ),
        patch.object(config, "hotspot_module_ref", return_value="provider.module"),
        patch.object(hotspot_provider, "_load_module", return_value=provider),
    ):
        assert hotspot_provider.initialize_hotspot_provider(gpu_id=1)
        assert hotspot_provider.provider_state()["selected_ops"] == [
            "router_logit_gemm",
            "router_sigmoid_topk",
        ]
