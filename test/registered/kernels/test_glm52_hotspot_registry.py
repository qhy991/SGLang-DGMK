"""CPU contract tests for GLM-5.2 hotspot provider registration."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import torch
from sglang.srt.layers.glm52_opt import config, hotspot_provider
from sglang.srt.layers.glm52_opt.context import op_context, set_forward_mode
from sglang.srt.layers.glm52_opt.dispatch import (
    try_dispatch_flashmla_sparse_decode,
    try_dispatch_moe_masked,
)
from sglang.srt.layers.glm52_opt.registry import list_enabled, lookup
from sglang.srt.model_executor.forward_batch_info import ForwardMode


def _restore_env(saved: dict[str, str | None]) -> None:
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def test_hotspot_profile_registers_exact_three_decode_ops():
    names = ("SGLANG_GLM52_OPT_PROFILE", "SGLANG_GLM52_OPT_OPS")
    saved = {name: os.environ.get(name) for name in names}
    try:
        os.environ["SGLANG_GLM52_OPT_PROFILE"] = "hotspot_candidates"
        os.environ.pop("SGLANG_GLM52_OPT_OPS", None)
        specs = {spec.op: spec for spec in list_enabled("decode")}
        assert set(specs) == {
            "dsa_decode_attn",
            "moe_gate_proj",
            "moe_down_proj",
        }

        flashmla = lookup("dsa_decode_attn", "decode", m=16)
        assert flashmla is not None
        assert flashmla.implementation == "hotspot_plugin"
        assert (flashmla.topk, flashmla.q_heads, flashmla.qk_dim) == (
            2048,
            64,
            576,
        )
        assert lookup("dsa_decode_attn", "decode", m=64) is None

        w13 = lookup("moe_gate_proj", "decode", m=32)
        assert w13 is not None
        assert (w13.num_groups, w13.slab_m, w13.n, w13.k) == (
            32,
            1024,
            4096,
            6144,
        )
        w2 = lookup("moe_down_proj", "decode", m=16)
        assert w2 is not None
        assert (w2.n, w2.k) == (6144, 2048)

        # User-facing fused names normalize to existing SGLang op contexts.
        os.environ["SGLANG_GLM52_OPT_OPS"] = "flashmla_sparse_decode,moe_w13,moe_w2"
        assert config.hotspot_candidate_ops() == {
            "dsa_decode_attn",
            "moe_gate_proj",
            "moe_down_proj",
        }
        os.environ["SGLANG_GLM52_OPT_OPS"] = "moe_w13"
        assert [spec.op for spec in list_enabled("decode")] == ["moe_gate_proj"]
        os.environ["SGLANG_GLM52_OPT_OPS"] = "moe_w13,moe_typo"
        with TestCase().assertRaisesRegex(ValueError, "moe_typo"):
            config.hotspot_candidate_ops()
    finally:
        _restore_env(saved)


def test_hotspot_profile_isolated_from_legacy_e2e_candidates():
    names = ("SGLANG_GLM52_OPT_PROFILE", "SGLANG_GLM52_OPT_OPS")
    saved = {name: os.environ.get(name) for name in names}
    try:
        os.environ["SGLANG_GLM52_OPT_PROFILE"] = "hotspot_candidates"
        os.environ.pop("SGLANG_GLM52_OPT_OPS", None)
        assert lookup("o_proj", "decode", m=16) is None
        assert lookup("index_q_upproj", "decode", m=16) is None
        assert lookup("dsa_decode_attn", "prefill", m=16) is None
    finally:
        _restore_env(saved)


def test_provider_requires_version_and_selected_callbacks():
    initializer = Mock()
    provider = SimpleNamespace(
        __name__="test_hotspot_provider",
        INFINI_KERNEL_API_VERSION=1,
        initialize=initializer,
        flashmla_sparse_decode=Mock(),
        moe_w13=Mock(),
        moe_w2=Mock(),
        PROVIDER_INFO={"build": "unit-test"},
    )
    hotspot_provider._reset_hotspot_provider_for_tests()
    with (
        patch.object(config, "is_enabled", return_value=True),
        patch.object(config, "profile_name", return_value="hotspot_candidates"),
        patch.object(
            config,
            "hotspot_candidate_ops",
            return_value=frozenset({"dsa_decode_attn", "moe_gate_proj"}),
        ),
        patch.object(config, "hotspot_module_ref", return_value="provider.module"),
        patch.object(hotspot_provider, "_load_module", return_value=provider),
    ):
        assert hotspot_provider.initialize_hotspot_provider(gpu_id=2)
        initializer.assert_called_once_with(gpu_id=2)
        state = hotspot_provider.provider_state()
        assert state["ready"] is True
        assert state["selected_ops"] == ["dsa_decode_attn", "moe_gate_proj"]
        assert state["provider_info"] == {"build": "unit-test"}

    hotspot_provider._reset_hotspot_provider_for_tests()
    del provider.moe_w13
    with (
        patch.object(config, "is_enabled", return_value=True),
        patch.object(config, "profile_name", return_value="hotspot_candidates"),
        patch.object(
            config,
            "hotspot_candidate_ops",
            return_value=frozenset({"moe_gate_proj"}),
        ),
        patch.object(config, "hotspot_module_ref", return_value="provider.module"),
        patch.object(hotspot_provider, "_load_module", return_value=provider),
        TestCase().assertRaisesRegex(RuntimeError, "initialization failed"),
    ):
        hotspot_provider.initialize_hotspot_provider(gpu_id=2)


def test_selected_moe_candidate_launches_once_and_errors_propagate():
    names = (
        "SGLANG_GLM52_OPT",
        "SGLANG_GLM52_OPT_PROFILE",
        "SGLANG_GLM52_OPT_OPS",
    )
    saved = {name: os.environ.get(name) for name in names}
    fake = SimpleNamespace(shape=(32, 1024, 6144), ndim=3)
    lhs = (fake, fake)
    rhs = (fake, fake)
    try:
        os.environ["SGLANG_GLM52_OPT"] = "1"
        os.environ["SGLANG_GLM52_OPT_PROFILE"] = "hotspot_candidates"
        os.environ["SGLANG_GLM52_OPT_OPS"] = "moe_w13"
        set_forward_mode(ForwardMode.DECODE, 16)
        with (
            op_context("moe_gate_proj"),
            patch(
                "sglang.srt.layers.glm52_opt.dispatch._moe_hotspot_abi_matches",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.glm52_opt.dispatch.run_hotspot_moe_masked",
                return_value=None,
            ) as candidate,
            patch("sglang.srt.layers.glm52_opt.dispatch._record_hit"),
        ):
            assert try_dispatch_moe_masked(lhs, rhs, fake, fake, 4)
        candidate.assert_called_once()

        with (
            op_context("moe_gate_proj"),
            patch(
                "sglang.srt.layers.glm52_opt.dispatch._moe_hotspot_abi_matches",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.glm52_opt.dispatch.run_hotspot_moe_masked",
                side_effect=RuntimeError("candidate launch failed"),
            ),
            TestCase().assertRaisesRegex(RuntimeError, "candidate launch failed"),
        ):
            try_dispatch_moe_masked(lhs, rhs, fake, fake, 4)
    finally:
        set_forward_mode(None)
        _restore_env(saved)


def test_abi_miss_falls_back_before_any_candidate_launch():
    names = (
        "SGLANG_GLM52_OPT",
        "SGLANG_GLM52_OPT_PROFILE",
        "SGLANG_GLM52_OPT_OPS",
    )
    saved = {name: os.environ.get(name) for name in names}
    fake = SimpleNamespace(shape=(32, 1024, 6144), ndim=3)
    pair = (fake, fake)
    try:
        os.environ["SGLANG_GLM52_OPT"] = "1"
        os.environ["SGLANG_GLM52_OPT_PROFILE"] = "hotspot_candidates"
        os.environ["SGLANG_GLM52_OPT_OPS"] = "moe_w13"
        set_forward_mode(ForwardMode.DECODE, 16)
        with (
            op_context("moe_gate_proj"),
            patch(
                "sglang.srt.layers.glm52_opt.dispatch._moe_hotspot_abi_matches",
                return_value=False,
            ),
            patch(
                "sglang.srt.layers.glm52_opt.dispatch.run_hotspot_moe_masked",
            ) as candidate,
            patch("sglang.srt.layers.glm52_opt.dispatch._record_miss"),
        ):
            assert not try_dispatch_moe_masked(pair, pair, fake, fake, 5)
        candidate.assert_not_called()
    finally:
        set_forward_mode(None)
        _restore_env(saved)


def test_selected_flashmla_candidate_preserves_public_return_contract():
    names = (
        "SGLANG_GLM52_OPT",
        "SGLANG_GLM52_OPT_PROFILE",
        "SGLANG_GLM52_OPT_OPS",
    )
    saved = {name: os.environ.get(name) for name in names}
    q = torch.empty((16, 1, 64, 576), dtype=torch.bfloat16)
    candidate_out = torch.empty((16, 1, 64, 512), dtype=torch.bfloat16)
    fake = torch.empty(1)
    try:
        os.environ["SGLANG_GLM52_OPT"] = "1"
        os.environ["SGLANG_GLM52_OPT_PROFILE"] = "hotspot_candidates"
        os.environ["SGLANG_GLM52_OPT_OPS"] = "flashmla_sparse_decode"
        set_forward_mode(ForwardMode.DECODE, 16)
        with (
            patch(
                "sglang.srt.layers.glm52_opt.dispatch._flashmla_hotspot_abi_matches",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.glm52_opt.dispatch._tensor_contract",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.glm52_opt.dispatch.run_flashmla_sparse_decode",
                return_value=(candidate_out, fake),
            ) as candidate,
            patch("sglang.srt.layers.glm52_opt.dispatch._record_hit"),
        ):
            result = try_dispatch_flashmla_sparse_decode(
                q=q,
                k_cache=fake,
                cache_seqlens=fake,
                head_dim_v=512,
                tile_scheduler_metadata=fake,
                num_splits=fake,
                softmax_scale=0.0625,
                indices=fake,
                block_table=fake,
                is_fp8_kvcache=True,
            )
        assert result is candidate_out
        candidate.assert_called_once()
    finally:
        set_forward_mode(None)
        _restore_env(saved)
