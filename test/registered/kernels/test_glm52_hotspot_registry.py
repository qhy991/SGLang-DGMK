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
    _moe_hotspot_abi_matches,
    try_dispatch_flashmla_sparse_decode,
    try_dispatch_fp8_gemm,
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


def test_only_w2_hotspot_spec_is_graph_only():
    names = ("SGLANG_GLM52_OPT_PROFILE", "SGLANG_GLM52_OPT_OPS")
    saved = {name: os.environ.get(name) for name in names}
    try:
        os.environ["SGLANG_GLM52_OPT_PROFILE"] = "hotspot_candidates"
        os.environ.pop("SGLANG_GLM52_OPT_OPS", None)
        specs = {spec.op: spec for spec in list_enabled("decode")}
        assert specs["moe_down_proj"].graph_only is True
        assert specs["moe_gate_proj"].graph_only is False
        assert specs["dsa_decode_attn"].graph_only is False
    finally:
        _restore_env(saved)


def test_w2_graph_only_env_parsing():
    saved = {"SGLANG_GLM52_W2_GRAPH_ONLY": os.environ.get("SGLANG_GLM52_W2_GRAPH_ONLY")}
    try:
        os.environ.pop("SGLANG_GLM52_W2_GRAPH_ONLY", None)
        assert config.graph_only_enabled("moe_down_proj") is True
        for value in ("0", "false", "no", "OFF", " off "):
            os.environ["SGLANG_GLM52_W2_GRAPH_ONLY"] = value
            assert config.graph_only_enabled("moe_down_proj") is False
        for value in ("1", "true", "on"):
            os.environ["SGLANG_GLM52_W2_GRAPH_ONLY"] = value
            assert config.graph_only_enabled("moe_down_proj") is True
        # An op with no registered override is never eager-forced.
        os.environ["SGLANG_GLM52_W2_GRAPH_ONLY"] = "0"
        assert config.graph_only_enabled("moe_gate_proj") is True
    finally:
        _restore_env(saved)


def _w2_graph_only_dispatch(*, capturing: bool, expected_m: int = 4):
    """Drive one W2 hotspot dispatch with a mocked capture state."""
    fake = SimpleNamespace(shape=(32, 1024, 6144), ndim=3)
    pair = (fake, fake)
    with (
        op_context("moe_down_proj"),
        patch(
            "sglang.srt.layers.glm52_opt.dispatch._is_cuda_graph_capturing",
            return_value=capturing,
        ),
        patch(
            "sglang.srt.layers.glm52_opt.dispatch._moe_hotspot_abi_matches",
            return_value=True,
        ) as abi,
        patch(
            "sglang.srt.layers.glm52_opt.dispatch.run_hotspot_moe_masked",
            return_value=None,
        ) as candidate,
        patch("sglang.srt.layers.glm52_opt.dispatch._record_hit") as hit,
        patch("sglang.srt.layers.glm52_opt.dispatch._record_miss") as miss,
    ):
        selected = try_dispatch_moe_masked(pair, pair, fake, fake, expected_m)
    return selected, abi, candidate, hit, miss


def test_w2_graph_only_declines_eager_before_any_provider_launch():
    names = (
        "SGLANG_GLM52_OPT",
        "SGLANG_GLM52_OPT_PROFILE",
        "SGLANG_GLM52_OPT_OPS",
        "SGLANG_GLM52_W2_GRAPH_ONLY",
    )
    saved = {name: os.environ.get(name) for name in names}
    try:
        os.environ["SGLANG_GLM52_OPT"] = "1"
        os.environ["SGLANG_GLM52_OPT_PROFILE"] = "hotspot_candidates"
        os.environ["SGLANG_GLM52_OPT_OPS"] = "moe_w2"
        os.environ.pop("SGLANG_GLM52_W2_GRAPH_ONLY", None)
        set_forward_mode(ForwardMode.DECODE, 16)

        selected, abi, candidate, hit, miss = _w2_graph_only_dispatch(capturing=False)
        assert not selected
        candidate.assert_not_called()
        hit.assert_not_called()
        # The eager decline must stay off the ABI and hit/miss lock paths.
        abi.assert_not_called()
        miss.assert_not_called()

        selected, _abi, candidate, hit, _miss = _w2_graph_only_dispatch(capturing=True)
        assert selected
        candidate.assert_called_once()
        hit.assert_called_once()
    finally:
        set_forward_mode(None)
        _restore_env(saved)


def test_w2_graph_only_can_be_disabled_for_diagnostic_eager_leaf():
    names = (
        "SGLANG_GLM52_OPT",
        "SGLANG_GLM52_OPT_PROFILE",
        "SGLANG_GLM52_OPT_OPS",
        "SGLANG_GLM52_W2_GRAPH_ONLY",
    )
    saved = {name: os.environ.get(name) for name in names}
    try:
        os.environ["SGLANG_GLM52_OPT"] = "1"
        os.environ["SGLANG_GLM52_OPT_PROFILE"] = "hotspot_candidates"
        os.environ["SGLANG_GLM52_OPT_OPS"] = "moe_w2"
        os.environ["SGLANG_GLM52_W2_GRAPH_ONLY"] = "0"
        set_forward_mode(ForwardMode.DECODE, 16)

        selected, _abi, candidate, _hit, _miss = _w2_graph_only_dispatch(capturing=False)
        assert selected
        candidate.assert_called_once()
    finally:
        set_forward_mode(None)
        _restore_env(saved)


def test_w13_hotspot_is_not_restricted_by_w2_graph_only():
    names = (
        "SGLANG_GLM52_OPT",
        "SGLANG_GLM52_OPT_PROFILE",
        "SGLANG_GLM52_OPT_OPS",
        "SGLANG_GLM52_W2_GRAPH_ONLY",
    )
    saved = {name: os.environ.get(name) for name in names}
    fake = SimpleNamespace(shape=(32, 1024, 4096), ndim=3)
    pair = (fake, fake)
    try:
        os.environ["SGLANG_GLM52_OPT"] = "1"
        os.environ["SGLANG_GLM52_OPT_PROFILE"] = "hotspot_candidates"
        os.environ["SGLANG_GLM52_OPT_OPS"] = "moe_w13"
        os.environ.pop("SGLANG_GLM52_W2_GRAPH_ONLY", None)
        set_forward_mode(ForwardMode.DECODE, 16)
        with (
            op_context("moe_gate_proj"),
            patch(
                "sglang.srt.layers.glm52_opt.dispatch._is_cuda_graph_capturing",
                return_value=False,
            ),
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
            assert try_dispatch_moe_masked(pair, pair, fake, fake, 4)
        candidate.assert_called_once()
    finally:
        set_forward_mode(None)
        _restore_env(saved)


def test_w2_expected_m_matrix_is_bound_to_forward_bucket():
    names = ("SGLANG_GLM52_OPT_PROFILE", "SGLANG_GLM52_OPT_OPS")
    saved = {name: os.environ.get(name) for name in names}
    fake = SimpleNamespace(device=torch.device("cpu"))
    pair = (fake, fake)
    try:
        os.environ["SGLANG_GLM52_OPT_PROFILE"] = "hotspot_candidates"
        os.environ["SGLANG_GLM52_OPT_OPS"] = "moe_w2"
        spec = lookup("moe_down_proj", "decode", m=16)
        assert spec is not None
        with patch(
            "sglang.srt.layers.glm52_opt.dispatch._tensor_contract",
            return_value=True,
        ):
            set_forward_mode(ForwardMode.DECODE, 16)
            assert all(
                _moe_hotspot_abi_matches(
                    spec, pair, pair, fake, fake, expected_m, 16
                )
                for expected_m in (4, 5)
            )
            assert not any(
                _moe_hotspot_abi_matches(
                    spec, pair, pair, fake, fake, expected_m, 16
                )
                for expected_m in (8, 9)
            )
            set_forward_mode(ForwardMode.DECODE, 32)
            assert all(
                _moe_hotspot_abi_matches(
                    spec, pair, pair, fake, fake, expected_m, 32
                )
                for expected_m in (8, 9)
            )
            assert not any(
                _moe_hotspot_abi_matches(
                    spec, pair, pair, fake, fake, expected_m, 32
                )
                for expected_m in (4, 5)
            )
            set_forward_mode(ForwardMode.EXTEND, 16)
            assert not _moe_hotspot_abi_matches(
                spec, pair, pair, fake, fake, 4, 16
            )
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


def test_o_proj_e2e_spec_is_graph_only():
    """Decode o_proj (fixed-N/K) is graph_only; the other E2E fp8_gemm is not."""
    names = ("SGLANG_GLM52_OPT", "SGLANG_GLM52_OPT_PROFILE", "SGLANG_GLM52_OPT_OPS")
    saved = {name: os.environ.get(name) for name in names}
    try:
        os.environ["SGLANG_GLM52_OPT"] = "1"
        os.environ["SGLANG_GLM52_OPT_PROFILE"] = "e2e_candidates"
        os.environ.pop("SGLANG_GLM52_OPT_OPS", None)
        o_proj = lookup("o_proj", "decode", m=16)
        assert o_proj is not None
        assert o_proj.kind == "fp8_gemm"
        assert o_proj.implementation == "fixed_nk"
        assert o_proj.graph_only is True
        # index_q_upproj is an explicit-only fixed-N/K candidate, not graph_only.
        os.environ["SGLANG_GLM52_OPT_OPS"] = "index_q_upproj"
        idx = lookup("index_q_upproj", "decode", m=16)
        assert idx is not None
        assert idx.implementation == "fixed_nk"
        assert idx.graph_only is False
    finally:
        _restore_env(saved)


def test_o_proj_graph_only_env_parsing():
    """SGLANG_GLM52_O_PROJ_GRAPH_ONLY toggles only o_proj; W2 env is independent."""
    names = ("SGLANG_GLM52_O_PROJ_GRAPH_ONLY", "SGLANG_GLM52_W2_GRAPH_ONLY")
    saved = {name: os.environ.get(name) for name in names}
    try:
        os.environ.pop("SGLANG_GLM52_O_PROJ_GRAPH_ONLY", None)
        assert config.graph_only_enabled("o_proj") is True
        for value in ("0", "false", "no", "OFF", " off "):
            os.environ["SGLANG_GLM52_O_PROJ_GRAPH_ONLY"] = value
            assert config.graph_only_enabled("o_proj") is False
        for value in ("1", "true", "on"):
            os.environ["SGLANG_GLM52_O_PROJ_GRAPH_ONLY"] = value
            assert config.graph_only_enabled("o_proj") is True
        # The W2 override must not leak into o_proj's decision and vice versa.
        os.environ["SGLANG_GLM52_O_PROJ_GRAPH_ONLY"] = "0"
        os.environ.pop("SGLANG_GLM52_W2_GRAPH_ONLY", None)
        assert config.graph_only_enabled("moe_down_proj") is True
        assert config.graph_only_enabled("o_proj") is False
    finally:
        _restore_env(saved)


def _o_proj_graph_only_dispatch(*, capturing: bool):
    """Drive one decode o_proj fp8_gemm dispatch with a mocked capture state."""
    input_2d = torch.zeros((16, 8))
    weight = torch.zeros((6144, 8))
    scale = torch.zeros(1, dtype=torch.int32)
    with (
        op_context("o_proj"),
        patch(
            "sglang.srt.layers.glm52_opt.dispatch._is_cuda_graph_capturing",
            return_value=capturing,
        ),
        patch(
            "sglang.srt.layers.glm52_opt.dispatch._fixed_nk_abi_matches",
            return_value=True,
        ) as abi,
        patch(
            "sglang.srt.layers.glm52_opt.dispatch.run_fp8_gemm",
            return_value=(True, "fixed_nk"),
        ) as candidate,
        patch("sglang.srt.layers.glm52_opt.dispatch._record_hit") as hit,
        patch("sglang.srt.layers.glm52_opt.dispatch._record_miss") as miss,
    ):
        result = try_dispatch_fp8_gemm(
            input_2d, weight, scale, scale, [128, 128], torch.bfloat16
        )
    return result, abi, candidate, hit, miss


def test_o_proj_graph_only_declines_eager_before_abi_and_run():
    names = (
        "SGLANG_GLM52_OPT",
        "SGLANG_GLM52_OPT_PROFILE",
        "SGLANG_GLM52_OPT_OPS",
        "SGLANG_GLM52_O_PROJ_GRAPH_ONLY",
    )
    saved = {name: os.environ.get(name) for name in names}
    try:
        os.environ["SGLANG_GLM52_OPT"] = "1"
        os.environ["SGLANG_GLM52_OPT_PROFILE"] = "e2e_candidates"
        os.environ["SGLANG_GLM52_OPT_OPS"] = "o_proj"
        os.environ.pop("SGLANG_GLM52_O_PROJ_GRAPH_ONLY", None)
        set_forward_mode(ForwardMode.DECODE, 16)

        # Eager, production graph-only on: decline to stock before the ABI check
        # and before the hit/miss lock; no candidate launch.
        result, abi, candidate, hit, miss = _o_proj_graph_only_dispatch(capturing=False)
        assert result is None
        candidate.assert_not_called()
        hit.assert_not_called()
        abi.assert_not_called()
        miss.assert_not_called()

        # Under graph capture the candidate is selected and the GEMM runs once.
        result, _abi, candidate, hit, _miss = _o_proj_graph_only_dispatch(capturing=True)
        assert result is not None
        candidate.assert_called_once()
        hit.assert_called_once()
    finally:
        set_forward_mode(None)
        _restore_env(saved)


def test_o_proj_graph_only_can_be_disabled_for_diagnostic_eager_leaf():
    names = (
        "SGLANG_GLM52_OPT",
        "SGLANG_GLM52_OPT_PROFILE",
        "SGLANG_GLM52_OPT_OPS",
        "SGLANG_GLM52_O_PROJ_GRAPH_ONLY",
    )
    saved = {name: os.environ.get(name) for name in names}
    try:
        os.environ["SGLANG_GLM52_OPT"] = "1"
        os.environ["SGLANG_GLM52_OPT_PROFILE"] = "e2e_candidates"
        os.environ["SGLANG_GLM52_OPT_OPS"] = "o_proj"
        os.environ["SGLANG_GLM52_O_PROJ_GRAPH_ONLY"] = "0"
        set_forward_mode(ForwardMode.DECODE, 16)

        result, _abi, candidate, _hit, _miss = _o_proj_graph_only_dispatch(capturing=False)
        assert result is not None
        candidate.assert_called_once()
    finally:
        set_forward_mode(None)
        _restore_env(saved)
