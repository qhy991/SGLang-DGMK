"""CPU contract tests for GLM-5.2 hotspot provider registration."""

from __future__ import annotations

import os
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import torch
from sglang.srt.layers.glm52_opt import config, hotspot_provider
from sglang.srt.layers.glm52_opt.context import op_context, set_forward_mode
from sglang.srt.layers.glm52_opt.dispatch import (
    _flashmla_hotspot_abi_matches,
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


def test_decode_graph_only_hotspot_specs_are_registered():
    names = ("SGLANG_GLM52_OPT_PROFILE", "SGLANG_GLM52_OPT_OPS")
    saved = {name: os.environ.get(name) for name in names}
    try:
        os.environ["SGLANG_GLM52_OPT_PROFILE"] = "hotspot_candidates"
        os.environ.pop("SGLANG_GLM52_OPT_OPS", None)
        specs = {spec.op: spec for spec in list_enabled("decode")}
        assert specs["moe_down_proj"].graph_only is True
        assert specs["moe_gate_proj"].graph_only is False
        assert specs["dsa_decode_attn"].graph_only is True
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

        selected, _abi, candidate, _hit, _miss = _w2_graph_only_dispatch(
            capturing=False
        )
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
                _moe_hotspot_abi_matches(spec, pair, pair, fake, fake, expected_m, 16)
                for expected_m in (4, 5)
            )
            assert not any(
                _moe_hotspot_abi_matches(spec, pair, pair, fake, fake, expected_m, 16)
                for expected_m in (8, 9)
            )
            set_forward_mode(ForwardMode.DECODE, 32)
            assert all(
                _moe_hotspot_abi_matches(spec, pair, pair, fake, fake, expected_m, 32)
                for expected_m in (8, 9)
            )
            assert not any(
                _moe_hotspot_abi_matches(spec, pair, pair, fake, fake, expected_m, 32)
                for expected_m in (4, 5)
            )
            set_forward_mode(ForwardMode.EXTEND, 16)
            assert not _moe_hotspot_abi_matches(spec, pair, pair, fake, fake, 4, 16)
    finally:
        set_forward_mode(None)
        _restore_env(saved)


def test_selected_flashmla_candidate_preserves_public_return_contract():
    names = (
        "SGLANG_GLM52_OPT",
        "SGLANG_GLM52_OPT_PROFILE",
        "SGLANG_GLM52_OPT_OPS",
        "SGLANG_GLM52_FLASHMLA_GRAPH_ONLY",
    )
    saved = {name: os.environ.get(name) for name in names}
    q = torch.empty((16, 1, 64, 576), dtype=torch.bfloat16)
    candidate_out = torch.empty((16, 1, 64, 512), dtype=torch.bfloat16)
    candidate_lse = torch.empty_strided((16, 64, 1), (64, 1, 64), dtype=torch.float32)
    fake = torch.empty(1)
    try:
        os.environ["SGLANG_GLM52_OPT"] = "1"
        os.environ["SGLANG_GLM52_OPT_PROFILE"] = "hotspot_candidates"
        os.environ["SGLANG_GLM52_OPT_OPS"] = "flashmla_sparse_decode"
        set_forward_mode(ForwardMode.DECODE, 16)
        with (
            patch(
                "sglang.srt.layers.glm52_opt.dispatch._is_cuda_graph_capturing",
                return_value=True,
            ),
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
                return_value=(candidate_out, candidate_lse),
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


def _flashmla_decode_call(q: torch.Tensor, fake: torch.Tensor):
    return try_dispatch_flashmla_sparse_decode(
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


@contextmanager
def _mock_flashmla_candidate(*, capturing: bool, provider_result, contract=None):
    if contract is None:
        contract = lambda *_args, **_kwargs: True
    with (
        patch(
            "sglang.srt.layers.glm52_opt.dispatch._is_cuda_graph_capturing",
            return_value=capturing,
        ),
        patch(
            "sglang.srt.layers.glm52_opt.dispatch._flashmla_hotspot_abi_matches",
            return_value=True,
        ) as abi,
        patch(
            "sglang.srt.layers.glm52_opt.dispatch._tensor_contract",
            side_effect=contract,
        ),
        patch(
            "sglang.srt.layers.glm52_opt.dispatch.run_flashmla_sparse_decode",
            return_value=provider_result,
        ) as provider,
        patch("sglang.srt.layers.glm52_opt.dispatch._record_hit") as record_hit,
        patch("sglang.srt.layers.glm52_opt.dispatch._record_miss") as record_miss,
    ):
        yield SimpleNamespace(
            abi=abi,
            provider=provider,
            record_hit=record_hit,
            record_miss=record_miss,
        )


def test_flashmla_graph_only_eager_decline_and_diagnostic_override():
    names = (
        "SGLANG_GLM52_OPT",
        "SGLANG_GLM52_OPT_PROFILE",
        "SGLANG_GLM52_OPT_OPS",
        "SGLANG_GLM52_FLASHMLA_GRAPH_ONLY",
    )
    saved = {name: os.environ.get(name) for name in names}
    q = torch.empty((16, 1, 64, 576), dtype=torch.bfloat16)
    candidate_out = torch.empty((16, 1, 64, 512), dtype=torch.bfloat16)
    candidate_lse = torch.empty_strided((16, 64, 1), (64, 1, 64), dtype=torch.float32)
    fake = torch.empty(1)
    try:
        os.environ["SGLANG_GLM52_OPT"] = "1"
        os.environ["SGLANG_GLM52_OPT_PROFILE"] = "hotspot_candidates"
        os.environ["SGLANG_GLM52_OPT_OPS"] = "flashmla_sparse_decode"
        os.environ.pop("SGLANG_GLM52_FLASHMLA_GRAPH_ONLY", None)
        set_forward_mode(ForwardMode.DECODE, 16)
        with _mock_flashmla_candidate(
            capturing=False, provider_result=(candidate_out, candidate_lse)
        ) as mocks:
            assert _flashmla_decode_call(q, fake) is None
        mocks.abi.assert_not_called()
        mocks.provider.assert_not_called()
        mocks.record_hit.assert_not_called()
        mocks.record_miss.assert_not_called()

        os.environ["SGLANG_GLM52_FLASHMLA_GRAPH_ONLY"] = "0"
        with _mock_flashmla_candidate(
            capturing=False, provider_result=(candidate_out, candidate_lse)
        ) as mocks:
            assert _flashmla_decode_call(q, fake) is candidate_out
        mocks.abi.assert_called_once()
        mocks.provider.assert_called_once()
        mocks.record_hit.assert_called_once()
    finally:
        set_forward_mode(None)
        _restore_env(saved)


def test_flashmla_dynamic_page_and_alignment_abi_gate():
    names = ("SGLANG_GLM52_OPT_PROFILE", "SGLANG_GLM52_OPT_OPS")
    saved = {name: os.environ.get(name) for name in names}
    device = torch.device("cuda:0")

    def tensor(
        shape,
        *,
        dtype=torch.int32,
        ptr=0x1000,
        contiguous=True,
        stride=None,
    ):
        if stride is None:
            running = 1
            inferred = []
            for extent in reversed(shape):
                inferred.append(running)
                running *= max(int(extent), 1)
            stride = tuple(reversed(inferred))
        return SimpleNamespace(
            ndim=len(shape),
            shape=shape,
            device=device,
            is_cuda=True,
            dtype=dtype,
            is_contiguous=lambda: contiguous,
            storage_offset=lambda: 0,
            data_ptr=lambda: ptr,
            stride=lambda: stride,
        )

    try:
        os.environ["SGLANG_GLM52_OPT_PROFILE"] = "hotspot_candidates"
        os.environ["SGLANG_GLM52_OPT_OPS"] = "flashmla_sparse_decode"
        spec = lookup("dsa_decode_attn", "decode", m=16)
        assert spec is not None
        set_forward_mode(ForwardMode.DECODE, 16)
        q = tensor((16, 1, 64, 576), dtype=torch.bfloat16)
        common = {
            "q": q,
            "cache_seqlens": tensor((16,)),
            "head_dim_v": 512,
            "tile_scheduler_metadata": tensor((148, 8)),
            "num_splits": tensor((17,)),
            "softmax_scale": 0.0625,
            "indices": tensor((16, 1, 2048)),
            "block_table": tensor((16, 0)),
            "is_fp8_kvcache": True,
        }
        with patch(
            "sglang.srt.layers.glm52_opt.dispatch._tensor_contract",
            return_value=True,
        ):
            for pages in (33, 2049, 4097):
                kv = tensor(
                    (pages, 64, 1, 656),
                    dtype=torch.float8_e4m3fn,
                )
                assert _flashmla_hotspot_abi_matches(spec, k_cache=kv, **common)

            for pages in (0, (2**31 - 1) // 64 + 1):
                kv = tensor(
                    (pages, 64, 1, 656),
                    dtype=torch.float8_e4m3fn,
                )
                assert not _flashmla_hotspot_abi_matches(spec, k_cache=kv, **common)

            misaligned = tensor(
                (33, 64, 1, 656),
                dtype=torch.float8_e4m3fn,
                ptr=0x1001,
            )
            assert not _flashmla_hotspot_abi_matches(spec, k_cache=misaligned, **common)

        def exact_fake_contract(value, *, shape, stride, dtype, device=None):
            return bool(
                value.is_cuda
                and value.dtype == dtype
                and tuple(value.shape) == shape
                and tuple(value.stride()) == stride
                and value.storage_offset() == 0
                and (device is None or value.device == device)
            )

        with patch(
            "sglang.srt.layers.glm52_opt.dispatch._tensor_contract",
            side_effect=exact_fake_contract,
        ):
            wrong_size_one_stride = tensor(
                (33, 64, 1, 656),
                dtype=torch.float8_e4m3fn,
                stride=(64 * 656, 656, 1, 1),
            )
            assert not _flashmla_hotspot_abi_matches(
                spec, k_cache=wrong_size_one_stride, **common
            )

            valid_kv = tensor(
                (33, 64, 1, 656),
                dtype=torch.float8_e4m3fn,
            )
            bad_block_table = tensor((16, 0), stride=(0, 1))
            assert not _flashmla_hotspot_abi_matches(
                spec,
                k_cache=valid_kv,
                **{**common, "block_table": bad_block_table},
            )
    finally:
        set_forward_mode(None)
        _restore_env(saved)


def test_flashmla_provider_validates_exact_output_and_lse_contract_on_cpu_meta():
    names = (
        "SGLANG_GLM52_OPT",
        "SGLANG_GLM52_OPT_PROFILE",
        "SGLANG_GLM52_OPT_OPS",
        "SGLANG_GLM52_FLASHMLA_GRAPH_ONLY",
    )
    saved = {name: os.environ.get(name) for name in names}
    q = torch.empty((16, 1, 64, 576), dtype=torch.bfloat16)
    candidate_out = torch.empty((16, 1, 64, 512), dtype=torch.bfloat16)
    correct_lse = torch.empty_strided((16, 64, 1), (64, 1, 64), dtype=torch.float32)
    fake = torch.empty(1)

    def cpu_meta_contract(tensor, *, shape, stride, dtype, device=None):
        return bool(
            tensor.dtype == dtype
            and tuple(tensor.shape) == shape
            and tuple(tensor.stride()) == stride
            and tensor.storage_offset() == 0
            and (device is None or tensor.device == device)
        )

    try:
        os.environ["SGLANG_GLM52_OPT"] = "1"
        os.environ["SGLANG_GLM52_OPT_PROFILE"] = "hotspot_candidates"
        os.environ["SGLANG_GLM52_OPT_OPS"] = "flashmla_sparse_decode"
        set_forward_mode(ForwardMode.DECODE, 16)

        with _mock_flashmla_candidate(
            capturing=True,
            provider_result=(candidate_out, correct_lse),
            contract=cpu_meta_contract,
        ):
            assert _flashmla_decode_call(q, fake) is candidate_out

        invalid_lses = (
            object(),
            torch.empty_strided((16, 1, 64), (64, 64, 1), dtype=torch.float32),
            torch.empty((16, 64, 1), dtype=torch.float32),
            torch.empty_strided((16, 64, 1), (64, 1, 64), dtype=torch.bfloat16),
            torch.empty_strided(
                (16, 64, 1),
                (64, 1, 64),
                dtype=torch.float32,
                device="meta",
            ),
        )
        for invalid_lse in invalid_lses:
            with (
                _mock_flashmla_candidate(
                    capturing=True,
                    provider_result=(candidate_out, invalid_lse),
                    contract=cpu_meta_contract,
                ) as mocks,
                TestCase().assertRaisesRegex(RuntimeError, "invalid LSE"),
            ):
                _flashmla_decode_call(q, fake)
            mocks.record_hit.assert_not_called()
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
        # Existing index_q_upproj graph-only behavior must remain unchanged.
        os.environ["SGLANG_GLM52_OPT_OPS"] = "index_q_upproj"
        idx = lookup("index_q_upproj", "decode", m=16)
        assert idx is not None
        assert idx.implementation == "fixed_nk"
        assert idx.graph_only is True
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
        result, _abi, candidate, hit, _miss = _o_proj_graph_only_dispatch(
            capturing=True
        )
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

        result, _abi, candidate, _hit, _miss = _o_proj_graph_only_dispatch(
            capturing=False
        )
        assert result is not None
        candidate.assert_called_once()
    finally:
        set_forward_mode(None)
        _restore_env(saved)


def test_fused_qkv_a_decode_e2e_spec_is_graph_only():
    """Decode fused_qkv_a_proj (fixed-N/K) is an explicit-only graph_only op."""
    names = ("SGLANG_GLM52_OPT", "SGLANG_GLM52_OPT_PROFILE", "SGLANG_GLM52_OPT_OPS")
    saved = {name: os.environ.get(name) for name in names}
    try:
        os.environ["SGLANG_GLM52_OPT"] = "1"
        os.environ["SGLANG_GLM52_OPT_PROFILE"] = "e2e_candidates"
        # Explicit-only: it must NOT appear in the default e2e set.
        os.environ.pop("SGLANG_GLM52_OPT_OPS", None)
        assert lookup("fused_qkv_a_proj", "decode", m=16) is None
        # With the explicit allowlist the decode candidate is selected.
        os.environ["SGLANG_GLM52_OPT_OPS"] = "fused_qkv_a_proj"
        qkv = lookup("fused_qkv_a_proj", "decode", m=16)
        assert qkv is not None
        assert qkv.kind == "fp8_gemm"
        assert qkv.implementation == "fixed_nk"
        assert qkv.graph_only is True
        assert (qkv.n, qkv.k) == (2624, 6144)
        assert qkv.profiler_name == "infini_kernel_glm52_fused_qkv_a_decode_nk"
        assert lookup("fused_qkv_a_proj", "decode", m=32) is not None
        assert lookup("fused_qkv_a_proj", "decode", m=64) is None
    finally:
        _restore_env(saved)


def test_fused_qkv_a_graph_only_env_parsing():
    """SGLANG_GLM52_FUSED_QKV_A_GRAPH_ONLY toggles only fused_qkv_a_proj."""
    names = (
        "SGLANG_GLM52_FUSED_QKV_A_GRAPH_ONLY",
        "SGLANG_GLM52_O_PROJ_GRAPH_ONLY",
        "SGLANG_GLM52_W2_GRAPH_ONLY",
    )
    saved = {name: os.environ.get(name) for name in names}
    try:
        os.environ.pop("SGLANG_GLM52_FUSED_QKV_A_GRAPH_ONLY", None)
        assert config.graph_only_enabled("fused_qkv_a_proj") is True
        for value in ("0", "false", "no", "OFF", " off "):
            os.environ["SGLANG_GLM52_FUSED_QKV_A_GRAPH_ONLY"] = value
            assert config.graph_only_enabled("fused_qkv_a_proj") is False
        for value in ("1", "true", "on"):
            os.environ["SGLANG_GLM52_FUSED_QKV_A_GRAPH_ONLY"] = value
            assert config.graph_only_enabled("fused_qkv_a_proj") is True
        # The fused_qkv_a override must not leak into o_proj / W2 and vice versa.
        os.environ["SGLANG_GLM52_FUSED_QKV_A_GRAPH_ONLY"] = "0"
        os.environ.pop("SGLANG_GLM52_O_PROJ_GRAPH_ONLY", None)
        os.environ.pop("SGLANG_GLM52_W2_GRAPH_ONLY", None)
        assert config.graph_only_enabled("fused_qkv_a_proj") is False
        assert config.graph_only_enabled("o_proj") is True
        assert config.graph_only_enabled("moe_down_proj") is True
    finally:
        _restore_env(saved)


def _fused_qkv_a_graph_only_dispatch(*, capturing: bool):
    """Drive one decode fused_qkv_a_proj fp8_gemm dispatch with a mocked capture."""
    input_2d = torch.zeros((16, 8))
    weight = torch.zeros((2624, 8))
    scale = torch.zeros(1, dtype=torch.int32)
    with (
        op_context("fused_qkv_a_proj"),
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


def test_fused_qkv_a_graph_only_declines_eager_before_abi_and_run():
    names = (
        "SGLANG_GLM52_OPT",
        "SGLANG_GLM52_OPT_PROFILE",
        "SGLANG_GLM52_OPT_OPS",
        "SGLANG_GLM52_FUSED_QKV_A_GRAPH_ONLY",
    )
    saved = {name: os.environ.get(name) for name in names}
    try:
        os.environ["SGLANG_GLM52_OPT"] = "1"
        os.environ["SGLANG_GLM52_OPT_PROFILE"] = "e2e_candidates"
        os.environ["SGLANG_GLM52_OPT_OPS"] = "fused_qkv_a_proj"
        os.environ.pop("SGLANG_GLM52_FUSED_QKV_A_GRAPH_ONLY", None)
        set_forward_mode(ForwardMode.DECODE, 16)

        # Eager, production graph-only on: decline to stock before the ABI check
        # and before the hit/miss lock; no candidate launch.
        result, abi, candidate, hit, miss = _fused_qkv_a_graph_only_dispatch(
            capturing=False
        )
        assert result is None
        candidate.assert_not_called()
        hit.assert_not_called()
        abi.assert_not_called()
        miss.assert_not_called()

        # Under graph capture the candidate is selected and the GEMM runs once.
        result, _abi, candidate, hit, _miss = _fused_qkv_a_graph_only_dispatch(
            capturing=True
        )
        assert result is not None
        candidate.assert_called_once()
        hit.assert_called_once()
    finally:
        set_forward_mode(None)
        _restore_env(saved)


def test_fused_qkv_a_graph_only_can_be_disabled_for_diagnostic_eager_leaf():
    names = (
        "SGLANG_GLM52_OPT",
        "SGLANG_GLM52_OPT_PROFILE",
        "SGLANG_GLM52_OPT_OPS",
        "SGLANG_GLM52_FUSED_QKV_A_GRAPH_ONLY",
    )
    saved = {name: os.environ.get(name) for name in names}
    try:
        os.environ["SGLANG_GLM52_OPT"] = "1"
        os.environ["SGLANG_GLM52_OPT_PROFILE"] = "e2e_candidates"
        os.environ["SGLANG_GLM52_OPT_OPS"] = "fused_qkv_a_proj"
        os.environ["SGLANG_GLM52_FUSED_QKV_A_GRAPH_ONLY"] = "0"
        set_forward_mode(ForwardMode.DECODE, 16)

        result, _abi, candidate, _hit, _miss = _fused_qkv_a_graph_only_dispatch(
            capturing=False
        )
        assert result is not None
        candidate.assert_called_once()
    finally:
        set_forward_mode(None)
        _restore_env(saved)
