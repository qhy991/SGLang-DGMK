"""CPU contract tests for GLM-5.2 hotspot provider registration."""

from __future__ import annotations

import os
from dataclasses import replace
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


def test_hotspot_profile_defaults_to_promotable_w13_and_keeps_diagnostics_explicit():
    names = ("SGLANG_GLM52_OPT_PROFILE", "SGLANG_GLM52_OPT_OPS")
    saved = {name: os.environ.get(name) for name in names}
    try:
        os.environ["SGLANG_GLM52_OPT_PROFILE"] = "hotspot_candidates"
        os.environ.pop("SGLANG_GLM52_OPT_OPS", None)
        specs = {spec.op: spec for spec in list_enabled("decode")}
        assert set(specs) == {"moe_gate_proj"}

        # Rejected FlashMLA PTX/SASS and W2 experiments remain registered for
        # explicit diagnostics, but a bare profile cannot select either.
        os.environ["SGLANG_GLM52_OPT_OPS"] = "flashmla_sparse_decode,moe_w2"
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
        assert w13 is None
        os.environ["SGLANG_GLM52_OPT_OPS"] = "moe_w13"
        w13_explicit = lookup("moe_gate_proj", "decode", m=32)
        assert w13_explicit is not None
        assert (
            w13_explicit.num_groups,
            w13_explicit.slab_m,
            w13_explicit.n,
            w13_explicit.k,
        ) == (
            32,
            1024,
            4096,
            6144,
        )
        os.environ["SGLANG_GLM52_OPT_OPS"] = "moe_w2"
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


def test_flashmla_hotspot_abi_requires_exact_bucket_page_count():
    from sglang.srt.layers.glm52_opt.dispatch import _flashmla_hotspot_abi_matches

    names = ("SGLANG_GLM52_OPT_PROFILE", "SGLANG_GLM52_OPT_OPS")
    saved = {name: os.environ.get(name) for name in names}
    device = object()

    class FakeTensor:
        def __init__(self, shape, dtype):
            self.shape = shape
            self.ndim = len(shape)
            self.dtype = dtype
            self.device = device
            self.is_cuda = True

        def is_contiguous(self):
            return True

        def storage_offset(self):
            return 0

    try:
        os.environ["SGLANG_GLM52_OPT_PROFILE"] = "hotspot_candidates"
        os.environ["SGLANG_GLM52_OPT_OPS"] = "flashmla_sparse_decode"
        with patch(
            "sglang.srt.layers.glm52_opt.dispatch._tensor_contract",
            return_value=True,
        ):
            for m, expected_pages in ((16, 2049), (32, 4097)):
                spec = lookup("dsa_decode_attn", "decode", m=m)
                assert spec is not None
                set_forward_mode(ForwardMode.DECODE, m)
                common = {
                    "q": FakeTensor((m, 1, 64, 576), torch.bfloat16),
                    "cache_seqlens": FakeTensor((m,), torch.int32),
                    "head_dim_v": 512,
                    "tile_scheduler_metadata": FakeTensor((148, 8), torch.int32),
                    "num_splits": FakeTensor((m + 1,), torch.int32),
                    "softmax_scale": 0.0625,
                    "indices": FakeTensor((m, 1, 2048), torch.int32),
                    "block_table": FakeTensor((m, 0), torch.int32),
                    "is_fp8_kvcache": True,
                }
                assert _flashmla_hotspot_abi_matches(
                    spec,
                    k_cache=FakeTensor(
                        (expected_pages, 64, 1, 656), torch.float8_e4m3fn
                    ),
                    **common,
                )
                for wrong_pages in (expected_pages - 1, expected_pages + 1):
                    assert not _flashmla_hotspot_abi_matches(
                        spec,
                        k_cache=FakeTensor(
                            (wrong_pages, 64, 1, 656), torch.float8_e4m3fn
                        ),
                        **common,
                    )
    finally:
        set_forward_mode(None)
        _restore_env(saved)


def test_flashmla_hotspot_abi_fails_closed_for_nonpromotional_fields():
    from sglang.srt.layers.glm52_opt.dispatch import _flashmla_hotspot_abi_matches

    names = ("SGLANG_GLM52_OPT_PROFILE", "SGLANG_GLM52_OPT_OPS")
    saved = {name: os.environ.get(name) for name in names}
    device = object()
    other_device = object()

    class FakeTensor:
        def __init__(
            self,
            shape,
            dtype,
            *,
            stride=None,
            tensor_device=device,
            is_cuda=True,
            contiguous=True,
            storage_offset=0,
        ):
            self.shape = tuple(shape)
            self.ndim = len(shape)
            self.dtype = dtype
            self.device = tensor_device
            self.is_cuda = is_cuda
            self._stride = (
                tuple(stride)
                if stride is not None
                else tuple(
                    1
                    if index == len(shape) - 1
                    else int(torch.tensor(shape[index + 1 :]).prod().item())
                    for index in range(len(shape))
                )
            )
            self._contiguous = contiguous
            self._storage_offset = storage_offset

        def stride(self):
            return self._stride

        def is_contiguous(self):
            return self._contiguous

        def storage_offset(self):
            return self._storage_offset

    try:
        os.environ["SGLANG_GLM52_OPT_PROFILE"] = "hotspot_candidates"
        os.environ["SGLANG_GLM52_OPT_OPS"] = "flashmla_sparse_decode"
        set_forward_mode(ForwardMode.DECODE, 16)
        spec = lookup("dsa_decode_attn", "decode", m=16)
        assert spec is not None
        kwargs = {
            "q": FakeTensor(
                (16, 1, 64, 576),
                torch.bfloat16,
                stride=(36864, 36864, 576, 1),
            ),
            "k_cache": FakeTensor(
                (2049, 64, 1, 656),
                torch.float8_e4m3fn,
                stride=(41984, 656, 656, 1),
            ),
            "cache_seqlens": FakeTensor((16,), torch.int32, stride=(1,)),
            "head_dim_v": 512,
            "tile_scheduler_metadata": FakeTensor(
                (148, 8), torch.int32, stride=(8, 1)
            ),
            "num_splits": FakeTensor((17,), torch.int32, stride=(1,)),
            "softmax_scale": 0.0625,
            "indices": FakeTensor(
                (16, 1, 2048), torch.int32, stride=(2048, 2048, 1)
            ),
            "block_table": FakeTensor((16, 0), torch.int32),
            "is_fp8_kvcache": True,
        }
        assert _flashmla_hotspot_abi_matches(spec, **kwargs)

        rejected = [
            {"head_dim_v": 511},
            {"softmax_scale": 0.125},
            {"is_fp8_kvcache": False},
            {
                "q": FakeTensor(
                    (16, 1, 64, 576),
                    torch.float16,
                    stride=(36864, 36864, 576, 1),
                )
            },
            {
                "q": FakeTensor(
                    (16, 1, 64, 576),
                    torch.bfloat16,
                    stride=(36864, 36864, 1, 64),
                )
            },
            {
                "q": FakeTensor(
                    (16, 1, 64, 576),
                    torch.bfloat16,
                    stride=(36864, 36864, 576, 1),
                    is_cuda=False,
                )
            },
            {
                "k_cache": FakeTensor(
                    (2048, 64, 1, 656),
                    torch.float8_e4m3fn,
                    stride=(41984, 656, 656, 1),
                )
            },
            {
                "k_cache": FakeTensor(
                    (2049, 64, 1, 656),
                    torch.bfloat16,
                    stride=(41984, 656, 656, 1),
                )
            },
            {
                "k_cache": FakeTensor(
                    (2049, 64, 1, 656),
                    torch.float8_e4m3fn,
                    stride=(41984, 656, 656, 1),
                    contiguous=False,
                )
            },
            {
                "k_cache": FakeTensor(
                    (2049, 64, 1, 656),
                    torch.float8_e4m3fn,
                    stride=(41984, 656, 656, 1),
                    tensor_device=other_device,
                )
            },
            {
                "cache_seqlens": FakeTensor(
                    (16,), torch.int64, stride=(1,)
                )
            },
            {
                "tile_scheduler_metadata": FakeTensor(
                    (147, 8), torch.int32, stride=(8, 1)
                )
            },
            {"num_splits": FakeTensor((16,), torch.int32, stride=(1,))},
            {
                "indices": FakeTensor(
                    (16, 1, 2048),
                    torch.int64,
                    stride=(2048, 2048, 1),
                )
            },
            {
                "indices": FakeTensor(
                    (16, 1, 2048),
                    torch.int32,
                    stride=(2048, 1, 1),
                )
            },
            {
                "block_table": FakeTensor(
                    (16, 1), torch.int32, tensor_device=device
                )
            },
            {
                "block_table": FakeTensor(
                    (16, 0), torch.int32, tensor_device=other_device
                )
            },
        ]
        for overrides in rejected:
            current = dict(kwargs)
            current.update(overrides)
            assert not _flashmla_hotspot_abi_matches(spec, **current)

        assert not _flashmla_hotspot_abi_matches(
            replace(spec, implementation="auto"), **kwargs
        )
        assert not _flashmla_hotspot_abi_matches(
            replace(spec, kind="bmm"), **kwargs
        )
        set_forward_mode(ForwardMode.TARGET_VERIFY, 16)
        assert not _flashmla_hotspot_abi_matches(spec, **kwargs)
    finally:
        set_forward_mode(None)
        _restore_env(saved)


def test_selected_flashmla_candidate_rejects_invalid_lse():
    names = (
        "SGLANG_GLM52_OPT",
        "SGLANG_GLM52_OPT_PROFILE",
        "SGLANG_GLM52_OPT_OPS",
    )
    saved = {name: os.environ.get(name) for name in names}
    q = torch.empty((16, 1, 64, 576), dtype=torch.bfloat16)
    candidate_out = torch.empty((16, 1, 64, 512), dtype=torch.bfloat16)
    invalid_lse = torch.empty((16, 1, 64), dtype=torch.float32)
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
                side_effect=(True, False),
            ),
            patch(
                "sglang.srt.layers.glm52_opt.dispatch.run_flashmla_sparse_decode",
                return_value=(candidate_out, invalid_lse),
            ) as candidate,
            patch("sglang.srt.layers.glm52_opt.dispatch._record_hit") as record_hit,
            TestCase().assertRaisesRegex(RuntimeError, "invalid LSE"),
        ):
            try_dispatch_flashmla_sparse_decode(
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
        candidate.assert_called_once()
        record_hit.assert_not_called()
    finally:
        set_forward_mode(None)
        _restore_env(saved)
