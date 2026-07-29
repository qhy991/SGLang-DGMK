"""CPU-only contract tests for the exact W13 API-v1 provider sources."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[3]
PROVIDER_ROOT = ROOT / "third_party" / "deepgemm_w13"


def _load(name: str):
    path = PROVIDER_ROOT / name
    spec = importlib.util.spec_from_file_location(f"test_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_provider_metadata_and_configs():
    two_sm = _load("provider_bm16_2sm.py")
    one_sm = _load("provider_bm16_1sm.py")
    assert two_sm.INFINI_KERNEL_API_VERSION == 1
    assert one_sm.INFINI_KERNEL_API_VERSION == 1
    assert two_sm._PROVIDER.config == (16, 128, 128, 12, 2)
    assert one_sm._PROVIDER.config == (16, 128, 128, 11, 1)
    for module in (two_sm, one_sm):
        assert set(module.PROVIDER_INFO) >= {"name", "git_commit", "build_id"}
        assert module.PROVIDER_INFO["name"].startswith(
            "infini_kernel_glm52_moe_w13_decode"
        )


def test_hot_callback_is_one_fail_closed_candidate_call():
    module = _load("provider_bm16_2sm.py")
    launcher = Mock(return_value=None)
    module._PROVIDER._launcher = launcher
    lhs, rhs, out, masked_m = object(), object(), object(), object()
    assert (
        module.moe_w13(
            lhs=lhs,
            rhs=rhs,
            out=out,
            masked_m=masked_m,
            expected_m=4,
        )
        is None
    )
    launcher.assert_called_once_with(
        lhs,
        rhs,
        out,
        masked_m,
        4,
        compiled_dims="nk",
        disable_ue8m0_cast=True,
        w13_config=(16, 128, 128, 12, 2),
    )

    launcher.side_effect = RuntimeError("selected candidate failed")
    with TestCase().assertRaisesRegex(RuntimeError, "selected candidate failed"):
        module.moe_w13(
            lhs=lhs,
            rhs=rhs,
            out=out,
            masked_m=masked_m,
            expected_m=5,
        )
    assert launcher.call_count == 2


def test_hot_callback_contains_no_startup_or_cuda_control_work():
    common = _load("provider_common.py")
    source = inspect.getsource(common.Provider.moe_w13)
    for forbidden in (
        "import ",
        "open(",
        "read_text",
        "cuda.",
        "synchronize",
        "set_pdl",
        "set_num_sms",
        "set_tc_util",
        "empty(",
        "zeros(",
    ):
        assert forbidden not in source
