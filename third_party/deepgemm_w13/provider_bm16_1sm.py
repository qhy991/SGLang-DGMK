"""Exact BM16 one-SM API-v1 provider for GLM-5.2 W13 decode."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_COMMON_PATH = Path(__file__).resolve().with_name("provider_common.py")
_SPEC = importlib.util.spec_from_file_location(
    "infini_kernel_glm52_moe_w13_decode_provider_common", _COMMON_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(_COMMON_PATH)
_COMMON = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_COMMON)

INFINI_KERNEL_API_VERSION = 1
PROVIDER_INFO = {
    "name": "infini_kernel_glm52_moe_w13_decode_bm16_1sm",
    "git_commit": _COMMON.CANDIDATE_COMMIT,
    "build_id": "bm16-1sm-stage11-api-v1",
}
_PROVIDER = _COMMON.Provider(
    name="bm16_1sm",
    config=(16, 128, 128, 11, 1),
)


def initialize(*, gpu_id: int | None) -> None:
    _PROVIDER.initialize(gpu_id=gpu_id)


def moe_w13(*, lhs, rhs, out, masked_m, expected_m):
    return _PROVIDER.moe_w13(
        lhs=lhs,
        rhs=rhs,
        out=out,
        masked_m=masked_m,
        expected_m=expected_m,
    )
