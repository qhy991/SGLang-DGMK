"""Round-2 SF-relay-bypass API-v1 provider for GLM-5.2 W13 decode.

Predeclared hypothesis H2-SF-BYPASS from
`profile/w13-bm16-r2-survivor-em4-20260730/REPORT.md`. Default-off experiment.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_COMMON_PATH = Path(__file__).resolve().with_name("provider_common.py")
_SPEC = importlib.util.spec_from_file_location(
    "infini_kernel_glm52_moe_w13_decode_provider_common_r2", _COMMON_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(_COMMON_PATH)
_COMMON = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_COMMON)

MANIFEST = Path(
    "/home/qinhaiyan/glm52-hotspot-goal-runs/cache/moe_w13_ptx_r2/deepgemm/w13_variants/manifest.json"
)
_DOCUMENT = json.loads(MANIFEST.read_text())
EXPECTED_SOURCE = {
    key: _DOCUMENT["source"][key]
    for key in (
        "base_commit",
        "candidate_commit",
        "candidate_diff_sha256",
        "stock_source_tree_sha256",
        "candidate_source_tree_sha256",
    )
}

INFINI_KERNEL_API_VERSION = 1
PROVIDER_INFO = {
    "name": "infini_kernel_glm52_moe_w13_decode_r2_bm16_1sm_sfbypass",
    "git_commit": EXPECTED_SOURCE["candidate_commit"],
    "build_id": "r2-bm16-1sm-stage11-sfrelaybypass-api-v1",
}
_PROVIDER = _COMMON.Provider(
    name="r2_bm16_1sm_sfbypass",
    config=(16, 128, 128, 11, 1, 1),
    manifest=MANIFEST,
    expected_source=EXPECTED_SOURCE,
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
