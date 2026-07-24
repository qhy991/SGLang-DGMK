"""EP8 DeepEP joint Config seed for current TP8/DP8 serving (no GPU required)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sglang.srt.utils.common import load_json_config

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EP4_PATH = _REPO_ROOT / "glm52_opt" / "deepep" / "ep4_joint_config_frontier.json"
_EP8_PATH = _REPO_ROOT / "glm52_opt" / "deepep" / "ep8_joint_config_seed.json"

# Same joint body as EP4 frontier (seed copy for EP8 serving wiring).
_EXPECTED_CANONICAL_SHA256 = (
    "b0dc47051e10c17a76f6b68b00c76a8690241e154c10f04e7b7cd42ecd3bfe0a"
)


def _canonical_sha256(config: dict) -> str:
    payload = json.dumps(config, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def test_ep8_joint_config_seed_schema_and_sha():
    assert _EP8_PATH.is_file(), f"missing {_EP8_PATH}"

    config = load_json_config(str(_EP8_PATH))
    assert set(config) == {"normal_dispatch", "normal_combine"}

    for key in ("normal_dispatch", "normal_combine"):
        block = config[key]
        assert set(block) == {
            "num_sms",
            "num_max_nvl_chunked_send_tokens",
            "num_max_nvl_chunked_recv_tokens",
            "num_max_rdma_chunked_send_tokens",
            "num_max_rdma_chunked_recv_tokens",
        }

    assert config["normal_dispatch"]["num_sms"] == config["normal_combine"]["num_sms"]
    assert config["normal_dispatch"]["num_sms"] == 24
    assert config["normal_dispatch"]["num_max_nvl_chunked_send_tokens"] == 32
    assert config["normal_combine"]["num_max_nvl_chunked_send_tokens"] == 16
    assert _canonical_sha256(config) == _EXPECTED_CANONICAL_SHA256


def test_ep8_seed_matches_ep4_frontier_body():
    """Seed starts as EP4 joint copy; EP8 acceptance may diverge later."""
    ep4 = load_json_config(str(_EP4_PATH))
    ep8 = load_json_config(str(_EP8_PATH))
    assert ep8 == ep4
