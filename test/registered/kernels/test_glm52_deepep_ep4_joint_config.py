"""Goal-25 EP4 DeepEP joint Config frontier (no GPU required)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sglang.srt.utils.common import load_json_config

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_PATH = _REPO_ROOT / "glm52_opt" / "deepep" / "ep4_joint_config_frontier.json"

# Canonical SHA from goal-25 EP4 joint-Config evidence summary.
_EXPECTED_CANONICAL_SHA256 = (
    "b0dc47051e10c17a76f6b68b00c76a8690241e154c10f04e7b7cd42ecd3bfe0a"
)


def _canonical_sha256(config: dict) -> str:
    payload = json.dumps(config, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def test_ep4_joint_config_frontier_schema_and_sha():
    assert _CONFIG_PATH.is_file(), f"missing {_CONFIG_PATH}"

    config = load_json_config(str(_CONFIG_PATH))
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
