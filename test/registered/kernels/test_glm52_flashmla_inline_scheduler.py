"""CPU provenance contracts for the inline M32 FlashMLA scheduler."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest
from sglang.srt.layers.glm52_opt import flashmla_scheduler_contract as scheduler

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MANIFEST_PATH = (
    _REPO_ROOT / "python/sglang/srt/layers/glm52_opt/hotspot_candidates/MANIFEST.json"
)
_PROVIDER_PATH = (
    _REPO_ROOT
    / "python/sglang/srt/layers/glm52_opt/hotspot_candidates"
    / "flashmla_bucketed_dynamic_provider.py"
)
_DSA_BACKEND_PATH = _REPO_ROOT / "python/sglang/srt/layers/attention/dsa_backend.py"


def _manifest_bucket(m: int) -> dict[str, object]:
    manifest = json.loads(_MANIFEST_PATH.read_text())
    matches = [
        entry
        for entry in manifest["binaries"]
        if entry.get("bucket_m") == m
        and str(entry.get("role", "")).startswith("decode_stack_dynamic_pages")
    ]
    assert len(matches) == 1
    return matches[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def test_scheduler_contract_is_exact_bijection_and_manifest_bound():
    assert scheduler.physical_to_canonical_rows(16) == tuple(range(148))
    m32 = scheduler.physical_to_canonical_rows(32)
    assert len(m32) == 148
    assert set(m32) == set(range(148))
    assert m32[:8] == (0, 4, 8, 12, 16, 20, 24, 28)
    assert m32[32:40] == (1, 5, 9, 13, 17, 21, 25, 29)
    assert m32[128:] == tuple(range(128, 148))

    for m in (16, 32):
        entry = _manifest_bucket(m)
        assert {
            key: entry[key] for key in scheduler.scheduler_contract(m)
        } == scheduler.scheduler_contract(m)


def test_manifest_pins_old_m16_and_new_inline_m32_dso():
    m16 = _manifest_bucket(16)
    m32 = _manifest_bucket(32)
    assert m16["main_variant"] == "r2a_prologue_overlap"
    assert (
        m16["module_name"] == "glm52_mla_sparse_combine_r2a_c2_dynamic_86b3787ea5a1c410"
    )
    assert m32["main_variant"] == "p1_consumer_scale_m32_rr_inline"
    assert (
        m32["module_name"]
        == "glm52_mla_sparse_combine_p1_c2_m32_rr_inline_dynamic_958c0de27a9b6629"
    )
    assert m32["build_id"].startswith("958c0de27a9b6629")
    assert (
        m32["promotion_status"]
        == "explicit_experimental_profile_only_pending_checkpoint_full_layer_whole_forward_tpot"
    )
    assert m32["performance_status"].startswith("containing-region paired gate passed")

    candidate_root = _MANIFEST_PATH.parent
    for entry in (m16, m32):
        dso = candidate_root / str(entry["so_file"])
        assert dso.is_file()
        assert _sha256(dso) == entry["sha256"]


def test_manifest_external_evidence_is_explicit_and_self_authenticating():
    evidence = _manifest_bucket(32)["external_evidence"]
    assert set(evidence) == {
        "build",
        "cache_and_tail_correctness",
        "invalid_indices_correctness",
        "containing_region_paired",
        "containing_region_paired_audit",
    }
    for item in evidence.values():
        assert Path(item["path"]).is_absolute()
        assert len(item["sha256"]) == 64
        assert item["summary"]
    assert Path(evidence["containing_region_paired_audit"]["path"]).name.endswith(
        ".audit_hardened_v2.json"
    )


def test_provider_info_exposes_scheduler_policy_hash_and_experimental_status():
    from sglang.srt.layers.glm52_opt.hotspot_candidates import (
        flashmla_bucketed_dynamic_provider as provider,
    )

    assert provider.PROVIDER_INFO["role"] == "experimental"
    assert provider.PROVIDER_INFO["canonical_scheduler_metadata"] is True
    assert provider.PROVIDER_INFO["framework_side_scheduler_reorder"] is False
    for m in (16, 32):
        bucket = provider.PROVIDER_INFO["buckets"][str(m)]
        assert {
            key: bucket[key] for key in scheduler.scheduler_contract(m)
        } == scheduler.scheduler_contract(m)
        assert bucket["provenance"]["promotion_status"].startswith(
            "explicit_experimental_profile_only"
        )


def test_explicit_dso_override_accepts_only_bucket_variant_token(tmp_path, monkeypatch):
    from sglang.srt.layers.glm52_opt.hotspot_candidates import (
        flashmla_bucketed_dynamic_provider as provider,
    )

    for m, filename in (
        (16, "local_r2a_c2_dynamic_build.so"),
        (32, "local_p1_c2_m32_rr_inline_dynamic_build.so"),
    ):
        path = tmp_path / filename
        path.write_bytes(f"M{m}".encode())
        env_name = str(provider._BUCKETS[m]["env"])
        monkeypatch.setenv(env_name, str(path))
        resolved, provenance = provider._resolve_bucket(m)
        assert resolved == path.resolve()
        assert provenance["variant_token"] in path.stem
        assert provenance["source"] == "explicit_development_override"
        monkeypatch.delenv(env_name)

    wrong_m16 = tmp_path / "local_p1_c2_m32_rr_inline_dynamic_build.so"
    wrong_m16.write_bytes(b"wrong-M16")
    monkeypatch.setenv("GLM52_FLASHMLA_M16_SO", str(wrong_m16))
    with pytest.raises(RuntimeError, match="r2a_c2_dynamic"):
        provider._resolve_bucket(16)
    monkeypatch.delenv("GLM52_FLASHMLA_M16_SO")

    legacy_m32 = tmp_path / "local_p1_c2_dynamic_build.so"
    legacy_m32.write_bytes(b"legacy-M32")
    monkeypatch.setenv("GLM52_FLASHMLA_M32_SO", str(legacy_m32))
    with pytest.raises(RuntimeError, match="p1_c2_m32_rr_inline_dynamic"):
        provider._resolve_bucket(32)


def test_dsa_leaf_passes_same_canonical_scheduler_to_candidate_and_stock():
    tree = ast.parse(_DSA_BACKEND_PATH.read_text())
    forward = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_forward_flashmla_kv"
    )
    scheduler_arguments: dict[str, list[str]] = {}
    for node in ast.walk(forward):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name not in (
            "try_dispatch_flashmla_sparse_decode",
            "flash_mla_with_kvcache",
        ):
            continue
        value = next(
            keyword.value
            for keyword in node.keywords
            if keyword.arg == "tile_scheduler_metadata"
        )
        scheduler_arguments.setdefault(name, []).append(ast.unparse(value))

    canonical = "metadata.flashmla_metadata.flashmla_metadata"
    assert scheduler_arguments["try_dispatch_flashmla_sparse_decode"] == [canonical]
    assert scheduler_arguments["flash_mla_with_kvcache"] == [canonical]

    leaf_source = ast.unparse(forward)
    assert "index_select" not in leaf_source
    assert "hotspot_flashmla_metadata" not in leaf_source
    provider_source = _PROVIDER_PATH.read_text()
    assert all(
        forbidden not in provider_source
        for forbidden in ("index_select", ".cpu(", ".item(", ".tolist(")
    )
