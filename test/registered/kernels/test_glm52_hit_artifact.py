"""CPU-only contracts for rank-safe GLM-5.2 dispatch evidence."""

from __future__ import annotations

import ast
import json
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.layers.glm52_opt import dispatch
from sglang.srt.layers.glm52_opt.context import get_layer_id, layer_context


@contextmanager
def _isolated_stats(path: Path | None):
    saved = (
        dispatch._HIT_FILE_RAW,
        dispatch._HIT_FILE,
        dispatch._PROCESS_IDENTITY,
        dispatch._STATIC_ARTIFACT_METADATA,
        dispatch._HIT_LOCK,
        dict(dispatch._HIT_COUNTS),
        dict(dispatch._MISS_COUNTS),
        dict(dispatch._SELECTED_SCOPE_COUNTS),
        dict(dispatch._MISS_SCOPE_COUNTS),
    )
    dispatch._HIT_FILE_RAW = "" if path is None else str(path)
    dispatch._HIT_FILE = path
    dispatch._STATIC_ARTIFACT_METADATA = {
        "process": {
            "pid": 123,
            "global_rank": 4,
            "local_rank": 1,
            "gpu_uuid": "GPU-unit-test",
        },
        "sglang": {
            "repo_root": "/repo",
            "commit": "abc123",
            "branch": "test-branch",
            "dirty": False,
        },
    }
    dispatch._HIT_COUNTS.clear()
    dispatch._MISS_COUNTS.clear()
    dispatch._SELECTED_SCOPE_COUNTS.clear()
    dispatch._MISS_SCOPE_COUNTS.clear()
    try:
        yield
    finally:
        (
            dispatch._HIT_FILE_RAW,
            dispatch._HIT_FILE,
            dispatch._PROCESS_IDENTITY,
            dispatch._STATIC_ARTIFACT_METADATA,
            dispatch._HIT_LOCK,
            hit_counts,
            miss_counts,
            selected_scope_counts,
            miss_scope_counts,
        ) = saved
        dispatch._HIT_COUNTS.clear()
        dispatch._HIT_COUNTS.update(hit_counts)
        dispatch._MISS_COUNTS.clear()
        dispatch._MISS_COUNTS.update(miss_counts)
        dispatch._SELECTED_SCOPE_COUNTS.clear()
        dispatch._SELECTED_SCOPE_COUNTS.update(selected_scope_counts)
        dispatch._MISS_SCOPE_COUNTS.clear()
        dispatch._MISS_SCOPE_COUNTS.update(miss_scope_counts)


def test_hit_file_path_is_process_unique_and_supports_templates():
    identity = {
        "pid": 91,
        "global_rank": "7",
        "local_rank": "2",
        "gpu_uuid": "GPU-deadbeef",
    }
    assert dispatch._resolve_hit_file("", identity) is None
    legacy = dispatch._resolve_hit_file("/tmp/glm52_hits.json", identity)
    assert legacy == Path("/tmp/glm52_hits.rank7.local2.pid91.json")

    template = dispatch._resolve_hit_file(
        "/tmp/glm52-{global_rank}-{local_rank}-{pid}-{gpu_uuid}.json",
        identity,
    )
    assert template == Path("/tmp/glm52-7-2-91-GPU-deadbeef.json")


def test_process_identity_reads_gpu_uuid_without_initializing_cuda(monkeypatch):
    monkeypatch.setenv("RANK", "6")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv(
        "NVIDIA_VISIBLE_DEVICES",
        "GPU-first,GPU-second",
    )
    before = dispatch.torch.cuda.is_initialized()
    identity = dispatch._process_identity()
    after = dispatch.torch.cuda.is_initialized()
    assert identity["global_rank"] == 6
    assert identity["local_rank"] == 1
    assert identity["gpu_uuid"] == "GPU-second"
    assert after is before


def test_fork_reset_drops_parent_path_counts_and_lock(tmp_path):
    child_identity = {
        "pid": 456,
        "global_rank": 5,
        "local_rank": 2,
        "gpu_uuid": "GPU-child",
    }
    with _isolated_stats(tmp_path / "parent.json"):
        dispatch._HIT_FILE_RAW = str(tmp_path / "hits.json")
        dispatch._HIT_COUNTS["parent"] = 3
        parent_lock = dispatch._HIT_LOCK
        dispatch._reset_artifact_after_fork()
        assert dispatch._HIT_FILE is None
        assert dispatch._HIT_COUNTS == {}
        assert dispatch._HIT_LOCK is not parent_lock
        with patch.object(dispatch, "_process_identity", return_value=child_identity):
            dispatch._initialize_process_artifact()
        assert dispatch._HIT_FILE == (tmp_path / "hits.rank5.local2.pid456.json")


def test_layer_context_is_nested_and_restored():
    assert get_layer_id() is None
    with layer_context(7):
        assert get_layer_id() == 7
        with layer_context(9):
            assert get_layer_id() == 9
        assert get_layer_id() == 7
    assert get_layer_id() is None


def test_artifact_has_scoped_counts_provider_identity_and_atomic_replace(tmp_path):
    target = tmp_path / "hits.rank4.local1.pid123.json"
    provider = {
        "ready": True,
        "module_ref": "provider.module",
        "module_name": "provider_module",
        "provider_info": {
            "buckets": {
                "16": {
                    "module_name": "flashmla_m16",
                    "extension_file": "/prebuilt/flashmla_m16.so",
                    "sha256": "012345",
                    "main_variant": "r2a",
                    "combine_variant": "c2",
                }
            }
        },
    }
    real_replace = os.replace
    replacements: list[tuple[Path, Path]] = []

    def record_replace(source, destination):
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    with (
        _isolated_stats(target),
        patch.object(dispatch, "provider_state", return_value=provider),
        patch.object(dispatch.config, "load_manifest", return_value={"run": "g4"}),
        patch.object(dispatch.os, "replace", side_effect=record_replace),
        layer_context(11),
    ):
        dispatch._record_hit(
            "hotspot_plugin/flashmla_sparse_decode",
            "dsa_decode_attn",
            "decode",
            m=16,
        )
        dispatch._record_miss(
            "flashmla_abi",
            "dsa_decode_attn",
            "decode",
            m=16,
        )
        payload = json.loads(target.read_text())

    assert replacements
    assert all(destination == target for _source, destination in replacements)
    assert not list(tmp_path.glob(".*.tmp.*"))
    assert payload["schema_version"] == 2
    assert payload["process"] == {
        "pid": 123,
        "global_rank": 4,
        "local_rank": 1,
        "gpu_uuid": "GPU-unit-test",
    }
    assert payload["sglang"]["commit"] == "abc123"
    assert payload["manifest"] == {"run": "g4"}
    assert payload["provider_state"] == provider
    assert payload["provider_dso_identities"] == [
        {
            "location": "provider_info.buckets.16",
            "module_name": "flashmla_m16",
            "extension_file": "/prebuilt/flashmla_m16.so",
            "sha256": "012345",
            "main_variant": "r2a",
            "combine_variant": "c2",
        }
    ]
    assert payload["counts"]["selected"] == [
        {
            "layer": 11,
            "op": "dsa_decode_attn",
            "phase": "decode",
            "m": 16,
            "kind": "hotspot_plugin/flashmla_sparse_decode",
            "count": 1,
        }
    ]
    assert payload["counts"]["misses"] == [
        {
            "layer": 11,
            "op": "dsa_decode_attn",
            "phase": "decode",
            "m": 16,
            "reason": "flashmla_abi",
            "count": 1,
        }
    ]
    assert (
        payload["hits"][
            "hotspot_plugin/flashmla_sparse_decode:dsa_decode_attn:decode:m16"
        ]
        == 1
    )


def test_dso_identity_carries_inline_scheduler_provenance():
    identities = dispatch._dso_identities(
        {
            "provider_info": {
                "buckets": {
                    "32": {
                        "module_name": "flashmla_m32_inline",
                        "sha256": "dso-sha",
                        "build_id": "build-sha",
                        "main_variant": "p1_consumer_scale_m32_rr_inline",
                        "combine_variant": "combine_c2_bucket_stages",
                        "promotion_status": "explicit_experimental_profile_only",
                        "scheduler_order": "split_major_round_robin_inline",
                        "scheduler_mapping_location": (
                            "main_kernel_physical_to_canonical_row"
                        ),
                        "scheduler_metadata_order": "request_major",
                        "scheduler_contract_version": (
                            "glm52-flashmla-inline-row-map-v1"
                        ),
                        "scheduler_permutation_sha256": "mapping-sha",
                    }
                }
            }
        }
    )
    assert identities == [
        {
            "location": "provider_info.buckets.32",
            "module_name": "flashmla_m32_inline",
            "sha256": "dso-sha",
            "build_id": "build-sha",
            "main_variant": "p1_consumer_scale_m32_rr_inline",
            "combine_variant": "combine_c2_bucket_stages",
            "promotion_status": "explicit_experimental_profile_only",
            "scheduler_order": "split_major_round_robin_inline",
            "scheduler_mapping_location": ("main_kernel_physical_to_canonical_row"),
            "scheduler_metadata_order": "request_major",
            "scheduler_contract_version": "glm52-flashmla-inline-row-map-v1",
            "scheduler_permutation_sha256": "mapping-sha",
        }
    ]


def test_empty_hit_file_disables_stats_without_lock_or_write():
    lock = Mock()
    with _isolated_stats(None), patch.object(dispatch, "_HIT_LOCK", lock):
        dispatch._record_hit("candidate", "dsa_decode_attn", "decode", m=16)
        dispatch._record_miss("abi", "dsa_decode_attn", "decode", m=16)
        assert dispatch._HIT_COUNTS == {}
        assert dispatch._MISS_COUNTS == {}
    lock.assert_not_called()


def test_first_selected_observation_flushes_each_layer(tmp_path):
    with (
        _isolated_stats(tmp_path / "hits.json"),
        patch.object(dispatch, "_flush_stats") as flush,
    ):
        with layer_context(10):
            dispatch._record_hit("candidate", "dsa_decode_attn", "decode", m=16)
        with layer_context(11):
            dispatch._record_hit("candidate", "dsa_decode_attn", "decode", m=16)
    assert flush.call_count == 2


def test_graph_only_decline_takes_no_stats_lock_or_write():
    spec = SimpleNamespace(graph_only=True, op="dsa_decode_attn")
    with (
        patch.object(dispatch.config, "graph_only_enabled", return_value=True),
        patch.object(dispatch, "_is_cuda_graph_capturing", return_value=False),
        patch.object(dispatch, "_HIT_LOCK", Mock()) as lock,
        patch.object(dispatch, "_flush_stats") as flush,
    ):
        assert dispatch._graph_only_declines(spec)
    lock.assert_not_called()
    flush.assert_not_called()


def test_dsa_backend_propagates_real_layer_to_both_hotspot_dispatches():
    repo_root = Path(__file__).resolve().parents[3]
    source_path = repo_root / "python/sglang/srt/layers/attention/dsa_backend.py"
    tree = ast.parse(source_path.read_text())
    wrapped_calls: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.With) or len(node.items) != 1:
            continue
        expression = node.items[0].context_expr
        if (
            not isinstance(expression, ast.Call)
            or not isinstance(expression.func, ast.Name)
            or expression.func.id != "layer_context"
            or len(expression.args) != 1
        ):
            continue
        argument = expression.args[0]
        if not (
            isinstance(argument, ast.Attribute)
            and isinstance(argument.value, ast.Name)
            and argument.value.id == "layer"
            and argument.attr == "layer_id"
        ):
            continue
        wrapped_calls.update(
            call.func.id
            for call in ast.walk(node)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        )
    assert {
        "try_dispatch_flashmla_sparse_decode",
        "try_dispatch_flashmla_sparse_prefill",
    } <= wrapped_calls
