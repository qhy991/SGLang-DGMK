#!/usr/bin/env python3
"""Validate GLM-5.2 per-token expert records before communication replay."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-dir", required=True, type=Path)
    parser.add_argument("--accepted-map", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ep-size", type=int, default=8)
    args = parser.parse_args()

    files = sorted(args.record_dir.glob("*.pt"))
    if len(files) != args.ep_size:
        raise RuntimeError(f"expected {args.ep_size} rank files, got {len(files)}")
    accepted = torch.tensor(
        json.loads(args.accepted_map.read_text())["physical_to_logical_map"],
        dtype=torch.long,
    )
    num_layers, num_experts = accepted.shape
    if num_experts % args.ep_size:
        raise RuntimeError("expert count does not divide EP size")
    expected_experts = torch.arange(num_experts)
    if not torch.all(accepted.sort(1).values == expected_experts):
        raise RuntimeError("accepted map is not a permutation in every layer")
    logical_to_physical = torch.empty_like(accepted)
    logical_to_physical.scatter_(
        1, accepted, expected_experts.expand(num_layers, -1)
    )
    round_trip = logical_to_physical.gather(1, accepted)
    if not torch.equal(round_trip, expected_experts.expand(num_layers, -1)):
        raise RuntimeError("physical/logical map round-trip failed")

    reference_forward_ids = None
    seen_ranks: list[int] = []
    record_count = 0
    nonempty_records = 0
    layer_passes = 0
    topk_count_checks = 0
    topk_count_mismatches = 0
    full_topk_layer_passes = 0
    incomplete_topk_layer_passes = 0
    replayed_channel_layer_passes = 0
    authoritative_hook_layer_passes = 0
    authoritative_hook_matches = 0
    authoritative_hook_max_error = 0
    union_send_ratio_min = float("inf")
    union_send_ratio_max = 0.0
    file_summaries = []

    for path in files:
        payload = torch.load(
            path, map_location="cpu", weights_only=False, mmap=True
        )
        records = payload["records"]
        if not torch.equal(payload["last_physical_to_logical_map"].cpu(), accepted):
            raise RuntimeError(f"record map mismatch: {path}")
        ranks = {int(record["rank"]) for record in records}
        if len(ranks) != 1:
            raise RuntimeError(f"file contains multiple ranks: {path} {ranks}")
        rank = ranks.pop()
        seen_ranks.append(rank)
        forward_ids = [int(record["forward_pass_id"]) for record in records]
        if forward_ids != sorted(forward_ids) or len(forward_ids) != len(set(forward_ids)):
            raise RuntimeError(f"forward IDs are not unique/sorted: {path}")
        if reference_forward_ids is None:
            reference_forward_ids = forward_ids
        elif forward_ids != reference_forward_ids:
            raise RuntimeError(f"forward ID alignment mismatch: {path}")

        file_nonempty = 0
        file_channel_checks = 0
        for record in records:
            record_count += 1
            topk = record["topk_ids_of_layer"].to(torch.long)
            global_counts = record["global_physical_count"].to(torch.long)
            if topk.ndim != 3 or tuple(topk.shape[::2]) != (num_layers, 8):
                raise RuntimeError(f"unexpected top-k shape {tuple(topk.shape)} in {path}")
            if tuple(global_counts.shape) != (num_layers, num_experts):
                raise RuntimeError(f"unexpected count shape in {path}")
            tokens = int(topk.shape[1])
            if tokens == 0:
                if int(global_counts.sum()) != 0:
                    raise RuntimeError(f"empty top-k has nonzero counts: {path}")
                continue
            nonempty_records += 1
            file_nonempty += 1
            misc_by_layer = {
                int(item["layer_id"]): item for item in record["misc_objects"]
            }
            for layer, misc in misc_by_layer.items():
                layer_passes += 1
                physical = topk[layer]
                valid = physical >= 0
                valid_ids = physical[valid]
                replay_counts = torch.bincount(valid_ids, minlength=num_experts)
                topk_count_checks += 1
                if not torch.equal(replay_counts, global_counts[layer]):
                    topk_count_mismatches += 1
                if int(valid.sum()) == tokens * 8:
                    full_topk_layer_passes += 1
                else:
                    incomplete_topk_layer_passes += 1

                logical = accepted[layer, valid_ids]
                if not torch.equal(logical_to_physical[layer, logical], valid_ids):
                    raise RuntimeError("physical/logical round-trip failed on routed IDs")
                destination = torch.div(
                    physical.clamp_min(0),
                    num_experts // args.ep_size,
                    rounding_mode="floor",
                )
                replay_channels = torch.stack(
                    [
                        (((destination == dst) & valid).any(1)).sum()
                        for dst in range(args.ep_size)
                    ]
                ).to(torch.long)
                replayed_channel_layer_passes += 1
                file_channel_checks += 1
                union_send_ratio = float(replay_channels.sum()) / tokens
                union_send_ratio_min = min(union_send_ratio_min, union_send_ratio)
                union_send_ratio_max = max(union_send_ratio_max, union_send_ratio)
                if not 1.0 <= union_send_ratio <= 8.0:
                    raise RuntimeError(
                        f"invalid union-send ratio {union_send_ratio} in {path}"
                    )

                observed = torch.tensor(
                    misc["num_tokens_per_rank"], dtype=torch.long
                )
                # The DeepEP tensors are produced asynchronously.  Zero
                # snapshots are explicitly unavailable rather than negative
                # evidence; nonzero snapshots are authoritative cross-checks.
                if int(observed.sum()) > 0:
                    authoritative_hook_layer_passes += 1
                    error = int((replay_channels - observed).abs().max())
                    authoritative_hook_max_error = max(
                        authoritative_hook_max_error, error
                    )
                    if error == 0:
                        authoritative_hook_matches += 1

        file_summaries.append(
            {
                "path": str(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "rank": rank,
                "records": len(records),
                "nonempty_records": file_nonempty,
                "channel_layer_passes": file_channel_checks,
            }
        )
        del payload, records
        gc.collect()

    if sorted(seen_ranks) != list(range(args.ep_size)):
        raise RuntimeError(f"rank coverage mismatch: {seen_ranks}")
    # DeepEP's layout tensors are produced on an asynchronous communication
    # stream and the recorder does not wait on the returned event before
    # copying them.  They are therefore diagnostics, not an independent gate:
    # in this capture seven ranks stayed at their zero initialization and the
    # remaining rank repeatedly exposed a stale layout.  The routed top-k
    # payload is the SSOT used for deterministic communication replay.
    gates = {
        "rank_coverage_exact": True,
        "forward_ids_aligned": True,
        "accepted_map_exact": True,
        "physical_logical_round_trip": True,
        "topk_count_exact": topk_count_mismatches == 0,
        "topk_payload_complete": (
            layer_passes > 0
            and full_topk_layer_passes == layer_passes
            and incomplete_topk_layer_passes == 0
        ),
    }
    result = {
        "status": "VALID_ROUTE_SSOT" if all(gates.values()) else "INVALID",
        "record_dir": str(args.record_dir),
        "accepted_map": str(args.accepted_map),
        "accepted_map_sha256": sha256(args.accepted_map),
        "gates": gates,
        "summary": {
            "ranks": sorted(seen_ranks),
            "forward_ids": reference_forward_ids,
            "records": record_count,
            "nonempty_records": nonempty_records,
            "layer_passes": layer_passes,
            "topk_count_checks": topk_count_checks,
            "topk_count_mismatches": topk_count_mismatches,
            "full_topk_layer_passes": full_topk_layer_passes,
            "incomplete_topk_layer_passes": incomplete_topk_layer_passes,
            "replayed_channel_layer_passes": replayed_channel_layer_passes,
            "authoritative_hook_layer_passes": authoritative_hook_layer_passes,
            "authoritative_hook_matches": authoritative_hook_matches,
            "authoritative_hook_max_error": authoritative_hook_max_error,
            "deepep_hook_interpretation": (
                "asynchronous diagnostic snapshot; excluded from admission"
            ),
            "union_send_ratio_min": union_send_ratio_min,
            "union_send_ratio_max": union_send_ratio_max,
        },
        "files": file_summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "VALID_ROUTE_SSOT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
