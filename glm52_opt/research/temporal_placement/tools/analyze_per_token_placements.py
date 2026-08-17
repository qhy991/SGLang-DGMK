#!/usr/bin/env python3
"""Replay exact GLM-5.2 token destinations under static expert placements.

The recorder payload contains physical IDs under the accepted placement.  We
first recover logical expert IDs from that authoritative map, then project the
same routing decisions through each target placement.  This separates routing
semantics from placement policy and preserves a single source of truth.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_map(path: Path, ep_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    physical_to_logical = torch.tensor(
        json.loads(path.read_text())["physical_to_logical_map"], dtype=torch.long
    )
    layers, experts = physical_to_logical.shape
    if experts % ep_size:
        raise RuntimeError(f"{path}: {experts} experts do not divide EP={ep_size}")
    expected = torch.arange(experts).expand(layers, -1)
    if not torch.equal(physical_to_logical.sort(1).values, expected):
        raise RuntimeError(f"{path}: placement is not a per-layer permutation")
    logical_to_physical = torch.empty_like(physical_to_logical)
    logical_to_physical.scatter_(1, physical_to_logical, expected)
    logical_to_rank = torch.div(
        logical_to_physical, experts // ep_size, rounding_mode="floor"
    )
    return physical_to_logical, logical_to_rank


def quantile(values: list[float], q: float) -> float:
    tensor = torch.tensor(values, dtype=torch.float64)
    return float(torch.quantile(tensor, q))


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    sequence = list(values)
    if not sequence:
        return {"count": 0}
    return {
        "count": len(sequence),
        "mean": float(sum(sequence) / len(sequence)),
        "p50": quantile(sequence, 0.50),
        "p90": quantile(sequence, 0.90),
        "p99": quantile(sequence, 0.99),
        "max": float(max(sequence)),
    }


def summarize_metrics(metrics: dict[str, list[float]]) -> dict[str, dict]:
    return {name: summarize(values) for name, values in sorted(metrics.items())}


def relative_change(candidate: float, reference: float) -> float | None:
    if reference == 0:
        return None
    return 100.0 * (candidate / reference - 1.0)


def compare_summaries(candidate: dict, reference: dict) -> dict:
    result = {}
    for metric in sorted(set(candidate) & set(reference)):
        result[metric] = {
            field: relative_change(candidate[metric][field], reference[metric][field])
            for field in ("mean", "p90", "p99", "max")
            if field in candidate[metric] and field in reference[metric]
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-dir", required=True, type=Path)
    parser.add_argument("--recorded-map", required=True, type=Path)
    parser.add_argument(
        "--placement",
        action="append",
        required=True,
        help="NAME=/absolute/path/to/physical_to_logical_map.json",
    )
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ep-size", type=int, default=8)
    args = parser.parse_args()

    placement_paths: dict[str, Path] = {}
    for spec in args.placement:
        name, separator, raw_path = spec.partition("=")
        if not separator or not name or not raw_path:
            raise RuntimeError(f"invalid --placement {spec!r}")
        if name in placement_paths:
            raise RuntimeError(f"duplicate placement name {name!r}")
        placement_paths[name] = Path(raw_path)
    if args.reference not in placement_paths:
        raise RuntimeError("--reference must name one of --placement")

    recorded_p2l, _ = load_map(args.recorded_map, args.ep_size)
    num_layers, num_experts = recorded_p2l.shape
    experts_per_rank = num_experts // args.ep_size
    placement_l2rank = {}
    for name, path in placement_paths.items():
        p2l, l2rank = load_map(path, args.ep_size)
        if p2l.shape != recorded_p2l.shape:
            raise RuntimeError(f"{path}: placement shape mismatch")
        placement_l2rank[name] = l2rank

    files = sorted(args.record_dir.glob("*.pt"))
    if len(files) != args.ep_size:
        raise RuntimeError(f"expected {args.ep_size} recorder files, got {len(files)}")

    source_metrics = {
        name: {"all": defaultdict(list), "train_even": defaultdict(list), "holdout_odd": defaultdict(list)}
        for name in placement_paths
    }
    # placement -> rank -> nonempty ordinal -> layer -> (compute counts, channels, tokens)
    aligned: dict[str, dict[int, list[dict[int, tuple[torch.Tensor, torch.Tensor, int]]]]] = {
        name: {} for name in placement_paths
    }
    file_summaries = []
    seen_ranks = []

    for path in files:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if not torch.equal(payload["last_physical_to_logical_map"].cpu(), recorded_p2l):
            raise RuntimeError(f"{path}: recorder map mismatch")
        ranks = {int(record["rank"]) for record in payload["records"]}
        if len(ranks) != 1:
            raise RuntimeError(f"{path}: mixed ranks {ranks}")
        source_rank = ranks.pop()
        seen_ranks.append(source_rank)
        per_placement_calls = {name: [] for name in placement_paths}
        nonempty_ordinal = 0
        token_counts = []
        forward_ids = []

        for record in payload["records"]:
            topk = record["topk_ids_of_layer"].to(torch.long)
            tokens = int(topk.shape[1])
            if tokens == 0:
                continue
            if topk.ndim != 3 or topk.shape[0] != num_layers or topk.shape[2] != 8:
                raise RuntimeError(f"{path}: unexpected top-k shape {tuple(topk.shape)}")
            token_counts.append(tokens)
            forward_ids.append(int(record["forward_pass_id"]))
            active_layers = sorted(int(item["layer_id"]) for item in record["misc_objects"])
            if len(active_layers) != 75 or active_layers != list(range(3, 78)):
                raise RuntimeError(f"{path}: unexpected active layers {active_layers}")
            for name in placement_paths:
                per_placement_calls[name].append({})

            split = "train_even" if nonempty_ordinal % 2 == 0 else "holdout_odd"
            for layer in active_layers:
                physical = topk[layer]
                if bool((physical < 0).any()) or bool((physical >= num_experts).any()):
                    raise RuntimeError(f"{path}: incomplete/out-of-range route IDs")
                logical = recorded_p2l[layer, physical]
                for name, l2rank in placement_l2rank.items():
                    destination = l2rank[layer, logical]
                    compute = torch.bincount(
                        destination.reshape(-1), minlength=args.ep_size
                    ).to(torch.long)
                    channels = torch.stack(
                        [(destination == dst).any(1).sum() for dst in range(args.ep_size)]
                    ).to(torch.long)
                    per_placement_calls[name][-1][layer] = (compute, channels, tokens)

                    total_assignments = tokens * 8
                    total_sends = int(channels.sum())
                    remote_sends = total_sends - int(channels[source_rank])
                    values = {
                        "compute_critical_ratio": float(compute.max()) / (total_assignments / args.ep_size),
                        "max_channel_fraction": float(channels.max()) / tokens,
                        "unique_sends_per_token": total_sends / tokens,
                        "remote_unique_sends_per_token": remote_sends / tokens,
                        "local_token_fraction": float(channels[source_rank]) / tokens,
                        "destination_channel_cv": float(channels.float().std(unbiased=False) / channels.float().mean()),
                    }
                    for bucket in ("all", split):
                        for metric, value in values.items():
                            source_metrics[name][bucket][metric].append(value)
            nonempty_ordinal += 1

        for name in placement_paths:
            aligned[name][source_rank] = per_placement_calls[name]
        file_summaries.append(
            {
                "path": str(path),
                "sha256": sha256(path),
                "rank": source_rank,
                "records": len(payload["records"]),
                "nonempty_records": nonempty_ordinal,
                "forward_ids_nonempty": forward_ids,
                "token_counts": token_counts,
            }
        )
        del payload
        gc.collect()

    if sorted(seen_ranks) != list(range(args.ep_size)):
        raise RuntimeError(f"rank coverage mismatch: {seen_ranks}")
    complete_ordinals = min(
        len(aligned[args.reference][rank]) for rank in range(args.ep_size)
    )
    if complete_ordinals == 0:
        raise RuntimeError("no complete ordinal-aligned window")

    collective_metrics = {
        name: {"all": defaultdict(list), "train_even": defaultdict(list), "holdout_odd": defaultdict(list)}
        for name in placement_paths
    }
    for name in placement_paths:
        for ordinal in range(complete_ordinals):
            split = "train_even" if ordinal % 2 == 0 else "holdout_odd"
            layer_sets = [set(aligned[name][rank][ordinal]) for rank in range(args.ep_size)]
            if any(layers != layer_sets[0] for layers in layer_sets[1:]):
                raise RuntimeError(f"layer mismatch at ordinal {ordinal}")
            for layer in sorted(layer_sets[0]):
                compute = torch.stack(
                    [aligned[name][rank][ordinal][layer][0] for rank in range(args.ep_size)]
                ).sum(0)
                channel_matrix = torch.stack(
                    [aligned[name][rank][ordinal][layer][1] for rank in range(args.ep_size)]
                )
                tokens_by_source = torch.tensor(
                    [aligned[name][rank][ordinal][layer][2] for rank in range(args.ep_size)],
                    dtype=torch.long,
                )
                total_assignments = int(compute.sum())
                total_sends = int(channel_matrix.sum())
                remote_mask = ~torch.eye(args.ep_size, dtype=torch.bool)
                remote_channels = channel_matrix[remote_mask]
                destination_ingress = channel_matrix.sum(0)
                source_egress = channel_matrix.sum(1)
                values = {
                    "compute_critical_ratio": float(compute.max()) / (total_assignments / args.ep_size),
                    "max_channel_tokens": float(channel_matrix.max()),
                    "max_channel_fraction_of_source": float(
                        (channel_matrix / tokens_by_source[:, None]).max()
                    ),
                    "max_remote_channel_tokens": float(remote_channels.max()),
                    "destination_ingress_critical_ratio": float(destination_ingress.max()) / (total_sends / args.ep_size),
                    "source_egress_critical_ratio": float(source_egress.max()) / (total_sends / args.ep_size),
                    "unique_sends_per_token": total_sends / int(tokens_by_source.sum()),
                    "remote_unique_sends_per_token": int(remote_channels.sum()) / int(tokens_by_source.sum()),
                }
                for bucket in ("all", split):
                    for metric, value in values.items():
                        collective_metrics[name][bucket][metric].append(value)

    summarized_source = {
        name: {bucket: summarize_metrics(metrics) for bucket, metrics in buckets.items()}
        for name, buckets in source_metrics.items()
    }
    summarized_collective = {
        name: {bucket: summarize_metrics(metrics) for bucket, metrics in buckets.items()}
        for name, buckets in collective_metrics.items()
    }
    reference = args.reference
    comparisons = {}
    for name in placement_paths:
        if name == reference:
            continue
        comparisons[name] = {
            "source_pass_percent_change": {
                bucket: compare_summaries(
                    summarized_source[name][bucket], summarized_source[reference][bucket]
                )
                for bucket in ("all", "train_even", "holdout_odd")
            },
            "ordinal_collective_percent_change": {
                bucket: compare_summaries(
                    summarized_collective[name][bucket],
                    summarized_collective[reference][bucket],
                )
                for bucket in ("all", "train_even", "holdout_odd")
            },
        }

    result = {
        "status": "VALID_EXACT_REPLAY",
        "semantics": {
            "routing_ssot": "recorded per-token physical top-k IDs",
            "logical_recovery": "recorded physical_to_logical map",
            "placement_projection": "target logical_to_physical rank",
            "channel_count": "unique source tokens with at least one expert on destination rank",
            "ordinal_collective_window": (
                "diagnostic alignment by nonempty recorder-call ordinal; source-pass metrics do not depend on this assumption"
            ),
            "lower_is_better": [
                "compute_critical_ratio",
                "max_channel_fraction",
                "unique_sends_per_token",
                "remote_unique_sends_per_token",
                "destination_channel_cv",
                "max_channel_tokens",
                "max_channel_fraction_of_source",
                "max_remote_channel_tokens",
                "destination_ingress_critical_ratio",
                "source_egress_critical_ratio",
            ],
        },
        "record_dir": str(args.record_dir),
        "recorded_map": {
            "path": str(args.recorded_map),
            "sha256": sha256(args.recorded_map),
        },
        "placements": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in placement_paths.items()
        },
        "reference": reference,
        "shape": {
            "layers": num_layers,
            "experts": num_experts,
            "ep_size": args.ep_size,
            "experts_per_rank": experts_per_rank,
            "complete_ordinal_window": complete_ordinals,
        },
        "source_pass_metrics": summarized_source,
        "ordinal_collective_metrics": summarized_collective,
        "comparisons": comparisons,
        "files": sorted(file_summaries, key=lambda item: item["rank"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
