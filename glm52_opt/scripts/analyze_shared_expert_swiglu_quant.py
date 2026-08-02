#!/usr/bin/env python3
"""Compare production shared-expert SwiGLU -> FP8 quant in nsys traces."""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ACTIVATION = "act_and_mul_kernel"
QUANT = "per_token_group_quant_8bit_v2_kernel"
FUSED = "silu_mul_quant_contig_kernel"
ROUTED_ACTIVATION = "silu_mul_quant_varlen_kernel"
DECODE_ACTIVATION_GRID = 8
DECODE_QUANT_GRID = 16
DECODE_FUSED_GRID = 16


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def distribution(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    return {
        "count": len(values),
        "min": min(values),
        "p10": percentile(values, 0.10),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "p90": percentile(values, 0.90),
        "max": max(values),
        "stdev": statistics.pstdev(values),
    }


def analyze(path: Path) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        names = dict(connection.execute("select id, value from StringIds"))
        ids = {value: key for key, value in names.items()}
        required = [ACTIVATION, QUANT, ROUTED_ACTIVATION]
        missing = [name for name in required if name not in ids]
        if missing:
            raise RuntimeError(f"missing kernel names in {path}: {missing}")
        selected_ids = [ids[ACTIVATION], ids[QUANT]]
        if FUSED in ids:
            selected_ids.append(ids[FUSED])
        placeholders = ",".join("?" for _ in selected_ids)
        rows = list(
            connection.execute(
                "select deviceId, streamId, start, end, shortName, gridX "
                "from CUPTI_ACTIVITY_KIND_KERNEL "
                f"where shortName in ({placeholders}) "
                "order by deviceId, streamId, start",
                selected_ids,
            )
        )
        graph_rows = list(
            connection.execute(
                "select deviceId, correlationId, min(start), max(end), count(*), "
                "sum(case when shortName = ? then 1 else 0 end) "
                "from CUPTI_ACTIVITY_KIND_KERNEL "
                "where graphId is not null and graphId != 0 "
                "group by deviceId, correlationId "
                "having sum(case when shortName = ? then 1 else 0 end) = 75 "
                "order by deviceId, min(start)",
                (ids[ROUTED_ACTIVATION], ids[ROUTED_ACTIVATION]),
            )
        )
    finally:
        connection.close()

    by_stream: dict[tuple[int, int], list[tuple[int, int, int, int]]] = (
        defaultdict(list)
    )
    for device, stream, start, end, name_id, grid_x in rows:
        by_stream[(int(device), int(stream))].append(
            (int(start), int(end), int(name_id), int(grid_x))
        )

    pair_samples: list[dict[str, float | int]] = []
    fused_samples: list[dict[str, float | int]] = []
    activation_id = ids[ACTIVATION]
    quant_id = ids[QUANT]
    fused_id = ids.get(FUSED)
    for (device, _stream), events in by_stream.items():
        for index, (start, end, name_id, grid_x) in enumerate(events):
            if (
                fused_id is not None
                and name_id == fused_id
                and grid_x == DECODE_FUSED_GRID
            ):
                fused_samples.append(
                    {"device": device, "duration_us": (end - start) / 1_000.0}
                )
            if name_id != activation_id or grid_x != DECODE_ACTIVATION_GRID:
                continue
            if index + 1 >= len(events):
                continue
            q_start, q_end, q_name_id, q_grid_x = events[index + 1]
            if q_name_id != quant_id or q_grid_x != DECODE_QUANT_GRID:
                continue
            pair_samples.append(
                {
                    "device": device,
                    "activation_us": (end - start) / 1_000.0,
                    "quant_us": (q_end - q_start) / 1_000.0,
                    "overlap_us": max(
                        0, min(end, q_end) - max(start, q_start)
                    )
                    / 1_000.0,
                    "pair_span_us": (max(end, q_end) - min(start, q_start))
                    / 1_000.0,
                    "pair_sum_us": (end - start + q_end - q_start) / 1_000.0,
                }
            )

    graph_by_device: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
    for device, _correlation, start, end, kernel_count, _routed_count in graph_rows:
        graph_by_device[int(device)].append(
            (int(start), int(end), int(kernel_count), int(_correlation))
        )
    replay_counts = {
        device: len(events) for device, events in graph_by_device.items()
    }
    aligned_replays: list[dict[str, float]] = []
    if len(graph_by_device) == 8 and len(set(replay_counts.values())) == 1:
        replay_count = next(iter(replay_counts.values()))
        for ordinal in range(replay_count):
            events = [graph_by_device[device][ordinal] for device in range(8)]
            starts = [event[0] for event in events]
            ends = [event[1] for event in events]
            spans = [event[1] - event[0] for event in events]
            aligned_replays.append(
                {
                    "rank_max_device_span_us": max(spans) / 1_000.0,
                    "rank_median_device_span_us": statistics.median(spans) / 1_000.0,
                    "cross_rank_window_us": (max(ends) - min(starts)) / 1_000.0,
                    "start_skew_us": (max(starts) - min(starts)) / 1_000.0,
                    "end_skew_us": (max(ends) - min(ends)) / 1_000.0,
                }
            )

    result: dict[str, Any] = {
        "sqlite": str(path),
        "contract": {
            "stock_sequence": [ACTIVATION, QUANT],
            "fused_kernel": FUSED,
            "decode_grids": {
                "activation": DECODE_ACTIVATION_GRID,
                "quant": DECODE_QUANT_GRID,
                "fused": DECODE_FUSED_GRID,
            },
        },
        "stock_pairs_per_device": {
            str(device): sum(int(sample["device"]) == device for sample in pair_samples)
            for device in range(8)
        },
        "fused_calls_per_device": {
            str(device): sum(int(sample["device"]) == device for sample in fused_samples)
            for device in range(8)
        },
        "stock_pair": {
            metric: distribution([float(sample[metric]) for sample in pair_samples])
            for metric in (
                "activation_us",
                "quant_us",
                "overlap_us",
                "pair_span_us",
                "pair_sum_us",
            )
        },
        "fused": {
            "duration_us": distribution(
                [float(sample["duration_us"]) for sample in fused_samples]
            )
        },
        "graph_replay": {
            "interpretation_guard": (
                "Independent nsys traces can have different profiler-induced cross-rank "
                "start tails. Treat rank-median device span as supporting evidence and do "
                "not promote from rank-max/cross-rank windows without an unprofiled gate."
            ),
            "replays_per_device": {
                str(device): replay_counts.get(device, 0) for device in range(8)
            },
            "kernel_count_per_rank_replay": distribution(
                [
                    float(event[2])
                    for events in graph_by_device.values()
                    for event in events
                ]
            ),
            **{
                metric: distribution([sample[metric] for sample in aligned_replays])
                for metric in (
                    "rank_max_device_span_us",
                    "rank_median_device_span_us",
                    "cross_rank_window_us",
                    "start_skew_us",
                    "end_skew_us",
                )
            },
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stock", type=Path)
    parser.add_argument("fused", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    stock = analyze(args.stock.resolve())
    fused = analyze(args.fused.resolve())
    stock_span = stock["stock_pair"]["pair_span_us"]
    fused_duration = fused["fused"]["duration_us"]
    if stock_span is None:
        raise SystemExit("stock trace has no M16 shared-expert activation->quant pairs")
    if fused_duration is None:
        raise SystemExit("fused trace has no M16 silu_mul_quant_contig_kernel calls")

    comparison = {
        "stock_pair_median_us": stock_span["median"],
        "fused_median_us": fused_duration["median"],
        "median_speedup": stock_span["median"] / fused_duration["median"],
        "median_saved_us_per_layer": stock_span["median"] - fused_duration["median"],
        "mean_speedup": stock_span["mean"] / fused_duration["mean"],
        "mean_saved_us_per_layer": stock_span["mean"] - fused_duration["mean"],
        "projected_median_saved_us_per_75_sparse_layers": 75
        * (stock_span["median"] - fused_duration["median"]),
        "projected_mean_saved_us_per_75_sparse_layers": 75
        * (stock_span["mean"] - fused_duration["mean"]),
    }
    stock_graph = stock["graph_replay"]
    fused_graph = fused["graph_replay"]
    for metric in (
        "rank_median_device_span_us",
        "rank_max_device_span_us",
        "cross_rank_window_us",
    ):
        stock_metric = stock_graph[metric]
        fused_metric = fused_graph[metric]
        if stock_metric is not None and fused_metric is not None:
            comparison[f"graph_{metric}_stock_median_us"] = stock_metric["median"]
            comparison[f"graph_{metric}_fused_median_us"] = fused_metric["median"]
            comparison[f"graph_{metric}_median_saved_us"] = (
                stock_metric["median"] - fused_metric["median"]
            )
    report = {
        "schema_version": 1,
        "stock": stock,
        "fused": fused,
        "comparison": comparison,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output_json:
        args.output_json.write_text(rendered)


if __name__ == "__main__":
    main()
