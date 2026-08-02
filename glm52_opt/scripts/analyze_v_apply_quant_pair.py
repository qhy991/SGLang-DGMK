#!/usr/bin/env python3
"""Pair a stock V-apply->quant->o_proj nsys trace with a fused trace.

The production stock path launches the BF16 V-apply BMM and the group-128
quantizer as separate CUDA Graph nodes.  The candidate replaces those two
nodes with ``_infini_v_apply_quant_kernel``.  This analyzer deliberately uses
the observed same-stream adjacency and wall-clock boundaries, because the
stock BMM and quantizer overlap through programmatic dependent launch (PDL).
Summing their individual kernel durations would therefore overstate the
removable latency.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


STOCK_BMM = "nvjet_sm103_tst_128x16_64x12_2x1_2cta_v_bz_TNT"
QUANT = "per_token_group_quant_8bit_v2_kernel"
FUSED = "_infini_v_apply_quant_kernel"
O_PROJ = "sm100_fp8_fp4_gemm_1d1d_impl"


def percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize an empty distribution")
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: Iterable[float]) -> dict[str, float]:
    samples = list(values)
    if not samples:
        raise ValueError("cannot summarize an empty distribution")
    return {
        "p10": percentile(samples, 0.10),
        "p50": percentile(samples, 0.50),
        "p90": percentile(samples, 0.90),
        "p99": percentile(samples, 0.99),
        "mean": statistics.fmean(samples),
        "min": min(samples),
        "max": max(samples),
    }


def short_name_id(
    connection: sqlite3.Connection, value: str, *, required: bool = True
) -> int | None:
    row = connection.execute(
        "select id from StringIds where value = ?", (value,)
    ).fetchone()
    if row is None:
        if required:
            raise RuntimeError(f"short CUDA kernel name is absent: {value}")
        return None
    return int(row[0])


def select_decode_graph(connection: sqlite3.Connection) -> int:
    row = connection.execute(
        "select graphId, count(*) as launches "
        "from CUPTI_ACTIVITY_KIND_KERNEL "
        "where graphId is not null "
        "group by graphId order by launches desc limit 1"
    ).fetchone()
    if row is None:
        raise RuntimeError("trace has no CUDA Graph kernels")
    return int(row[0])


def graph_summary(
    connection: sqlite3.Connection, graph_id: int
) -> tuple[dict[str, Any], dict[tuple[int, int], tuple[int, int]]]:
    device_node_counts = list(
        connection.execute(
            "select deviceId, graphNodeId, count(*) "
            "from CUPTI_ACTIVITY_KIND_KERNEL where graphId = ? "
            "group by deviceId, graphNodeId",
            (graph_id,),
        )
    )
    if not device_node_counts:
        raise RuntimeError(f"graph {graph_id} has no kernels")

    devices = sorted({int(row[0]) for row in device_node_counts})
    nodes = {int(row[1]) for row in device_node_counts}
    if len(device_node_counts) != len(devices) * len(nodes):
        raise RuntimeError("not every graph node is present on every device")
    replay_counts_by_device: dict[int, int] = {}
    for device in devices:
        counts = {
            int(row[2]) for row in device_node_counts if int(row[0]) == device
        }
        if len(counts) != 1:
            raise RuntimeError(
                f"device {device} graph nodes have non-uniform replay counts: "
                f"{sorted(counts)}"
            )
        replay_counts_by_device[device] = counts.pop()

    # The nth occurrence of every graph node belongs to the nth replay on that
    # device.  This remains correct when graph streams overlap and avoids using
    # host launch timestamps as guessed replay boundaries.
    spans = list(
        connection.execute(
            "with numbered as ("
            "  select deviceId, graphNodeId, start, end, "
            "         row_number() over ("
            "           partition by deviceId, graphNodeId order by start"
            "         ) as replay_index "
            "  from CUPTI_ACTIVITY_KIND_KERNEL where graphId = ?"
            ") "
            "select deviceId, replay_index, min(start), max(end), count(*) "
            "from numbered group by deviceId, replay_index "
            "order by deviceId, replay_index",
            (graph_id,),
        )
    )
    expected_span_count = sum(replay_counts_by_device.values())
    if len(spans) != expected_span_count:
        raise RuntimeError(
            f"expected {expected_span_count} graph spans, observed {len(spans)}"
        )
    if {int(row[4]) for row in spans} != {len(nodes)}:
        raise RuntimeError("a reconstructed replay does not contain every graph node")

    span_by_device_replay: dict[tuple[int, int], tuple[int, int]] = {}
    duration_by_device: dict[int, list[float]] = defaultdict(list)
    for device, replay_index, start, end, _count in spans:
        key = (int(device), int(replay_index) - 1)
        span_by_device_replay[key] = (int(start), int(end))
        duration_by_device[int(device)].append((int(end) - int(start)) / 1_000.0)

    per_device_p50_us = {
        str(device): percentile(values, 0.50)
        for device, values in sorted(duration_by_device.items())
    }
    aligned_rank_median_us = []
    aligned_rank_max_us = []
    aligned_replays = min(replay_counts_by_device.values())
    for replay_index in range(aligned_replays):
        rank_durations = [
            (
                span_by_device_replay[(device, replay_index)][1]
                - span_by_device_replay[(device, replay_index)][0]
            )
            / 1_000.0
            for device in devices
        ]
        aligned_rank_median_us.append(statistics.median(rank_durations))
        aligned_rank_max_us.append(max(rank_durations))

    total_launches = int(
        connection.execute(
            "select count(*) from CUPTI_ACTIVITY_KIND_KERNEL where graphId = ?",
            (graph_id,),
        ).fetchone()[0]
    )
    summary = {
        "graph_id": graph_id,
        "devices": devices,
        "replays_per_device": {
            str(device): count
            for device, count in sorted(replay_counts_by_device.items())
        },
        "aligned_complete_replays": aligned_replays,
        "total_replays": expected_span_count,
        "nodes_per_replay": len(nodes),
        "total_kernel_launches": total_launches,
        "all_rank_replay_span_us": distribution(
            (end - start) / 1_000.0 for start, end in span_by_device_replay.values()
        ),
        "per_device_p50_us": per_device_p50_us,
        "median_of_device_p50_us": statistics.median(per_device_p50_us.values()),
        "aligned_replay_rank_median_us": distribution(aligned_rank_median_us),
        "aligned_replay_rank_max_us": distribution(aligned_rank_max_us),
    }
    return summary, span_by_device_replay


def kernel_count(
    connection: sqlite3.Connection, graph_id: int, short_name: str
) -> int:
    identifier = short_name_id(connection, short_name, required=False)
    if identifier is None:
        return 0
    return int(
        connection.execute(
            "select count(*) from CUPTI_ACTIVITY_KIND_KERNEL "
            "where graphId = ? and shortName = ?",
            (graph_id, identifier),
        ).fetchone()[0]
    )


def chain_summary(
    connection: sqlite3.Connection, graph_id: int, mode: str
) -> dict[str, Any]:
    names = {
        int(identifier): value
        for identifier, value in connection.execute("select id, value from StringIds")
    }
    target_name = STOCK_BMM if mode == "stock" else FUSED
    target_id = short_name_id(connection, target_name)

    rows = list(
        connection.execute(
            "select deviceId, streamId, start, end, shortName, graphNodeId "
            "from CUPTI_ACTIVITY_KIND_KERNEL where graphId = ? "
            "order by deviceId, streamId, start, end",
            (graph_id,),
        )
    )
    by_stream: dict[tuple[int, int], list[tuple[int, int, int, int]]] = defaultdict(list)
    for device, stream, start, end, name_id, node_id in rows:
        by_stream[(int(device), int(stream))].append(
            (int(start), int(end), int(name_id), int(node_id))
        )

    samples: list[dict[str, float | int]] = []
    rejected_neighbors: dict[str, int] = defaultdict(int)
    for (device, _stream), events in by_stream.items():
        for index, (first_start, first_end, name_id, _node_id) in enumerate(events):
            if name_id != target_id:
                continue
            if mode == "stock":
                if index + 2 >= len(events):
                    rejected_neighbors["truncated"] += 1
                    continue
                second_start, second_end, second_name_id, _ = events[index + 1]
                o_start, o_end, o_name_id, _ = events[index + 2]
                observed = (names.get(second_name_id), names.get(o_name_id))
                expected = (QUANT, O_PROJ)
                if observed != expected:
                    rejected_neighbors[str(observed)] += 1
                    continue
                samples.append(
                    {
                        "device": device,
                        "first_kernel_us": (first_end - first_start) / 1_000.0,
                        "second_kernel_us": (second_end - second_start) / 1_000.0,
                        "first_second_overlap_us": max(
                            0, first_end - second_start
                        )
                        / 1_000.0,
                        "producer_span_us": (max(first_end, second_end) - first_start)
                        / 1_000.0,
                        "o_proj_start_offset_us": (o_start - first_start) / 1_000.0,
                        "o_proj_us": (o_end - o_start) / 1_000.0,
                        "chain_boundary_us": (o_end - first_start) / 1_000.0,
                    }
                )
            else:
                if index + 1 >= len(events):
                    rejected_neighbors["truncated"] += 1
                    continue
                o_start, o_end, o_name_id, _ = events[index + 1]
                if names.get(o_name_id) != O_PROJ:
                    rejected_neighbors[str(names.get(o_name_id))] += 1
                    continue
                samples.append(
                    {
                        "device": device,
                        "first_kernel_us": (first_end - first_start) / 1_000.0,
                        "producer_span_us": (first_end - first_start) / 1_000.0,
                        "o_proj_start_offset_us": (o_start - first_start) / 1_000.0,
                        "o_proj_us": (o_end - o_start) / 1_000.0,
                        "chain_boundary_us": (o_end - first_start) / 1_000.0,
                    }
                )

    if not samples:
        raise RuntimeError(f"no valid {mode} V-apply chains found")
    expected_targets = kernel_count(connection, graph_id, target_name)
    if len(samples) != expected_targets:
        raise RuntimeError(
            f"matched {len(samples)} of {expected_targets} {target_name} launches; "
            f"rejected neighbors: {dict(rejected_neighbors)}"
        )

    metric_names = [name for name in samples[0] if name != "device"]
    return {
        "mode": mode,
        "target_short_name": target_name,
        "samples": len(samples),
        "samples_per_device": {
            str(device): sum(int(sample["device"]) == device for sample in samples)
            for device in sorted({int(sample["device"]) for sample in samples})
        },
        "adjacency_contract": (
            [STOCK_BMM, QUANT, O_PROJ]
            if mode == "stock"
            else [FUSED, O_PROJ]
        ),
        "metrics_us": {
            metric: distribution(float(sample[metric]) for sample in samples)
            for metric in metric_names
        },
    }


def analyze(path: Path, mode: str) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        graph_id = select_decode_graph(connection)
        graph, _spans = graph_summary(connection, graph_id)
        chain = chain_summary(connection, graph_id, mode)
        counts = {
            name: kernel_count(connection, graph_id, name)
            for name in (STOCK_BMM, QUANT, FUSED, O_PROJ)
        }
        return {
            "sqlite": str(path),
            "mode": mode,
            "graph": graph,
            "target_kernel_counts_in_graph": counts,
            "chain": chain,
        }
    finally:
        connection.close()


def reduction(baseline: float, candidate: float) -> dict[str, float]:
    return {
        "baseline_us": baseline,
        "candidate_us": candidate,
        "delta_us": candidate - baseline,
        "reduction_us": baseline - candidate,
        "reduction_pct": 100.0 * (baseline - candidate) / baseline,
        "speedup": baseline / candidate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stock", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    stock = analyze(args.stock.resolve(), "stock")
    candidate = analyze(args.candidate.resolve(), "candidate")
    stock_chain = stock["chain"]["metrics_us"]
    candidate_chain = candidate["chain"]["metrics_us"]
    stock_graph = stock["graph"]
    candidate_graph = candidate["graph"]

    report = {
        "contract": {
            "workload": "GLM-5.2 decode, fixed KV=32768, global BS=128, local M=16, DP8/TP8/EP8",
            "stock_region": [STOCK_BMM, QUANT, O_PROJ],
            "candidate_region": [FUSED, O_PROJ],
            "timing_rule": (
                "Compare observed wall-clock boundaries. Do not add stock BMM and "
                "quant durations because their PDL intervals overlap."
            ),
        },
        "stock": stock,
        "candidate": candidate,
        "paired": {
            "nodes_per_replay_delta": (
                candidate_graph["nodes_per_replay"]
                - stock_graph["nodes_per_replay"]
            ),
            "producer_span_p50": reduction(
                stock_chain["producer_span_us"]["p50"],
                candidate_chain["producer_span_us"]["p50"],
            ),
            "o_proj_start_offset_p50": reduction(
                stock_chain["o_proj_start_offset_us"]["p50"],
                candidate_chain["o_proj_start_offset_us"]["p50"],
            ),
            "chain_boundary_p50": reduction(
                stock_chain["chain_boundary_us"]["p50"],
                candidate_chain["chain_boundary_us"]["p50"],
            ),
            "full_graph_median_of_device_p50": reduction(
                stock_graph["median_of_device_p50_us"],
                candidate_graph["median_of_device_p50_us"],
            ),
            "full_graph_aligned_rank_median_p50": reduction(
                stock_graph["aligned_replay_rank_median_us"]["p50"],
                candidate_graph["aligned_replay_rank_median_us"]["p50"],
            ),
            "full_graph_aligned_rank_max_p50": reduction(
                stock_graph["aligned_replay_rank_max_us"]["p50"],
                candidate_graph["aligned_replay_rank_max_us"]["p50"],
            ),
        },
        "decision_guard": (
            "A fused-kernel duration win is insufficient. Promotion requires the "
            "producer boundary, downstream chain, and full graph to improve beyond "
            "trace jitter, followed by order-balanced unprofiled serving A/B."
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output_json:
        args.output_json.write_text(rendered)


if __name__ == "__main__":
    main()
