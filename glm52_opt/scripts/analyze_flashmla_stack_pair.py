#!/usr/bin/env python3
"""Pair production P1+c2 and r2a+c2 FlashMLA decode nsys traces.

Both stacks launch the main sparse-decode kernel followed by the same c2 combine
kernel.  The combine uses PDL and can begin before the main kernel ends, so the
removable latency is the observed main-to-combine wall-clock span rather than the
sum of the two kernel durations.  The next V-apply BMM is included as a stable
downstream boundary.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from analyze_v_apply_quant_pair import (
    distribution,
    graph_summary,
    kernel_count,
    reduction,
    select_decode_graph,
    short_name_id,
)


P1_MAIN = "infini_kernel_glm52_flashmla_sparse_decode_p1_consumer_scale_main"
R2A_MAIN = "infini_kernel_glm52_flashmla_sparse_decode_r2a_prologue_overlap_main"
COMBINE = "infini_kernel_glm52_flashmla_sparse_decode_combine_c2_bucket_stages"
NEXT_V_BMM = "nvjet_sm103_tst_128x16_64x12_2x1_2cta_v_bz_TNT"


def launch_signature(
    connection: sqlite3.Connection, graph_id: int, short_name: str
) -> dict[str, Any]:
    identifier = short_name_id(connection, short_name)
    rows = list(
        connection.execute(
            "select distinct registersPerThread, gridX, gridY, gridZ, "
            "blockX, blockY, blockZ, staticSharedMemory, dynamicSharedMemory "
            "from CUPTI_ACTIVITY_KIND_KERNEL where graphId = ? and shortName = ?",
            (graph_id, identifier),
        )
    )
    if len(rows) != 1:
        raise RuntimeError(
            f"expected one launch signature for {short_name}, observed {rows}"
        )
    row = rows[0]
    return {
        "registers_per_thread": int(row[0]),
        "grid": [int(row[1]), int(row[2]), int(row[3])],
        "block": [int(row[4]), int(row[5]), int(row[6])],
        "static_shared_memory_bytes": int(row[7]),
        "dynamic_shared_memory_bytes": int(row[8]),
    }


def stack_summary(
    connection: sqlite3.Connection, graph_id: int, mode: str
) -> dict[str, Any]:
    target_name = P1_MAIN if mode == "stock" else R2A_MAIN
    target_id = short_name_id(connection, target_name)
    names = {
        int(identifier): value
        for identifier, value in connection.execute("select id, value from StringIds")
    }
    rows = list(
        connection.execute(
            "select deviceId, streamId, start, end, shortName, graphNodeId "
            "from CUPTI_ACTIVITY_KIND_KERNEL where graphId = ? "
            "order by deviceId, streamId, start, end",
            (graph_id,),
        )
    )
    by_stream: dict[tuple[int, int], list[tuple[int, int, int, int]]] = defaultdict(
        list
    )
    for device, stream, start, end, name_id, node_id in rows:
        by_stream[(int(device), int(stream))].append(
            (int(start), int(end), int(name_id), int(node_id))
        )

    samples: list[dict[str, float | int]] = []
    rejected: dict[str, int] = defaultdict(int)
    for (device, _stream), events in by_stream.items():
        for index, (main_start, main_end, name_id, _node_id) in enumerate(events):
            if name_id != target_id:
                continue
            if index + 2 >= len(events):
                rejected["truncated"] += 1
                continue
            combine_start, combine_end, combine_id, _ = events[index + 1]
            next_start, next_end, next_id, _ = events[index + 2]
            observed = (names.get(combine_id), names.get(next_id))
            if observed != (COMBINE, NEXT_V_BMM):
                rejected[str(observed)] += 1
                continue
            samples.append(
                {
                    "device": device,
                    "main_kernel_us": (main_end - main_start) / 1_000.0,
                    "combine_start_offset_us": (combine_start - main_start)
                    / 1_000.0,
                    "combine_kernel_us": (combine_end - combine_start) / 1_000.0,
                    "main_combine_overlap_us": max(0, main_end - combine_start)
                    / 1_000.0,
                    "stack_boundary_us": (max(main_end, combine_end) - main_start)
                    / 1_000.0,
                    "next_v_bmm_start_offset_us": (next_start - main_start)
                    / 1_000.0,
                    "through_next_v_bmm_us": (next_end - main_start) / 1_000.0,
                }
            )

    expected = kernel_count(connection, graph_id, target_name)
    if not samples or len(samples) != expected:
        raise RuntimeError(
            f"matched {len(samples)} of {expected} {target_name} launches; "
            f"rejected neighbors: {dict(rejected)}"
        )
    metrics = [key for key in samples[0] if key != "device"]
    return {
        "mode": mode,
        "main_short_name": target_name,
        "samples": len(samples),
        "samples_per_device": {
            str(device): sum(int(sample["device"]) == device for sample in samples)
            for device in sorted({int(sample["device"]) for sample in samples})
        },
        "adjacency_contract": [target_name, COMBINE, NEXT_V_BMM],
        "main_launch": launch_signature(connection, graph_id, target_name),
        "combine_launch": launch_signature(connection, graph_id, COMBINE),
        "metrics_us": {
            metric: distribution(float(sample[metric]) for sample in samples)
            for metric in metrics
        },
    }


def analyze(path: Path, mode: str) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        graph_id = select_decode_graph(connection)
        graph, _spans = graph_summary(connection, graph_id)
        stack = stack_summary(connection, graph_id, mode)
        counts = {
            name: kernel_count(connection, graph_id, name)
            for name in (P1_MAIN, R2A_MAIN, COMBINE, NEXT_V_BMM)
        }
        return {
            "sqlite": str(path),
            "mode": mode,
            "graph": graph,
            "target_kernel_counts_in_graph": counts,
            "stack": stack,
        }
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stock", type=Path, help="P1+c2 production nsys SQLite")
    parser.add_argument("candidate", type=Path, help="r2a+c2 production nsys SQLite")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    stock = analyze(args.stock.resolve(), "stock")
    candidate = analyze(args.candidate.resolve(), "candidate")
    stock_metrics = stock["stack"]["metrics_us"]
    candidate_metrics = candidate["stack"]["metrics_us"]
    stock_graph = stock["graph"]
    candidate_graph = candidate["graph"]
    report = {
        "contract": {
            "workload": (
                "GLM-5.2 decode, fixed KV=32768, global BS=128, local M=16, "
                "DP8/TP8/EP8"
            ),
            "stock_stack": [P1_MAIN, COMBINE],
            "candidate_stack": [R2A_MAIN, COMBINE],
            "downstream_boundary": NEXT_V_BMM,
            "timing_rule": (
                "Compare observed main-to-combine and downstream wall-clock "
                "boundaries; main and combine overlap through PDL."
            ),
        },
        "stock": stock,
        "candidate": candidate,
        "paired": {
            "nodes_per_replay_delta": (
                candidate_graph["nodes_per_replay"]
                - stock_graph["nodes_per_replay"]
            ),
            "main_kernel_p50": reduction(
                stock_metrics["main_kernel_us"]["p50"],
                candidate_metrics["main_kernel_us"]["p50"],
            ),
            "combine_start_offset_p50": reduction(
                stock_metrics["combine_start_offset_us"]["p50"],
                candidate_metrics["combine_start_offset_us"]["p50"],
            ),
            "stack_boundary_p50": reduction(
                stock_metrics["stack_boundary_us"]["p50"],
                candidate_metrics["stack_boundary_us"]["p50"],
            ),
            "next_v_bmm_start_offset_p50": reduction(
                stock_metrics["next_v_bmm_start_offset_us"]["p50"],
                candidate_metrics["next_v_bmm_start_offset_us"]["p50"],
            ),
            "through_next_v_bmm_p50": reduction(
                stock_metrics["through_next_v_bmm_us"]["p50"],
                candidate_metrics["through_next_v_bmm_us"]["p50"],
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
            "A main-kernel win is insufficient. Promotion requires the overlapped "
            "stack and downstream boundary to improve in an otherwise identical "
            "production graph, then an order-balanced unprofiled serving A/B."
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output_json:
        args.output_json.write_text(rendered)


if __name__ == "__main__":
    main()
