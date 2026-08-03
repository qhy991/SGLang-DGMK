#!/usr/bin/env python3
"""Compare W13 -> masked SwiGLU+quant -> W2 regions in nsys SQLite traces."""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ACTIVATION_SHORT_NAMES = (
    "silu_mul_quant_varlen_kernel",
    "task25_silu_mul_quant_grid_stride_kernel",
)
DEEPGEMM_SHORT_NAME = "sm100_fp8_fp4_gemm_1d1d_impl"


def nearest_percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * percentile)]


def distribution(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty distribution")
    return {
        "p10": nearest_percentile(values, 0.10),
        "median": statistics.median(values),
        "p90": nearest_percentile(values, 0.90),
        "mean": statistics.fmean(values),
    }


def short_name_id(connection: sqlite3.Connection, value: str) -> int:
    row = connection.execute(
        "select id from StringIds where value = ?", (value,)
    ).fetchone()
    if row is None:
        raise RuntimeError(f"short CUDA kernel name is absent: {value}")
    return int(row[0])


def first_short_name_id(
    connection: sqlite3.Connection, values: tuple[str, ...]
) -> tuple[str, int]:
    for value in values:
        row = connection.execute(
            "select id from StringIds where value = ?", (value,)
        ).fetchone()
        if row is not None:
            return value, int(row[0])
    raise RuntimeError(
        "none of the activation CUDA kernel short names are present: "
        + ", ".join(values)
    )


def analyze(path: Path, max_neighbor_gap_us: float) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        activation_short_name, activation_id = first_short_name_id(
            connection, ACTIVATION_SHORT_NAMES
        )
        deepgemm_id = short_name_id(connection, DEEPGEMM_SHORT_NAME)
        bounds = connection.execute(
            "select min(start), max(end) from CUPTI_ACTIVITY_KIND_KERNEL "
            "where shortName = ?",
            (activation_id,),
        ).fetchone()
        if bounds is None or bounds[0] is None or bounds[1] is None:
            raise RuntimeError(f"no {activation_short_name} launches in {path}")
        lower = max(int(bounds[0]) - 1_000_000, 0)
        upper = int(bounds[1]) + 1_000_000
        rows = list(
            connection.execute(
                "select deviceId, streamId, start, end, shortName, gridX "
                "from CUPTI_ACTIVITY_KIND_KERNEL "
                "where start < ? and end > ? and shortName in (?, ?) "
                "order by deviceId, streamId, start",
                (upper, lower, activation_id, deepgemm_id),
            )
        )
    finally:
        connection.close()

    by_stream: dict[tuple[int, int], list[tuple[int, int, int, int]]] = defaultdict(list)
    for device, stream, start, end, name_id, grid_x in rows:
        by_stream[(int(device), int(stream))].append(
            (int(start), int(end), int(name_id), int(grid_x))
        )

    max_gap_ns = int(max_neighbor_gap_us * 1_000)
    samples: list[dict[str, float | int]] = []
    for (device, _stream), events in by_stream.items():
        for index, (start, end, name_id, activation_grid) in enumerate(events):
            if name_id != activation_id:
                continue
            previous = next(
                (events[i] for i in range(index - 1, -1, -1) if events[i][2] == deepgemm_id),
                None,
            )
            following = next(
                (events[i] for i in range(index + 1, len(events)) if events[i][2] == deepgemm_id),
                None,
            )
            if previous is None or following is None:
                continue
            w13_start, w13_end, _, w13_grid = previous
            w2_start, w2_end, _, w2_grid = following
            if start - w13_start > max_gap_ns or w2_start - end > max_gap_ns:
                continue
            samples.append(
                {
                    "device": device,
                    "activation_us": (end - start) / 1_000,
                    "w13_us": (w13_end - w13_start) / 1_000,
                    "w2_us": (w2_end - w2_start) / 1_000,
                    "w13_activation_overlap_us": max(0, w13_end - start) / 1_000,
                    "activation_to_w2_gap_us": (w2_start - end) / 1_000,
                    "region_us": (w2_end - w13_start) / 1_000,
                    # PDL can overlap the start of activation with W13.  This is
                    # the activation interval that is actually on region critical path.
                    "activation_removable_us": max(0, w2_start - w13_end) / 1_000,
                    "activation_grid": activation_grid,
                    "w13_grid": w13_grid,
                    "w2_grid": w2_grid,
                }
            )

    if not samples:
        raise RuntimeError(f"no adjacent W13/activation/W2 triples found in {path}")

    metrics = (
        "activation_us",
        "w13_us",
        "w2_us",
        "w13_activation_overlap_us",
        "activation_to_w2_gap_us",
        "region_us",
        "activation_removable_us",
    )
    result: dict[str, Any] = {
        "sqlite": str(path),
        "activation_short_name": activation_short_name,
        "activation_bounds_s": [int(bounds[0]) / 1e9, int(bounds[1]) / 1e9],
        "triples": len(samples),
        "per_device_triples": {
            str(device): sum(int(sample["device"]) == device for sample in samples)
            for device in range(8)
        },
        "grids": {
            key: sorted({int(sample[key]) for sample in samples})
            for key in ("activation_grid", "w13_grid", "w2_grid")
        },
        "grid_counts": {
            key: {
                str(grid): sum(int(sample[key]) == grid for sample in samples)
                for grid in sorted({int(sample[key]) for sample in samples})
            }
            for key in ("activation_grid", "w13_grid", "w2_grid")
        },
        "activation_grid_counts_by_device": {
            str(device): {
                str(grid): sum(
                    int(sample["device"]) == device
                    and int(sample["activation_grid"]) == grid
                    for sample in samples
                )
                for grid in sorted(
                    {int(sample["activation_grid"]) for sample in samples}
                )
            }
            for device in range(8)
        },
    }
    for metric in metrics:
        result[metric] = distribution([float(sample[metric]) for sample in samples])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", nargs="+", type=Path, help="baseline first, then candidates")
    parser.add_argument("--max-neighbor-gap-us", type=float, default=500.0)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    traces = [analyze(path.resolve(), args.max_neighbor_gap_us) for path in args.sqlite]
    report: dict[str, Any] = {"traces": traces, "comparisons": []}
    baseline = traces[0]
    for candidate in traces[1:]:
        baseline_region = float(baseline["region_us"]["median"])
        candidate_region = float(candidate["region_us"]["median"])
        baseline_activation = float(baseline["activation_us"]["median"])
        candidate_activation = float(candidate["activation_us"]["median"])
        report["comparisons"].append(
            {
                "baseline": baseline["sqlite"],
                "candidate": candidate["sqlite"],
                "activation_median_speedup": baseline_activation / candidate_activation,
                "region_median_speedup": baseline_region / candidate_region,
                "region_median_reduction_pct": 100
                * (baseline_region - candidate_region)
                / baseline_region,
            }
        )

    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output_json:
        args.output_json.write_text(rendered)


if __name__ == "__main__":
    main()
