#!/usr/bin/env python3
"""Aggregate concatenated fixed-N/K benchmark JSON objects from stdin."""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict


def _objects(text: str) -> list[dict]:
    decoder = json.JSONDecoder()
    objects = []
    marker = '{\n  "device"'
    offset = 0
    while True:
        start = text.find(marker, offset)
        if start < 0:
            break
        value, consumed = decoder.raw_decode(text[start:])
        objects.append(value)
        offset = start + consumed
    return objects


def main() -> None:
    runs = _objects(sys.stdin.read())
    if not runs:
        raise RuntimeError("no benchmark JSON objects found on stdin")
    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for run in runs:
        if run.get("schema") != "glm52-diagnostic-fixed-nk-v1":
            raise RuntimeError(f"unexpected schema {run.get('schema')!r}")
        for result in run["results"]:
            key = (result["op"], result["phase"], int(result["m"]))
            grouped[key].append(result)

    rows = []
    timing_keys = (
        "eager_stock_us",
        "eager_candidate_us",
        "eager_speedup",
        "graph_stock_us",
        "graph_candidate_us",
        "graph_speedup",
    )
    for (op, phase, m), values in sorted(grouped.items()):
        first = values[0]
        row = {
            "op": op,
            "phase": phase,
            "m": m,
            "n": first["n"],
            "k": first["k"],
            "process_count": len(values),
            "all_exact": all(value["exact"] for value in values),
            "all_graph_exact": all(value["graph_exact"] for value in values),
            "min_hit_count": min(value["hit_count"] for value in values),
        }
        for key in timing_keys:
            samples = [float(value[key]) for value in values]
            row[f"{key}_median"] = float(statistics.median(samples))
            if key.endswith("speedup"):
                row[f"{key}_min"] = min(samples)
                row[f"{key}_max"] = max(samples)
        rows.append(row)

    print(
        json.dumps(
            {
                "schema": "glm52-diagnostic-fixed-nk-aggregate-v1",
                "independent_processes": len(runs),
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
