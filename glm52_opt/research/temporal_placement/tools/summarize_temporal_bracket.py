#!/usr/bin/env python3
"""Summarize a frozen control-candidate-control temporal-placement bracket."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def improvement(control: float, candidate: float, higher_is_better: bool) -> float:
    sign = 1 if higher_is_better else -1
    return sign * 100.0 * (candidate - control) / control


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("samples", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--candidate-name", default="glm52-temporal-critical-rank-placement-v1"
    )
    args = parser.parse_args()
    rows = list(csv.DictReader(args.samples.open(), delimiter="\t"))
    if [row["arm"] for row in rows] != ["control", "candidate", "control"]:
        raise RuntimeError(f"unexpected bracket order: {[row['arm'] for row in rows]}")
    if any(int(row["successful_requests"]) != 110 for row in rows):
        raise RuntimeError("not every bracket arm completed 110 requests")

    controls = [rows[0], rows[2]]
    candidate = rows[1]
    fields = {
        "median_ttft_ms": False,
        "p90_ttft_ms": False,
        "total_token_throughput": True,
    }
    control_median = {
        field: statistics.median(float(row[field]) for row in controls)
        for field in fields
    }
    candidate_values = {field: float(candidate[field]) for field in fields}
    improvements = {
        field: improvement(control_median[field], candidate_values[field], higher)
        for field, higher in fields.items()
    }
    drift = {
        field: 100.0
        * abs(float(controls[1][field]) - float(controls[0][field]))
        / control_median[field]
        for field in fields
    }
    relative_gates = {
        "median_ttft_improves_ge_1pct": improvements["median_ttft_ms"] >= 1.0,
        "p90_ttft_nonregression_1pct": improvements["p90_ttft_ms"] >= -1.0,
        "throughput_improves_ge_1pct": improvements["total_token_throughput"] >= 1.0,
        "control_median_drift_le_15pct": drift["median_ttft_ms"] <= 15.0,
        "control_p90_drift_le_15pct": drift["p90_ttft_ms"] <= 15.0,
        "control_throughput_drift_le_10pct": drift["total_token_throughput"] <= 10.0,
    }
    absolute_gates = {
        "candidate_median_ttft_le_2000ms": candidate_values["median_ttft_ms"] <= 2000,
        "candidate_p90_ttft_le_5000ms": candidate_values["p90_ttft_ms"] <= 5000,
        "candidate_throughput_ge_438000": candidate_values["total_token_throughput"] >= 438000,
    }
    relative_pass = all(relative_gates.values())
    absolute_pass = all(absolute_gates.values())
    if relative_pass and absolute_pass:
        status = "ADMITTED_FOR_PAIRED_SCREEN"
    elif relative_pass:
        status = "RELATIVE_WIN_BLOCKED_BY_HOST_ANCHOR"
    else:
        status = "REJECTED_DEVELOPMENT"

    result = {
        "status": status,
        "design": "fresh-server control-candidate-control",
        "candidate": args.candidate_name,
        "control": "accepted-N6-static-placement",
        "control_median": control_median,
        "candidate_measured": candidate_values,
        "improvement_percent": improvements,
        "control_anchor_drift_percent": drift,
        "relative_gates": relative_gates,
        "absolute_gates": absolute_gates,
        "samples": rows,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
