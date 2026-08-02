#!/usr/bin/env python3
"""Validate and analyze a fixed-KV baseline/candidate/baseline decode series."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Callable


PROTOCOL = "fixed-kv-decode-series-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-runs", type=int, default=50)
    parser.add_argument("--cache-hit-tolerance", type=float, default=0.001)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260802)
    return parser.parse_args()


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def describe(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p10": percentile(values, 0.10),
        "p90": percentile(values, 0.90),
        "min": min(values),
        "max": max(values),
    }


def bootstrap_ci(
    values: list[float],
    statistic: Callable[[list[float]], float],
    *,
    samples: int,
    seed: int,
) -> dict[str, float | int]:
    rng = random.Random(seed)
    n = len(values)
    estimates = [
        statistic([values[rng.randrange(n)] for _ in range(n)])
        for _ in range(samples)
    ]
    return {
        "samples": samples,
        "point": statistic(values),
        "ci95_low": percentile(estimates, 0.025),
        "ci95_high": percentile(estimates, 0.975),
    }


def load_rows(
    path: Path,
    *,
    expected_label: str,
    expected_runs: int,
    cache_hit_tolerance: float,
) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != expected_runs:
        raise SystemExit(f"{path}: expected {expected_runs} rows, found {len(rows)}")
    for index, row in enumerate(rows, 1):
        if row.get("protocol") != PROTOCOL:
            raise SystemExit(f"{path}:{index}: invalid protocol {row.get('protocol')!r}")
        if row.get("label") != expected_label:
            raise SystemExit(f"{path}:{index}: invalid label {row.get('label')!r}")
        if row.get("is_warmup"):
            raise SystemExit(f"{path}:{index}: warmup leaked into measured rows")
        if row.get("measured_index") != index:
            raise SystemExit(f"{path}:{index}: non-contiguous measured_index")
        input_len = int(row["input_len"])
        prefix_len = int(row["prefix_len"])
        target = prefix_len / float(input_len)
        hit = float(row["cache_hit_rate"])
        if abs(hit - target) > cache_hit_tolerance:
            raise SystemExit(
                f"{path}:{index}: cache_hit_rate={hit:.9f}, target={target:.9f}"
            )
        if not row.get("prompt_set_id"):
            raise SystemExit(f"{path}:{index}: missing prompt_set_id")
    return rows


def itl_ms(row: dict[str, Any]) -> float:
    stored = row.get("itl_ms")
    calculated = (
        (float(row["latency"]) - float(row["last_ttft"]))
        / float(row["output_len"])
        * 1000.0
    )
    if stored is not None and not math.isclose(
        float(stored), calculated, rel_tol=0.0, abs_tol=1e-9
    ):
        raise SystemExit(
            f"stored/calculated ITL mismatch: {float(stored):.12f} != {calculated:.12f}"
        )
    return calculated


def main() -> None:
    args = parse_args()
    if args.expected_runs <= 0 or args.bootstrap_samples <= 0:
        raise SystemExit("expected-runs and bootstrap-samples must be positive")

    labels = ("p1_before", "r2a", "p1_after")
    paths = (args.before, args.candidate, args.after)
    rows = [
        load_rows(
            path,
            expected_label=label,
            expected_runs=args.expected_runs,
            cache_hit_tolerance=args.cache_hit_tolerance,
        )
        for path, label in zip(paths, labels)
    ]

    prompt_ids = [[row["prompt_set_id"] for row in arm] for arm in rows]
    if prompt_ids[1] != prompt_ids[0] or prompt_ids[2] != prompt_ids[0]:
        raise SystemExit("ordered prompt_set_id sequence differs across A-B-A arms")

    contract_keys = (
        "protocol",
        "batch_size",
        "input_len",
        "prefix_len",
        "uncached_tokens_per_request",
        "output_len",
    )
    reference_contract = {key: rows[0][0][key] for key in contract_keys}
    for arm, label in zip(rows, labels):
        for index, row in enumerate(arm, 1):
            observed = {key: row[key] for key in contract_keys}
            if observed != reference_contract:
                raise SystemExit(f"{label}:{index}: workload contract mismatch")

    series = [[itl_ms(row) for row in arm] for arm in rows]
    before, candidate, after = series
    candidate_vs_before = [left - middle for left, middle in zip(before, candidate)]
    candidate_vs_after = [right - middle for middle, right in zip(candidate, after)]
    bracket_effect = [
        (left + right) / 2.0 - middle
        for left, middle, right in zip(before, candidate, after)
    ]
    baseline_drift = [right - left for left, right in zip(before, after)]

    arm_stats = {label: describe(values) for label, values in zip(labels, series)}
    midpoint_of_baseline_medians = (
        float(arm_stats["p1_before"]["median"])
        + float(arm_stats["p1_after"]["median"])
    ) / 2.0
    candidate_median = float(arm_stats["r2a"]["median"])
    reduction = midpoint_of_baseline_medians - candidate_median
    baseline_median_drift = float(arm_stats["p1_after"]["median"]) - float(
        arm_stats["p1_before"]["median"]
    )

    result: dict[str, Any] = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "inputs": {
            label: {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for label, path in zip(labels, paths)
        },
        "contract": reference_contract,
        "validation": {
            "exact_ordered_prompt_sets": len(prompt_ids[0]),
            "ordered_prompt_sets_sha256": hashlib.sha256(
                json.dumps(prompt_ids[0], separators=(",", ":")).encode()
            ).hexdigest(),
            "cache_hit_tolerance": args.cache_hit_tolerance,
            "all_rows_valid": True,
        },
        "arms_itl_ms": arm_stats,
        "point_estimate": {
            "baseline_midpoint_of_medians_itl_ms": midpoint_of_baseline_medians,
            "candidate_median_itl_ms": candidate_median,
            "candidate_reduction_itl_ms": reduction,
            "candidate_reduction_pct": reduction
            / midpoint_of_baseline_medians
            * 100.0,
            "candidate_speedup": midpoint_of_baseline_medians / candidate_median,
            "candidate_lower_than_both_baseline_medians": candidate_median
            < min(
                float(arm_stats["p1_before"]["median"]),
                float(arm_stats["p1_after"]["median"]),
            ),
            "baseline_median_drift_itl_ms": baseline_median_drift,
            "effect_exceeds_abs_baseline_median_drift": reduction
            > abs(baseline_median_drift),
        },
        "paired_distributions_itl_ms": {
            "p1_before_minus_candidate": describe(candidate_vs_before),
            "p1_after_minus_candidate": describe(candidate_vs_after),
            "baseline_midpoint_minus_candidate": describe(bracket_effect),
            "p1_after_minus_p1_before": describe(baseline_drift),
        },
        "paired_bootstrap_ci95_itl_ms": {
            "p1_before_minus_candidate_median": bootstrap_ci(
                candidate_vs_before,
                statistics.median,
                samples=args.bootstrap_samples,
                seed=args.seed + 3,
            ),
            "p1_after_minus_candidate_median": bootstrap_ci(
                candidate_vs_after,
                statistics.median,
                samples=args.bootstrap_samples,
                seed=args.seed + 4,
            ),
            "baseline_midpoint_minus_candidate_mean": bootstrap_ci(
                bracket_effect,
                statistics.mean,
                samples=args.bootstrap_samples,
                seed=args.seed,
            ),
            "baseline_midpoint_minus_candidate_median": bootstrap_ci(
                bracket_effect,
                statistics.median,
                samples=args.bootstrap_samples,
                seed=args.seed + 1,
            ),
            "p1_after_minus_p1_before_median": bootstrap_ci(
                baseline_drift,
                statistics.median,
                samples=args.bootstrap_samples,
                seed=args.seed + 2,
            ),
        },
    }
    median_ci = result["paired_bootstrap_ci95_itl_ms"][
        "baseline_midpoint_minus_candidate_median"
    ]
    before_ci = result["paired_bootstrap_ci95_itl_ms"][
        "p1_before_minus_candidate_median"
    ]
    after_ci = result["paired_bootstrap_ci95_itl_ms"][
        "p1_after_minus_candidate_median"
    ]
    result["decision"] = (
        "pass"
        if result["point_estimate"]["candidate_lower_than_both_baseline_medians"]
        and result["point_estimate"]["effect_exceeds_abs_baseline_median_drift"]
        and before_ci["ci95_low"] > 0.0
        and after_ci["ci95_low"] > 0.0
        and median_ci["ci95_low"] > 0.0
        else "inconclusive"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    point = result["point_estimate"]
    print(
        f"fixed-KV A-B-A {result['decision']}: "
        f"P1 {arm_stats['p1_before']['median']:.6f} -> "
        f"r2a {candidate_median:.6f} -> "
        f"P1 {arm_stats['p1_after']['median']:.6f} ms; "
        f"midpoint reduction={point['candidate_reduction_itl_ms']:+.6f} ms "
        f"({point['candidate_reduction_pct']:+.4f}%), "
        f"paired-median CI95=[{median_ci['ci95_low']:+.6f}, "
        f"{median_ci['ci95_high']:+.6f}] ms"
    )


if __name__ == "__main__":
    main()
