#!/usr/bin/env python3

"""Analyze paired fresh-process FlashMLA GLM-5.2 A/B measurements."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path


METRICS = (
    "eager_main_plus_combine",
    "eager_metadata_plus_main_plus_combine",
    "cuda_graph_main_plus_combine",
)
LABEL = re.compile(r"^(baseline|candidate)-b(16|32)-p([123])$")
NON_TARGET_LABEL = re.compile(
    r"^(baseline|candidate)-nontarget-(h128|topk128)-p([123])$"
)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize empty values")
    return {
        "count": len(values),
        "p10": percentile(values, 0.10),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "mean": statistics.fmean(values),
        "stddev": statistics.pstdev(values),
    }


def load_records(
    directory: Path,
) -> tuple[dict[tuple[str, int, int], dict], dict[tuple[str, str, int], dict]]:
    records: dict[tuple[str, int, int], dict] = {}
    non_target: dict[tuple[str, str, int], dict] = {}
    extension_hashes: dict[str, set[str]] = {"baseline": set(), "candidate": set()}
    for path in sorted(directory.glob("*.json")):
        with path.open() as source:
            record = json.load(source)
        match = LABEL.fullmatch(record.get("label", ""))
        if match is None:
            non_target_match = NON_TARGET_LABEL.fullmatch(record.get("label", ""))
            if non_target_match is None:
                continue
            arm, case_name, pair_text = non_target_match.groups()
            pair = int(pair_text)
            key = (arm, case_name, pair)
            if key in non_target:
                raise ValueError(f"duplicate non-target record for {key}: {path}")
            if record.get("correctness", {}).get("verdict") != "PASS":
                raise ValueError(f"correctness did not pass in {path}")
            if record.get("timings") is None:
                raise ValueError(f"timings missing in {path}")
            if record.get("shape", {}).get("seed") != 20260729 + pair:
                raise ValueError(f"seed mismatch in {path}")
            extension_hash = record.get("environment", {}).get("extension_sha256")
            if not extension_hash:
                raise ValueError(f"extension SHA256 missing in {path}")
            extension_hashes[arm].add(extension_hash)
            non_target[key] = record
            continue
        arm, batch_text, pair_text = match.groups()
        batch_size = int(batch_text)
        pair = int(pair_text)
        key = (arm, batch_size, pair)
        if key in records:
            raise ValueError(f"duplicate A/B record for {key}: {path}")
        if record.get("correctness", {}).get("verdict") != "PASS":
            raise ValueError(f"correctness did not pass in {path}")
        if record.get("timings") is None:
            raise ValueError(f"timings missing in {path}")
        shape = record.get("shape", {})
        if shape.get("batch_size") != batch_size:
            raise ValueError(f"batch mismatch in {path}")
        if shape.get("length_pattern") != "production":
            raise ValueError(f"non-production length pattern in {path}")
        expected_seed = 20260729 + pair
        if shape.get("seed") != expected_seed:
            raise ValueError(
                f"seed mismatch in {path}: {shape.get('seed')} != {expected_seed}"
            )
        if record.get("fixture", {}).get("tensors") is None:
            raise ValueError(f"fixture provenance missing in {path}")
        extension_hash = record.get("environment", {}).get("extension_sha256")
        if not extension_hash:
            raise ValueError(f"extension SHA256 missing in {path}")
        extension_hashes[arm].add(extension_hash)
        records[key] = record

    expected = {
        (arm, batch_size, pair)
        for arm in ("baseline", "candidate")
        for batch_size in (16, 32)
        for pair in (1, 2, 3)
    }
    missing = sorted(expected - records.keys())
    extra = sorted(records.keys() - expected)
    if missing or extra:
        raise ValueError(f"A/B record set mismatch: missing={missing}, extra={extra}")
    for arm, hashes in extension_hashes.items():
        if len(hashes) != 1:
            raise ValueError(f"{arm} used mixed extension hashes: {sorted(hashes)}")
    if extension_hashes["baseline"] == extension_hashes["candidate"]:
        raise ValueError("baseline and candidate extension hashes are identical")
    expected_non_target = {
        (arm, case_name, pair)
        for arm in ("baseline", "candidate")
        for case_name in ("h128", "topk128")
        for pair in (1, 2, 3)
    }
    missing_non_target = sorted(expected_non_target - non_target.keys())
    extra_non_target = sorted(non_target.keys() - expected_non_target)
    if missing_non_target or extra_non_target:
        raise ValueError(
            "non-target record set mismatch: "
            f"missing={missing_non_target}, extra={extra_non_target}"
        )
    return records, non_target


def aggregate(
    records: dict[tuple[str, int, int], dict],
    non_target: dict[tuple[str, str, int], dict],
) -> dict[str, object]:
    result: dict[str, object] = {
        "schema_version": 2,
        "experimental_unit": "paired fresh process",
        "promotion_rule": {
            "all_pair_p50_candidate_over_baseline_max": 0.97,
            "all_pair_p90_candidate_over_baseline_max": 1.01,
            "preferred_all_pair_p50_max": 0.95,
            "required_batches": [16, 32],
            "required_pairs_per_batch": 3,
            "order": {"1": "baseline_then_candidate", "2": "candidate_then_baseline", "3": "baseline_then_candidate"},
        },
        "batches": {},
    }
    all_promotable = True
    all_preferred = True
    any_regression = False
    for batch_size in (16, 32):
        batch_result: dict[str, object] = {}
        for metric in METRICS:
            pooled: dict[str, dict[str, float | int]] = {}
            for arm in ("baseline", "candidate"):
                raw = [
                    float(value)
                    for pair in (1, 2, 3)
                    for value in records[(arm, batch_size, pair)]["timings"][metric][
                        "raw"
                    ]
                ]
                if len(raw) != 3000:
                    raise ValueError(
                        f"{arm} b{batch_size} {metric} expected 3000 raw samples, "
                        f"got {len(raw)}"
                    )
                pooled[arm] = summary(raw)

            pairs = []
            p50_ratios = []
            p90_ratios = []
            for pair in (1, 2, 3):
                baseline_record = records[("baseline", batch_size, pair)]
                candidate_record = records[("candidate", batch_size, pair)]
                baseline_fixture = baseline_record["fixture"]["tensors"]
                candidate_fixture = candidate_record["fixture"]["tensors"]
                for tensor_name in (
                    "q",
                    "packed_kv",
                    "indices",
                    "cache_seqlens",
                    "metadata",
                    "num_splits",
                ):
                    if (
                        baseline_fixture[tensor_name]["sha256"]
                        != candidate_fixture[tensor_name]["sha256"]
                    ):
                        raise ValueError(
                            f"pair {pair} b{batch_size} input mismatch: {tensor_name}"
                        )
                baseline_timing = baseline_record["timings"][metric]
                candidate_timing = candidate_record["timings"][metric]
                if baseline_timing["count"] != 1000 or candidate_timing["count"] != 1000:
                    raise ValueError(
                        f"pair {pair} b{batch_size} {metric} must have 1000 samples"
                    )
                p50_ratio = candidate_timing["p50"] / baseline_timing["p50"]
                p90_ratio = candidate_timing["p90"] / baseline_timing["p90"]
                p50_ratios.append(p50_ratio)
                p90_ratios.append(p90_ratio)
                pairs.append(
                    {
                        "pair": pair,
                        "order": (
                            "baseline_then_candidate"
                            if pair in (1, 3)
                            else "candidate_then_baseline"
                        ),
                        "baseline_label": baseline_record["label"],
                        "candidate_label": candidate_record["label"],
                        "baseline_extension_sha256": baseline_record["environment"][
                            "extension_sha256"
                        ],
                        "candidate_extension_sha256": candidate_record["environment"][
                            "extension_sha256"
                        ],
                        "baseline_p50_us": baseline_timing["p50"],
                        "candidate_p50_us": candidate_timing["p50"],
                        "candidate_over_baseline_p50": p50_ratio,
                        "baseline_p90_us": baseline_timing["p90"],
                        "candidate_p90_us": candidate_timing["p90"],
                        "candidate_over_baseline_p90": p90_ratio,
                    }
                )

            promotable = max(p50_ratios) <= 0.97 and max(p90_ratios) <= 1.01
            preferred = max(p50_ratios) <= 0.95 and max(p90_ratios) <= 1.01
            regression = max(p50_ratios) > 1.01 or max(p90_ratios) > 1.01
            all_promotable &= promotable
            all_preferred &= preferred
            any_regression |= regression
            batch_result[metric] = {
                "pooled_descriptive_only": pooled,
                "pairs": pairs,
                "paired_p50_ratios": summary(p50_ratios),
                "paired_p90_ratios": summary(p90_ratios),
                "all_pair_p50_max": max(p50_ratios),
                "all_pair_p90_max": max(p90_ratios),
                "promotion_gate": promotable,
                "preferred_5_percent_gate": preferred,
                "regression": regression,
            }
        result["batches"][str(batch_size)] = batch_result
    non_target_result: dict[str, object] = {}
    all_non_target_stable = True
    for case_name in ("h128", "topk128"):
        case_result: dict[str, object] = {}
        for metric in METRICS:
            pairs = []
            p50_ratios = []
            p90_ratios = []
            for pair in (1, 2, 3):
                baseline_record = non_target[("baseline", case_name, pair)]
                candidate_record = non_target[("candidate", case_name, pair)]
                for tensor_name in (
                    "q",
                    "packed_kv",
                    "indices",
                    "cache_seqlens",
                    "metadata",
                    "num_splits",
                ):
                    baseline_sha = baseline_record["fixture"]["tensors"][tensor_name][
                        "sha256"
                    ]
                    candidate_sha = candidate_record["fixture"]["tensors"][tensor_name][
                        "sha256"
                    ]
                    if baseline_sha != candidate_sha:
                        raise ValueError(
                            f"non-target {case_name} pair {pair} mismatch: {tensor_name}"
                        )
                baseline_timing = baseline_record["timings"][metric]
                candidate_timing = candidate_record["timings"][metric]
                if baseline_timing["count"] != 1000 or candidate_timing["count"] != 1000:
                    raise ValueError(
                        f"non-target {case_name} pair {pair} must have 1000 samples"
                    )
                p50_ratio = candidate_timing["p50"] / baseline_timing["p50"]
                p90_ratio = candidate_timing["p90"] / baseline_timing["p90"]
                p50_ratios.append(p50_ratio)
                p90_ratios.append(p90_ratio)
                pairs.append(
                    {
                        "pair": pair,
                        "order": (
                            "baseline_then_candidate"
                            if pair in (1, 3)
                            else "candidate_then_baseline"
                        ),
                        "candidate_over_baseline_p50": p50_ratio,
                        "candidate_over_baseline_p90": p90_ratio,
                    }
                )
            stable = max(p50_ratios) <= 1.01 and max(p90_ratios) <= 1.01
            all_non_target_stable &= stable
            any_regression |= not stable
            case_result[metric] = {
                "pairs": pairs,
                "paired_p50_ratios": summary(p50_ratios),
                "paired_p90_ratios": summary(p90_ratios),
                "all_pair_p50_max": max(p50_ratios),
                "all_pair_p90_max": max(p90_ratios),
                "no_material_regression_gate": stable,
            }
        non_target_result[case_name] = case_result
    result["non_target"] = non_target_result
    result["non_target_gate"] = all_non_target_stable
    all_promotable &= all_non_target_stable
    all_preferred &= all_non_target_stable
    result["preferred_5_percent"] = all_preferred
    if all_promotable:
        result["verdict"] = "WIN"
    elif any_regression:
        result["verdict"] = "REGRESS"
    else:
        result["verdict"] = "FLAT"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records, non_target = load_records(args.benchmark_directory)
    result = aggregate(records, non_target)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
