#!/usr/bin/env python3
"""Fail-closed five-pair summary for CP8 FlashMLA metadata reuse."""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

from summarize_cp8_metadata_reuse_screen import load_arm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lines = args.manifest.read_text().splitlines()
    if len(lines) != 11:
        raise RuntimeError(f"expected header plus ten arms, got {len(lines)} lines")
    header = lines[0].split("\t")
    expected_header = [
        "label",
        "arm",
        "launcher_log",
        "result_json",
        "correctness_json",
        "server_log",
    ]
    if header != expected_header:
        raise RuntimeError(f"manifest header drift: {header}")
    records = []
    for line in lines[1:]:
        values = line.split("\t")
        if len(values) != len(header):
            raise RuntimeError(f"malformed manifest row: {line}")
        records.append(load_arm(dict(zip(header, values, strict=True))))

    grouped: dict[str, list[dict]] = collections.defaultdict(list)
    for record in records:
        grouped[record["label"][:2]].append(record)
    if sorted(grouped) != ["p1", "p2", "p3", "p4", "p5"]:
        raise RuntimeError(f"unexpected pair labels: {sorted(grouped)}")

    canonical_input = records[0]["result"]["workload_contract"]["input_ids_sha256"]
    canonical_sentinel = records[0]["result"]["workload_contract"][
        "sentinel_ids_sha256"
    ]
    canonical_output = records[0]["result"]["output_token_ids"]
    canonical_probe_input = records[0]["correctness"]["input_ids_sha256"]
    canonical_probe_output = records[0]["correctness"]["output_token_ids"]
    pairs = []
    for pair_id in sorted(grouped):
        pair_records = grouped[pair_id]
        if len(pair_records) != 2 or {row["arm"] for row in pair_records} != {
            "baseline",
            "candidate",
        }:
            raise RuntimeError(f"pair {pair_id} must contain one arm each")
        for record in pair_records:
            result = record["result"]
            correctness = record["correctness"]
            if (
                result["workload_contract"]["input_ids_sha256"] != canonical_input
                or result["workload_contract"]["sentinel_ids_sha256"]
                != canonical_sentinel
                or result["output_token_ids"] != canonical_output
                or correctness["input_ids_sha256"] != canonical_probe_input
                or correctness["output_token_ids"] != canonical_probe_output
            ):
                raise RuntimeError(f"cross-arm exact contract drift: {record['label']}")
        baseline = next(row for row in pair_records if row["arm"] == "baseline")
        candidate = next(row for row in pair_records if row["arm"] == "candidate")
        bm = baseline["result"]["measured_metrics"]
        cm = candidate["result"]["measured_metrics"]
        p50 = (bm["median_ttft_ms"] - cm["median_ttft_ms"]) / bm[
            "median_ttft_ms"
        ]
        p90 = (bm["p90_ttft_ms"] - cm["p90_ttft_ms"]) / bm["p90_ttft_ms"]
        logprob_abs = [
            abs(a - b)
            for a, b in zip(
                baseline["correctness"]["output_token_logprobs"],
                candidate["correctness"]["output_token_logprobs"],
                strict=True,
            )
        ]
        max_abs = max(logprob_abs)
        mean_abs = statistics.fmean(logprob_abs)
        if max_abs > 1e-3 or mean_abs > 1e-4:
            raise RuntimeError(
                f"pair {pair_id} logprob gate failed: max={max_abs} mean={mean_abs}"
            )
        pairs.append(
            {
                "pair": pair_id,
                "order": "".join(
                    "B" if row["arm"] == "baseline" else "C"
                    for row in pair_records
                ),
                "baseline_result": baseline["result_json"],
                "candidate_result": candidate["result_json"],
                "baseline_p50_ms": bm["median_ttft_ms"],
                "candidate_p50_ms": cm["median_ttft_ms"],
                "p50_reduction_fraction": p50,
                "baseline_p90_ms": bm["p90_ttft_ms"],
                "candidate_p90_ms": cm["p90_ttft_ms"],
                "p90_reduction_fraction": p90,
                "selected_token_logprob_max_abs": max_abs,
                "selected_token_logprob_mean_abs": mean_abs,
                "candidate_metadata_hits": candidate["metadata_hits"],
                "candidate_selected_ranks": candidate["selected_ranks"],
                "baseline_server_log_sha256": baseline["server_log_sha256"],
                "candidate_server_log_sha256": candidate["server_log_sha256"],
            }
        )

    p50_values = [row["p50_reduction_fraction"] for row in pairs]
    p90_values = [row["p90_reduction_fraction"] for row in pairs]
    summary = {
        "p50_wins": sum(value > 0 for value in p50_values),
        "p90_non_regressions": sum(value >= 0 for value in p90_values),
        "paired_median_p50_reduction_fraction": statistics.median(p50_values),
        "paired_median_p90_reduction_fraction": statistics.median(p90_values),
        "max_selected_token_logprob_abs": max(
            row["selected_token_logprob_max_abs"] for row in pairs
        ),
        "max_pair_mean_selected_token_logprob_abs": max(
            row["selected_token_logprob_mean_abs"] for row in pairs
        ),
    }
    gate = {
        "p50_wins_min": 4,
        "p90_non_regressions_min": 4,
        "paired_median_p50_reduction_fraction_min": 0.01,
        "paired_median_p90_reduction_fraction_min": 0.0,
        "selected_token_logprob_max_abs_max": 0.001,
        "selected_token_logprob_mean_abs_max": 0.0001,
    }
    promoted = (
        summary["p50_wins"] >= gate["p50_wins_min"]
        and summary["p90_non_regressions"] >= gate["p90_non_regressions_min"]
        and summary["paired_median_p50_reduction_fraction"]
        >= gate["paired_median_p50_reduction_fraction_min"]
        and summary["paired_median_p90_reduction_fraction"]
        >= gate["paired_median_p90_reduction_fraction_min"]
    )
    out = {
        "schema": "glm52-cp8-metadata-reuse-five-pairs-v1",
        "status": "PASS",
        "scope": "frozen no-profiler exact-input CP8/EP8 five-pair endpoint screen",
        "input_ids_sha256": canonical_input,
        "sentinel_ids_sha256": canonical_sentinel,
        "output_token_ids_sha256": __import__("hashlib")
        .sha256(json.dumps(canonical_output).encode())
        .hexdigest(),
        "pairs": pairs,
        "summary": summary,
        "promotion_gate": gate,
        "decision": "ADVANCE_TO_MATCHED_NSYS" if promoted else "STOP_KEEP_C01",
        "limitations": "Only the frozen eager 100K cached-prefill CP8/EP8 cell is covered.",
    }
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
