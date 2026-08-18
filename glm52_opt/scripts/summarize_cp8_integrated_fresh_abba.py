#!/usr/bin/env python3
"""Fail-closed summary for five fresh-server control/integrated CP8 pairs."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import statistics
from pathlib import Path


FATAL = (
    "Traceback (most recent call last)",
    "illegal memory access",
    "CUDA out of memory",
    "torch.OutOfMemoryError",
    "KV cache pool is full",
)
COMBINED_HIT = "e2e_prefill/cp8_combined_indexer_halves:index_score:prefill:m1252"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_result(path: Path, arm: str) -> dict:
    row = json.loads(path.read_text())
    if row.get("status") != "PASS":
        raise RuntimeError(f"{arm} result is not PASS: {path}")
    if row["parallelism_contract"] != {
        "tp": 8,
        "dp": 1,
        "attention_cp": 8,
        "attention_tp": 1,
        "ep": 8,
        "request_concurrency": 11,
    }:
        raise RuntimeError(f"{arm} parallelism drift: {path}")
    workload = row["workload_contract"]
    expected = {
        "requests": 110,
        "logical_prefix_tokens": 90_000,
        "page_aligned_cached_tokens": 89_984,
        "suffix_tokens": 10_000,
        "output_tokens": 1,
        "scheduled_tokens_per_cp_rank": 10_048,
        "unique_suffix_first_token": True,
    }
    if {key: workload.get(key) for key in expected} != expected:
        raise RuntimeError(f"{arm} workload drift: {path}")
    evidence = row["server_evidence"]
    if (
        row["measured_metrics"]["successful_requests"] != 110
        or row["measured_metrics"]["total_generated_tokens"] != 110
        or len(row["measured_metrics"]["ttft_samples_ms"]) != 110
        or len(row["output_token_ids"]) != 110
        or evidence["matching_prefill_lines"] != 880
        or evidence["per_rank_matching_lines"]
        != {str(rank): 110 for rank in range(8)}
        or evidence["positive_cache_values"] != [89_984]
        or evidence["scheduled_values"] != [10_048]
    ):
        raise RuntimeError(f"{arm} result evidence is incomplete: {path}")
    return row


def load_correctness(path: Path, label: str) -> dict:
    row = json.loads(path.read_text())
    if (
        row.get("schema_version") != 1
        or row.get("label") != label
        or row.get("input_len") != 100_000
        or row.get("prefix_len") != 90_000
        or row.get("suffix_len") != 10_000
        or row.get("requests") != 11
        or row.get("request_mode") != "sequential_batch_size_one"
        or len(row.get("output_token_ids", [])) != 11
        or len(row.get("output_token_logprobs", [])) != 11
    ):
        raise RuntimeError(f"correctness contract drift: {path}")
    return row


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
        raise RuntimeError(f"unexpected manifest header: {header}")

    records = []
    for line in lines[1:]:
        values = line.split("\t")
        if len(values) != len(header):
            raise RuntimeError(f"malformed manifest line: {line}")
        record = dict(zip(header, values, strict=True))
        result_path = Path(record["result_json"])
        correctness_path = Path(record["correctness_json"])
        server_log = Path(record["server_log"])
        hits_path = server_log.parent / "hits.json"
        for artifact in (result_path, correctness_path, server_log, hits_path):
            if not artifact.is_file():
                raise RuntimeError(f"missing arm artifact: {artifact}")
        result = load_result(result_path, record["arm"])
        correctness = load_correctness(correctness_path, f"{record['label']}_{record['arm']}")
        if Path(result["server_log"]).resolve() != server_log.resolve():
            raise RuntimeError(f"result/server-log mismatch: {record}")
        text = server_log.read_text(errors="replace")
        fatal = [pattern for pattern in FATAL if pattern in text]
        if fatal:
            raise RuntimeError(f"fatal server log for {record['label']}: {fatal}")

        zigzag = "Enabled DSA context parallel: strategy=zigzag" in text
        deepep120 = (
            "Use DeepEP Config: {'normal_dispatch': {'num_sms': 120}, "
            "'normal_combine': {'num_sms': 120}}" in text
        )
        router_ranks = set(
            re.findall(
                r"ATTN_CP(\d+).*GLM-5\.2 router static-placement fusion selected:",
                text,
            )
        )
        if not zigzag or not deepep120 or router_ranks != {str(i) for i in range(8)}:
            raise RuntimeError(
                f"base integration contract failed for {record['label']}: "
                f"zigzag={zigzag} deepep120={deepep120} router={sorted(router_ranks)}"
            )

        direct_pattern = re.compile(
            r"ATTN_CP(\d+) TP\d+ EP\d+\] GLM-5\.2 direct packed CP MLA-KV "
            r"selected: local_M=(\d+) global_M=(\d+)"
        )
        direct_hits = collections.Counter(direct_pattern.findall(text))
        combined_ranks = set(
            re.findall(
                r"ATTN_CP(\d+).*glm52_opt HIT "
                r"e2e_prefill/cp8_combined_indexer_halves:index_score:prefill:m1252",
                text,
            )
        )
        hits = json.loads(hits_path.read_text()).get("hits", {})
        combined_count = int(hits.get(COMBINED_HIT, 0))
        if record["arm"] == "integrated":
            expected_direct = {
                (str(rank), "10048", "80384"): 78 for rank in range(8)
            }
            if dict(direct_hits) != expected_direct:
                raise RuntimeError(
                    f"direct path hit contract failed for {record['label']}: {direct_hits}"
                )
            if "GLM-5.2 direct packed CP MLA-KV rejected:" in text:
                raise RuntimeError(f"direct rejection in {record['label']}")
            if combined_ranks != {str(i) for i in range(8)} or combined_count <= 0:
                raise RuntimeError(
                    f"combined path hit contract failed for {record['label']}: "
                    f"ranks={sorted(combined_ranks)} count={combined_count}"
                )
        elif direct_hits or combined_ranks or combined_count:
            raise RuntimeError(f"control emitted integrated-path hits: {record['label']}")

        records.append(
            {
                **record,
                "result": result,
                "correctness": correctness,
                "combined_hit_count": combined_count,
                "server_log_sha256": hashlib.sha256(server_log.read_bytes()).hexdigest(),
            }
        )

    grouped: dict[str, list[dict]] = collections.defaultdict(list)
    for record in records:
        grouped[record["label"][:2]].append(record)
    if sorted(grouped) != ["p1", "p2", "p3", "p4", "p5"]:
        raise RuntimeError(f"unexpected pair labels: {sorted(grouped)}")

    canonical_input = records[0]["result"]["workload_contract"]["input_ids_sha256"]
    canonical_sentinel = records[0]["result"]["workload_contract"]["sentinel_ids_sha256"]
    canonical_output = records[0]["result"]["output_token_ids"]
    canonical_probe_input = records[0]["correctness"]["input_ids_sha256"]
    canonical_probe_output = records[0]["correctness"]["output_token_ids"]
    pairs = []
    for pair_id in sorted(grouped):
        pair_records = grouped[pair_id]
        if len(pair_records) != 2 or {row["arm"] for row in pair_records} != {
            "control",
            "integrated",
        }:
            raise RuntimeError(f"pair {pair_id} does not contain one arm each")
        for record in pair_records:
            result = record["result"]
            correctness = record["correctness"]
            if (
                result["workload_contract"]["input_ids_sha256"] != canonical_input
                or result["workload_contract"]["sentinel_ids_sha256"] != canonical_sentinel
                or result["output_token_ids"] != canonical_output
                or correctness["input_ids_sha256"] != canonical_probe_input
                or correctness["output_token_ids"] != canonical_probe_output
            ):
                raise RuntimeError(f"cross-arm exact contract drift: {record['label']}")
        control = next(row for row in pair_records if row["arm"] == "control")
        integrated = next(row for row in pair_records if row["arm"] == "integrated")
        control_metrics = control["result"]["measured_metrics"]
        integrated_metrics = integrated["result"]["measured_metrics"]
        p50_improvement = (
            control_metrics["median_ttft_ms"] - integrated_metrics["median_ttft_ms"]
        ) / control_metrics["median_ttft_ms"]
        p90_improvement = (
            control_metrics["p90_ttft_ms"] - integrated_metrics["p90_ttft_ms"]
        ) / control_metrics["p90_ttft_ms"]
        logprob_abs = [
            abs(a - b)
            for a, b in zip(
                control["correctness"]["output_token_logprobs"],
                integrated["correctness"]["output_token_logprobs"],
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
                    "C" if row["arm"] == "control" else "I" for row in pair_records
                ),
                "control_result": control["result_json"],
                "integrated_result": integrated["result_json"],
                "control_p50_ms": control_metrics["median_ttft_ms"],
                "integrated_p50_ms": integrated_metrics["median_ttft_ms"],
                "p50_improvement_fraction": p50_improvement,
                "control_p90_ms": control_metrics["p90_ttft_ms"],
                "integrated_p90_ms": integrated_metrics["p90_ttft_ms"],
                "p90_improvement_fraction": p90_improvement,
                "selected_token_logprob_max_abs": max_abs,
                "selected_token_logprob_mean_abs": mean_abs,
                "combined_hit_count": integrated["combined_hit_count"],
            }
        )

    p50_improvements = [row["p50_improvement_fraction"] for row in pairs]
    p90_improvements = [row["p90_improvement_fraction"] for row in pairs]
    p50_wins = sum(value > 0 for value in p50_improvements)
    p90_wins = sum(value >= 0 for value in p90_improvements)
    median_p50 = statistics.median(p50_improvements)
    median_p90 = statistics.median(p90_improvements)
    promote = p50_wins >= 4 and p90_wins >= 4 and median_p50 >= 0.01 and median_p90 >= 0
    result = {
        "schema": "glm52-cp8-integrated-prefill-fresh-abba-v1",
        "status": "PASS",
        "manifest": str(args.manifest),
        "input_ids_sha256": canonical_input,
        "sentinel_ids_sha256": canonical_sentinel,
        "output_token_ids_sha256": hashlib.sha256(
            json.dumps(canonical_output, separators=(",", ":")).encode()
        ).hexdigest(),
        "correctness_probe_input_ids_sha256": canonical_probe_input,
        "correctness_probe_output_token_ids_sha256": hashlib.sha256(
            json.dumps(canonical_probe_output, separators=(",", ":")).encode()
        ).hexdigest(),
        "pairs": pairs,
        "summary": {
            "p50_wins": p50_wins,
            "p90_wins": p90_wins,
            "median_p50_improvement_fraction": median_p50,
            "median_p90_improvement_fraction": median_p90,
            "max_selected_token_logprob_abs": max(
                row["selected_token_logprob_max_abs"] for row in pairs
            ),
            "max_pair_mean_selected_token_logprob_abs": max(
                row["selected_token_logprob_mean_abs"] for row in pairs
            ),
        },
        "promotion_gate": {
            "p50_wins_min": 4,
            "p90_wins_min": 4,
            "median_p50_improvement_fraction_min": 0.01,
            "median_p90_improvement_fraction_min": 0.0,
            "selected_token_logprob_max_abs_max": 1e-3,
            "selected_token_logprob_mean_abs_max": 1e-4,
            "decision": "PROMOTE_TO_MATCHED_ATTRIBUTION" if promote else "STOP",
        },
        "limitations": (
            "exactly the frozen eager CP8/EP8 90K-cache + 10K-input prefill cell; "
            "decode, CUDA Graph, other shapes, and online arrival processes are untested"
        ),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
