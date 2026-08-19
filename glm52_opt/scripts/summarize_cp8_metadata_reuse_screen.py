#!/usr/bin/env python3
"""Fail-closed summary for one fresh c01/candidate metadata-reuse screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path


METADATA_HIT = "e2e_prefill/cp8_flashmla_metadata_reuse:metadata:prefill:m1252"
FATAL = (
    "Traceback (most recent call last)",
    "illegal memory access",
    "CUDA out of memory",
    "KV cache pool is full",
    "outside its frozen contract",
    "not layer-invariant",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_arm(fields: dict[str, str]) -> dict:
    result_path = Path(fields["result_json"])
    correctness_path = Path(fields["correctness_json"])
    server_log = Path(fields["server_log"])
    hits_path = server_log.parent / "hits.json"
    for path in (result_path, correctness_path, server_log, hits_path):
        if not path.is_file():
            raise RuntimeError(f"missing {fields['arm']} artifact: {path}")
    result = json.loads(result_path.read_text())
    correctness = json.loads(correctness_path.read_text())
    hits = json.loads(hits_path.read_text()).get("hits", {})
    text = server_log.read_text(errors="replace")
    fatal = [pattern for pattern in FATAL if pattern in text]
    if fatal:
        raise RuntimeError(f"fatal markers in {fields['arm']}: {fatal}")
    expected_parallel = {
        "tp": 8,
        "dp": 1,
        "attention_cp": 8,
        "attention_tp": 1,
        "ep": 8,
        "request_concurrency": 11,
    }
    metrics = result.get("measured_metrics", {})
    evidence = result.get("server_evidence", {})
    if (
        result.get("status") != "PASS"
        or result.get("parallelism_contract") != expected_parallel
        or metrics.get("successful_requests") != 110
        or metrics.get("total_generated_tokens") != 110
        or len(metrics.get("ttft_samples_ms", [])) != 110
        or len(result.get("output_token_ids", [])) != 110
        or evidence.get("matching_prefill_lines") != 880
        or evidence.get("per_rank_matching_lines")
        != {str(rank): 110 for rank in range(8)}
        or evidence.get("positive_cache_values") != [89_984]
        or evidence.get("scheduled_values") != [10_048]
    ):
        raise RuntimeError(f"incomplete {fields['arm']} result contract")
    if (
        correctness.get("requests") != 11
        or len(correctness.get("output_token_ids", [])) != 11
        or len(correctness.get("output_token_logprobs", [])) != 11
    ):
        raise RuntimeError(f"incomplete {fields['arm']} correctness contract")
    selected_ranks = set(
        re.findall(
            r"ATTN_CP(\d+).*glm52_opt HIT "
            r"e2e_prefill/cp8_flashmla_metadata_reuse:metadata:prefill:m1252",
            text,
        )
    )
    metadata_hits = int(hits.get(METADATA_HIT, 0))
    if fields["arm"] == "candidate":
        if selected_ranks != {str(rank) for rank in range(8)} or metadata_hits <= 0:
            raise RuntimeError(
                f"candidate metadata path missing: ranks={sorted(selected_ranks)} "
                f"hits={metadata_hits}"
            )
    elif selected_ranks or metadata_hits:
        raise RuntimeError("baseline unexpectedly selected metadata reuse")
    return {
        **fields,
        "result": result,
        "correctness": correctness,
        "metadata_hits": metadata_hits,
        "selected_ranks": sorted(selected_ranks),
        "server_log_sha256": hashlib.sha256(server_log.read_bytes()).hexdigest(),
    }


def main() -> None:
    args = parse_args()
    lines = args.manifest.read_text().splitlines()
    if len(lines) != 3:
        raise RuntimeError(f"expected header and two arms, got {len(lines)} lines")
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
    if {row["arm"] for row in records} != {"baseline", "candidate"}:
        raise RuntimeError("screen requires one baseline and one candidate")
    baseline = next(row for row in records if row["arm"] == "baseline")
    candidate = next(row for row in records if row["arm"] == "candidate")
    for key in ("input_ids_sha256", "sentinel_ids_sha256"):
        if (
            baseline["result"]["workload_contract"][key]
            != candidate["result"]["workload_contract"][key]
        ):
            raise RuntimeError(f"workload hash drift: {key}")
    if baseline["result"]["output_token_ids"] != candidate["result"]["output_token_ids"]:
        raise RuntimeError("110-token output trajectory mismatch")
    if (
        baseline["correctness"]["input_ids_sha256"]
        != candidate["correctness"]["input_ids_sha256"]
        or baseline["correctness"]["output_token_ids"]
        != candidate["correctness"]["output_token_ids"]
    ):
        raise RuntimeError("correctness input or token trajectory mismatch")
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
        raise RuntimeError(f"logprob gate failed: max={max_abs} mean={mean_abs}")
    bm = baseline["result"]["measured_metrics"]
    cm = candidate["result"]["measured_metrics"]
    p50 = (bm["median_ttft_ms"] - cm["median_ttft_ms"]) / bm["median_ttft_ms"]
    p90 = (bm["p90_ttft_ms"] - cm["p90_ttft_ms"]) / bm["p90_ttft_ms"]
    decision = (
        "ADVANCE_TO_FIVE_PAIRS"
        if p50 >= 0.01 and p90 >= -0.02
        else "STOP_OR_REDESIGN_AFTER_SINGLE_PAIR_SCREEN"
    )
    out = {
        "schema": "glm52-cp8-metadata-reuse-screen-v1",
        "status": "PASS",
        "scope": "single fresh-server pair screen only; not promotion evidence",
        "baseline": {
            "p50_ms": bm["median_ttft_ms"],
            "p90_ms": bm["p90_ttft_ms"],
            "result": baseline["result_json"],
            "server_log_sha256": baseline["server_log_sha256"],
        },
        "candidate": {
            "p50_ms": cm["median_ttft_ms"],
            "p90_ms": cm["p90_ttft_ms"],
            "result": candidate["result_json"],
            "metadata_hits": candidate["metadata_hits"],
            "selected_ranks": candidate["selected_ranks"],
            "server_log_sha256": candidate["server_log_sha256"],
        },
        "observed": {
            "p50_reduction_fraction": p50,
            "p90_reduction_fraction": p90,
            "tokens_exact": True,
            "selected_token_logprob_max_abs": max_abs,
            "selected_token_logprob_mean_abs": mean_abs,
        },
        "screen_rule": {
            "p50_reduction_min": 0.01,
            "p90_reduction_min": -0.02,
            "note": "This rule only decides whether to spend five-pair budget; the frozen promotion rule remains stricter.",
        },
        "decision": decision,
    }
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
