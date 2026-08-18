#!/usr/bin/env python3
"""Fail-closed summary for matched CP8 packed-MLA-KV serving runs."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import statistics
from pathlib import Path


EXPECTED_PARALLELISM = {
    "tp": 8,
    "dp": 1,
    "attention_cp": 8,
    "attention_tp": 1,
    "ep": 8,
    "request_concurrency": 11,
}
EXPECTED_WORKLOAD = {
    "requests": 110,
    "logical_prefix_tokens": 90_000,
    "page_aligned_cached_tokens": 89_984,
    "suffix_tokens": 10_000,
    "output_tokens": 1,
    "scheduled_tokens_per_cp_rank": 10_048,
    "unique_suffix_first_token": True,
}
FATAL_SERVER_PATTERNS = (
    "Traceback (most recent call last)",
    "illegal memory access",
    "CUBLAS_STATUS_EXECUTION_FAILED",
    "CUDA out of memory",
    "torch.OutOfMemoryError",
    "KV cache pool is full",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, action="append", required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--candidate-server-log", type=Path, required=True)
    parser.add_argument(
        "--candidate-kind", choices=("packed_nccl", "direct_multimem"), default="packed_nccl"
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_arm(paths: list[Path], name: str) -> dict:
    if len(paths) != 3:
        raise RuntimeError(f"{name} requires exactly three repeats, got {len(paths)}")
    rows = [json.loads(path.read_text()) for path in paths]
    for path, row in zip(paths, rows, strict=True):
        if row.get("status") != "PASS":
            raise RuntimeError(f"{name} is not PASS: {path}")
        if row["parallelism_contract"] != EXPECTED_PARALLELISM:
            raise RuntimeError(
                f"{name} parallelism contract drift in {path}: "
                f"{row['parallelism_contract']}"
            )
        observed_workload = {
            key: row["workload_contract"].get(key) for key in EXPECTED_WORKLOAD
        }
        if observed_workload != EXPECTED_WORKLOAD:
            raise RuntimeError(
                f"{name} workload contract drift in {path}: {observed_workload}"
            )
        if row["measured_metrics"]["total_generated_tokens"] != 110:
            raise RuntimeError(f"{name} does not contain 110 output tokens: {path}")
        if row["measured_metrics"]["successful_requests"] != 110:
            raise RuntimeError(f"{name} does not contain 110 successes: {path}")
        if len(row["measured_metrics"]["ttft_samples_ms"]) != 110:
            raise RuntimeError(f"{name} does not contain 110 TTFT samples: {path}")
        if len(row["output_token_ids"]) != 110:
            raise RuntimeError(f"{name} does not contain 110 output IDs: {path}")

        server_evidence = row["server_evidence"]
        expected_per_rank = {str(rank): 110 for rank in range(8)}
        if (
            server_evidence["matching_prefill_lines"] != 880
            or server_evidence["per_rank_matching_lines"] != expected_per_rank
            or server_evidence["positive_cache_values"] != [89_984]
            or server_evidence["scheduled_values"] != [10_048]
        ):
            raise RuntimeError(
                f"{name} cache/shape evidence drift in {path}: {server_evidence}"
            )

    input_hashes = {row["workload_contract"]["input_ids_sha256"] for row in rows}
    sentinel_hashes = {
        row["workload_contract"]["sentinel_ids_sha256"] for row in rows
    }
    output_sequences = {tuple(row["output_token_ids"]) for row in rows}
    server_logs = {row["server_log"] for row in rows}
    if (
        len(input_hashes) != 1
        or len(sentinel_hashes) != 1
        or len(output_sequences) != 1
        or len(server_logs) != 1
    ):
        raise RuntimeError(f"{name} repeats do not share one exact input/output contract")

    p50 = [row["measured_metrics"]["median_ttft_ms"] for row in rows]
    p90 = [row["measured_metrics"]["p90_ttft_ms"] for row in rows]
    return {
        "result_paths": [str(path) for path in paths],
        "p50_ms": p50,
        "p90_ms": p90,
        "cross_repeat_p50_median_ms": statistics.median(p50),
        "cross_repeat_p90_median_ms": statistics.median(p90),
        "input_ids_sha256": next(iter(input_hashes)),
        "sentinel_ids_sha256": next(iter(sentinel_hashes)),
        "output_token_ids": list(next(iter(output_sequences))),
        "server_log": next(iter(server_logs)),
    }


def main() -> None:
    args = parse_args()
    control = load_arm(args.control, "control")
    candidate = load_arm(args.candidate, "candidate")
    for key in ("input_ids_sha256", "sentinel_ids_sha256", "output_token_ids"):
        if control[key] != candidate[key]:
            raise RuntimeError(f"candidate differs from control for {key}")
    if Path(candidate["server_log"]).resolve() != args.candidate_server_log.resolve():
        raise RuntimeError(
            "candidate result/server-log mismatch: "
            f"results={candidate['server_log']} argument={args.candidate_server_log}"
        )

    text = args.candidate_server_log.read_text(errors="replace")
    rejection_marker = (
        "GLM-5.2 direct packed CP MLA-KV rejected:"
        if args.candidate_kind == "direct_multimem"
        else "GLM-5.2 packed CP MLA-KV rejected:"
    )
    rejected = [
        line
        for line in text.splitlines()
        if rejection_marker in line
    ]
    fatal = [pattern for pattern in FATAL_SERVER_PATTERNS if pattern in text]
    if rejected or fatal:
        raise RuntimeError(
            f"candidate server log is not clean: rejected={rejected[:8]} fatal={fatal}"
        )
    if args.candidate_kind == "direct_multimem":
        pattern = re.compile(
            r"ATTN_CP(\d+) TP\d+ EP\d+\] GLM-5\.2 direct packed CP MLA-KV "
            r"selected: local_M=(\d+) global_M=(\d+)"
        )
    else:
        pattern = re.compile(
            r"ATTN_CP(\d+) TP\d+ EP\d+\] GLM-5\.2 packed CP MLA-KV "
            r"communication selected: local_M=(\d+)"
        )
    hits = collections.Counter(pattern.findall(text))
    hit_ranks = sorted({int(key[0]) for key in hits})
    local_shapes = sorted({int(key[1]) for key in hits})
    global_shapes = (
        sorted({int(key[2]) for key in hits})
        if args.candidate_kind == "direct_multimem"
        else []
    )
    if (
        hit_ranks != list(range(8))
        or local_shapes != [10_048]
        or (args.candidate_kind == "direct_multimem" and global_shapes != [80_384])
    ):
        raise RuntimeError(
            "candidate-path contract failed: "
            f"ranks={hit_ranks} local_M={local_shapes} global_M={global_shapes}"
        )

    control_p50 = control["cross_repeat_p50_median_ms"]
    candidate_p50 = candidate["cross_repeat_p50_median_ms"]
    control_p90 = control["cross_repeat_p90_median_ms"]
    candidate_p90 = candidate["cross_repeat_p90_median_ms"]
    p50_improvement = (control_p50 - candidate_p50) / control_p50
    p90_change = (candidate_p90 - control_p90) / control_p90
    promotion_passed = p50_improvement >= 0.01 and p90_change <= 0.0
    result = {
        "schema": "glm52-cp8-packed-mla-kv-serving-ab-v2",
        "status": "PASS",
        "candidate_kind": args.candidate_kind,
        "control": control,
        "candidate": candidate,
        "candidate_vs_control": {
            "p50_delta_ms": candidate_p50 - control_p50,
            "p50_change_fraction": (candidate_p50 - control_p50) / control_p50,
            "p90_delta_ms": candidate_p90 - control_p90,
            "p90_change_fraction": p90_change,
        },
        "promotion_gate": {
            "required_p50_improvement_fraction": 0.01,
            "required_p90_change_fraction_max": 0.0,
            "observed_p50_improvement_fraction": p50_improvement,
            "observed_p90_change_fraction": p90_change,
            "decision": "PROMOTE" if promotion_passed else "STOP",
        },
        "candidate_hit_counts": {
            (
                f"rank{key[0]}_localM{key[1]}_globalM{key[2]}"
                if len(key) == 3
                else f"rank{key[0]}_localM{key[1]}"
            ): count
            for key, count in sorted(hits.items())
        },
        "candidate_server_log": str(args.candidate_server_log),
        "candidate_server_log_sha256": hashlib.sha256(
            args.candidate_server_log.read_bytes()
        ).hexdigest(),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
