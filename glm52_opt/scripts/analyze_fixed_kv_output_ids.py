#!/usr/bin/env python3
"""Classify and localize fixed-KV A/B/A greedy-output differences."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROTOCOL = "fixed-kv-decode-series-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--before-label", default="winners_before")
    parser.add_argument("--candidate-label", default="swiglu")
    parser.add_argument("--after-label", default="winners_after")
    parser.add_argument(
        "--require-exact",
        action="store_true",
        help="return a non-zero status after writing the report unless all IDs match",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_ids(ids: list[list[int]]) -> bytes:
    return json.dumps(ids, separators=(",", ":")).encode()


def load_record(path: Path, expected_label: str) -> dict[str, Any]:
    record = json.loads(path.read_text())
    if record.get("protocol") != PROTOCOL:
        raise SystemExit(f"{path}: invalid protocol {record.get('protocol')!r}")
    if record.get("label") != expected_label:
        raise SystemExit(
            f"{path}: invalid label {record.get('label')!r}; "
            f"expected {expected_label!r}"
        )
    if not record.get("prompt_set_id"):
        raise SystemExit(f"{path}: missing prompt_set_id")
    if not record.get("output_token_ids_sha256"):
        raise SystemExit(f"{path}: missing output_token_ids_sha256")

    ids = record.get("output_token_ids")
    if ids is None:
        return record
    if not isinstance(ids, list) or any(not isinstance(row, list) for row in ids):
        raise SystemExit(f"{path}: output_token_ids is not a two-dimensional list")
    normalized = [[int(value) for value in row] for row in ids]
    batch_size = int(record["batch_size"])
    output_len = int(record["output_len"])
    if len(normalized) != batch_size or any(
        len(row) != output_len for row in normalized
    ):
        raise SystemExit(
            f"{path}: output shape does not match batch={batch_size}, "
            f"output_len={output_len}"
        )
    token_count = sum(len(row) for row in normalized)
    if token_count != int(record["output_token_count"]):
        raise SystemExit(
            f"{path}: token count {token_count} != "
            f"{record['output_token_count']}"
        )
    observed_hash = hashlib.sha256(canonical_ids(normalized)).hexdigest()
    if observed_hash != record["output_token_ids_sha256"]:
        raise SystemExit(
            f"{path}: output ID hash {observed_hash} != "
            f"{record['output_token_ids_sha256']}"
        )
    record["output_token_ids"] = normalized
    return record


def pairwise_diff(
    left: list[list[int]], right: list[list[int]]
) -> dict[str, Any]:
    if len(left) != len(right) or any(
        len(left_row) != len(right_row)
        for left_row, right_row in zip(left, right)
    ):
        raise SystemExit("A/B/A correctness records have different output shapes")

    output_len = len(left[0]) if left else 0
    by_position = [0] * output_len
    mismatched_requests = 0
    mismatched_tokens = 0
    first_mismatch: dict[str, int] | None = None
    for request_index, (left_row, right_row) in enumerate(zip(left, right)):
        request_differs = False
        for position, (left_id, right_id) in enumerate(zip(left_row, right_row)):
            if left_id == right_id:
                continue
            request_differs = True
            mismatched_tokens += 1
            by_position[position] += 1
            if first_mismatch is None:
                first_mismatch = {
                    "request_index": request_index,
                    "output_position": position,
                    "left_token_id": left_id,
                    "right_token_id": right_id,
                }
        mismatched_requests += int(request_differs)

    total_tokens = sum(len(row) for row in left)
    return {
        "exact": mismatched_tokens == 0,
        "total_tokens": total_tokens,
        "mismatched_tokens": mismatched_tokens,
        "mismatch_fraction": (
            mismatched_tokens / float(total_tokens) if total_tokens else 0.0
        ),
        "total_requests": len(left),
        "mismatched_requests": mismatched_requests,
        "mismatch_by_output_position": by_position,
        "first_mismatch": first_mismatch,
    }


def main() -> None:
    args = parse_args()
    labels = (args.before_label, args.candidate_label, args.after_label)
    if len(set(labels)) != 3 or any(not label for label in labels):
        raise SystemExit("A/B/A labels must be non-empty and distinct")
    paths = (args.before, args.candidate, args.after)
    records = [
        load_record(path, label) for path, label in zip(paths, labels)
    ]

    contract_keys = (
        "protocol",
        "input_source_id",
        "prompt_set_id",
        "sequence_index",
        "batch_size",
        "output_len",
        "output_token_count",
    )
    reference_contract = {
        key: records[0].get(key) for key in contract_keys
    }
    for label, record in zip(labels, records):
        observed = {key: record.get(key) for key in contract_keys}
        if observed != reference_contract:
            raise SystemExit(
                f"{label}: correctness-probe contract differs from first arm"
            )

    hashes = [record["output_token_ids_sha256"] for record in records]
    full_ids_available = all(
        isinstance(record.get("output_token_ids"), list) for record in records
    )
    exact_all = len(set(hashes)) == 1
    if hashes[0] != hashes[2]:
        classification = "baseline_nondeterministic"
    elif not exact_all:
        classification = "candidate_mismatch"
    else:
        classification = "exact_all"

    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "classification": classification,
        "exact_all": exact_all,
        "full_ids_available": full_ids_available,
        "contract": reference_contract,
        "inputs": {
            label: {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "output_token_ids_sha256": record[
                    "output_token_ids_sha256"
                ],
            }
            for label, path, record in zip(labels, paths, records)
        },
    }
    if full_ids_available:
        before_ids, candidate_ids, after_ids = (
            record["output_token_ids"] for record in records
        )
        report["pairwise"] = {
            f"{labels[0]}_vs_{labels[1]}": pairwise_diff(
                before_ids, candidate_ids
            ),
            f"{labels[0]}_vs_{labels[2]}": pairwise_diff(
                before_ids, after_ids
            ),
            f"{labels[1]}_vs_{labels[2]}": pairwise_diff(
                candidate_ids, after_ids
            ),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"fixed-KV output classification={classification} "
        f"exact_all={exact_all} full_ids={full_ids_available}; "
        f"report={args.output}"
    )
    if args.require_exact and not exact_all:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
