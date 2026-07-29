#!/usr/bin/env python3

"""Assert GLM-5.2 FlashMLA launch routing from one nsys kernel summary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

FLAT_KERNEL = "flash_fwd_splitkv_mla_fp8_sparse_kernel_glm52_flat_page64_v32"
GENERIC_KERNEL = "flash_fwd_splitkv_mla_fp8_sparse_kernel"


def analyze_dispatch(
    *,
    input_record: dict[str, object],
    csv_text: str,
    expected_state: str,
    expected_h_q: int,
    expected_region: str,
    expected_iterations: int,
    expected_flat: int,
    expected_generic: int,
) -> dict[str, object]:
    rows = list(csv.reader(csv_text.splitlines()))
    header_index = next(
        (
            index
            for index, row in enumerate(rows)
            if "Instances" in row and "Name" in row
        ),
        None,
    )
    if header_index is None:
        return {
            "verdict": "FAIL",
            "checks": {},
            "matched_kernel_rows": [],
            "failures": ["cannot find Instances/Name columns"],
        }
    header = rows[header_index]
    instances_index = header.index("Instances")
    name_index = header.index("Name")
    flat_instances = 0
    generic_instances = 0
    matched_rows: list[dict[str, object]] = []
    for row in rows[header_index + 1 :]:
        if len(row) <= max(instances_index, name_index):
            continue
        name = row[name_index]
        try:
            instances = int(row[instances_index].replace(",", ""))
        except ValueError:
            continue
        if FLAT_KERNEL in name:
            flat_instances += instances
            matched_rows.append(
                {"kind": "flat", "instances": instances, "name": name}
            )
        elif GENERIC_KERNEL in name:
            generic_instances += instances
            matched_rows.append(
                {"kind": "generic", "instances": instances, "name": name}
            )

    observed_state = input_record.get("dispatch", {})
    if isinstance(observed_state, dict):
        observed_state = observed_state.get("state")
    observed_h_q = input_record.get("shape", {})
    if isinstance(observed_h_q, dict):
        observed_h_q = observed_h_q.get("h_q")
    observed_profile = input_record.get("profile", {})
    if isinstance(observed_profile, dict):
        observed_region = observed_profile.get("region")
        observed_iterations = observed_profile.get("iterations")
    else:
        observed_region = None
        observed_iterations = None
    checks = {
        "dispatch_state": {"actual": observed_state, "expected": expected_state},
        "original_h_q": {"actual": observed_h_q, "expected": expected_h_q},
        "profile_region": {
            "actual": observed_region,
            "expected": expected_region,
        },
        "profile_iterations": {
            "actual": observed_iterations,
            "expected": expected_iterations,
        },
        "flat_instances": {"actual": flat_instances, "expected": expected_flat},
        "generic_instances": {
            "actual": generic_instances,
            "expected": expected_generic,
        },
    }
    failures = [
        f"{name}: {check['actual']!r} != {check['expected']!r}"
        for name, check in checks.items()
        if check["actual"] != check["expected"]
    ]
    return {
        "verdict": "PASS" if not failures else "FAIL",
        "checks": checks,
        "matched_kernel_rows": matched_rows,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-state",
        choices=("disabled", "miss", "hit"),
        required=True,
    )
    parser.add_argument("--expected-h-q", type=int, required=True)
    parser.add_argument(
        "--expected-region",
        choices=("operator", "containing", "graph"),
        required=True,
    )
    parser.add_argument("--expected-iterations", type=int, required=True)
    parser.add_argument("--expected-flat", type=int, required=True)
    parser.add_argument("--expected-generic", type=int, required=True)
    args = parser.parse_args()

    evidence = analyze_dispatch(
        input_record=json.loads(args.input.read_text()),
        csv_text=args.csv.read_text(),
        expected_state=args.expected_state,
        expected_h_q=args.expected_h_q,
        expected_region=args.expected_region,
        expected_iterations=args.expected_iterations,
        expected_flat=args.expected_flat,
        expected_generic=args.expected_generic,
    )
    evidence["input"] = args.input.name
    evidence["csv"] = args.csv.name
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    if evidence["verdict"] != "PASS":
        raise SystemExit("; ".join(evidence["failures"]))


if __name__ == "__main__":
    main()
