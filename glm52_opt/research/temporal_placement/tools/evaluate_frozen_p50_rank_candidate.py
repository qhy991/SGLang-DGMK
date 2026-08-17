#!/usr/bin/env python3
"""Evaluate a frozen placement on an external per-token route capture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pct(candidate: float, reference: float) -> float:
    return 100.0 * (candidate / reference - 1.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer-summary", required=True, type=Path)
    parser.add_argument("--selection-summary", required=True, type=Path)
    parser.add_argument("--candidate-map", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    layers = [item for item in json.loads(args.layer_summary.read_text())["per_layer"] if item["active"]]
    selection = json.loads(args.selection_summary.read_text())
    candidate_hash = sha256(args.candidate_map)
    if selection["status"] != "ADMITTED_FOR_FULL_REPLAY":
        raise RuntimeError(f"candidate was not frozen: {selection['status']}")
    if candidate_hash != selection["output_map_sha256"]:
        raise RuntimeError("candidate map hash mismatch")
    tolerance = selection["optimization"]["holdout_tolerances_pct"]
    send_tolerance = float(tolerance["send_and_channel_scalars"])
    rank_tolerance = float(tolerance["rank_maxima"])
    compute_tolerance = float(tolerance["compute_mean_p50_p90"])

    scalar_metrics = (
        "compute_mean",
        "compute_p50",
        "compute_p90",
        "max_channel_p50",
        "max_channel_p90",
        "unique_sends_per_token",
        "remote_unique_sends_per_token",
    )
    vector_kinds = (
        "source_compute_mean",
        "source_compute_p50",
        "source_unique_sends",
        "source_remote_unique_sends",
        "destination_compute",
        "destination_channels",
    )
    splits = ("train_even", "holdout_odd")
    results = []

    for split in splits:
        for metric in scalar_metrics:
            reference_total = 0.0
            candidate_total = 0.0
            for item in layers:
                values = item["splits"][split]
                ref = float(values["reference"][metric])
                cand = float(values["proposal"][metric])
                if metric in (
                    "unique_sends_per_token",
                    "remote_unique_sends_per_token",
                ):
                    tokens = int(values["reference"]["tokens"])
                    ref *= tokens
                    cand *= tokens
                reference_total += ref
                candidate_total += cand
            change = pct(candidate_total, reference_total)
            limit = (
                compute_tolerance
                if metric in ("compute_mean", "compute_p50", "compute_p90")
                else send_tolerance
            )
            results.append(
                {
                    "name": f"{split}:scalar:{metric}",
                    "reference": reference_total,
                    "candidate": candidate_total,
                    "percent_change": change,
                    "tolerance_pct": limit,
                    "passed": change <= limit + 1e-12,
                }
            )

        reference_vectors = {kind: [0.0] * 8 for kind in vector_kinds}
        candidate_vectors = {kind: [0.0] * 8 for kind in vector_kinds}
        for item in layers:
            values = item["splits"][split]
            for placement, output in (
                ("reference", reference_vectors),
                ("proposal", candidate_vectors),
            ):
                source_ranks = values[placement]["source_ranks"]
                for rank in range(8):
                    output["source_compute_mean"][rank] += float(
                        source_ranks[str(rank)]["compute_mean"]
                    )
                    output["source_compute_p50"][rank] += float(
                        source_ranks[str(rank)]["compute_p50"]
                    )
                    output["source_unique_sends"][rank] += float(
                        source_ranks[str(rank)]["unique_sends"]
                    )
                    output["source_remote_unique_sends"][rank] += float(
                        source_ranks[str(rank)]["remote_unique_sends"]
                    )
                    output["destination_compute"][rank] += float(
                        values[placement]["compute_by_destination"][rank]
                    )
                    output["destination_channels"][rank] += float(
                        values[placement]["channels_by_destination"][rank]
                    )
        for kind in vector_kinds:
            reference_max = max(reference_vectors[kind])
            candidate_max = max(candidate_vectors[kind])
            change = pct(candidate_max, reference_max)
            results.append(
                {
                    "name": f"{split}:rank_max:{kind}",
                    "reference_max": reference_max,
                    "candidate_max": candidate_max,
                    "reference_rank": reference_vectors[kind].index(reference_max),
                    "candidate_rank": candidate_vectors[kind].index(candidate_max),
                    "percent_change": change,
                    "tolerance_pct": rank_tolerance,
                    "passed": change <= rank_tolerance + 1e-12,
                }
            )

    failures = [item for item in results if not item["passed"]]
    result = {
        "status": "PASS_EXTERNAL_PROXY" if not failures else "REJECTED_EXTERNAL_PROXY",
        "candidate_map": {"path": str(args.candidate_map), "sha256": candidate_hash},
        "selection_summary": {
            "path": str(args.selection_summary),
            "sha256": sha256(args.selection_summary),
        },
        "layer_summary": {"path": str(args.layer_summary), "sha256": sha256(args.layer_summary)},
        "tolerances_pct": {
            "compute_mean_p50_p90": compute_tolerance,
            "send_and_channel_scalars": send_tolerance,
            "rank_maxima": rank_tolerance,
        },
        "results": results,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(
        {
            "status": result["status"],
            "failures": failures,
            "results": [
                {
                    "name": item["name"],
                    "percent_change": item["percent_change"],
                    "tolerance_pct": item["tolerance_pct"],
                    "passed": item["passed"],
                }
                for item in results
            ],
        },
        indent=2,
        sort_keys=True,
    ))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
