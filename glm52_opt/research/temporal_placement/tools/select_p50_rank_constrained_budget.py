#!/usr/bin/env python3
"""Select proposal layers with P50 and rank-level zero-regression budgets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import Bounds, LinearConstraint, milp


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_p2l(path: Path) -> torch.Tensor:
    result = torch.tensor(
        json.loads(path.read_text())["physical_to_logical_map"], dtype=torch.long
    )
    expected = torch.arange(result.shape[1]).expand(result.shape[0], -1)
    if not torch.equal(result.sort(1).values, expected):
        raise RuntimeError(f"invalid placement permutation: {path}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer-summary", required=True, type=Path)
    parser.add_argument("--reference-map", required=True, type=Path)
    parser.add_argument("--proposal-map", required=True, type=Path)
    parser.add_argument("--output-map", required=True, type=Path)
    parser.add_argument("--output-summary", required=True, type=Path)
    parser.add_argument("--selection-split", default="train_even")
    parser.add_argument("--mean-objective-weight", type=float, default=0.5)
    parser.add_argument("--p50-objective-weight", type=float, default=1.0)
    parser.add_argument("--p90-objective-weight", type=float, default=0.25)
    parser.add_argument(
        "--holdout-send-tolerance-pct", type=float, default=0.05
    )
    parser.add_argument(
        "--holdout-rank-max-tolerance-pct", type=float, default=0.10
    )
    args = parser.parse_args()

    source = json.loads(args.layer_summary.read_text())
    reference = load_p2l(args.reference_map)
    proposal = load_p2l(args.proposal_map)
    if reference.shape != proposal.shape:
        raise RuntimeError("placement shape mismatch")
    active = [item for item in source["per_layer"] if item["active"]]
    layers = [int(item["layer"]) for item in active]
    if layers != list(range(3, reference.shape[0])):
        raise RuntimeError("unexpected active layer coverage")
    ep_size = 8
    splits = ("train_even", "holdout_odd")
    if args.selection_split not in splits:
        raise RuntimeError(f"unknown selection split {args.selection_split!r}")

    scalar_metrics = (
        "compute_mean",
        "compute_p50",
        "compute_p90",
        "max_channel_p50",
        "max_channel_p90",
        "unique_sends_per_token",
        "remote_unique_sends_per_token",
    )
    scalar_delta: dict[str, dict[str, np.ndarray]] = {split: {} for split in splits}
    scalar_reference: dict[str, dict[str, float]] = {split: {} for split in splits}
    for split in splits:
        for metric in scalar_metrics:
            coefficients = []
            reference_total = 0.0
            for item in active:
                values = item["splits"][split]
                ref = float(values["reference"][metric])
                prop = float(values["proposal"][metric])
                if metric in ("unique_sends_per_token", "remote_unique_sends_per_token"):
                    tokens = int(values["reference"]["tokens"])
                    ref *= tokens
                    prop *= tokens
                coefficients.append(prop - ref)
                reference_total += ref
            scalar_delta[split][metric] = np.asarray(coefficients, dtype=np.float64)
            scalar_reference[split][metric] = reference_total

    # Vector budgets model the rank-level bottlenecks hidden by a global mean.
    # Each matrix has shape [layers, ranks].
    vector_kinds = (
        "source_compute_mean",
        "source_compute_p50",
        "source_unique_sends",
        "source_remote_unique_sends",
        "destination_compute",
        "destination_channels",
    )
    vector_reference: dict[str, dict[str, np.ndarray]] = {split: {} for split in splits}
    vector_delta: dict[str, dict[str, np.ndarray]] = {split: {} for split in splits}
    for split in splits:
        reference_rows = {kind: [] for kind in vector_kinds}
        proposal_rows = {kind: [] for kind in vector_kinds}
        for item in active:
            values = item["splits"][split]
            for placement, rows in (("reference", reference_rows), ("proposal", proposal_rows)):
                placement_values = values[placement]
                source_ranks = placement_values["source_ranks"]
                rows["source_compute_mean"].append(
                    [float(source_ranks[str(rank)]["compute_mean"]) for rank in range(ep_size)]
                )
                rows["source_compute_p50"].append(
                    [float(source_ranks[str(rank)]["compute_p50"]) for rank in range(ep_size)]
                )
                rows["source_unique_sends"].append(
                    [float(source_ranks[str(rank)]["unique_sends"]) for rank in range(ep_size)]
                )
                rows["source_remote_unique_sends"].append(
                    [float(source_ranks[str(rank)]["remote_unique_sends"]) for rank in range(ep_size)]
                )
                rows["destination_compute"].append(
                    [float(value) for value in placement_values["compute_by_destination"]]
                )
                rows["destination_channels"].append(
                    [float(value) for value in placement_values["channels_by_destination"]]
                )
        for kind in vector_kinds:
            ref = np.asarray(reference_rows[kind], dtype=np.float64)
            prop = np.asarray(proposal_rows[kind], dtype=np.float64)
            vector_reference[split][kind] = ref
            vector_delta[split][kind] = prop - ref

    objective = np.zeros(len(layers), dtype=np.float64)
    split = args.selection_split
    objective += (
        args.mean_objective_weight
        * scalar_delta[split]["compute_mean"]
        / scalar_reference[split]["compute_mean"]
    )
    objective += (
        args.p50_objective_weight
        * scalar_delta[split]["compute_p50"]
        / scalar_reference[split]["compute_p50"]
    )
    objective += (
        args.p90_objective_weight
        * scalar_delta[split]["compute_p90"]
        / scalar_reference[split]["compute_p90"]
    )

    rows = []
    upper_bounds = []
    constraint_names = []
    for metric in scalar_metrics:
        rows.append(scalar_delta[split][metric] / scalar_reference[split][metric])
        upper_bounds.append(0.0)
        constraint_names.append(f"{split}:scalar:{metric}")
    for kind in vector_kinds:
        ref = vector_reference[split][kind]
        delta = vector_delta[split][kind]
        ref_total = ref.sum(axis=0)
        ref_max = float(ref_total.max())
        if ref_max <= 0:
            raise RuntimeError(f"zero rank budget for {split}:{kind}")
        for rank in range(ep_size):
            rows.append(delta[:, rank] / ref_max)
            upper_bounds.append((ref_max - ref_total[rank]) / ref_max)
            constraint_names.append(f"{split}:rank_max:{kind}:rank{rank}")
    matrix = np.stack(rows)
    upper = np.asarray(upper_bounds, dtype=np.float64)
    constraint = LinearConstraint(
        matrix, lb=np.full(len(rows), -np.inf), ub=upper
    )
    solution = milp(
        c=objective,
        integrality=np.ones(len(layers), dtype=np.int32),
        bounds=Bounds(np.zeros(len(layers)), np.ones(len(layers))),
        constraints=constraint,
        options={"time_limit": 60.0, "mip_rel_gap": 0.0},
    )
    if not solution.success or solution.x is None:
        raise RuntimeError(f"MILP failed: status={solution.status} {solution.message}")
    selected_mask = solution.x > 0.5
    selected_float = selected_mask.astype(np.float64)
    selected_layers = [layer for layer, chosen in zip(layers, selected_mask) if chosen]

    hybrid = reference.clone()
    if selected_layers:
        hybrid[selected_layers] = proposal[selected_layers]
    args.output_map.parent.mkdir(parents=True, exist_ok=True)
    args.output_map.write_text(
        json.dumps({"physical_to_logical_map": hybrid.tolist()}, indent=2) + "\n"
    )

    selection_results = []
    lhs = matrix @ selected_float
    for name, value, limit in zip(constraint_names, lhs, upper):
        selection_results.append(
            {
                "name": name,
                "normalized_value": float(value),
                "normalized_upper_bound": float(limit),
                "passed": bool(value <= limit + 1e-12),
            }
        )

    holdout_results = []
    for holdout_split in splits:
        if holdout_split == args.selection_split:
            continue
        for metric in scalar_metrics:
            value = float(
                (scalar_delta[holdout_split][metric] / scalar_reference[holdout_split][metric])
                @ selected_float
            )
            if metric in (
                "max_channel_p50",
                "max_channel_p90",
                "unique_sends_per_token",
                "remote_unique_sends_per_token",
            ):
                tolerance_pct = args.holdout_send_tolerance_pct
            else:
                tolerance_pct = 0.0
            holdout_results.append(
                {
                    "name": f"{holdout_split}:scalar:{metric}",
                    "percent_change": 100.0 * value,
                    "tolerance_pct": tolerance_pct,
                    "passed": bool(100.0 * value <= tolerance_pct + 1e-12),
                }
            )
        for kind in vector_kinds:
            ref = vector_reference[holdout_split][kind]
            delta = vector_delta[holdout_split][kind]
            reference_total = ref.sum(axis=0)
            candidate_total = reference_total + delta.T @ selected_float
            ref_max = float(reference_total.max())
            candidate_max = float(candidate_total.max())
            holdout_results.append(
                {
                    "name": f"{holdout_split}:rank_max:{kind}",
                    "reference_max": ref_max,
                    "candidate_max": candidate_max,
                    "percent_change": 100.0 * (candidate_max / ref_max - 1.0),
                    "tolerance_pct": args.holdout_rank_max_tolerance_pct,
                    "passed": bool(
                        100.0 * (candidate_max / ref_max - 1.0)
                        <= args.holdout_rank_max_tolerance_pct + 1e-12
                    ),
                }
            )

    selection_passed = all(item["passed"] for item in selection_results)
    holdout_passed = all(item["passed"] for item in holdout_results)
    status = (
        "ADMITTED_FOR_FULL_REPLAY"
        if selected_layers and selection_passed and holdout_passed
        else "REJECTED_HOLDOUT"
        if selected_layers and selection_passed
        else "REJECTED_OFFLINE"
    )
    result = {
        "status": status,
        "primitive": "whole-layer proposal selection with reference fallback",
        "optimization": {
            "solver": "scipy.optimize.milp",
            "solver_status": int(solution.status),
            "solver_message": solution.message,
            "objective": float(solution.fun),
            "selection_split": args.selection_split,
            "holdout_used_by_solver": False,
            "objective_weights": {
                "compute_mean": args.mean_objective_weight,
                "compute_p50": args.p50_objective_weight,
                "compute_p90": args.p90_objective_weight,
            },
            "rank_constraints": list(vector_kinds),
            "holdout_tolerances_pct": {
                "compute_mean_p50_p90": 0.0,
                "send_and_channel_scalars": args.holdout_send_tolerance_pct,
                "rank_maxima": args.holdout_rank_max_tolerance_pct,
            },
        },
        "inputs": {
            "layer_summary": {"path": str(args.layer_summary), "sha256": sha256(args.layer_summary)},
            "reference_map": {"path": str(args.reference_map), "sha256": sha256(args.reference_map)},
            "proposal_map": {"path": str(args.proposal_map), "sha256": sha256(args.proposal_map)},
        },
        "selected_layers": selected_layers,
        "selected_layer_count": len(selected_layers),
        "fallback_layers": [
            layer for layer in range(reference.shape[0]) if layer not in selected_layers
        ],
        "selection_constraints": selection_results,
        "holdout": {
            "read_after_solution_frozen": True,
            "passed": holdout_passed,
            "results": holdout_results,
        },
        "output_map": str(args.output_map),
        "output_map_sha256": sha256(args.output_map),
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(
        {
            "status": status,
            "selected_layers": selected_layers,
            "selected_layer_count": len(selected_layers),
            "objective": float(solution.fun),
            "holdout_passed": holdout_passed,
            "holdout_failures": [
                item for item in holdout_results if not item["passed"]
            ],
            "output_map": str(args.output_map),
            "output_map_sha256": sha256(args.output_map),
        },
        indent=2,
        sort_keys=True,
    ))
    return 0 if status == "ADMITTED_FOR_FULL_REPLAY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
