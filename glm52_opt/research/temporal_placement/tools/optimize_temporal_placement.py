#!/usr/bin/env python3
"""Build a fail-closed temporal expert-placement candidate for GLM-5.2.

The accepted placement balances counts aggregated over the full recording.  This
tool keeps that map as the local fallback and only swaps experts between EP
ranks when a layer's train-window critical-rank objective improves.  Every rank
retains exactly the same number of physical experts, and aggregate imbalance is
bounded relative to the accepted map.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", required=True, type=Path)
    parser.add_argument("--accepted-map", required=True, type=Path)
    parser.add_argument("--output-map", required=True, type=Path)
    parser.add_argument("--output-summary", required=True, type=Path)
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument("--exclude-leading-nonzero-steps", type=int, default=1)
    parser.add_argument("--samples-per-round", type=int, default=1024)
    parser.add_argument("--rounds", type=int, default=240)
    parser.add_argument("--patience", type=int, default=32)
    parser.add_argument("--aggregate-relative-cap", type=float, default=0.015)
    parser.add_argument("--p90-weight", type=float, default=0.20)
    parser.add_argument("--max-weight", type=float, default=0.05)
    parser.add_argument("--aggregate-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=565849983)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def q90(values: torch.Tensor) -> torch.Tensor:
    """Row-wise nearest-rank P90 for a [candidate, step] tensor."""
    index = max(1, (9 * values.shape[1] + 9) // 10)
    return values.kthvalue(index, dim=1).values


def objective(
    rank_loads: torch.Tensor,
    aggregate_ratio: torch.Tensor,
    *,
    ep_size: int,
    p90_weight: float,
    max_weight: float,
    aggregate_weight: float,
) -> torch.Tensor:
    # rank_loads is [candidate, step, rank] and each step sums to one.
    ratios = rank_loads.max(dim=2).values * ep_size
    return (
        ratios.mean(dim=1)
        + p90_weight * q90(ratios)
        + max_weight * ratios.max(dim=1).values
        + aggregate_weight * aggregate_ratio
    )


def placement_metrics(
    counts: torch.Tensor, placement: torch.Tensor, ep_size: int
) -> dict[str, float | list[float]]:
    num_steps, num_layers, num_experts = counts.shape
    per_rank = num_experts // ep_size
    ordered = counts.gather(
        2, placement.unsqueeze(0).expand(num_steps, -1, -1)
    )
    loads = ordered.reshape(num_steps, num_layers, ep_size, per_rank).sum(3)
    totals = loads.sum(2)
    active = totals > 0
    ratios = (loads.max(2).values * ep_size / totals.clamp_min(1))[active]
    step_ratios = loads.max(2).values.sum(1) * ep_size / totals.sum(1).clamp_min(1)
    aggregate = counts.sum(0).gather(1, placement)
    aggregate_loads = aggregate.reshape(num_layers, ep_size, per_rank).sum(2)
    aggregate_totals = aggregate_loads.sum(1)
    aggregate_active = aggregate_totals > 0
    aggregate_ratios = (
        aggregate_loads.max(1).values
        * ep_size
        / aggregate_totals.clamp_min(1)
    )[aggregate_active]

    def quantile(tensor: torch.Tensor, value: float) -> float:
        return float(torch.quantile(tensor, value).item())

    return {
        "layer_step_ratio_mean": float(ratios.mean().item()),
        "layer_step_ratio_p50": quantile(ratios, 0.50),
        "layer_step_ratio_p90": quantile(ratios, 0.90),
        "layer_step_ratio_p99": quantile(ratios, 0.99),
        "layer_step_ratio_max": float(ratios.max().item()),
        "step_critical_ratio_mean": float(step_ratios.mean().item()),
        "step_critical_ratio_p50": quantile(step_ratios, 0.50),
        "step_critical_ratio_p90": quantile(step_ratios, 0.90),
        "step_critical_ratio_max": float(step_ratios.max().item()),
        "step_critical_ratios": [float(x) for x in step_ratios],
        "aggregate_ratio_mean": float(aggregate_ratios.mean().item()),
        "aggregate_ratio_p90": quantile(aggregate_ratios, 0.90),
        "aggregate_ratio_max": float(aggregate_ratios.max().item()),
    }


def optimize_layer(
    normalized_train: torch.Tensor,
    accepted_positions: torch.Tensor,
    aggregate_counts: torch.Tensor,
    *,
    layer_id: int,
    ep_size: int,
    samples_per_round: int,
    rounds: int,
    patience: int,
    aggregate_relative_cap: float,
    p90_weight: float,
    max_weight: float,
    aggregate_weight: float,
    seed: int,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    num_experts = accepted_positions.numel()
    per_rank = num_experts // ep_size
    placement = accepted_positions.clone()
    train_loads = normalized_train[:, placement].reshape(
        normalized_train.shape[0], ep_size, per_rank
    ).sum(2)
    aggregate_total = aggregate_counts.sum().clamp_min(1)
    aggregate_loads = aggregate_counts.gather(0, placement).reshape(
        ep_size, per_rank
    ).sum(1)
    accepted_aggregate_ratio = float(
        (aggregate_loads.max() * ep_size / aggregate_total).item()
    )
    aggregate_limit = accepted_aggregate_ratio * (1 + aggregate_relative_cap)
    current_aggregate_ratio = torch.tensor(
        [accepted_aggregate_ratio], dtype=torch.float64
    )
    current_score = float(
        objective(
            train_loads.unsqueeze(0),
            current_aggregate_ratio,
            ep_size=ep_size,
            p90_weight=p90_weight,
            max_weight=max_weight,
            aggregate_weight=aggregate_weight,
        )[0].item()
    )
    initial_score = current_score
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + layer_id * 1_000_003)
    stale = 0
    accepted_swaps = 0

    for _ in range(rounds):
        first_rank = torch.randint(
            ep_size, (samples_per_round,), generator=generator
        )
        rank_offset = torch.randint(
            1, ep_size, (samples_per_round,), generator=generator
        )
        second_rank = (first_rank + rank_offset) % ep_size
        first_pos = first_rank * per_rank + torch.randint(
            per_rank, (samples_per_round,), generator=generator
        )
        second_pos = second_rank * per_rank + torch.randint(
            per_rank, (samples_per_round,), generator=generator
        )
        first_expert = placement[first_pos]
        second_expert = placement[second_pos]

        candidates = train_loads.unsqueeze(0).expand(
            samples_per_round, -1, -1
        ).clone()
        delta = normalized_train[:, second_expert].T - normalized_train[:, first_expert].T
        row = torch.arange(samples_per_round).unsqueeze(1)
        step = torch.arange(normalized_train.shape[0]).unsqueeze(0)
        candidates[row, step, first_rank.unsqueeze(1)] += delta
        candidates[row, step, second_rank.unsqueeze(1)] -= delta

        aggregate_delta = aggregate_counts[second_expert] - aggregate_counts[first_expert]
        candidate_aggregate = aggregate_loads.unsqueeze(0).expand(
            samples_per_round, -1
        ).clone()
        sample = torch.arange(samples_per_round)
        candidate_aggregate[sample, first_rank] += aggregate_delta
        candidate_aggregate[sample, second_rank] -= aggregate_delta
        aggregate_ratios = (
            candidate_aggregate.max(1).values * ep_size / aggregate_total
        )
        scores = objective(
            candidates,
            aggregate_ratios,
            ep_size=ep_size,
            p90_weight=p90_weight,
            max_weight=max_weight,
            aggregate_weight=aggregate_weight,
        )
        scores[aggregate_ratios > aggregate_limit] = torch.inf
        best_score, best_index = scores.min(0)
        if float(best_score.item()) + 1e-12 >= current_score:
            stale += 1
            if stale >= patience:
                break
            continue

        idx = int(best_index.item())
        p = int(first_pos[idx].item())
        q = int(second_pos[idx].item())
        placement[p], placement[q] = placement[q].clone(), placement[p].clone()
        train_loads = candidates[idx].clone()
        aggregate_loads = candidate_aggregate[idx].clone()
        current_score = float(best_score.item())
        accepted_swaps += 1
        stale = 0

    final_aggregate_ratio = float(
        (aggregate_loads.max() * ep_size / aggregate_total).item()
    )
    return placement, {
        "accepted_swaps": accepted_swaps,
        "initial_train_score": initial_score,
        "final_train_score": current_score,
        "train_score_improvement_percent": 100 * (initial_score - current_score) / initial_score,
        "accepted_aggregate_ratio": accepted_aggregate_ratio,
        "final_aggregate_ratio": final_aggregate_ratio,
        "aggregate_ratio_limit": aggregate_limit,
    }


def main() -> None:
    args = parse_args()
    record = torch.load(args.record, map_location="cpu", weights_only=True)
    counts = record["logical_count"].to(torch.float64)
    nonzero = torch.nonzero(counts.sum((1, 2)) > 0).flatten()
    if len(nonzero) <= args.exclude_leading_nonzero_steps + 2:
        raise RuntimeError("insufficient nonzero steps after exclusions")
    selected = nonzero[args.exclude_leading_nonzero_steps :]
    counts = counts[selected]
    # Alternating chronological steps form the development/holdout split.  The
    # holdout is never consulted by the swap selector.
    train_indices = torch.arange(0, counts.shape[0], 2)
    holdout_indices = torch.arange(1, counts.shape[0], 2)
    train = counts[train_indices]
    holdout = counts[holdout_indices]

    accepted_json = json.loads(args.accepted_map.read_text())
    accepted = torch.tensor(
        accepted_json["physical_to_logical_map"], dtype=torch.long
    )
    if accepted.shape != counts.shape[1:]:
        raise RuntimeError(
            f"map/record shape mismatch: {tuple(accepted.shape)} vs {tuple(counts.shape[1:])}"
        )
    num_layers, num_experts = accepted.shape
    if num_experts % args.ep_size:
        raise RuntimeError("experts must divide evenly over EP ranks")
    expected = torch.arange(num_experts)
    if not torch.all(accepted.sort(1).values == expected):
        raise RuntimeError("accepted map is not a permutation in every layer")

    candidate = accepted.clone()
    layer_details: dict[str, dict[str, float | int | str]] = {}
    changed_layers = 0
    for layer in range(num_layers):
        train_layer = train[:, layer]
        layer_totals = train_layer.sum(1)
        if not torch.all(layer_totals > 0):
            layer_details[str(layer)] = {"status": "local-fallback-inactive-layer"}
            continue
        normalized = train_layer / layer_totals.unsqueeze(1)
        optimized, details = optimize_layer(
            normalized,
            accepted[layer],
            # Aggregate load is an invariant rather than a fitted temporal
            # target, so bound it against the full selected recording.  The
            # holdout time-series is still never used by the swap objective.
            counts[:, layer].sum(0),
            layer_id=layer,
            ep_size=args.ep_size,
            samples_per_round=args.samples_per_round,
            rounds=args.rounds,
            patience=args.patience,
            aggregate_relative_cap=args.aggregate_relative_cap,
            p90_weight=args.p90_weight,
            max_weight=args.max_weight,
            aggregate_weight=args.aggregate_weight,
            seed=args.seed,
        )
        if details["accepted_swaps"] > 0:
            candidate[layer] = optimized
            changed_layers += 1
            details["status"] = "temporal-optimized"
        else:
            details["status"] = "local-fallback-no-train-win"
        layer_details[str(layer)] = details

    if not torch.all(candidate.sort(1).values == expected):
        raise RuntimeError("candidate map violates the per-layer permutation invariant")
    accepted_train = placement_metrics(train, accepted, args.ep_size)
    candidate_train = placement_metrics(train, candidate, args.ep_size)
    accepted_holdout = placement_metrics(holdout, accepted, args.ep_size)
    candidate_holdout = placement_metrics(holdout, candidate, args.ep_size)
    accepted_all = placement_metrics(counts, accepted, args.ep_size)
    candidate_all = placement_metrics(counts, candidate, args.ep_size)

    # Global fail-closed admission for the offline development candidate.  A
    # service screen is still required before promotion.
    train_win = (
        candidate_train["step_critical_ratio_mean"]
        < accepted_train["step_critical_ratio_mean"]
    )
    holdout_nonregression = (
        candidate_holdout["step_critical_ratio_mean"]
        <= accepted_holdout["step_critical_ratio_mean"]
    )
    aggregate_cap = (
        candidate_all["aggregate_ratio_max"]
        <= accepted_all["aggregate_ratio_max"] * (1 + args.aggregate_relative_cap)
    )
    status = (
        "ADMITTED_FOR_CORRECTNESS_ONLY"
        if train_win and holdout_nonregression and aggregate_cap
        else "REJECTED_OFFLINE"
    )

    args.output_map.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_map.write_text(
        json.dumps({"physical_to_logical_map": candidate.tolist()}, indent=2) + "\n"
    )
    summary = {
        "status": status,
        "candidate": "glm52-temporal-critical-rank-placement-v1",
        "source_record": str(args.record),
        "source_record_sha256": sha256(args.record),
        "accepted_map": str(args.accepted_map),
        "accepted_map_sha256": sha256(args.accepted_map),
        "output_map": str(args.output_map),
        "output_map_sha256": sha256(args.output_map),
        "invariants": {
            "ep_size": args.ep_size,
            "experts_per_rank": num_experts // args.ep_size,
            "per_layer_permutation": True,
            "inactive_layer_local_fallback": True,
            "aggregate_constraint_source": "all-selected-step marginal counts",
            "changed_layers": changed_layers,
            "nonzero_step_ids": [int(x) for x in nonzero],
            "selected_step_ids": [int(nonzero[x]) for x in range(args.exclude_leading_nonzero_steps, len(nonzero))],
            "train_selected_offsets": [int(x) for x in train_indices],
            "holdout_selected_offsets": [int(x) for x in holdout_indices],
        },
        "objective": {
            "p90_weight": args.p90_weight,
            "max_weight": args.max_weight,
            "aggregate_weight": args.aggregate_weight,
            "aggregate_relative_cap": args.aggregate_relative_cap,
            "samples_per_round": args.samples_per_round,
            "rounds": args.rounds,
            "patience": args.patience,
            "seed": args.seed,
        },
        "gates": {
            "train_step_critical_mean_win": train_win,
            "holdout_step_critical_mean_nonregression": holdout_nonregression,
            "aggregate_max_within_relative_cap": aggregate_cap,
        },
        "metrics": {
            "train": {"accepted": accepted_train, "candidate": candidate_train},
            "holdout": {"accepted": accepted_holdout, "candidate": candidate_holdout},
            "all": {"accepted": accepted_all, "candidate": candidate_all},
        },
        "layer_details": layer_details,
    }
    args.output_summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
