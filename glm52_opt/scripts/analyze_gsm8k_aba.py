#!/usr/bin/env python3
"""Compare retained GSM8K results from baseline/candidate/baseline runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path, expected_label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("protocol") != "sglang-builtin-gsm8k-retained-v1":
        raise SystemExit(f"unexpected protocol in {path}")
    if payload.get("label") != expected_label:
        raise SystemExit(
            f"unexpected label in {path}: {payload.get('label')!r}"
        )
    return payload


def _pair(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_samples = left["samples"]
    right_samples = right["samples"]
    if len(left_samples) != len(right_samples):
        raise SystemExit("sample count mismatch")
    differing_text = []
    differing_answer = []
    differing_correctness = []
    left_only_correct = []
    right_only_correct = []
    for lhs, rhs in zip(left_samples, right_samples):
        if lhs["dataset_index"] != rhs["dataset_index"]:
            raise SystemExit("dataset index mismatch")
        index = lhs["dataset_index"]
        if lhs["response_sha256"] != rhs["response_sha256"]:
            differing_text.append(index)
        if lhs["extracted_answer"] != rhs["extracted_answer"]:
            differing_answer.append(index)
        if lhs["correct"] != rhs["correct"]:
            differing_correctness.append(index)
            if lhs["correct"]:
                left_only_correct.append(index)
            else:
                right_only_correct.append(index)
    total = len(left_samples)
    return {
        "total": total,
        "exact_response_matches": total - len(differing_text),
        "extracted_answer_matches": total - len(differing_answer),
        "correctness_matches": total - len(differing_correctness),
        "differing_response_indices": differing_text,
        "differing_answer_indices": differing_answer,
        "differing_correctness_indices": differing_correctness,
        "left_only_correct_indices": left_only_correct,
        "right_only_correct_indices": right_only_correct,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--before-label", default="winners_before")
    parser.add_argument("--candidate-label", default="swiglu")
    parser.add_argument("--after-label", default="winners_after")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    before = _load(args.before, args.before_label)
    candidate = _load(args.candidate, args.candidate_label)
    after = _load(args.after, args.after_label)
    runs = [before, candidate, after]
    incomplete = {
        run["label"]: int(run["metrics"].get("empty_responses", -1))
        for run in runs
        if int(run["metrics"].get("empty_responses", -1)) != 0
    }
    if incomplete:
        raise SystemExit(f"quality gate contains empty responses: {incomplete}")

    dataset_fingerprints = {
        (
            run["dataset"]["sha256"],
            tuple(run["dataset"]["selected_dataset_indices"]),
            tuple(sample["prompt_sha256"] for sample in run["samples"]),
        )
        for run in runs
    }
    if len(dataset_fingerprints) != 1:
        raise SystemExit("A/B/A runs do not contain exactly the same prompts")
    request_contracts = {
        (
            run["request"]["temperature"],
            run["request"]["top_p"],
            run["request"]["max_tokens"],
            run["request"]["num_threads"],
            run["request"]["enable_thinking"],
        )
        for run in runs
    }
    if len(request_contracts) != 1:
        raise SystemExit("A/B/A request contracts differ")

    scores = [float(run["metrics"]["score"]) for run in runs]
    before_after = _pair(before, after)
    before_candidate = _pair(before, candidate)
    candidate_after = _pair(candidate, after)
    exact_metric_match = scores[0] == scores[1] == scores[2]
    within_restart_bracket = min(scores[0], scores[2]) <= scores[1] <= max(
        scores[0], scores[2]
    )
    if exact_metric_match:
        classification = "exact_metric_match"
    elif within_restart_bracket:
        classification = "within_baseline_restart_bracket"
    else:
        classification = "metric_shift_outside_baseline_restart_bracket"

    stable_baseline_correct = {
        sample["dataset_index"]
        for left, sample in zip(before["samples"], after["samples"])
        if left["correct"] and sample["correct"]
    }
    candidate_by_index = {
        sample["dataset_index"]: sample for sample in candidate["samples"]
    }
    candidate_regressions = sorted(
        index
        for index in stable_baseline_correct
        if not candidate_by_index[index]["correct"]
    )

    payload = {
        "protocol": "sglang-gsm8k-aba-analysis-v1",
        "classification": classification,
        "exact_metric_match": exact_metric_match,
        "candidate_within_baseline_restart_bracket": within_restart_bracket,
        "dataset": before["dataset"],
        "request_contract": before["request"],
        "runs": {
            run["label"]: run["metrics"]
            for run in runs
        },
        "score_sequence": {
            args.before_label: scores[0],
            args.candidate_label: scores[1],
            args.after_label: scores[2],
        },
        "candidate_minus_baseline_midpoint": scores[1]
        - (scores[0] + scores[2]) / 2.0,
        "stable_baseline_correct_candidate_regression_indices": candidate_regressions,
        "pairwise": {
            f"{args.before_label}_vs_{args.candidate_label}": before_candidate,
            f"{args.before_label}_vs_{args.after_label}": before_after,
            f"{args.candidate_label}_vs_{args.after_label}": candidate_after,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    total = before["metrics"]["total"]
    lines = [
        f"# GSM8K {total}-example A/B/A quality summary",
        "",
        f"Classification: **{classification}**.",
        "",
        "| arm | correct | total | accuracy | latency (s) | output tok/s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        metric = run["metrics"]
        lines.append(
            f"| {run['label']} | {metric['correct']} | {metric['total']} | "
            f"{100.0 * metric['score']:.2f}% | {metric['latency_s']:.3f} | "
            f"{metric['output_throughput_token_s']:.2f} |"
        )
    lines += [
        "",
        f"- Candidate minus baseline-score midpoint: "
        f"{100.0 * payload['candidate_minus_baseline_midpoint']:+.2f} percentage points.",
        f"- Baseline self-restart correctness differences: "
        f"{len(before_after['differing_correctness_indices'])}/{total}.",
        f"- Before/candidate correctness differences: "
        f"{len(before_candidate['differing_correctness_indices'])}/{total}.",
        f"- Stable-baseline-correct questions regressed by candidate: "
        f"{len(candidate_regressions)} ({candidate_regressions}).",
        "- Exact response text is diagnostic only; the primary gate is answer accuracy "
        "and per-question correctness under the frozen prompt set.",
    ]
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
