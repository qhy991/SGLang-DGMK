#!/usr/bin/env python3
"""Paired B300 benchmark for folding the decode padding mask into the router.

The reference is the exact production boundary: unified Triton router followed
by ``mask_topk_ids``.  The candidate passes the live CUDA int32 token-count
scalar into the router and writes ``-1`` for padded rows in the router store.
Both eager and CUDA-graph measurements are reported; promotion should use the
graph result because GLM-5.2 decode executes this boundary inside its graph.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from sglang.jit_kernel.dsv4 import mask_topk_ids
from sglang.jit_kernel.moe_fused_gate import moe_fused_gate


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def describe(values: list[float]) -> dict[str, float]:
    return {
        "p10": percentile(values, 0.10),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "min": min(values),
        "max": max(values),
    }


def flush_l2(buffer: torch.Tensor) -> None:
    buffer.zero_()
    torch.cuda.synchronize()


def elapsed_us(call: Callable[[], Any]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    call()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) * 1e3)


def capture(call: Callable[[], Any], warmup: int) -> Callable[[], None]:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    torch.cuda.synchronize()
    return graph.replay


def paired_measure(
    reference: Callable[[], Any],
    candidate: Callable[[], Any],
    flush: torch.Tensor,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        reference()
        candidate()
    torch.cuda.synchronize()

    reference_us: list[float] = []
    candidate_us: list[float] = []
    speedups: list[float] = []
    for index in range(repeat):
        order = (
            (("reference", reference), ("candidate", candidate))
            if index % 2 == 0
            else (("candidate", candidate), ("reference", reference))
        )
        sample: dict[str, float] = {}
        for label, call in order:
            flush_l2(flush)
            sample[label] = elapsed_us(call)
        reference_us.append(sample["reference"])
        candidate_us.append(sample["candidate"])
        speedups.append(sample["reference"] / sample["candidate"])
    return {
        "reference_us": describe(reference_us),
        "candidate_us": describe(candidate_us),
        "paired_speedup": describe(speedups),
        "paired_samples": speedups,
    }


def analyze_num_valid(
    num_valid: int,
    scores: torch.Tensor,
    bias: torch.Tensor,
    flush: torch.Tensor,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    ntn = torch.tensor(num_valid, dtype=torch.int32, device="cuda")

    def reference() -> tuple[torch.Tensor, torch.Tensor]:
        weights, ids = moe_fused_gate(
            scores,
            bias,
            8,
            scoring_func="sigmoid",
            renormalize=True,
            routed_scaling_factor=2.5,
        )
        mask_topk_ids(ids, ntn)
        return weights, ids

    def candidate() -> tuple[torch.Tensor, torch.Tensor]:
        return moe_fused_gate(
            scores,
            bias,
            8,
            scoring_func="sigmoid",
            renormalize=True,
            routed_scaling_factor=2.5,
            num_token_non_padded=ntn,
        )

    ref_weights, ref_ids = reference()
    fused_weights, fused_ids = candidate()
    torch.cuda.synchronize()
    if not torch.equal(fused_weights, ref_weights):
        raise AssertionError(f"num_valid={num_valid}: top-k weights changed")
    if not torch.equal(fused_ids, ref_ids):
        raise AssertionError(f"num_valid={num_valid}: top-k IDs changed")

    eager = paired_measure(reference, candidate, flush, warmup, repeat)
    reference_graph = capture(reference, warmup)
    candidate_graph = capture(candidate, warmup)
    graph = paired_measure(
        reference_graph, candidate_graph, flush, warmup, repeat
    )
    return {
        "num_token_non_padded": num_valid,
        "correctness": {
            "topk_ids_exact": True,
            "topk_weights_exact": True,
        },
        "measurements": {"eager": eager, "graph": graph},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--captured-m", type=int, default=16)
    parser.add_argument("--num-valid", default="1,2,4,8,12,15,16")
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--repeat", type=int, default=40)
    parser.add_argument("--flush-mib", type=int, default=256)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability != (10, 3):
        raise RuntimeError(f"expected B300 SM103, got SM{capability[0]}{capability[1]}")

    torch.manual_seed(20260802)
    scores = torch.randn(
        args.captured_m, 256, dtype=torch.float32, device="cuda"
    )
    bias = 0.05 * torch.randn(256, dtype=torch.float32, device="cuda")
    flush = torch.empty(
        args.flush_mib * 1024 * 1024 // 2,
        dtype=torch.bfloat16,
        device="cuda",
    )
    valid_counts = [int(value) for value in args.num_valid.split(",")]
    if any(value < 0 or value > args.captured_m for value in valid_counts):
        raise ValueError("num-valid values must be within the captured row count")

    output = {
        "schema_version": 1,
        "gpu": torch.cuda.get_device_name(),
        "capability": list(capability),
        "captured_m": args.captured_m,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "flush_mib": args.flush_mib,
        "reference": "moe_fused_gate + mask_topk_ids",
        "candidate": "moe_fused_gate(num_token_non_padded=CUDA scalar)",
        "results": [
            analyze_num_valid(
                value, scores, bias, flush, args.warmup, args.repeat
            )
            for value in valid_counts
        ],
    }
    encoded = json.dumps(output, indent=2, sort_keys=True)
    print(encoded)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
