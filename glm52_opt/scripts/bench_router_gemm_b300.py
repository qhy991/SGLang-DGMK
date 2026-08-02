#!/usr/bin/env python3
"""Compare the B300 router JIT kernel with SGLang's exact production fallback.

The generic Kernel Harness router region uses an FP32 ``F.linear`` reference.
SGLang's actual CUDA fallback is ``linear_bf16_fp32`` (BF16 inputs with FP32
output), so promotion decisions must compare against that callable directly.
This script measures both the projection and the projection+top-k region in
eager and CUDA-graph modes for every production decode graph bucket.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from sglang.jit_kernel.dsv3_router_gemm import dsv3_router_gemm
from sglang.jit_kernel.dsv4 import linear_bf16_fp32
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


def capture(call: Callable[[], Any], warmup: int) -> tuple[Callable[[], None], Any]:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = call()
    torch.cuda.synchronize()
    return graph.replay, output


def make_region_call(
    projection: Callable[[], torch.Tensor],
    correction_bias: torch.Tensor,
) -> Callable[[], tuple[torch.Tensor, torch.Tensor]]:
    def call() -> tuple[torch.Tensor, torch.Tensor]:
        return moe_fused_gate(
            projection(),
            correction_bias,
            8,
            scoring_func="sigmoid",
            renormalize=True,
            routed_scaling_factor=2.5,
            apply_routed_scaling_factor_on_output=False,
        )

    return call


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


def analyze_shape(
    m: int,
    warmup: int,
    repeat: int,
    flush: torch.Tensor,
) -> dict[str, Any]:
    torch.manual_seed(20260802 + m)
    hidden = torch.randn(m, 6144, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(256, 6144, dtype=torch.bfloat16, device="cuda")
    correction_bias = 0.05 * torch.randn(256, dtype=torch.float32, device="cuda")
    jit_output = torch.empty(m, 256, dtype=torch.float32, device="cuda")

    def reference_projection() -> torch.Tensor:
        return linear_bf16_fp32(hidden, weight)

    def candidate_projection() -> torch.Tensor:
        return dsv3_router_gemm(
            hidden,
            weight,
            out_dtype=torch.float32,
            output=jit_output,
        )

    reference_logits = reference_projection()
    candidate_logits = candidate_projection().clone()
    reference_region = make_region_call(reference_projection, correction_bias)
    candidate_region = make_region_call(candidate_projection, correction_bias)
    reference_weights, reference_ids = reference_region()
    candidate_weights, candidate_ids = candidate_region()
    torch.cuda.synchronize()

    ids_equal = bool(torch.equal(candidate_ids, reference_ids))
    logits_abs = float((candidate_logits - reference_logits).abs().max().item())
    weights_abs = float((candidate_weights - reference_weights).abs().max().item())
    if not ids_equal:
        raise AssertionError(f"M={m}: JIT router changes top-k expert IDs")
    torch.testing.assert_close(
        candidate_weights,
        reference_weights,
        atol=2e-5,
        rtol=2e-5,
    )

    result: dict[str, Any] = {
        "M": m,
        "correctness": {
            "topk_ids_exact": ids_equal,
            "logits_max_abs": logits_abs,
            "topk_weights_max_abs": weights_abs,
        },
        "measurements": {},
    }
    for region_name, reference_call, candidate_call in (
        ("projection", reference_projection, candidate_projection),
        ("projection_topk", reference_region, candidate_region),
    ):
        eager = paired_measure(
            reference_call, candidate_call, flush, warmup, repeat
        )
        reference_graph, _ = capture(reference_call, warmup)
        candidate_graph, _ = capture(candidate_call, warmup)
        graph = paired_measure(
            reference_graph, candidate_graph, flush, warmup, repeat
        )
        result["measurements"][region_name] = {"eager": eager, "graph": graph}
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", default="1,2,4,8,12,16")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--flush-mib", type=int, default=256)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (10, 3):
        raise RuntimeError(f"expected B300 SM103, got SM{major}{minor}")
    shapes = [int(value) for value in args.shapes.split(",")]
    flush = torch.empty(
        args.flush_mib * 1024 * 1024 // 2,
        dtype=torch.bfloat16,
        device="cuda",
    )
    output = {
        "schema_version": 1,
        "gpu": torch.cuda.get_device_name(),
        "capability": [major, minor],
        "shapes": shapes,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "flush_mib": args.flush_mib,
        "reference": "sglang.jit_kernel.dsv4.linear_bf16_fp32",
        "candidate": "sglang.jit_kernel.dsv3_router_gemm(out_dtype=float32)",
        "results": [
            analyze_shape(m, args.warmup, args.repeat, flush) for m in shapes
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
