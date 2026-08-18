#!/usr/bin/env python3
"""B300 correctness/performance gate for destination-rank MoE route skew.

The production host bucket supplies a 128-CTA pool at local M=16.  These cases
deliberately make ``sum(masked_m)`` larger than 128 and compare every active FP8
code and packed UE8M0 byte against the unchanged stock kernel.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Iterable

import torch

from sglang.jit_kernel.dsv4 import silu_and_mul_masked_post_quant
from sglang.srt.layers.glm52_opt.swiglu_quant import silu_mul_quant_packed_into


EXPERTS = 32
SLAB = 8192
GATE_UP = 4096
HIDDEN = 2048
PACKED_GROUPS = 4
TOPK = 8


def _round_robin_counts(total: int) -> list[int]:
    base, remainder = divmod(total, EXPERTS)
    return [base + (expert < remainder) for expert in range(EXPERTS)]


def _cases() -> Iterable[tuple[str, list[int]]]:
    yield "balanced_128", _round_robin_counts(128)
    yield "observed_rank_skew_143", _round_robin_counts(143)
    yield "two_pool_waves_256", [256, *([0] * (EXPERTS - 1))]
    yield "ep8_receive_upper_1024", [1024, *([0] * (EXPERTS - 1))]


def _launch_stock(
    gate_up: torch.Tensor,
    output: torch.Tensor,
    scales: torch.Tensor,
    masked_m: torch.Tensor,
) -> None:
    silu_and_mul_masked_post_quant(
        gate_up,
        output,
        scales,
        128,
        masked_m,
        scale_ue8m0=True,
        topk=TOPK,
        transposed=True,
    )


def _launch_candidate(
    gate_up: torch.Tensor,
    output: torch.Tensor,
    scales: torch.Tensor,
    masked_m: torch.Tensor,
    variant: str,
) -> None:
    silu_mul_quant_packed_into(
        gate_up,
        output,
        scales,
        masked_m,
        num_real_tokens=16,
        variant=variant,
    )


def _median_us(fn, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="cuda_grid_stride")
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability != (10, 3):
        raise RuntimeError(f"B300 sm_103 required, found {capability}")

    torch.manual_seed(20260803)
    gate_up = torch.randn(
        (EXPERTS, SLAB, GATE_UP), device="cuda", dtype=torch.bfloat16
    )
    reference = torch.empty(
        (EXPERTS, SLAB, HIDDEN), device="cuda", dtype=torch.float8_e4m3fn
    )
    candidate = torch.empty_like(reference)
    reference_scales = torch.empty(
        (EXPERTS, PACKED_GROUPS, SLAB), device="cuda", dtype=torch.int32
    )
    candidate_scales = torch.empty_like(reference_scales)

    rows = []
    for name, counts in _cases():
        masked_m = torch.tensor(counts, device="cuda", dtype=torch.int32)
        reference.fill_(float("nan"))
        candidate.fill_(float("nan"))
        reference_scales.fill_(0x7F7F7F7F)
        candidate_scales.fill_(0x7F7F7F7F)

        _launch_stock(gate_up, reference, reference_scales, masked_m)
        _launch_candidate(
            gate_up, candidate, candidate_scales, masked_m, args.variant
        )
        torch.cuda.synchronize()

        value_mismatches = 0
        scale_mismatches = 0
        for expert, count in enumerate(counts):
            if count == 0:
                continue
            value_mismatches += int(
                (reference[expert, :count] != candidate[expert, :count])
                .sum()
                .item()
            )
            scale_mismatches += int(
                (
                    reference_scales[expert, :, :count]
                    != candidate_scales[expert, :, :count]
                )
                .sum()
                .item()
            )

        reference_us = _median_us(
            lambda: _launch_stock(gate_up, reference, reference_scales, masked_m),
            args.warmup,
            args.repeat,
        )
        candidate_us = _median_us(
            lambda: _launch_candidate(
                gate_up, candidate, candidate_scales, masked_m, args.variant
            ),
            args.warmup,
            args.repeat,
        )
        row = {
            "case": name,
            "cta_pool": 16 * TOPK,
            "received_assignments": sum(counts),
            "max_expert_rows": max(counts),
            "value_mismatches": value_mismatches,
            "packed_scale_mismatches": scale_mismatches,
            "reference_us": round(reference_us, 6),
            "candidate_us": round(candidate_us, 6),
            "speedup": round(reference_us / candidate_us, 6),
        }
        print(json.dumps(row, sort_keys=True), flush=True)
        rows.append(row)

    if any(
        row["value_mismatches"] or row["packed_scale_mismatches"] for row in rows
    ):
        raise SystemExit("route-complete correctness FAILED")
    print(json.dumps({"status": "PASS", "cases": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
