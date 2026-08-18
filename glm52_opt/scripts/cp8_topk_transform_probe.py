#!/usr/bin/env python3
"""Production-shape GLM DSA fused TopK-transform screen.

This closes the KDA v24 migration question independently of MQA: compare the
current one-launch ``fast_topk_transform_ragged_fused`` final-index ABI against
``fast_topk_v2`` followed by a separate raw-index offset/materialization pass.
The representative CP8 combined-indexer shape is M1252, context 100032, and
TopK2048.  This is a single-GPU operator test, not an E2E candidate.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import torch
from sgl_kernel import fast_topk_transform_ragged_fused, fast_topk_v2


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - position) + ordered[hi] * (position - lo)


def time_one(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    end.synchronize()
    if out.numel() == 0:
        raise RuntimeError("empty top-k output")
    return float(start.elapsed_time(end))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1252)
    parser.add_argument("--context", type=int, default=100_032)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if torch.cuda.get_device_capability() != (10, 3):
        raise RuntimeError("frozen probe requires B300 SM103")
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260818)
    score = torch.randn(
        (args.m, args.context),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    half = args.m // 2
    first_lengths = torch.arange(
        args.context - 2 * half + 1,
        args.context - half + 1,
        dtype=torch.int32,
        device="cuda",
    )
    second_lengths = torch.arange(
        args.context - half + 1,
        args.context + 1,
        dtype=torch.int32,
        device="cuda",
    )
    lengths = torch.cat((first_lengths, second_lengths))[: args.m].contiguous()
    row_starts = torch.zeros(args.m, dtype=torch.int32, device="cuda")
    offsets = torch.zeros(args.m, dtype=torch.int32, device="cuda")

    def separate():
        raw = fast_topk_v2(score, lengths, args.topk, row_starts=row_starts)
        return torch.where(raw >= 0, raw + offsets[:, None], raw)

    def fused():
        return fast_topk_transform_ragged_fused(
            score=score,
            lengths=lengths,
            topk_indices_offset=offsets,
            topk=args.topk,
            row_starts=row_starts,
        )

    reference = separate()
    candidate = fused()
    torch.cuda.synchronize()
    exact_order = torch.equal(reference, candidate)
    exact_set = torch.equal(
        torch.sort(reference, dim=-1).values,
        torch.sort(candidate, dim=-1).values,
    )
    if not exact_set:
        raise RuntimeError("fused TopK-transform changes the selected index set")

    for iteration in range(args.warmup):
        if iteration % 2:
            fused()
            separate()
        else:
            separate()
            fused()
    torch.cuda.synchronize()

    separate_ms, fused_ms, orders = [], [], []
    for iteration in range(args.repeat + 1):
        if iteration % 2:
            orders.append("FS")
            fused_ms.append(time_one(fused))
            separate_ms.append(time_one(separate))
        else:
            orders.append("SF")
            separate_ms.append(time_one(separate))
            fused_ms.append(time_one(fused))
    separate_ms = separate_ms[1:]
    fused_ms = fused_ms[1:]
    speedups = [baseline / candidate for baseline, candidate in zip(separate_ms, fused_ms)]
    median_speedup = statistics.median(speedups)
    result = {
        "schema": "glm52-cp8-fused-topk-transform-v1",
        "status": "PASS",
        "gpu": torch.cuda.get_device_name(),
        "m": args.m,
        "context": args.context,
        "topk": args.topk,
        "exact_index_order": exact_order,
        "exact_index_set": exact_set,
        "separate_median_ms": statistics.median(separate_ms),
        "fused_median_ms": statistics.median(fused_ms),
        "median_speedup": median_speedup,
        "speedup_p10": percentile(speedups, 0.10),
        "speedup_p90": percentile(speedups, 0.90),
        "separate_ms": separate_ms,
        "fused_ms": fused_ms,
        "pair_order_after_discard": orders[1:],
        "decision": (
            "ALREADY_EXPRESSED_POSITIVE"
            if median_speedup >= 1.01
            else "TESTED_NULL_OR_NEGATIVE"
        ),
        "scope": "TopK final-index transform only; MQA logits materialization is outside this probe",
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="", flush=True)
    if args.output:
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
