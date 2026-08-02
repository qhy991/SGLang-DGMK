#!/usr/bin/env python3
"""B300 micro A/B for router -> padded mask -> DeepEP int64 IDs.

The stock boundary creates int32 top-k IDs, masks CUDA-graph padding, then
casts the 16x8 tensor to the int64 ABI required by DeepEP low-latency dispatch.
The candidate writes masked int64 IDs directly from the router kernel.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from bench_router_pad_mask_b300 import capture, paired_measure
from sglang.jit_kernel.dsv4 import mask_topk_ids
from sglang.jit_kernel.moe_fused_gate import moe_fused_gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--captured-m", type=int, default=16)
    parser.add_argument("--num-valid", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--repeat", type=int, default=12)
    parser.add_argument("--flush-mib", type=int, default=32)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability != (10, 3):
        raise RuntimeError(f"expected B300 SM103, got SM{capability[0]}{capability[1]}")
    if not 0 <= args.num_valid <= args.captured_m:
        raise ValueError("num-valid must be within the captured row count")

    torch.manual_seed(20260802)
    scores = torch.randn(
        args.captured_m, 256, dtype=torch.float32, device="cuda"
    )
    bias = 0.05 * torch.randn(256, dtype=torch.float32, device="cuda")
    num_token_non_padded = torch.tensor(
        args.num_valid, dtype=torch.int32, device="cuda"
    )
    flush = torch.empty(
        args.flush_mib * 1024 * 1024 // 2,
        dtype=torch.bfloat16,
        device="cuda",
    )

    def reference() -> tuple[torch.Tensor, torch.Tensor]:
        weights, ids = moe_fused_gate(
            scores,
            bias,
            8,
            scoring_func="sigmoid",
            renormalize=True,
            routed_scaling_factor=2.5,
        )
        mask_topk_ids(ids, num_token_non_padded)
        return weights, ids.to(torch.int64)

    def candidate() -> tuple[torch.Tensor, torch.Tensor]:
        return moe_fused_gate(
            scores,
            bias,
            8,
            scoring_func="sigmoid",
            renormalize=True,
            routed_scaling_factor=2.5,
            num_token_non_padded=num_token_non_padded,
            output_ids_int64=True,
        )

    reference_weights, reference_ids = reference()
    candidate_weights, candidate_ids = candidate()
    torch.cuda.synchronize()
    if not torch.equal(candidate_weights, reference_weights):
        raise AssertionError("top-k weights changed")
    if not torch.equal(candidate_ids, reference_ids):
        raise AssertionError("masked int64 top-k IDs changed")
    if candidate_ids.dtype != torch.int64:
        raise AssertionError(f"candidate IDs have dtype {candidate_ids.dtype}")

    eager = paired_measure(reference, candidate, flush, args.warmup, args.repeat)
    reference_graph = capture(reference, args.warmup)
    candidate_graph = capture(candidate, args.warmup)
    graph = paired_measure(
        reference_graph, candidate_graph, flush, args.warmup, args.repeat
    )
    report = {
        "schema_version": 1,
        "gpu": torch.cuda.get_device_name(),
        "capability": list(capability),
        "captured_m": args.captured_m,
        "num_token_non_padded": args.num_valid,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "flush_mib": args.flush_mib,
        "reference": "moe_fused_gate(int32) + mask_topk_ids + to(int64)",
        "candidate": (
            "moe_fused_gate(num_token_non_padded, output_ids_int64=True)"
        ),
        "correctness": {
            "topk_weights_exact": True,
            "masked_topk_ids_exact": True,
            "candidate_ids_dtype": "torch.int64",
        },
        "measurements": {"eager": eager, "graph": graph},
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
