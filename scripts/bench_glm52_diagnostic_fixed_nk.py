#!/usr/bin/env python3
"""Fair production-wrapper A/B for the exhaustive GLM-5.2 fixed-N/K matrix."""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
from contextlib import contextmanager

import torch

SHAPES = (
    ("fused_qkv_a_proj", 2624, 6144),
    ("q_b_proj", 16384, 2048),
    ("o_proj", 6144, 16384),
    ("dense_gate_up_proj", 4096, 6144),
    ("dense_down_proj", 6144, 2048),
    ("index_q_upproj", 4096, 2048),
    ("index_k_proj", 128, 6144),
)
CASES = tuple(
    (op, "decode", m, n, k)
    for op, n, k in SHAPES
    for m in (16, 32)
) + tuple(
    (op, "prefill", 4096, n, k)
    for op, n, k in SHAPES
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--series", type=int, default=3)
    return parser.parse_args()


@contextmanager
def _selected(op: str, enabled: bool):
    previous = os.environ.get("SGLANG_GLM52_OPT")
    os.environ["SGLANG_GLM52_OPT"] = "1" if enabled else "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("SGLANG_GLM52_OPT", None)
        else:
            os.environ["SGLANG_GLM52_OPT"] = previous


def _measure(fn, repeats: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) * 1000.0 / repeats)


def _capture(fn) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    for _ in range(3):
        out = fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph, out


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _bench_case(
    case: tuple[str, str, int, int, int],
    *,
    repeats: int,
    series: int,
) -> dict[str, object]:
    from sglang.srt.layers.glm52_opt.context import op_context, set_forward_mode
    from sglang.srt.layers.glm52_opt.dispatch import _HIT_COUNTS
    from sglang.srt.layers.quantization.fp8_utils import (
        deepgemm_w8a8_block_fp8_linear_with_fallback,
    )
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    op, phase, m, n, k = case
    mode = ForwardMode.DECODE if phase == "decode" else ForwardMode.EXTEND
    set_forward_mode(mode, m)
    os.environ["SGLANG_GLM52_OPT_OPS"] = op
    os.environ["SGLANG_GLM52_OPT_M_BUCKETS"] = f"{op}:{m}"

    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260730 + m + n + k)
    x = torch.randn(
        (m, k), generator=generator, device="cuda", dtype=torch.bfloat16
    )
    weight = torch.randn(
        (n, k), generator=generator, device="cuda", dtype=torch.bfloat16
    ).to(torch.float8_e4m3fn)
    weight_scale = torch.full(
        (k // 128 // 4, n),
        0x7F7F7F7F,
        device="cuda",
        dtype=torch.int32,
    ).T

    def run(enabled: bool):
        with _selected(op, enabled), op_context(op):
            return deepgemm_w8a8_block_fp8_linear_with_fallback(
                x, weight, [128, 128], weight_scale
            )

    stock = run(False)
    candidate = run(True)
    exact = bool(torch.equal(stock, candidate))

    eager_stock: list[float] = []
    eager_candidate: list[float] = []
    for index in range(series):
        order = (False, True, True, False) if index % 2 == 0 else (
            True,
            False,
            False,
            True,
        )
        for enabled in order:
            with _selected(op, enabled):
                value = _measure(lambda: run(enabled), repeats)
            (eager_candidate if enabled else eager_stock).append(value)

    with _selected(op, False):
        stock_graph, stock_graph_out = _capture(lambda: run(False))
    with _selected(op, True):
        candidate_graph, candidate_graph_out = _capture(lambda: run(True))
    graph_exact = bool(torch.equal(stock_graph_out, candidate_graph_out))

    graph_stock: list[float] = []
    graph_candidate: list[float] = []
    for index in range(series):
        order = (False, True, True, False) if index % 2 == 0 else (
            True,
            False,
            False,
            True,
        )
        for enabled in order:
            graph = candidate_graph if enabled else stock_graph
            value = _measure(graph.replay, repeats)
            (graph_candidate if enabled else graph_stock).append(value)

    eager_stock_us = _median(eager_stock)
    eager_candidate_us = _median(eager_candidate)
    graph_stock_us = _median(graph_stock)
    graph_candidate_us = _median(graph_candidate)
    hit_key = f"fp8_gemm/fixed_nk:{op}:{phase}:m{m}"
    result = {
        "op": op,
        "phase": phase,
        "m": m,
        "n": n,
        "k": k,
        "exact": exact,
        "graph_exact": graph_exact,
        "hit_count": int(_HIT_COUNTS.get(hit_key, 0)),
        "eager_stock_us": eager_stock_us,
        "eager_candidate_us": eager_candidate_us,
        "eager_speedup": eager_stock_us / eager_candidate_us,
        "graph_stock_us": graph_stock_us,
        "graph_candidate_us": graph_candidate_us,
        "graph_speedup": graph_stock_us / graph_candidate_us,
    }

    set_forward_mode(None)
    return result


def main() -> None:
    args = _parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    os.environ["SGLANG_GLM52_OPT"] = "1"
    os.environ["SGLANG_GLM52_OPT_PROFILE"] = "diagnostic_all"
    os.environ["SGLANG_GLM52_ALLOW_ABI_ADAPTER"] = "0"
    os.environ["SGLANG_GLM52_OPT_HIT_FILE"] = (
        f"/tmp/glm52-fixed-nk-shard-{args.shard_index}-hits.json"
    )
    # This benchmark compiles only the selected exact shapes on first use.
    # Server-wide DeepGEMM precompile enumeration is startup work and must not
    # enter either timed arm.
    from sglang.srt.layers.deep_gemm_wrapper import compile_utils

    compile_utils._ENABLE_JIT_DEEPGEMM_PRECOMPILE = False

    selected_cases = CASES[args.shard_index :: args.num_shards]
    results = []
    for case in selected_cases:
        results.append(
            _bench_case(case, repeats=args.repeats, series=args.series)
        )
        gc.collect()
        torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "schema": "glm52-diagnostic-fixed-nk-v1",
                "device": torch.cuda.get_device_name(),
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "repeats": args.repeats,
                "series": args.series,
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
