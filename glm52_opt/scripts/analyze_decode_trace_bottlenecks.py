#!/usr/bin/env python3
"""Rank fixed-KV decode kernels and cross-rank wait imbalance from nsys SQLite."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ACTIVATION_SHORT_NAME = "silu_mul_quant_varlen_kernel"


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def describe(values: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    return {
        "mean": mean,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p99": percentile(values, 0.99),
        "max": max(values),
        "cv": statistics.pstdev(values) / mean if mean else 0.0,
    }


def deepgemm_shape(name: str) -> tuple[int, int, int, str] | None:
    if "deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl" not in name:
        return None
    dimensions = [int(value) for value in re.findall(r"\(unsigned int\)(\d+)", name)]
    gemm_type = re.search(r"\(deep_gemm::GemmType\)(\d+)", name)
    if len(dimensions) < 7 or gemm_type is None:
        return None
    return dimensions[4], dimensions[5], dimensions[6], gemm_type.group(1)


def category(name: str) -> str:
    lowered = name.lower()
    if "nccl" in lowered:
        return "communication_nccl"
    if "deep_ep" in lowered or "deepep" in lowered:
        return "communication_deepep"
    shape = deepgemm_shape(name)
    if shape is not None:
        k, n, m, gemm_type = shape
        if (k, n, m, gemm_type) == (4096, 6144, 16, "2"):
            return "moe_w13"
        if (k, n, m, gemm_type) == (6144, 2048, 16, "2"):
            return "moe_w2"
        return "deepgemm_other"
    if "flash_fwd" in lowered or "flashmla" in lowered or "flash_mla" in lowered:
        return "flashmla"
    if "silu_mul_quant_varlen" in lowered:
        return "moe_swiglu_quant"
    if "per_token_group_quant" in lowered or "quantize_k_cache" in lowered:
        return "quantization"
    if "router" in lowered or "topk" in lowered:
        return "router_topk"
    if "rmsnorm" in lowered or "rms_norm" in lowered or "rope" in lowered:
        return "normalization_rope"
    if "concat_mla_absorb" in lowered or "paged_mqa" in lowered or "mqa_logits" in lowered:
        return "indexer_mla_helpers"
    if "elementwise_kernel" in lowered or "vectorized_elementwise_kernel" in lowered:
        return "torch_elementwise"
    if "reduce_kernel" in lowered or "devicereduce" in lowered or "devicescan" in lowered:
        return "torch_reduction"
    return "other"


def concise_name(name: str) -> str:
    shape = deepgemm_shape(name)
    if shape is not None:
        k, n, m, gemm_type = shape
        return f"deepgemm_k{k}_n{n}_m{m}_type{gemm_type}"
    markers = (
        ("ncclDevKernel_AllGather_RING_LL", "nccl_allgather_ring_ll"),
        ("deep_ep::intranode::notify_dispatch", "deepep_notify_dispatch"),
        ("deep_ep::intranode::cached_notify_combine", "deepep_cached_notify_combine"),
        ("deep_ep::intranode::dispatch<", "deepep_dispatch"),
        ("deep_ep::intranode::combine<", "deepep_combine"),
        ("deep_ep::internode_ll::dispatch", "deepep_ll_dispatch"),
        ("deep_ep::internode_ll::combine", "deepep_ll_combine"),
        ("silu_mul_quant_varlen_kernel", "moe_swiglu_quant"),
        ("flashmla_sparse_decode_p1_consumer", "flashmla_p1_consumer"),
        ("flash_fwd_splitkv_mla", "flashmla_splitkv"),
        ("flashmla_sparse_decode_combine", "flashmla_combine"),
        ("per_token_group_quant", "per_token_group_quant"),
        ("_router_triton_kernel", "router_topk"),
        ("router_gemm_kernel", "router_gemm"),
    )
    for marker, label in markers:
        if marker in name:
            return label
    return name[:180]


WAIT_PATTERNS = {
    "nccl_allgather": "%ncclDevKernel_AllGather_RING_LL%",
    "deepep_notify_dispatch": "%deep_ep::intranode::notify_dispatch<%",
    "deepep_cached_notify_combine": "%deep_ep::intranode::cached_notify_combine<%",
    "deepep_dispatch": "%deep_ep::intranode::dispatch<%",
    "deepep_combine": "%deep_ep::intranode::combine<%",
    "deepep_ll_dispatch": "%deep_ep::internode_ll::dispatch<%",
    "deepep_ll_combine": "%deep_ep::internode_ll::combine<%",
}


def analyze(path: Path, top_n: int) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        activation_row = connection.execute(
            "select id from StringIds where value = ?", (ACTIVATION_SHORT_NAME,)
        ).fetchone()
        if activation_row is None:
            raise RuntimeError(f"missing {ACTIVATION_SHORT_NAME} in {path}")
        lower, upper = connection.execute(
            "select min(start), max(end) from CUPTI_ACTIVITY_KIND_KERNEL "
            "where shortName = ?",
            (int(activation_row[0]),),
        ).fetchone()
        if lower is None or upper is None:
            raise RuntimeError(f"empty decode bounds in {path}")

        grouped = list(
            connection.execute(
                "select k.demangledName, k.deviceId, count(*), sum(k.end-k.start) "
                "from CUPTI_ACTIVITY_KIND_KERNEL k "
                "where k.start >= ? and k.start < ? "
                "group by k.demangledName, k.deviceId",
                (int(lower), int(upper)),
            )
        )
        names = {
            int(identifier): value
            for identifier, value in connection.execute("select id, value from StringIds")
        }

        by_kernel: dict[int, list[list[int]]] = defaultdict(
            lambda: [[0, 0] for _ in range(8)]
        )
        by_category: dict[str, list[int]] = defaultdict(lambda: [0 for _ in range(8)])
        per_device_total = [0 for _ in range(8)]
        for name_id, device, count, duration in grouped:
            device = int(device)
            duration = int(duration)
            by_kernel[int(name_id)][device] = [int(count), duration]
            by_category[category(names.get(int(name_id), ""))][device] += duration
            per_device_total[device] += duration

        total = sum(per_device_total)
        category_rows = []
        for label, device_ns in sorted(
            by_category.items(), key=lambda item: -sum(item[1])
        ):
            mean = statistics.fmean(device_ns)
            category_rows.append(
                {
                    "category": label,
                    "summed_ms": sum(device_ns) / 1e6,
                    "summed_pct": 100 * sum(device_ns) / total,
                    "per_device_ms": [value / 1e6 for value in device_ns],
                    "rank_cv": statistics.pstdev(device_ns) / mean if mean else 0.0,
                    "rank_max_over_min": max(device_ns) / max(min(device_ns), 1),
                }
            )

        kernel_rows = []
        for name_id, device_values in sorted(
            by_kernel.items(),
            key=lambda item: -sum(value[1] for value in item[1]),
        )[:top_n]:
            launch_count = sum(value[0] for value in device_values)
            device_ns = [value[1] for value in device_values]
            mean_device = statistics.fmean(device_ns)
            name = names.get(name_id, "")
            kernel_rows.append(
                {
                    "name": concise_name(name),
                    "category": category(name),
                    "calls": launch_count,
                    "summed_ms": sum(device_ns) / 1e6,
                    "summed_pct": 100 * sum(device_ns) / total,
                    "mean_launch_us": sum(device_ns) / max(launch_count, 1) / 1e3,
                    "per_device_ms": [value / 1e6 for value in device_ns],
                    "rank_cv": statistics.pstdev(device_ns) / mean_device
                    if mean_device
                    else 0.0,
                }
            )

        waits: dict[str, Any] = {}
        for label, pattern in WAIT_PATTERNS.items():
            rows = list(
                connection.execute(
                    "select k.deviceId, k.start, k.end "
                    "from CUPTI_ACTIVITY_KIND_KERNEL k "
                    "join StringIds s on s.id = k.demangledName "
                    "where k.start >= ? and k.start < ? and s.value like ? "
                    "order by k.deviceId, k.start",
                    (int(lower), int(upper), pattern),
                )
            )
            per_device: dict[int, list[tuple[int, int]]] = defaultdict(list)
            for device, start, end in rows:
                per_device[int(device)].append((int(start), int(end)))
            if not per_device:
                continue
            waits[label] = {
                "per_device": {
                    str(device): {
                        "count": len(events),
                        "duration_us": describe(
                            [(end - start) / 1e3 for start, end in events]
                        ),
                    }
                    for device, events in sorted(per_device.items())
                }
            }
            counts = {len(events) for events in per_device.values()}
            if len(per_device) == 8 and len(counts) == 1:
                count = counts.pop()
                start_skews = []
                duration_spreads = []
                for index in range(count):
                    starts = [per_device[device][index][0] for device in range(8)]
                    durations = [
                        per_device[device][index][1] - per_device[device][index][0]
                        for device in range(8)
                    ]
                    start_skews.append((max(starts) - min(starts)) / 1e3)
                    duration_spreads.append((max(durations) - min(durations)) / 1e3)
                waits[label]["aligned_rank_start_skew_us"] = describe(start_skews)
                waits[label]["aligned_rank_duration_spread_us"] = describe(
                    duration_spreads
                )

        return {
            "sqlite": str(path),
            "decode_bounds_s": [int(lower) / 1e9, int(upper) / 1e9],
            "decode_window_ms": (int(upper) - int(lower)) / 1e6,
            "summed_kernel_ms": total / 1e6,
            "per_device_summed_kernel_ms": [value / 1e6 for value in per_device_total],
            "categories": category_rows,
            "top_kernels": kernel_rows,
            "wait_and_skew": waits,
            "interpretation_guard": (
                "Summed CUDA durations double-count overlap and persistent/waiting kernels; "
                "use rank-max event latency and the containing-region/serving gates for decisions."
            ),
        }
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", nargs="+", type=Path)
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    report = {"traces": [analyze(path.resolve(), args.top_n) for path in args.sqlite]}
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output_json:
        args.output_json.write_text(rendered)


if __name__ == "__main__":
    main()
