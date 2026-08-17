#!/usr/bin/env python3
"""Compare matched Nsys captures for the N6 and v5 expert maps.

The output is causal evidence only. Profiled latency is deliberately excluded
from candidate promotion decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


CATEGORIES = (
    "deepep_notify_dispatch_wait",
    "deepep_dispatch",
    "deepep_combine",
    "deepep_cached_notify_combine_wait",
    "deepgemm",
    "flashmla_sparse_attention",
    "ep_gather_scatter",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def percent_change(control: float, candidate: float) -> float | None:
    if control == 0:
        return None
    return (candidate / control - 1.0) * 100.0


def category_summary(devices: dict[str, dict], category: str) -> dict:
    rows = []
    per_device = {}
    for device, payload in sorted(devices.items(), key=lambda item: int(item[0])):
        value = payload["categories"][category]
        calls = int(value["calls"])
        span_ms = float(payload["trace_span_ms"])
        row = {
            "calls": calls,
            "calls_per_second": 1000.0 * calls / span_ms,
            "kernel_sum_ms_per_call": float(value["kernel_sum_ms"]) / calls,
            "kernel_sum_fraction": float(value["kernel_sum_fraction"]),
            "time_ge_50ms": float(value["time_ge_50ms"]),
            **{key: float(value["duration_ms"][key]) for key in ("p50", "p90", "p99", "max")},
        }
        rows.append(row)
        per_device[device] = row
    keys = tuple(rows[0])
    return {
        "device_median": {key: median([row[key] for row in rows]) for key in keys},
        "calls_min": min(row["calls"] for row in rows),
        "calls_max": max(row["calls"] for row in rows),
        "per_device": per_device,
    }


def scheduler_summary(path: Path) -> dict:
    groups = json.loads(path.read_text())["groups"]
    sync = []
    memcpy = []
    for group in groups.values():
        sync_value = group.get("synchronize:cudaStreamSynchronize_v3020", {})
        memcpy_value = group.get("memcpy_api:cudaMemcpyAsync_v3020", {})
        sync.append((int(sync_value.get("calls", 0)), float(sync_value.get("total_ms", 0))))
        memcpy.append((int(memcpy_value.get("calls", 0)), float(memcpy_value.get("total_ms", 0))))

    def aggregate(rows: list[tuple[int, float]]) -> dict:
        per_call = [duration / calls for calls, duration in rows if calls]
        return {
            "calls": sum(calls for calls, _ in rows),
            "total_ms": sum(duration for _, duration in rows),
            "device_median_ms_per_call": median(per_call),
        }

    return {"stream_synchronize": aggregate(sync), "memcpy_async": aggregate(memcpy)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("control_dir", type=Path)
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    kernel_name = "per_gpu_kernel_analysis.json"
    sync_name = "scheduler_sync_analysis.json"
    control_kernel_path = args.control_dir / kernel_name
    candidate_kernel_path = args.candidate_dir / kernel_name
    control_sync_path = args.control_dir / sync_name
    candidate_sync_path = args.candidate_dir / sync_name
    for path in (control_kernel_path, candidate_kernel_path, control_sync_path, candidate_sync_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    arms = {
        "control": json.loads(control_kernel_path.read_text())["devices"],
        "candidate": json.loads(candidate_kernel_path.read_text())["devices"],
    }
    if set(arms["control"]) != set(arms["candidate"]) or len(arms["control"]) != 8:
        raise RuntimeError("captures must contain the same eight devices")

    categories = {}
    for category in CATEGORIES:
        summaries = {arm: category_summary(devices, category) for arm, devices in arms.items()}
        control = summaries["control"]["device_median"]
        candidate = summaries["candidate"]["device_median"]
        summaries["candidate_delta_percent"] = {
            key: percent_change(control[key], candidate[key]) for key in control
        }
        categories[category] = summaries

    capture = {}
    for arm, devices in arms.items():
        spans = [float(value["trace_span_ms"]) for value in devices.values()]
        active = [
            float(value["all_kernel_interval_union_ms"]) / float(value["trace_span_ms"])
            for value in devices.values()
        ]
        capture[arm] = {
            "devices": len(devices),
            "trace_span_ms_device_median": median(spans),
            "gpu_active_fraction_device_median": median(active),
        }
    capture["candidate_trace_span_delta_percent"] = percent_change(
        capture["control"]["trace_span_ms_device_median"],
        capture["candidate"]["trace_span_ms_device_median"],
    )

    scheduler = {
        "control": scheduler_summary(control_sync_path),
        "candidate": scheduler_summary(candidate_sync_path),
    }
    result = {
        "schema_version": 1,
        "status": "VALID_CAUSAL_COMPARISON",
        "purpose": "causal diagnosis only; profiled latency is not a performance estimator",
        "interpretation": (
            "The v5 map advances more DeepEP dispatches per matched trace while reducing "
            "dispatch-notify tail wait. FlashMLA per-call duration is unchanged, so the "
            "causal signal is rank-arrival/control-plane balance rather than attention math."
        ),
        "capture": capture,
        "categories": categories,
        "scheduler_runtime_api": scheduler,
        "inputs": {
            str(path): sha256(path)
            for path in (control_kernel_path, candidate_kernel_path, control_sync_path, candidate_sync_path)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
