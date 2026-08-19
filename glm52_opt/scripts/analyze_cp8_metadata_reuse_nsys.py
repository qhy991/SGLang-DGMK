#!/usr/bin/env python3
"""Compare matched c01/candidate metadata-reuse Nsight kernel composition.

The sums below span all eight GPUs and are deliberately *not* interpreted as
wall-clock latency.  They prove route coverage, expose removed/materialized
work, and rank follow-up boundaries after the no-profiler serving gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path


FAMILIES = {
    "direct_scatter": ("_direct_two_block_scatter",),
    "quantize_k": ("_quantize_k_cache_fast_kernel",),
    "mla_cache_store": ("set_mla_kv_buffer_kernel",),
    "indexer_mqa": ("sm100_mqa_logits",),
    "indexer_topk": ("topk_transform_prefill_kernel",),
    "indexer_k_gather": ("_get_k_triton_kernel",),
    "indexer_scale_gather": ("_get_s_triton_kernel",),
    "sparse_attention": ("flash_fwd_splitkv_mla_fp8_sparse_kernel",),
    "nccl_allgather": ("ncclDevKernel_AllGather",),
    "deepep_dispatch": ("=dispatch",),
    "deepep_dispatch_notify": ("notify_dispatch",),
    "deepep_combine": ("=combine",),
    "deepep_combine_notify": ("cached_notify_combine",),
    "deepgemm_moe": ("sm100_fp8_fp4_gemm_1d1d_impl",),
    "torch_cat": ("CatArrayBatchedCopy",),
    "mla_metadata": ("get_mla_metadata_kernel",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-sqlite", type=Path, required=True)
    parser.add_argument("--candidate-sqlite", type=Path, required=True)
    parser.add_argument("--baseline-log", type=Path, required=True)
    parser.add_argument("--candidate-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=80)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_kernels(path: Path) -> list[dict]:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            """
            SELECT s.value, COUNT(*), SUM(k.end - k.start)
            FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
            JOIN StringIds AS s ON s.id = k.shortName
            GROUP BY k.shortName
            ORDER BY SUM(k.end - k.start) DESC
            """
        ).fetchall()
    finally:
        connection.close()
    return [
        {"name": name, "instances": int(count), "total_time_ms": ns / 1e6}
        for name, count, ns in rows
    ]


def family_summary(rows: list[dict], needles: tuple[str, ...]) -> dict:
    def matches(name: str) -> bool:
        return any(
            name == needle[1:] if needle.startswith("=") else needle in name
            for needle in needles
        )

    matched = [row for row in rows if matches(row["name"])]
    return {
        "instances": sum(row["instances"] for row in matched),
        "total_time_ms": sum(row["total_time_ms"] for row in matched),
        "names": [row["name"] for row in matched],
    }


def capture_markers(path: Path) -> dict:
    text = path.read_text(errors="replace")
    return {
        "start": text.count("nsys capture START"),
        "stop": text.count("nsys capture STOP"),
        "direct_selected": text.count("GLM-5.2 direct packed CP MLA-KV selected:"),
        "combined_selected": text.count(
            "glm52_opt HIT e2e_prefill/cp8_combined_indexer_halves"
        ),
        "metadata_reuse_selected": text.count(
            "glm52_opt HIT e2e_prefill/cp8_flashmla_metadata_reuse"
        ),
    }


def resolve_server_log(launcher_log: Path) -> Path:
    outputs = [
        line.split("=", 1)[1].strip()
        for line in launcher_log.read_text(errors="replace").splitlines()
        if line.startswith("[INFO] OUT=")
    ]
    if len(outputs) != 1:
        raise RuntimeError(
            f"expected exactly one server output in {launcher_log}, got {outputs}"
        )
    server_log = Path(outputs[0]) / "server.log"
    if not server_log.is_file() or server_log.stat().st_size == 0:
        raise RuntimeError(f"missing matched server log: {server_log}")
    return server_log


def main() -> None:
    args = parse_args()
    if args.top <= 0:
        raise ValueError("--top must be positive")
    for path in (
        args.baseline_sqlite,
        args.candidate_sqlite,
        args.baseline_log,
        args.candidate_log,
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"missing or empty matched artifact: {path}")

    kernels = {
        "baseline": load_kernels(args.baseline_sqlite),
        "candidate": load_kernels(args.candidate_sqlite),
    }
    server_logs = {
        "baseline": resolve_server_log(args.baseline_log),
        "candidate": resolve_server_log(args.candidate_log),
    }
    families = {
        arm: {
            family: family_summary(rows, needles)
            for family, needles in FAMILIES.items()
        }
        for arm, rows in kernels.items()
    }
    family_deltas = {}
    for family in FAMILIES:
        baseline = families["baseline"][family]
        candidate = families["candidate"][family]
        family_deltas[family] = {
            "instance_delta": candidate["instances"] - baseline["instances"],
            "aggregate_time_delta_ms": (
                candidate["total_time_ms"] - baseline["total_time_ms"]
            ),
        }

    markers = {arm: capture_markers(path) for arm, path in server_logs.items()}
    for arm in ("baseline", "candidate"):
        if markers[arm]["start"] != 8 or markers[arm]["stop"] != 8:
            raise RuntimeError(f"incomplete {arm} capture markers: {markers[arm]}")
        if markers[arm]["direct_selected"] != 624:
            raise RuntimeError(f"direct path coverage drift: {arm} {markers[arm]}")
        if markers[arm]["combined_selected"] != 16:
            raise RuntimeError(f"combined path coverage drift: {arm} {markers[arm]}")
    if markers["baseline"]["metadata_reuse_selected"] != 0:
        raise RuntimeError("baseline unexpectedly selected metadata reuse")
    if markers["candidate"]["metadata_reuse_selected"] != 16:
        raise RuntimeError(f"candidate metadata marker drift: {markers['candidate']}")
    for family in ("sparse_attention", "deepgemm_moe", "deepep_combine"):
        if (
            families["baseline"][family]["instances"]
            != families["candidate"][family]["instances"]
        ):
            raise RuntimeError(f"work coverage drift for {family}")
    baseline_metadata = families["baseline"]["mla_metadata"]["instances"]
    candidate_metadata = families["candidate"]["mla_metadata"]["instances"]
    if baseline_metadata != 6_320 or candidate_metadata != 160:
        raise RuntimeError(
            "unexpected metadata call counts: "
            f"baseline={baseline_metadata} candidate={candidate_metadata}"
        )

    report = {
        "schema": "glm52-cp8-metadata-reuse-matched-nsys-v1",
        "status": "PASS",
        "scope": (
            "kernel-composition attribution only; cross-device sums are not wall time "
            "and the five-pair no-profiler TTFT campaign remains authoritative"
        ),
        "artifacts": {
            "baseline_sqlite": str(args.baseline_sqlite.resolve()),
            "candidate_sqlite": str(args.candidate_sqlite.resolve()),
            "baseline_log": str(args.baseline_log.resolve()),
            "candidate_log": str(args.candidate_log.resolve()),
            "baseline_server_log": str(server_logs["baseline"].resolve()),
            "candidate_server_log": str(server_logs["candidate"].resolve()),
            "sha256": {
                "baseline_sqlite": sha256(args.baseline_sqlite),
                "candidate_sqlite": sha256(args.candidate_sqlite),
                "baseline_log": sha256(args.baseline_log),
                "candidate_log": sha256(args.candidate_log),
                "baseline_server_log": sha256(server_logs["baseline"]),
                "candidate_server_log": sha256(server_logs["candidate"]),
            },
        },
        "capture_markers": markers,
        "families": families,
        "family_deltas_candidate_minus_baseline": family_deltas,
        "causal_checks": {
            "equal_sparse_attention_instances": True,
            "equal_moe_gemm_instances": True,
            "equal_deepep_combine_instances": True,
            "metadata_instances_baseline": baseline_metadata,
            "metadata_instances_candidate": candidate_metadata,
            "metadata_instance_reduction_fraction": 1
            - candidate_metadata / baseline_metadata,
            "ncu_decision": "not required because the candidate removes repeated metadata launches without changing the metadata kernel implementation",
        },
        "top_kernels": {
            arm: rows[: args.top] for arm, rows in kernels.items()
        },
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
