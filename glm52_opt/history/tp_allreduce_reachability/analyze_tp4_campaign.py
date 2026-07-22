#!/usr/bin/env python3
"""Validate and summarize one locked TP4 AllReduce diagnostic campaign.

This analyzer is CPU-only.  It reads the immutable artifacts produced by
``run_locked_tp4_campaign.sh`` and emits one JSON document.  A non-zero exit
means that the campaign is incomplete or internally inconsistent; failed
``attempt`` rows are evidence, not campaign failures, when their logs exist.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class Case:
    short_name: str
    task: str
    rows: int
    message_bytes: int
    execution_mode: str
    stream: str = "nondefault"


CASES = (
    Case(
        "m16",
        "tp4_allreduce_decode_m16",
        16,
        16 * 6144 * 2,
        "cuda_graph",
    ),
    Case(
        "m32",
        "tp4_allreduce_decode_m32",
        32,
        32 * 6144 * 2,
        "cuda_graph",
    ),
    Case(
        "prefill",
        "tp4_allreduce_prefill",
        8192,
        8192 * 6144 * 2,
        "eager",
    ),
)

STATUS_FIELDS = (
    "requirement",
    "step",
    "exit_code",
    "started_utc",
    "finished_utc",
    "log",
)

REQUIRED_STATUS_STEPS = {
    "environment/source_identity",
    "environment/check_env",
    "environment/nvidia_smi",
    "environment/topology",
    "environment/nvlink_status",
    "environment/nvlink_throughput_before",
    "environment/nvlink_throughput_after",
    "environment/nvidia_smi_after",
    "environment/source_identity_after",
    "producer_abi/linear_attn_o_decode_m16",
    "producer_abi/linear_attn_o_decode_m32",
    *(f"reachability/{case.short_name}" for case in CASES),
    *(f"baseline/{case.short_name}_run{run}" for case in CASES for run in range(1, 4)),
    *(f"paired/{case.short_name}_reference_control" for case in CASES),
    *(f"profile/{case.short_name}_nsys" for case in CASES),
    *(
        f"semantics/{case.task}.{mode}.{stream}"
        for case in CASES
        for mode, stream in (
            ("eager", "default"),
            ("eager", "nondefault"),
            ("cuda_graph", "nondefault"),
        )
    ),
}

EXPECTED_ATTEMPT_STEPS = {
    "backend_scout/custom_allreduce",
    *(
        f"paired/{case.short_name}_c10d_{variant}"
        for case in CASES
        for variant in ("inplace", "outplace")
    ),
}


class Analyzer:
    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        self.errors: list[dict[str, str]] = []
        self.status_rows: dict[str, dict[str, Any]] = {}
        self.expected_shas: dict[str, str] = {}
        self.expected_roots: dict[str, str] = {}

    def error(self, code: str, location: str, detail: str) -> None:
        self.errors.append({"code": code, "location": location, "detail": detail})

    def resolve_log(self, raw_path: str, *, step: Optional[str] = None) -> Path:
        if step is not None:
            copied = self.root / f"{step}.log"
            if copied.is_file():
                return copied.resolve()
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = self.root / path
        if path.exists():
            return path.resolve()
        # A copied campaign may retain its original absolute path.  The ledger
        # step still determines the canonical location inside the copy.
        return path

    def read_json(self, relative: str) -> Optional[dict[str, Any]]:
        path = self.root / relative
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            self.error("missing_json", relative, "required JSON file is absent")
            return None
        except (OSError, json.JSONDecodeError) as exc:
            self.error("invalid_json", relative, f"{type(exc).__name__}: {exc}")
            return None
        if not isinstance(value, dict):
            self.error("invalid_json_type", relative, "top-level value must be an object")
            return None
        return value

    def load_status(self) -> None:
        path = self.root / "status.tsv"
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                if tuple(reader.fieldnames or ()) != STATUS_FIELDS:
                    self.error(
                        "invalid_status_header",
                        "status.tsv",
                        f"expected {list(STATUS_FIELDS)!r}, got {reader.fieldnames!r}",
                    )
                    return
                for line_number, row in enumerate(reader, start=2):
                    step = row.get("step", "")
                    if not step:
                        self.error(
                            "empty_status_step",
                            f"status.tsv:{line_number}",
                            "step is empty",
                        )
                        continue
                    if step in self.status_rows:
                        self.error(
                            "duplicate_status_step",
                            f"status.tsv:{line_number}",
                            step,
                        )
                        continue
                    try:
                        exit_code = int(row.get("exit_code", ""))
                    except ValueError:
                        self.error(
                            "invalid_exit_code",
                            f"status.tsv:{line_number}",
                            repr(row.get("exit_code")),
                        )
                        continue
                    parsed: dict[str, Any] = dict(row)
                    parsed["exit_code"] = exit_code
                    parsed["line_number"] = line_number
                    self.status_rows[step] = parsed
        except FileNotFoundError:
            self.error("missing_status", "status.tsv", "status ledger is absent")
        except OSError as exc:
            self.error("status_read_error", "status.tsv", f"{type(exc).__name__}: {exc}")

        for step, row in sorted(self.status_rows.items()):
            if row["requirement"] == "required" and row["exit_code"] != 0:
                self.error(
                    "failed_required_status",
                    f"status.tsv:{row['line_number']}",
                    f"{step} exited {row['exit_code']}",
                )
            if row["requirement"] not in {"required", "attempt"}:
                self.error(
                    "invalid_requirement",
                    f"status.tsv:{row['line_number']}",
                    repr(row["requirement"]),
                )
            log = self.resolve_log(str(row.get("log", "")), step=step)
            if not log.is_file() or log.stat().st_size == 0:
                self.error(
                    "missing_status_log",
                    f"status.tsv:{row['line_number']}",
                    f"{step}: {log}",
                )

        for step in sorted(REQUIRED_STATUS_STEPS):
            row = self.status_rows.get(step)
            if row is None:
                self.error("missing_required_status", "status.tsv", step)
            elif row["requirement"] != "required":
                self.error(
                    "misclassified_required_status",
                    f"status.tsv:{row['line_number']}",
                    step,
                )
        for step in sorted(EXPECTED_ATTEMPT_STEPS):
            if step not in self.status_rows:
                self.error("missing_attempt_status", "status.tsv", step)

    @staticmethod
    def _command_payload(path: Path) -> list[str]:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return [
            line.strip()
            for line in lines
            if line.strip()
            and not line.startswith("started_utc=")
            and not line.startswith("command=")
            and not line.startswith("timeout_seconds=")
        ]

    def load_source_identity(self) -> dict[str, Any]:
        sha_pattern = set("0123456789abcdef")

        def is_sha(value: str) -> bool:
            return len(value) == 40 and set(value) <= sha_pattern

        start_path = self.root / "environment/source_identity.log"
        after_path = self.root / "environment/source_identity_after.log"
        try:
            start_payload = self._command_payload(start_path)
            after_payload = self._command_payload(after_path)
        except OSError as exc:
            self.error(
                "source_identity_read_error",
                "environment",
                f"{type(exc).__name__}: {exc}",
            )
            return {}

        start_shas = [value for value in start_payload if is_sha(value)]
        after_shas = [value for value in after_payload if is_sha(value)]
        if len(start_shas) != 2:
            self.error(
                "invalid_source_identity",
                "environment/source_identity.log",
                f"expected exactly two commit SHA lines, got {start_shas!r}",
            )
            return {}
        if any(not is_sha(value) for value in start_payload):
            self.error(
                "dirty_source_identity",
                "environment/source_identity.log",
                f"unexpected git-status output: {start_payload!r}",
            )
        if len(after_shas) != 2:
            self.error(
                "invalid_source_identity_after",
                "environment/source_identity_after.log",
                f"expected exactly two commit SHA lines, got {after_shas!r}",
            )
            return {}
        if start_shas != after_shas:
            self.error(
                "source_sha_changed",
                "environment/source_identity_after.log",
                f"start={start_shas!r}, after={after_shas!r}",
            )

        after_non_shas = [value for value in after_payload if not is_sha(value)]
        if len(after_non_shas) != 2:
            self.error(
                "dirty_source_identity_after",
                "environment/source_identity_after.log",
                f"expected two repository paths and no status output, got {after_payload!r}",
            )
        else:
            self.expected_roots = {
                "kernel_harness_git": after_non_shas[0],
                "sglang_git": after_non_shas[1],
            }

        self.expected_shas = {
            "kernel_harness_git": start_shas[0],
            "sglang_git": start_shas[1],
        }
        return {
            "kernel_harness_sha": start_shas[0],
            "sglang_sha": start_shas[1],
            "unchanged_after_campaign": start_shas == after_shas,
            "clean_at_start": not any(not is_sha(value) for value in start_payload),
            "clean_at_end": len(after_non_shas) == 2,
        }

    def validate_source(self, result: dict[str, Any], location: str) -> None:
        environment = result.get("environment")
        if not isinstance(environment, dict):
            self.error("missing_environment", location, "environment object is absent")
            return
        for key in ("kernel_harness_git", "sglang_git"):
            metadata = environment.get(key)
            if not isinstance(metadata, dict):
                self.error("missing_git_metadata", location, key)
                continue
            if metadata.get("dirty") is not False or metadata.get("status") not in ([], None):
                self.error(
                    "dirty_result_source",
                    location,
                    f"{key}: dirty={metadata.get('dirty')!r}, status={metadata.get('status')!r}",
                )
            expected_sha = self.expected_shas.get(key)
            if expected_sha is not None and metadata.get("sha") != expected_sha:
                self.error(
                    "result_source_sha_mismatch",
                    location,
                    f"{key}: expected {expected_sha}, got {metadata.get('sha')!r}",
                )
            expected_root = self.expected_roots.get(key)
            if expected_root is not None and metadata.get("root") != expected_root:
                self.error(
                    "result_source_root_mismatch",
                    location,
                    f"{key}: expected {expected_root!r}, got {metadata.get('root')!r}",
                )

    def validate_common_result(
        self,
        result: dict[str, Any],
        case: Case,
        location: str,
        *,
        expected_mode: Optional[str] = None,
        expected_stream: Optional[str] = None,
        expected_trace: Optional[bool] = False,
    ) -> None:
        workload = result.get("workload")
        if not isinstance(workload, dict):
            self.error("missing_workload", location, "workload object is absent")
            return
        expected_workload = {
            "name": case.task,
            "family": "allreduce",
            "world_size": 4,
        }
        for key, expected in expected_workload.items():
            if workload.get(key) != expected:
                self.error(
                    "workload_mismatch",
                    location,
                    f"{key}: expected {expected!r}, got {workload.get(key)!r}",
                )
        params = workload.get("params")
        expected_params = {
            "local_tokens": case.rows,
            "hidden": 6144,
            "dtype": "bfloat16",
        }
        if not isinstance(params, dict):
            self.error("missing_workload_params", location, "params object is absent")
        else:
            for key, expected in expected_params.items():
                if params.get(key) != expected:
                    self.error(
                        "workload_param_mismatch",
                        location,
                        f"{key}: expected {expected!r}, got {params.get(key)!r}",
                    )
        if result.get("scope") != "tp4_allreduce_diagnostic_only":
            self.error("scope_mismatch", location, repr(result.get("scope")))
        correctness = result.get("correctness")
        if not isinstance(correctness, dict) or correctness.get("passed") is not True:
            self.error("correctness_failed", location, repr(correctness))
        allreduce = result.get("allreduce")
        if not isinstance(allreduce, dict) or allreduce.get("message_bytes") != case.message_bytes:
            self.error(
                "message_bytes_mismatch",
                location,
                f"expected {case.message_bytes}, got {None if not isinstance(allreduce, dict) else allreduce.get('message_bytes')!r}",
            )
        execution = result.get("execution")
        if not isinstance(execution, dict):
            self.error("missing_execution", location, "execution object is absent")
        else:
            if expected_mode is not None and execution.get("mode") != expected_mode:
                self.error(
                    "execution_mode_mismatch",
                    location,
                    f"expected {expected_mode!r}, got {execution.get('mode')!r}",
                )
            if expected_stream is not None and execution.get("stream") != expected_stream:
                self.error(
                    "execution_stream_mismatch",
                    location,
                    f"expected {expected_stream!r}, got {execution.get('stream')!r}",
                )
        trace = result.get("all_reduce_trace")
        if expected_trace is not None:
            if not isinstance(trace, dict) or trace.get("enabled") is not expected_trace:
                self.error(
                    "trace_enablement_mismatch",
                    location,
                    f"expected enabled={expected_trace}, got {trace!r}",
                )
        self.validate_source(result, location)

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"line {line_number} is not an object")
                records.append(value)
        return records

    def validate_trace(self, case: Case) -> dict[str, Any]:
        relative = f"reachability/{case.short_name}.result.json"
        result = self.read_json(relative)
        summary: dict[str, Any] = {
            "task": case.task,
            "shape": [case.rows, 6144],
            "dtype": "torch.bfloat16",
            "message_bytes": case.message_bytes,
            "rank_files": [],
        }
        if result is None:
            return summary
        self.validate_common_result(
            result,
            case,
            relative,
            expected_mode=case.execution_mode,
            expected_stream=case.stream,
            expected_trace=True,
        )
        trace = result.get("all_reduce_trace")
        if not isinstance(trace, dict):
            return summary
        for key, expected in (
            ("evidence_valid", True),
            ("measurement_instrumented", True),
        ):
            if trace.get(key) is not expected:
                self.error("invalid_trace_evidence", relative, f"{key}={trace.get(key)!r}")
        for key in ("errors_by_rank", "invalid_reasons"):
            if trace.get(key) not in ([], None):
                self.error("trace_reports_errors", relative, f"{key}={trace.get(key)!r}")

        failure_rows = trace.get("failure_counts_by_rank")
        failure_by_rank: dict[int, dict[str, Any]] = {}
        if not isinstance(failure_rows, list):
            self.error("missing_trace_failure_counts", relative, repr(failure_rows))
        else:
            for item in failure_rows:
                if not isinstance(item, dict) or not isinstance(item.get("rank"), int):
                    self.error("invalid_trace_failure_row", relative, repr(item))
                    continue
                rank = item["rank"]
                counts = item.get("counts")
                if rank in failure_by_rank:
                    self.error("duplicate_trace_failure_rank", relative, str(rank))
                if not isinstance(counts, dict):
                    self.error("invalid_trace_failure_counts", relative, repr(item))
                    counts = {}
                if any(not isinstance(value, int) or value != 0 for value in counts.values()):
                    self.error("nonzero_trace_failure", relative, f"rank {rank}: {counts!r}")
                failure_by_rank[rank] = counts
            if set(failure_by_rank) != set(range(4)):
                self.error(
                    "trace_failure_rank_set_mismatch",
                    relative,
                    f"expected [0, 1, 2, 3], got {sorted(failure_by_rank)!r}",
                )

        paths = trace.get("paths_by_rank")
        path_by_rank: dict[int, Path] = {}
        if not isinstance(paths, list):
            self.error("missing_trace_paths", relative, repr(paths))
        else:
            for item in paths:
                if not isinstance(item, dict) or not isinstance(item.get("rank"), int):
                    self.error("invalid_trace_path_row", relative, repr(item))
                    continue
                rank = item["rank"]
                raw_path = item.get("path")
                if rank in path_by_rank:
                    self.error("duplicate_trace_path_rank", relative, str(rank))
                    continue
                if not isinstance(raw_path, str) or not raw_path:
                    self.error("invalid_trace_path", relative, repr(item))
                    continue
                path = Path(raw_path).expanduser()
                if not path.is_absolute():
                    path = self.root / path
                copied = self.root / "reachability" / path.name
                if copied.is_file():
                    path = copied
                path_by_rank[rank] = path.resolve()
            if set(path_by_rank) != set(range(4)):
                self.error(
                    "trace_path_rank_set_mismatch",
                    relative,
                    f"expected [0, 1, 2, 3], got {sorted(path_by_rank)!r}",
                )
            if len(set(path_by_rank.values())) != len(path_by_rank):
                self.error("duplicate_trace_paths", relative, repr(path_by_rank))

        observed_files = {
            path.resolve()
            for path in (self.root / "reachability").glob(f"{case.short_name}.*.jsonl")
            if path.is_file()
        }
        if observed_files != set(path_by_rank.values()):
            self.error(
                "trace_file_set_mismatch",
                f"reachability/{case.short_name}.*.jsonl",
                f"ledger={sorted(map(str, path_by_rank.values()))!r}, files={sorted(map(str, observed_files))!r}",
            )

        for rank in sorted(path_by_rank):
            path = path_by_rank[rank]
            try:
                records = self._read_jsonl(path)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                self.error(
                    "invalid_trace_jsonl",
                    str(path),
                    f"{type(exc).__name__}: {exc}",
                )
                continue
            if not records:
                self.error("empty_trace_jsonl", str(path), "no records")
                continue
            capture_observations: list[Optional[bool]] = []
            backends: set[str] = set()
            for record_index, record in enumerate(records):
                record_location = f"{path}:record[{record_index}]"
                group = record.get("group")
                if not isinstance(group, dict):
                    self.error("missing_trace_group", record_location, repr(group))
                else:
                    expected_group = {
                        "ranks": [0, 1, 2, 3],
                        "world_size": 4,
                        "rank": rank,
                        "rank_in_group": rank,
                    }
                    for key, expected in expected_group.items():
                        if group.get(key) != expected:
                            self.error(
                                "trace_group_mismatch",
                                record_location,
                                f"{key}: expected {expected!r}, got {group.get(key)!r}",
                            )
                for tensor_key in ("input", "output"):
                    tensor = record.get(tensor_key)
                    expected_tensor = {
                        "present": True,
                        "shape": [case.rows, 6144],
                        "stride": [6144, 1],
                        "dtype": "torch.bfloat16",
                        "numel": case.rows * 6144,
                        "element_size": 2,
                        "bytes": case.message_bytes,
                    }
                    if not isinstance(tensor, dict):
                        self.error("missing_trace_tensor", record_location, tensor_key)
                        continue
                    for key, expected in expected_tensor.items():
                        if tensor.get(key) != expected:
                            self.error(
                                "trace_tensor_mismatch",
                                record_location,
                                f"{tensor_key}.{key}: expected {expected!r}, got {tensor.get(key)!r}",
                            )
                    if tensor.get("errors") not in (None, {}):
                        self.error(
                            "trace_tensor_probe_error",
                            record_location,
                            f"{tensor_key}: {tensor.get('errors')!r}",
                        )
                graph = record.get("graph")
                if not isinstance(graph, dict):
                    self.error("missing_trace_graph", record_location, repr(graph))
                    capture_observations.append(None)
                else:
                    capture_observations.append(graph.get("cuda_stream_capturing"))
                    if "probe_error" in graph:
                        self.error("trace_graph_probe_error", record_location, repr(graph))
                stream = record.get("stream")
                if not isinstance(stream, dict) or stream.get("present") is not True:
                    self.error("missing_trace_stream", record_location, repr(stream))
                elif "probe_error" in stream:
                    self.error("trace_stream_probe_error", record_location, repr(stream))
                selection = record.get("selection")
                if not isinstance(selection, dict):
                    self.error("missing_trace_selection", record_location, repr(selection))
                else:
                    if selection.get("predicate_errors") not in (None, {}):
                        self.error(
                            "trace_predicate_error",
                            record_location,
                            repr(selection.get("predicate_errors")),
                        )
                    backend = selection.get("selected_backend")
                    if not isinstance(backend, str) or not backend:
                        self.error("missing_selected_backend", record_location, repr(backend))
                    else:
                        backends.add(backend)
            if case.execution_mode == "cuda_graph":
                if True not in capture_observations:
                    self.error(
                        "missing_cuda_graph_capture_observation",
                        str(path),
                        repr(capture_observations),
                    )
            elif True in capture_observations or False not in capture_observations:
                self.error(
                    "invalid_eager_graph_observation",
                    str(path),
                    repr(capture_observations),
                )
            summary["rank_files"].append(
                {
                    "rank": rank,
                    "path": str(path),
                    "records": len(records),
                    "cuda_stream_capturing_observations": capture_observations,
                    "selected_backends": sorted(backends),
                    "sha256": file_sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
        summary["trace_evidence_valid"] = trace.get("evidence_valid")
        return summary

    @staticmethod
    def _sample_values(result: dict[str, Any], side: str, metric: str) -> list[float]:
        raw = result.get("raw_samples")
        if not isinstance(raw, dict) or not isinstance(raw.get(side), list):
            raise ValueError(f"raw_samples.{side} is absent")
        values: list[float] = []
        for index, sample in enumerate(raw[side]):
            try:
                value = float(sample[metric]["rank_max_ms"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"raw_samples.{side}[{index}].{metric}: {exc}") from exc
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"raw_samples.{side}[{index}].{metric} must be finite and > 0"
                )
            values.append(value)
        return values

    def validate_sample_count(
        self,
        result: dict[str, Any],
        location: str,
        side: str,
        expected: int,
    ) -> list[float]:
        try:
            values = self._sample_values(result, side, "ready_region")
        except ValueError as exc:
            self.error("invalid_raw_samples", location, str(exc))
            return []
        if len(values) != expected:
            self.error(
                "sample_count_mismatch",
                location,
                f"raw_samples.{side}: expected {expected}, got {len(values)}",
            )
        return values

    def validate_reported_median(
        self,
        result: dict[str, Any],
        location: str,
        side: str,
        values: list[float],
    ) -> Optional[float]:
        if not values:
            return None
        try:
            reported = float(result[side]["ready_region"]["median_ms"])
        except (KeyError, TypeError, ValueError) as exc:
            self.error("missing_reported_median", location, f"{side}: {exc}")
            return None
        derived = statistics.median(values)
        if not math.isclose(reported, derived, rel_tol=1e-12, abs_tol=1e-12):
            self.error(
                "reported_median_mismatch",
                location,
                f"{side}: reported={reported}, derived={derived}",
            )
        return reported

    def validate_semantics(self) -> dict[str, Any]:
        checks: list[dict[str, Any]] = []
        for case in CASES:
            for mode, stream in (
                ("eager", "default"),
                ("eager", "nondefault"),
                ("cuda_graph", "nondefault"),
            ):
                relative = f"semantics/{case.task}.{mode}.{stream}.json"
                result = self.read_json(relative)
                if result is None:
                    continue
                self.validate_common_result(
                    result,
                    case,
                    relative,
                    expected_mode=mode,
                    expected_stream=stream,
                    expected_trace=False,
                )
                values = self.validate_sample_count(result, relative, "reference", 10)
                self.validate_reported_median(result, relative, "reference", values)
                checks.append(
                    {
                        "task": case.task,
                        "mode": mode,
                        "stream": stream,
                        "samples": len(values),
                    }
                )
        return {"checks": checks, "expected_checks": 9}

    def validate_baselines(self) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for case in CASES:
            run_summaries: list[dict[str, Any]] = []
            medians: list[float] = []
            for run in range(1, 4):
                relative = f"baseline/{case.short_name}_run{run}.json"
                result = self.read_json(relative)
                if result is None:
                    continue
                self.validate_common_result(
                    result,
                    case,
                    relative,
                    expected_mode=case.execution_mode,
                    expected_stream=case.stream,
                    expected_trace=False,
                )
                if result.get("candidate") is not None:
                    self.error("unexpected_baseline_candidate", relative, "candidate is present")
                values = self.validate_sample_count(result, relative, "reference", 100)
                median = self.validate_reported_median(
                    result, relative, "reference", values
                )
                if median is not None:
                    medians.append(median)
                run_summaries.append(
                    {
                        "run": run,
                        "ready_region_rank_max_median_ms": median,
                        "samples": len(values),
                    }
                )
            output[case.short_name] = {
                "task": case.task,
                "runs": run_summaries,
                "three_run_medians_ms": medians,
                "median_of_run_medians_ms": (
                    statistics.median(medians) if len(medians) == 3 else None
                ),
                "min_run_median_ms": min(medians) if len(medians) == 3 else None,
                "max_run_median_ms": max(medians) if len(medians) == 3 else None,
            }
        return output

    def validate_paired_result(
        self,
        result: dict[str, Any],
        case: Case,
        location: str,
        *,
        expected_reference_candidate: bool,
    ) -> dict[str, Any]:
        self.validate_common_result(
            result,
            case,
            location,
            expected_mode=case.execution_mode,
            expected_stream=case.stream,
            expected_trace=False,
        )
        reference_values = self.validate_sample_count(
            result, location, "reference", 100
        )
        candidate_values = self.validate_sample_count(
            result, location, "candidate", 100
        )
        reference_median = self.validate_reported_median(
            result, location, "reference", reference_values
        )
        candidate_median = self.validate_reported_median(
            result, location, "candidate", candidate_values
        )
        candidate = result.get("candidate")
        if not isinstance(candidate, dict):
            self.error("missing_candidate_summary", location, repr(candidate))
            return {}
        manifest = candidate.get("manifest")
        if not isinstance(manifest, dict):
            self.error("missing_candidate_manifest", location, repr(manifest))
        elif expected_reference_candidate:
            entrypoint = Path(str(manifest.get("entrypoint", ""))).name
            if entrypoint != "reference.py":
                self.error(
                    "reference_control_candidate_mismatch",
                    location,
                    f"expected reference.py, got {entrypoint!r}",
                )

        ratios: list[float] = []
        if len(reference_values) == len(candidate_values):
            ratios = [
                reference / candidate_value
                for reference, candidate_value in zip(reference_values, candidate_values)
            ]
        reported_speedup = candidate.get("speedup")
        derived_speedup = statistics.median(ratios) if ratios else None
        if isinstance(reported_speedup, (int, float)) and derived_speedup is not None:
            if not math.isclose(
                float(reported_speedup), derived_speedup, rel_tol=1e-12, abs_tol=1e-12
            ):
                self.error(
                    "reported_speedup_mismatch",
                    location,
                    f"reported={reported_speedup}, derived={derived_speedup}",
                )
        else:
            self.error("missing_reported_speedup", location, repr(reported_speedup))

        return {
            "reference_ready_region_rank_max_median_ms": reference_median,
            "candidate_ready_region_rank_max_median_ms": candidate_median,
            "reported_paired_median_speedup": reported_speedup,
            "derived_paired_median_speedup": derived_speedup,
            "paired_p10_speedup": candidate.get("paired_p10_speedup"),
            "paired_p90_speedup": candidate.get("paired_p90_speedup"),
            "derived_min_speedup": min(ratios) if ratios else None,
            "derived_max_speedup": max(ratios) if ratios else None,
            "derived_median_delta_percent": (
                (derived_speedup - 1.0) * 100.0 if derived_speedup is not None else None
            ),
            "samples": len(ratios),
            "gate_passed": result.get("gate", {}).get("passed"),
            "disposition": result.get("disposition"),
            "candidate_manifest_sha256": (
                manifest.get("manifest_sha256") if isinstance(manifest, dict) else None
            ),
        }

    def failed_log_summary(self, row: dict[str, Any]) -> dict[str, Any]:
        path = self.resolve_log(str(row.get("log", "")), step=str(row["step"]))
        summary: dict[str, Any] = {
            "step": row["step"],
            "exit_code": row["exit_code"],
            "path": str(path),
            "present": path.is_file(),
        }
        if not path.is_file():
            return summary
        payload = path.read_bytes()
        lines = payload.decode("utf-8", errors="replace").splitlines()
        summary.update(
            {
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "tail": lines[-20:],
            }
        )
        return summary

    def validate_paired(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
        controls: dict[str, Any] = {}
        candidates: dict[str, Any] = {}
        selected_candidate_variant: dict[str, str] = {}
        failed_logs: list[dict[str, Any]] = []
        for case in CASES:
            control_relative = f"paired/{case.short_name}_reference_control.json"
            control = self.read_json(control_relative)
            if control is not None:
                controls[case.short_name] = self.validate_paired_result(
                    control,
                    case,
                    control_relative,
                    expected_reference_candidate=True,
                )
            case_candidates: list[dict[str, Any]] = []
            persisted: list[str] = []
            for variant in ("inplace", "outplace"):
                step = f"paired/{case.short_name}_c10d_{variant}"
                relative = f"{step}.json"
                path = self.root / relative
                row = self.status_rows.get(step)
                if path.is_file() and path.stat().st_size > 0:
                    persisted.append(variant)
                    if row is None:
                        self.error("persisted_attempt_without_status", relative, step)
                    elif row["exit_code"] != 0:
                        self.error(
                            "persisted_attempt_failed_status",
                            relative,
                            f"status exit={row['exit_code']}",
                        )
                    result = self.read_json(relative)
                    if result is not None:
                        summary = self.validate_paired_result(
                            result,
                            case,
                            relative,
                            expected_reference_candidate=False,
                        )
                        summary.update({"variant": variant, "result_path": relative})
                        case_candidates.append(summary)
                else:
                    if row is not None and row["exit_code"] == 0:
                        self.error(
                            "successful_attempt_missing_result",
                            relative,
                            step,
                        )
                    if row is not None and row["exit_code"] != 0:
                        failed_logs.append(self.failed_log_summary(row))
            if persisted:
                selected_candidate_variant[case.short_name] = (
                    "outplace" if "outplace" in persisted else "inplace"
                )
            candidates[case.short_name] = {
                "persisted_results": case_candidates,
                "failed_attempts": [
                    item
                    for item in failed_logs
                    if item["step"].startswith(f"paired/{case.short_name}_")
                ],
            }
        candidates["failed_attempt_logs"] = failed_logs
        return controls, candidates, selected_candidate_variant

    def validate_profile_result(
        self,
        relative: str,
        case: Case,
        *,
        candidate: bool,
    ) -> None:
        result = self.read_json(relative)
        if result is None:
            return
        self.validate_common_result(
            result,
            case,
            relative,
            expected_mode=case.execution_mode,
            expected_stream=case.stream,
            expected_trace=False,
        )
        self.validate_sample_count(result, relative, "reference", 20)
        expected_range = (
            f"serving_native/{case.task}/{case.execution_mode}/{case.stream}/"
            f"{'paired' if candidate else 'reference'}"
        )
        execution = result.get("execution")
        observed_range = (
            execution.get("nvtx_measurement_range")
            if isinstance(execution, dict)
            else None
        )
        if observed_range != expected_range:
            self.error(
                "profile_nvtx_range_mismatch",
                relative,
                f"expected {expected_range!r}, got {observed_range!r}",
            )
        if candidate:
            self.validate_sample_count(result, relative, "candidate", 20)
            if not isinstance(result.get("candidate"), dict):
                self.error("missing_profile_candidate", relative, repr(result.get("candidate")))
        elif result.get("candidate") is not None:
            self.error("unexpected_profile_candidate", relative, "candidate is present")

    def load_profile_selection(self) -> dict[str, str]:
        path = self.root / "profile/c10d_profile_selection.tsv"
        if not path.is_file():
            return {}
        selected: dict[str, str] = {}
        try:
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    fields = line.rstrip("\n").split("\t")
                    if len(fields) != 3 or fields[0] not in {case.short_name for case in CASES}:
                        self.error(
                            "invalid_profile_selection",
                            f"profile/c10d_profile_selection.tsv:{line_number}",
                            repr(fields),
                        )
                        continue
                    short_name, candidate_path, result_path = fields
                    if short_name in selected:
                        self.error(
                            "duplicate_profile_selection",
                            f"profile/c10d_profile_selection.tsv:{line_number}",
                            short_name,
                        )
                    variant = "outplace" if candidate_path.endswith("allreduce_torch_outplace.py") else "inplace"
                    expected_result = f"{short_name}_c10d_{variant}.json"
                    if Path(result_path).name != expected_result:
                        self.error(
                            "profile_selection_result_mismatch",
                            f"profile/c10d_profile_selection.tsv:{line_number}",
                            f"candidate={candidate_path!r}, result={result_path!r}",
                        )
                    selected[short_name] = variant
        except OSError as exc:
            self.error(
                "profile_selection_read_error",
                "profile/c10d_profile_selection.tsv",
                f"{type(exc).__name__}: {exc}",
            )
        return selected

    def validate_profiles(self, expected_selected: dict[str, str]) -> dict[str, Any]:
        recorded_selected = self.load_profile_selection()
        output: dict[str, Any] = {}
        for case in CASES:
            report = self.root / f"profile/{case.short_name}.nsys-rep"
            report_summary = file_summary(report)
            if not report_summary["present"] or report_summary.get("bytes", 0) == 0:
                self.error(
                    "missing_nsys_report",
                    f"profile/{case.short_name}.nsys-rep",
                    "stock report absent or empty",
                )
            self.validate_profile_result(
                f"profile/{case.short_name}.result.json", case, candidate=False
            )
            stock_stats = self.root / f"profile/{case.short_name}.stats.log"
            expected_stock_range = (
                f"serving_native/{case.task}/{case.execution_mode}/"
                f"{case.stream}/reference"
            )
            if not stock_stats.is_file() or stock_stats.stat().st_size == 0:
                self.error(
                    "missing_nsys_stats",
                    f"profile/{case.short_name}.stats.log",
                    "stock stats are absent or empty",
                )
            elif expected_stock_range not in stock_stats.read_text(
                encoding="utf-8", errors="replace"
            ):
                self.error(
                    "missing_nsys_nvtx_range",
                    f"profile/{case.short_name}.stats.log",
                    expected_stock_range,
                )
            item: dict[str, Any] = {
                "stock": report_summary,
                "stock_stats_log": file_summary(stock_stats),
                "candidate": None,
            }
            expected_variant = expected_selected.get(case.short_name)
            recorded_variant = recorded_selected.get(case.short_name)
            if expected_variant is None:
                skipped_step = f"profile/{case.short_name}_c10d_skipped"
                row = self.status_rows.get(skipped_step)
                if row is None:
                    self.error("missing_candidate_profile_status", "status.tsv", skipped_step)
                elif row["requirement"] != "attempt" or row["exit_code"] == 0:
                    self.error(
                        "invalid_candidate_profile_skip",
                        f"status.tsv:{row['line_number']}",
                        f"requirement={row['requirement']!r}, exit={row['exit_code']}",
                    )
                if recorded_variant is not None:
                    self.error(
                        "unexpected_profile_selection",
                        "profile/c10d_profile_selection.tsv",
                        case.short_name,
                    )
            else:
                step = f"profile/{case.short_name}_c10d_nsys"
                row = self.status_rows.get(step)
                if row is None:
                    self.error("missing_candidate_profile_status", "status.tsv", step)
                elif row["requirement"] != "required" or row["exit_code"] != 0:
                    self.error(
                        "failed_candidate_profile_status",
                        f"status.tsv:{row['line_number']}",
                        f"requirement={row['requirement']!r}, exit={row['exit_code']}",
                    )
                if recorded_variant != expected_variant:
                    self.error(
                        "candidate_profile_selection_mismatch",
                        "profile/c10d_profile_selection.tsv",
                        f"{case.short_name}: expected {expected_variant!r}, got {recorded_variant!r}",
                    )
                candidate_report = self.root / f"profile/{case.short_name}_c10d.nsys-rep"
                candidate_summary = file_summary(candidate_report)
                if not candidate_summary["present"] or candidate_summary.get("bytes", 0) == 0:
                    self.error(
                        "missing_candidate_nsys_report",
                        f"profile/{case.short_name}_c10d.nsys-rep",
                        "candidate report absent or empty",
                    )
                self.validate_profile_result(
                    f"profile/{case.short_name}_c10d.result.json",
                    case,
                    candidate=True,
                )
                candidate_stats = (
                    self.root / f"profile/{case.short_name}_c10d.stats.log"
                )
                expected_candidate_range = (
                    f"serving_native/{case.task}/{case.execution_mode}/"
                    f"{case.stream}/paired"
                )
                if (
                    not candidate_stats.is_file()
                    or candidate_stats.stat().st_size == 0
                ):
                    self.error(
                        "missing_candidate_nsys_stats",
                        f"profile/{case.short_name}_c10d.stats.log",
                        "candidate stats are absent or empty",
                    )
                elif expected_candidate_range not in candidate_stats.read_text(
                    encoding="utf-8", errors="replace"
                ):
                    self.error(
                        "missing_candidate_nsys_nvtx_range",
                        f"profile/{case.short_name}_c10d.stats.log",
                        expected_candidate_range,
                    )
                item["candidate"] = {
                    "variant": expected_variant,
                    "report": candidate_summary,
                    "stats_log": file_summary(candidate_stats),
                }
            output[case.short_name] = item

        extra_recorded = set(recorded_selected) - {case.short_name for case in CASES}
        if extra_recorded:
            self.error(
                "unexpected_profile_selection_cases",
                "profile/c10d_profile_selection.tsv",
                repr(sorted(extra_recorded)),
            )
        return output

    def analyze(self) -> dict[str, Any]:
        if not self.root.is_dir():
            self.error("missing_campaign_root", str(self.root), "directory is absent")
            return self.finish({})
        self.load_status()
        source_identity = self.load_source_identity()
        traces = {case.short_name: self.validate_trace(case) for case in CASES}
        semantics = self.validate_semantics()
        baselines = self.validate_baselines()
        controls, candidates, selected = self.validate_paired()
        profiles = self.validate_profiles(selected)
        return self.finish(
            {
                "source_identity": source_identity,
                "reachability": traces,
                "semantics": semantics,
                "baselines": baselines,
                "reference_control_noise": controls,
                "c10d_attempts": candidates,
                "nsight_systems": profiles,
            }
        )

    def finish(self, sections: dict[str, Any]) -> dict[str, Any]:
        required_rows = [
            row for row in self.status_rows.values() if row["requirement"] == "required"
        ]
        attempts = [
            row for row in self.status_rows.values() if row["requirement"] == "attempt"
        ]
        return {
            "schema_version": 1,
            "scope": "tp4_allreduce_diagnostic_only",
            "production_acceptance": (
                "TP4 diagnostic evidence cannot be relabeled as TP8/DP8/EP8 production acceptance."
            ),
            "campaign_root": str(self.root),
            "valid": not self.errors,
            "status_summary": {
                "rows": len(self.status_rows),
                "required_rows": len(required_rows),
                "required_failed": sum(row["exit_code"] != 0 for row in required_rows),
                "attempt_rows": len(attempts),
                "attempt_failed": sum(row["exit_code"] != 0 for row in attempts),
            },
            **sections,
            "errors": self.errors,
        }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_summary(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {"path": str(path), "present": path.is_file()}
    if path.is_file():
        summary.update({"bytes": path.stat().st_size, "sha256": file_sha256(path)})
    return summary


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = JsonArgumentParser(description=__doc__)
    parser.add_argument("campaign_root")
    parser.add_argument("--output")
    return parser.parse_args(argv)


def render(result: dict[str, Any]) -> str:
    return json.dumps(result, indent=2, sort_keys=True) + "\n"


def main(argv: Optional[Iterable[str]] = None) -> int:
    try:
        args = parse_args(argv)
    except (SystemExit, ValueError) as exc:
        result = {
            "schema_version": 1,
            "scope": "tp4_allreduce_diagnostic_only",
            "valid": False,
            "errors": [
                {
                    "code": "argument_error",
                    "location": "command_line",
                    "detail": str(exc),
                }
            ],
        }
        sys.stdout.write(render(result))
        return 2

    result = Analyzer(Path(args.campaign_root)).analyze()
    payload = render(result)
    if args.output:
        output = Path(args.output).expanduser()
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(payload, encoding="utf-8")
        except OSError as exc:
            result["valid"] = False
            result["errors"].append(
                {
                    "code": "output_write_error",
                    "location": str(output),
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )
            payload = render(result)
    sys.stdout.write(payload)
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
