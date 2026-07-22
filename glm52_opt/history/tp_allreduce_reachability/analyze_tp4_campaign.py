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
import os
import re
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

BACKEND_SCOUT_MESSAGE_SIZES = (
    4 * 1024,
    16 * 1024,
    64 * 1024,
    128 * 1024,
    3 * 64 * 1024,
    4 * 64 * 1024,
    3 * 128 * 1024,
    4 * 128 * 1024,
    5 * 128 * 1024,
    6 * 128 * 1024,
    7 * 128 * 1024,
    1 * 1024 * 1024,
    2 * 1024 * 1024,
    3 * 1024 * 1024,
    4 * 1024 * 1024,
    8 * 1024 * 1024,
    16 * 1024 * 1024,
    32 * 1024 * 1024,
)
BACKEND_SCOUT_PROVIDERS = ("nccl", "aot", "jit", "fi")
SCHEDULED_START_LEAD_NS = 5_000_000
START_RECORD_ENVELOPE_LIMIT_NS = 500_000
MAX_PAIR_ATTEMPTS = 10
ALIGNMENT_ADMISSION_FIELDS = (
    "scheduled_start_arrival_ns_by_rank",
    "scheduled_start_target_ns_by_rank",
    "start_record_bracket_ns_by_rank",
    "start_record_envelope_span_ns",
)
BACKEND_SCOUT_HEADER = (
    "message_bytes",
    *(f"{provider}(us)" for provider in BACKEND_SCOUT_PROVIDERS),
    *(f"{provider}(GB/s)" for provider in BACKEND_SCOUT_PROVIDERS),
)


def baseline_relative_run_spread(medians: list[float]) -> float:
    """Return max/min - 1 for exactly three positive run medians."""

    if len(medians) != 3 or any(
        not math.isfinite(value) or value <= 0 for value in medians
    ):
        raise ValueError(f"expected three positive finite medians, got {medians!r}")
    return max(medians) / min(medians) - 1.0


def nvtx_bounds_from_csv(path: Path, expected_range: str) -> tuple[int, int]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            matches = [
                row
                for row in csv.DictReader(handle)
                if row.get("Name") in {expected_range, f":{expected_range}"}
            ]
    except OSError as exc:
        raise ValueError(f"cannot read NVTX trace {path}: {exc}") from exc
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {expected_range!r} NVTX range, got {matches!r}"
        )
    try:
        start = int(matches[0]["Start (ns)"])
        end = int(matches[0]["End (ns)"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid NVTX bounds in {path}: {matches[0]!r}") from exc
    if start <= 0 or end <= start:
        raise ValueError(f"invalid NVTX interval in {path}: {start}/{end}")
    return start, end


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}, got {type(value).__name__}")
    return value


def _resolution_status_rows(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "status.tsv"
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if tuple(reader.fieldnames or ()) != STATUS_FIELDS:
                raise ValueError(
                    f"invalid status header in {path}: {reader.fieldnames!r}"
                )
            rows: dict[str, dict[str, Any]] = {}
            for line_number, raw in enumerate(reader, start=2):
                step = raw.get("step", "")
                if not step or step in rows:
                    raise ValueError(
                        f"invalid or duplicate status step at {path}:{line_number}: {step!r}"
                    )
                try:
                    exit_code = int(raw.get("exit_code", ""))
                except ValueError as exc:
                    raise ValueError(
                        f"invalid status exit at {path}:{line_number}"
                    ) from exc
                log = Path(raw.get("log", ""))
                copied_log = root / f"{step}.log"
                if copied_log.is_file():
                    log = copied_log
                try:
                    log.relative_to(root)
                except ValueError as exc:
                    raise ValueError(
                        f"status log escapes campaign root at {path}:{line_number}: {log}"
                    ) from exc
                rows[step] = {
                    **raw,
                    "exit_code": exit_code,
                    "line_number": line_number,
                    "log": str(log),
                }
    except OSError as exc:
        raise ValueError(f"cannot read status ledger {path}: {exc}") from exc
    return rows


def _canonical_abi_mismatch_failures(log_text: str) -> list[list[dict[str, Any]]]:
    prefix = "SharedValidationError: candidate correctness failed collectively: "
    records: list[list[dict[str, Any]]] = []
    shared_validation_lines = [
        line.strip()
        for line in log_text.splitlines()
        if "SharedValidationError:" in line
    ]
    for line in shared_validation_lines:
        start = line.find(prefix)
        if start < 0:
            raise ValueError(f"unexpected shared-validation failure: {line}")
        payload = line[start + len(prefix) :]
        try:
            failures = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed shared-validation payload: {payload}") from exc
        if not isinstance(failures, list) or len(failures) != 4:
            raise ValueError(f"expected four collective failures, got {failures!r}")
        by_rank: dict[int, dict[str, Any]] = {}
        expected_fields = {
            "error",
            "rank",
            "scheduled_start_arrival_ns_by_rank",
            "scheduled_start_target_ns",
            "start_record_bracket_ns",
        }
        for failure in failures:
            if (
                not isinstance(failure, dict)
                or set(failure) != expected_fields
            ):
                raise ValueError(f"unexpected collective failure record: {failure!r}")
            rank = failure.get("rank")
            error = failure.get("error")
            if (
                not isinstance(rank, int)
                or isinstance(rank, bool)
                or rank in by_rank
                or not isinstance(error, str)
                or not error.startswith(
                    "AssertionError: candidate destructive/alias ABI differs from reference:"
                )
            ):
                raise ValueError(f"unexpected collective ABI failure: {failure!r}")
            arrivals = failure.get("scheduled_start_arrival_ns_by_rank")
            target = failure.get("scheduled_start_target_ns")
            bracket = failure.get("start_record_bracket_ns")
            if (
                not isinstance(arrivals, list)
                or len(arrivals) != 4
                or any(
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value <= 0
                    for value in arrivals
                )
                or not isinstance(target, int)
                or isinstance(target, bool)
                or target != max(arrivals) + SCHEDULED_START_LEAD_NS
            ):
                raise ValueError(
                    f"invalid failed-attempt scheduled start: {failure!r}"
                )
            if (
                not isinstance(bracket, list)
                or len(bracket) != 2
                or any(
                    not isinstance(value, int) or isinstance(value, bool)
                    for value in bracket
                )
                or bracket[0] > bracket[1]
                or bracket[0] < target
            ):
                raise ValueError(f"invalid failed-attempt start bracket: {failure!r}")
            by_rank[rank] = failure
        if set(by_rank) != set(range(4)):
            raise ValueError(f"collective ABI failure rank set mismatch: {by_rank!r}")
        common_arrivals = by_rank[0]["scheduled_start_arrival_ns_by_rank"]
        common_target = by_rank[0]["scheduled_start_target_ns"]
        if any(
            failure["scheduled_start_arrival_ns_by_rank"] != common_arrivals
            or failure["scheduled_start_target_ns"] != common_target
            for failure in by_rank.values()
        ):
            raise ValueError(
                f"collective ABI failure start metadata differs by rank: {by_rank!r}"
            )
        records.append([by_rank[rank] for rank in range(4)])
    if not records:
        raise ValueError("canonical collective ABI-mismatch failure is absent")
    return records


def resolve_c10d_abi(root: Path, case: Case, harness_root: Path) -> dict[str, Any]:
    """Fail-closed CPU resolution used before candidate profiling."""

    root = root.resolve()
    harness_root = harness_root.resolve()
    rows = _resolution_status_rows(root)
    control_path = root / f"paired/{case.short_name}_reference_control.json"
    control_step = f"paired/{case.short_name}_reference_control"
    control_row = rows.get(control_step)
    if (
        control_row is None
        or control_row.get("requirement") != "required"
        or control_row.get("exit_code") != 0
    ):
        raise ValueError(f"successful required status is absent for {control_step}")
    control = _load_json_object(control_path)
    if control.get("workload", {}).get("name") != case.task:
        raise ValueError(f"reference-control workload mismatch in {control_path}")
    contracts = control.get("allreduce", {}).get("reference_contract_by_rank")
    if (
        not isinstance(contracts, list)
        or len(contracts) != 4
        or any(not isinstance(contract, dict) for contract in contracts)
    ):
        raise ValueError(f"reference contracts are absent or malformed: {contracts!r}")
    contract_keys = {
        (
            contract.get("output_aliases_local"),
            contract.get("local_poststate"),
        )
        for contract in contracts
    }
    if len(contract_keys) != 1:
        raise ValueError(f"reference alias contract differs by rank: {contracts!r}")
    for rank, contract in enumerate(contracts):
        output = contract.get("output")
        expected_output = {
            "shape": [case.rows, 6144],
            "stride": [6144, 1],
            "dtype": "torch.bfloat16",
            "device": f"cuda:{rank}",
        }
        if (
            output != expected_output
            or contract.get("source_immutable") is not True
            or contract.get("exact_values") is not True
        ):
            raise ValueError(
                f"reference contract mismatch for rank {rank}: {contract!r}"
            )
    contract_key = next(iter(contract_keys))
    expected_variant = {
        (True, "reduced"): "inplace",
        (False, "source"): "outplace",
    }.get(contract_key)
    if expected_variant is None:
        raise ValueError(f"unsupported reference alias contract: {contracts!r}")

    source_identity_path = root / "environment/source_identity.log"
    try:
        source_payload = Analyzer._command_payload(source_identity_path)
    except OSError as exc:
        raise ValueError(f"cannot read source identity {source_identity_path}: {exc}") from exc
    if (
        len(source_payload) != 2
        or any(re.fullmatch(r"[0-9a-f]{40}", value) is None for value in source_payload)
    ):
        raise ValueError(f"invalid clean source identity: {source_payload!r}")
    sglang_root = Path(__file__).resolve().parents[3]
    validator = Analyzer(root)
    validator.expected_roots = {
        "kernel_harness_git": str(harness_root),
        "sglang_git": str(sglang_root),
    }
    validator.expected_shas = {
        "kernel_harness_git": source_payload[0],
        "sglang_git": source_payload[1],
    }
    validator.validate_paired_result(
        control,
        case,
        str(control_path.relative_to(root)),
        expected_entrypoint="reference.py",
    )
    if validator.errors:
        raise ValueError(
            "reference control failed full paired validation: "
            + json.dumps(validator.errors, sort_keys=True)
        )

    selected: Optional[dict[str, Any]] = None
    failed: Optional[dict[str, Any]] = None
    for variant, filename in (
        ("inplace", "allreduce_torch.py"),
        ("outplace", "allreduce_torch_outplace.py"),
    ):
        step = f"paired/{case.short_name}_c10d_{variant}"
        row = rows.get(step)
        if row is None or row.get("requirement") != "attempt":
            raise ValueError(f"missing attempt status for {step}")
        result_path = root / f"{step}.json"
        result_present = result_path.is_file() and result_path.stat().st_size > 0
        if row["exit_code"] == 0:
            if not result_present:
                raise ValueError(f"successful attempt has no result: {result_path}")
            result = _load_json_object(result_path)
            if result.get("workload", {}).get("name") != case.task:
                raise ValueError(f"candidate workload mismatch in {result_path}")
            allreduce = result.get("allreduce")
            if not isinstance(allreduce, dict):
                raise ValueError(f"candidate AllReduce evidence is absent in {result_path}")
            if allreduce.get("reference_contract_by_rank") != contracts:
                raise ValueError(f"candidate/reference contracts differ in {result_path}")
            if allreduce.get("candidate_contract_by_rank") != contracts:
                raise ValueError(f"candidate ABI does not match reference in {result_path}")
            candidate = result.get("candidate")
            manifest = candidate.get("manifest") if isinstance(candidate, dict) else None
            if not isinstance(manifest, dict):
                raise ValueError(f"candidate manifest is absent in {result_path}")
            candidate_path = (
                harness_root / "serving_native" / "candidates" / filename
            ).resolve()
            if manifest.get("entrypoint") != str(candidate_path) or manifest.get(
                "requested_path"
            ) != str(candidate_path):
                raise ValueError(f"candidate path mismatch in {result_path}")
            entrypoint_sha = manifest.get("entrypoint_sha256_at_import")
            manifest_sha = manifest.get("manifest_sha256")
            if (
                not candidate_path.is_file()
                or not isinstance(entrypoint_sha, str)
                or not re.fullmatch(r"[0-9a-f]{64}", entrypoint_sha)
                or file_sha256(candidate_path) != entrypoint_sha
                or not isinstance(manifest_sha, str)
                or not re.fullmatch(r"[0-9a-f]{64}", manifest_sha)
            ):
                raise ValueError(f"candidate source/manifest digest mismatch in {result_path}")
            if selected is not None:
                raise ValueError("more than one c10d ABI result succeeded")
            candidate_validator = Analyzer(root)
            candidate_validator.expected_roots = dict(validator.expected_roots)
            candidate_validator.expected_shas = dict(validator.expected_shas)
            candidate_validator.validate_paired_result(
                result,
                case,
                str(result_path.relative_to(root)),
                expected_entrypoint=filename,
            )
            if candidate_validator.errors:
                raise ValueError(
                    "candidate result failed full paired validation: "
                    + json.dumps(candidate_validator.errors, sort_keys=True)
                )
            selected = {
                "variant": variant,
                "candidate_path": str(candidate_path),
                "candidate_entrypoint_sha256": entrypoint_sha,
                "candidate_manifest_sha256": manifest_sha,
                "result_path": str(result_path.relative_to(root)),
                "result_sha256": file_sha256(result_path),
            }
        else:
            if row["exit_code"] != 1 or result_present:
                raise ValueError(
                    f"unexpected failed-attempt state for {step}: "
                    f"exit={row['exit_code']}, result_present={result_present}"
                )
            log_path = Path(str(row["log"]))
            try:
                log_text = log_path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                raise ValueError(f"cannot read failed attempt log {log_path}: {exc}") from exc
            failures = _canonical_abi_mismatch_failures(log_text)
            if failed is not None:
                raise ValueError("more than one c10d ABI attempt failed")
            failed = {
                "variant": variant,
                "exit_code": row["exit_code"],
                "log_path": str(log_path.relative_to(root)),
                "log_sha256": file_sha256(log_path),
                "canonical_failure_records": len(failures),
            }

    if selected is None or failed is None:
        raise ValueError("exactly one successful and one canonical failed attempt are required")
    if selected["variant"] != expected_variant or failed["variant"] == expected_variant:
        raise ValueError(
            f"resolved variant mismatch: expected={expected_variant}, "
            f"selected={selected['variant']}, failed={failed['variant']}"
        )
    return {
        "schema_version": 1,
        "scope": "tp4_allreduce_diagnostic_only",
        "short_name": case.short_name,
        "task": case.task,
        "expected_variant": expected_variant,
        "reference_control_path": str(control_path.relative_to(root)),
        "reference_control_sha256": file_sha256(control_path),
        "selected": selected,
        "failed": failed,
    }

STATUS_FIELDS = (
    "requirement",
    "step",
    "exit_code",
    "started_utc",
    "finished_utc",
    "log",
)

REQUIRED_STATUS_STEPS = {
    "environment/lock_receipt",
    "environment/source_identity",
    "environment/check_env",
    "environment/nvidia_smi",
    "environment/compute_processes_before",
    "environment/topology",
    "environment/p2p_capability",
    "environment/nvlink_status",
    "environment/nvlink_throughput_before",
    "environment/nvlink_throughput_after",
    "environment/nvidia_smi_after",
    "environment/compute_processes_after",
    "environment/source_identity_after",
    "backend_scout/custom_allreduce",
    "producer_abi/linear_attn_o_decode_m16",
    "producer_abi/linear_attn_o_decode_m32",
    *(f"reachability/{case.short_name}" for case in CASES),
    *(f"baseline/{case.short_name}_run{run}" for case in CASES for run in range(1, 4)),
    *(f"paired/{case.short_name}_reference_control" for case in CASES),
    *(f"paired/{case.short_name}_c10d_abi_resolution" for case in CASES),
    *(f"profile/{case.short_name}_nsys" for case in CASES),
    *(f"profile/{case.short_name}_c10d_nsys" for case in CASES),
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
    *(
        f"paired/{case.short_name}_c10d_{variant}"
        for case in CASES
        for variant in ("inplace", "outplace")
    ),
}

TRACE_PREDICATE_KEYS = {
    "world_size_is_one",
    "input_is_cpu",
    "cpu_shm_eligible",
    "hpu_present",
    "hpu_disabled",
    "hpu_eligible",
    "xpu_present",
    "xpu_disabled",
    "xpu_eligible",
    "npu_present",
    "npu_disabled",
    "npu_eligible",
    "pymscclpp_present",
    "pymscclpp_should",
    "pymscclpp_disabled",
    "pymscclpp_eligible",
    "symmetric_memory_enabled",
    "pynccl_present",
    "pynccl_disabled",
    "pynccl_symmetric_eligible",
    "custom_present",
    "custom_should",
    "custom_disabled",
    "custom_eligible",
    "quick_present",
    "quick_should",
    "quick_disabled",
    "quick_eligible",
    "torch_symm_mem_present",
    "torch_symm_mem_should",
    "torch_symm_mem_disabled",
    "torch_symm_mem_eligible",
    "piecewise_cuda_graph",
    "piecewise_pynccl_eligible",
    "pynccl_inplace_eligible",
    "torch_symm_mem_inplace_eligible",
}

TRACE_GPU_BACKENDS = {
    "pynccl_symmetric_inplace",
    "custom_all_reduce_outplace",
    "quick_all_reduce_outplace",
    "pymscclpp_outplace",
    "torch_symm_mem_outplace",
    "pynccl_piecewise_outplace",
    "pynccl_inplace",
    "torch_symm_mem_inplace",
    "torch_distributed_inplace",
}


def selected_backend_from_predicates(predicates: dict[str, Any]) -> str:
    """Independently replay the CUDA branch priority from pure trace data."""
    if predicates.get("world_size_is_one"):
        return "identity"
    if predicates.get("input_is_cpu"):
        return "cpu_shm" if predicates.get("cpu_shm_eligible") else "cpu_c10d"
    if predicates.get("hpu_eligible"):
        return "hpu"
    if predicates.get("xpu_eligible"):
        return "xpu_torch_distributed_inplace"
    if predicates.get("npu_eligible"):
        return "npu"
    if predicates.get("pynccl_symmetric_eligible"):
        return "pynccl_symmetric_inplace"
    if predicates.get("custom_eligible"):
        return "custom_all_reduce_outplace"
    if predicates.get("quick_eligible"):
        return "quick_all_reduce_outplace"
    if predicates.get("pymscclpp_eligible"):
        return "pymscclpp_outplace"
    if predicates.get("torch_symm_mem_eligible"):
        return "torch_symm_mem_outplace"
    if predicates.get("piecewise_pynccl_eligible"):
        return "pynccl_piecewise_outplace"
    if predicates.get("pynccl_inplace_eligible"):
        return "pynccl_inplace"
    if predicates.get("torch_symm_mem_inplace_eligible"):
        return "torch_symm_mem_inplace"
    return "torch_distributed_inplace"


def trace_record_matches_selected_execution(
    case: Case, capture_state: Optional[bool]
) -> bool:
    """Match the dispatch snapshot: graph capture for graph cases, eager otherwise."""
    return (
        capture_state is True
        if case.execution_mode == "cuda_graph"
        else capture_state is False
    )


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

    def _environment_payload(self, relative: str) -> list[str]:
        path = self.root / relative
        try:
            return self._command_payload(path)
        except OSError as exc:
            self.error(
                "environment_payload_read_error",
                relative,
                f"{type(exc).__name__}: {exc}",
            )
            return []

    def _validate_check_env(self) -> dict[str, Any]:
        relative = "environment/check_env.log"
        payload = self._environment_payload(relative)
        required_exact = {
            "gpu:         NVIDIA B200 sm_100",
            "visible gpus:4  CUDA_VISIBLE_DEVICES=0,1,2,3",
            "M3 kernels:  present in SGLANG_DIR",
            "Environment check passed.",
        }
        if not required_exact <= set(payload):
            self.error(
                "check_env_contract_mismatch",
                relative,
                repr(payload),
            )
        prefixes = {
            "python": "python:",
            "venv": "venv:",
            "cuda_home": "cuda home:",
            "sglang": "sglang:",
            "torch_cuda": "torch/cuda:",
        }
        observed: dict[str, str] = {}
        for key, prefix in prefixes.items():
            matches = [line for line in payload if line.startswith(prefix)]
            if len(matches) != 1:
                self.error(
                    "check_env_field_mismatch",
                    relative,
                    f"{key}: {matches!r}",
                )
            elif matches:
                observed[key] = matches[0].split(":", 1)[1].strip()
        harness_root = self.expected_roots.get("kernel_harness_git")
        sglang_root = self.expected_roots.get("sglang_git")
        if harness_root is not None:
            expected_python = str(Path(harness_root) / ".venv/bin/python")
            expected_venv = str(Path(harness_root) / ".venv")
            if observed.get("python") != expected_python:
                self.error(
                    "check_env_python_mismatch",
                    relative,
                    f"expected={expected_python!r}, got={observed.get('python')!r}",
                )
            if observed.get("venv") != expected_venv:
                self.error(
                    "check_env_venv_mismatch",
                    relative,
                    f"expected={expected_venv!r}, got={observed.get('venv')!r}",
                )
        sglang_sha = self.expected_shas.get("sglang_git")
        if sglang_root is not None and sglang_sha is not None:
            match = re.fullmatch(
                rf"checkout {re.escape(sglang_root)} \(([0-9a-f]+)\)",
                observed.get("sglang", ""),
            )
            observed_abbrev = match.group(1) if match is not None else ""
            if len(observed_abbrev) < 7 or not sglang_sha.startswith(observed_abbrev):
                self.error(
                    "check_env_sglang_mismatch",
                    relative,
                    (
                        f"expected checkout {sglang_root!r} with a prefix of "
                        f"{sglang_sha!r}, got={observed.get('sglang')!r}"
                    ),
                )
        if not re.fullmatch(r"[^\s]+ / [^\s]+", observed.get("torch_cuda", "")):
            self.error(
                "check_env_torch_cuda_mismatch",
                relative,
                repr(observed.get("torch_cuda")),
            )
        return {"fields": observed, **file_summary(self.root / relative)}

    def _parse_gpu_inventory(
        self, relative: str, *, final: bool
    ) -> dict[int, dict[str, Any]]:
        payload = self._environment_payload(relative)
        expected_header = (
            "index, uuid, clocks.current.sm [MHz], clocks.current.memory [MHz], "
            "power.draw [W], temperature.gpu"
            if final
            else "index, uuid, name, pci.bus_id, clocks.current.sm [MHz], "
            "clocks.current.memory [MHz], power.draw [W], temperature.gpu"
        )
        if not payload or payload[0] != expected_header:
            self.error("gpu_inventory_header_mismatch", relative, repr(payload[:1]))
            return {}
        devices: dict[int, dict[str, Any]] = {}
        expected_columns = 6 if final else 8
        for line in payload[1:]:
            fields = next(csv.reader([line], skipinitialspace=True))
            if len(fields) != expected_columns:
                self.error("gpu_inventory_row_mismatch", relative, repr(fields))
                continue
            try:
                rank = int(fields[0])
                uuid = fields[1]
                offset = 2
                name = None
                pci_bus_id = None
                if not final:
                    name = fields[offset]
                    pci_bus_id = fields[offset + 1]
                    offset += 2
                sm_clock = int(fields[offset].removesuffix(" MHz"))
                memory_clock = int(fields[offset + 1].removesuffix(" MHz"))
                power = float(fields[offset + 2].removesuffix(" W"))
                temperature = int(fields[offset + 3])
            except (ValueError, IndexError) as exc:
                self.error(
                    "gpu_inventory_value_mismatch",
                    relative,
                    f"{fields!r}: {exc}",
                )
                continue
            if (
                rank not in range(4)
                or rank in devices
                or re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", uuid) is None
                or sm_clock <= 0
                or memory_clock <= 0
                or not math.isfinite(power)
                or power <= 0
                or temperature < 0
                or (not final and name != "NVIDIA B200")
                or (
                    not final
                    and re.fullmatch(
                        r"[0-9A-Fa-f]{8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]",
                        str(pci_bus_id),
                    )
                    is None
                )
            ):
                self.error("gpu_inventory_contract_mismatch", relative, repr(fields))
            devices[rank] = {
                "uuid": uuid,
                "name": name,
                "pci_bus_id": pci_bus_id,
                "sm_clock_mhz": sm_clock,
                "memory_clock_mhz": memory_clock,
                "power_w": power,
                "temperature_c": temperature,
            }
        if set(devices) != set(range(4)) or len(
            {device["uuid"] for device in devices.values()}
        ) != 4:
            self.error("gpu_inventory_rank_set_mismatch", relative, repr(devices))
        return devices

    def _validate_topology(self) -> dict[str, Any]:
        relative = "environment/topology.log"
        payload = self._environment_payload(relative)
        text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", "\n".join(payload))
        rows: dict[int, list[str]] = {}
        gpu_lines = [
            " ".join(line.split())
            for line in text.splitlines()
            if line.strip().startswith("GPU")
        ]
        header_count = sum(
            line.startswith(
                "GPU0 GPU1 GPU2 GPU3 CPU Affinity NUMA Affinity GPU NUMA ID"
            )
            for line in gpu_lines
        )
        if header_count != 1:
            self.error("topology_header_mismatch", relative, repr(gpu_lines))
        for line in gpu_lines:
            if line.startswith(
                "GPU0 GPU1 GPU2 GPU3 CPU Affinity NUMA Affinity GPU NUMA ID"
            ):
                continue
            match = re.fullmatch(r"GPU([0-3])\s+(.+)", line)
            if match is None:
                self.error("topology_unexpected_gpu_row", relative, repr(line))
                continue
            fields = match.group(2).split()
            if len(fields) < 4:
                self.error("topology_row_mismatch", relative, repr(line))
                continue
            rank = int(match.group(1))
            if rank in rows:
                self.error("topology_duplicate_rank", relative, str(rank))
            rows[rank] = fields[:4]
        exact = set(rows) == set(range(4)) and all(
            fields[rank] == "X"
            and all(
                field == "NV18" for peer, field in enumerate(fields) if peer != rank
            )
            for rank, fields in rows.items()
        )
        if not exact:
            self.error("topology_not_full_nvlink", relative, repr(rows))
        return {"matrix": rows, **file_summary(self.root / relative)}

    def _parse_nvlink(
        self,
        relative: str,
        *,
        throughput: bool,
        expected_uuids: dict[int, str],
    ) -> dict[int, dict[str, Any]]:
        payload = self._environment_payload(relative)
        devices: dict[int, dict[str, Any]] = {}
        current_rank: Optional[int] = None
        for line in payload:
            header = re.fullmatch(
                r"GPU ([0-3]): NVIDIA B200 \(UUID: (GPU-[0-9a-fA-F-]{36})\)",
                line,
            )
            if header is not None:
                current_rank = int(header.group(1))
                if current_rank in devices:
                    self.error("nvlink_duplicate_gpu", relative, str(current_rank))
                devices[current_rank] = {"uuid": header.group(2), "links": {}}
                continue
            if current_rank is None:
                self.error("nvlink_unexpected_payload", relative, repr(line))
                continue
            if throughput:
                match = re.fullmatch(
                    r"Link ([0-9]+): Data (Tx|Rx): ([0-9]+) KiB", line
                )
                if match is None:
                    self.error("nvlink_throughput_row_mismatch", relative, repr(line))
                    continue
                link = int(match.group(1))
                direction = match.group(2).lower()
                record = devices[current_rank]["links"].setdefault(link, {})
                if direction in record:
                    self.error(
                        "nvlink_duplicate_direction",
                        relative,
                        f"rank={current_rank}, link={link}, direction={direction}",
                    )
                record[direction] = int(match.group(3))
            else:
                match = re.fullmatch(
                    r"Link ([0-9]+): ([0-9]+(?:\.[0-9]+)?) GB/s", line
                )
                if match is None:
                    self.error("nvlink_status_row_mismatch", relative, repr(line))
                    continue
                link = int(match.group(1))
                if link in devices[current_rank]["links"]:
                    self.error(
                        "nvlink_duplicate_status",
                        relative,
                        f"rank={current_rank}, link={link}",
                    )
                try:
                    rate = float(match.group(2))
                except ValueError as exc:
                    self.error(
                        "nvlink_status_value_mismatch",
                        relative,
                        f"{line!r}: {exc}",
                    )
                    continue
                devices[current_rank]["links"][link] = rate
        for rank in range(4):
            device = devices.get(rank)
            if device is None:
                self.error("nvlink_gpu_set_mismatch", relative, str(rank))
                continue
            if device["uuid"] != expected_uuids.get(rank):
                self.error(
                    "nvlink_uuid_mismatch",
                    relative,
                    f"rank={rank}, uuid={device['uuid']!r}",
                )
            links = device["links"]
            if set(links) != set(range(18)):
                self.error("nvlink_link_set_mismatch", relative, f"rank={rank}")
            elif throughput:
                if any(
                    set(record) != {"tx", "rx"}
                    or any(value < 0 for value in record.values())
                    for record in links.values()
                ):
                    self.error("nvlink_throughput_contract_mismatch", relative, f"rank={rank}")
            elif any(
                not isinstance(rate, float)
                or not math.isfinite(rate)
                or rate <= 0
                for rate in links.values()
            ):
                self.error("nvlink_status_contract_mismatch", relative, f"rank={rank}")
        return devices

    def validate_environment_evidence(self) -> dict[str, Any]:
        lock_path = self.root / "environment/lock_receipt.log"
        lock_payload: list[str] = []
        try:
            lock_payload = self._command_payload(lock_path)
        except OSError as exc:
            self.error(
                "lock_receipt_read_error",
                "environment/lock_receipt.log",
                f"{type(exc).__name__}: {exc}",
            )
        expected_lock_lines = {
            "CUDA_VISIBLE_DEVICES=0,1,2,3",
            *(
                f"fd={9 + rank} expected=/home/qinhaiyan/glm52-goal-runs/locks/"
                f"gpu{rank}.lock actual=/home/qinhaiyan/glm52-goal-runs/locks/"
                f"gpu{rank}.lock"
                for rank in range(4)
            ),
        }
        if set(lock_payload) != expected_lock_lines:
            self.error(
                "invalid_lock_receipt",
                "environment/lock_receipt.log",
                f"expected={sorted(expected_lock_lines)!r}, got={sorted(lock_payload)!r}",
            )

        process_snapshots: dict[str, Any] = {}
        expected_header = (
            "timestamp, gpu_uuid, pid, process_name, used_gpu_memory [MiB]"
        )
        for phase in ("before", "after"):
            relative = f"environment/compute_processes_{phase}.log"
            path = self.root / relative
            try:
                payload = self._command_payload(path)
            except OSError as exc:
                self.error(
                    "compute_process_snapshot_read_error",
                    relative,
                    f"{type(exc).__name__}: {exc}",
                )
                continue
            if payload != [expected_header]:
                self.error(
                    "unexpected_compute_process",
                    relative,
                    repr(payload),
                )
            process_snapshots[phase] = {
                "path": str(path),
                "no_compute_processes": payload == [expected_header],
                "sha256": file_sha256(path) if path.is_file() else None,
            }

        p2p_path = self.root / "environment/p2p_capability.log"
        p2p_counts: dict[str, int] = {}
        try:
            text = p2p_path.read_text(encoding="utf-8", errors="replace")
            text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
            capability_markers = re.findall(
                r"^capability=([^\s]+)\s*$", text, flags=re.MULTILINE
            )
            sections = re.split(
                r"^capability=([^\s]+)\s*$", text, flags=re.MULTILINE
            )
            observed_sections: dict[str, str] = {}
            for index in range(1, len(sections), 2):
                observed_sections[sections[index]] = sections[index + 1]
            if capability_markers != ["r", "w", "n"]:
                self.error(
                    "p2p_capability_section_mismatch",
                    "environment/p2p_capability.log",
                    repr(capability_markers),
                )
            for capability, section in observed_sections.items():
                header_pattern = re.compile(
                    r"^GPU0\s+GPU1\s+GPU2\s+GPU3$"
                )
                gpu_lines = [
                    line.strip()
                    for line in section.splitlines()
                    if line.strip().startswith("GPU")
                ]
                header_count = sum(
                    header_pattern.fullmatch(line) is not None for line in gpu_lines
                )
                if header_count != 1:
                    self.error(
                        "p2p_capability_header_mismatch",
                        "environment/p2p_capability.log",
                        f"capability={capability}, header_count={header_count}",
                    )
                matrix_rows: dict[int, list[str]] = {}
                for line in gpu_lines:
                    if header_pattern.fullmatch(line) is not None:
                        continue
                    match = re.fullmatch(r"GPU([0-3])\s+(.+)", line)
                    if match is None:
                        self.error(
                            "p2p_capability_unexpected_gpu_row",
                            "environment/p2p_capability.log",
                            f"capability={capability}, row={line!r}",
                        )
                        continue
                    rank = int(match.group(1))
                    fields = match.group(2).split()
                    if len(fields) != 4:
                        self.error(
                            "p2p_capability_malformed_rank_row",
                            "environment/p2p_capability.log",
                            f"capability={capability}, row={line!r}",
                        )
                        continue
                    if rank in matrix_rows:
                        self.error(
                            "p2p_capability_duplicate_rank",
                            "environment/p2p_capability.log",
                            f"capability={capability}, rank={rank}",
                        )
                    matrix_rows[rank] = fields
                ok_count = sum(
                    field == "OK"
                    for fields in matrix_rows.values()
                    for field in fields
                )
                p2p_counts[capability] = ok_count
                exact_matrix = set(matrix_rows) == set(range(4)) and all(
                    fields[rank] == "X"
                    and all(
                        field == "OK"
                        for peer, field in enumerate(fields)
                        if peer != rank
                    )
                    for rank, fields in matrix_rows.items()
                )
                if not exact_matrix or ok_count != 12:
                    self.error(
                        "p2p_capability_not_full_mesh",
                        "environment/p2p_capability.log",
                        (
                            f"capability={capability}, rows={matrix_rows!r}, "
                            f"directed_ok_edges={ok_count}"
                        ),
                    )
        except OSError as exc:
            self.error(
                "p2p_capability_read_error",
                "environment/p2p_capability.log",
                f"{type(exc).__name__}: {exc}",
            )
        check_env = self._validate_check_env()
        gpu_start = self._parse_gpu_inventory(
            "environment/nvidia_smi.log", final=False
        )
        gpu_after = self._parse_gpu_inventory(
            "environment/nvidia_smi_after.log", final=True
        )
        start_uuids = {
            rank: str(device.get("uuid")) for rank, device in gpu_start.items()
        }
        after_uuids = {
            rank: str(device.get("uuid")) for rank, device in gpu_after.items()
        }
        if start_uuids != after_uuids:
            self.error(
                "gpu_inventory_changed",
                "environment/nvidia_smi_after.log",
                f"start={start_uuids!r}, after={after_uuids!r}",
            )
        topology = self._validate_topology()
        nvlink_status = self._parse_nvlink(
            "environment/nvlink_status.log",
            throughput=False,
            expected_uuids=start_uuids,
        )
        nvlink_before = self._parse_nvlink(
            "environment/nvlink_throughput_before.log",
            throughput=True,
            expected_uuids=start_uuids,
        )
        nvlink_after = self._parse_nvlink(
            "environment/nvlink_throughput_after.log",
            throughput=True,
            expected_uuids=start_uuids,
        )
        nvlink_delta_by_rank_kib = {
            rank: {"tx": 0, "rx": 0, "total": 0} for rank in range(4)
        }
        for rank in range(4):
            before_links = nvlink_before.get(rank, {}).get("links", {})
            after_links = nvlink_after.get(rank, {}).get("links", {})
            for link in range(18):
                for direction in ("tx", "rx"):
                    before_value = before_links.get(link, {}).get(direction)
                    after_value = after_links.get(link, {}).get(direction)
                    if not isinstance(before_value, int) or not isinstance(
                        after_value, int
                    ):
                        continue
                    if after_value < before_value:
                        self.error(
                            "nvlink_counter_regressed",
                            "environment/nvlink_throughput_after.log",
                            (
                                f"rank={rank}, link={link}, direction={direction}, "
                                f"before={before_value}, after={after_value}"
                            ),
                        )
                    else:
                        delta = after_value - before_value
                        nvlink_delta_by_rank_kib[rank][direction] += delta
                        nvlink_delta_by_rank_kib[rank]["total"] += delta
        if nvlink_before and nvlink_after:
            for rank, deltas in nvlink_delta_by_rank_kib.items():
                if deltas["tx"] <= 0 or deltas["rx"] <= 0:
                    self.error(
                        "nvlink_counter_delta_missing",
                        "environment/nvlink_throughput_after.log",
                        f"rank={rank}, deltas={deltas!r}",
                    )
        nvlink_delta_kib = sum(
            deltas["total"] for deltas in nvlink_delta_by_rank_kib.values()
        )
        return {
            "lock_receipt": file_summary(lock_path),
            "compute_process_snapshots": process_snapshots,
            "p2p_full_mesh_directed_ok_edges": p2p_counts,
            "p2p_capability": file_summary(p2p_path),
            "check_env": check_env,
            "gpu_inventory_start": gpu_start,
            "gpu_inventory_after": gpu_after,
            "topology": topology,
            "nvlink_status": nvlink_status,
            "nvlink_throughput_before": nvlink_before,
            "nvlink_throughput_after": nvlink_after,
            "nvlink_counter_delta_by_rank_kib": nvlink_delta_by_rank_kib,
            "nvlink_counter_delta_kib": nvlink_delta_kib,
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
            if metadata.get("dirty") is not False or metadata.get("status") != []:
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

    def validate_reference_contracts(
        self,
        allreduce: dict[str, Any],
        case: Case,
        location: str,
    ) -> None:
        contracts = allreduce.get("reference_contract_by_rank")
        if not isinstance(contracts, list) or len(contracts) != 4:
            self.error("invalid_reference_contracts", location, repr(contracts))
            return
        alias_poststates: set[tuple[bool, str]] = set()
        for rank, contract in enumerate(contracts):
            contract_location = f"{location}:reference_contract_by_rank[{rank}]"
            if not isinstance(contract, dict):
                self.error("invalid_reference_contract", contract_location, repr(contract))
                continue
            output = contract.get("output")
            expected_output = {
                "shape": [case.rows, 6144],
                "stride": [6144, 1],
                "dtype": "torch.bfloat16",
                "device": f"cuda:{rank}",
            }
            if not isinstance(output, dict):
                self.error("missing_reference_output_contract", contract_location, repr(output))
            else:
                for key, expected in expected_output.items():
                    if output.get(key) != expected:
                        self.error(
                            "reference_output_contract_mismatch",
                            contract_location,
                            f"{key}: expected {expected!r}, got {output.get(key)!r}",
                        )
            alias = contract.get("output_aliases_local")
            poststate = contract.get("local_poststate")
            if not isinstance(alias, bool) or poststate not in {"source", "reduced"}:
                self.error(
                    "invalid_reference_alias_contract",
                    contract_location,
                    f"alias={alias!r}, local_poststate={poststate!r}",
                )
            else:
                alias_poststates.add((alias, poststate))
                if alias and poststate != "reduced":
                    self.error(
                        "aliased_reference_not_destructive",
                        contract_location,
                        f"local_poststate={poststate!r}",
                    )
            for key in ("source_immutable", "exact_values"):
                if contract.get(key) is not True:
                    self.error(
                        "invalid_reference_contract_flag",
                        contract_location,
                        f"{key}={contract.get(key)!r}",
                    )
        if len(alias_poststates) != 1:
            self.error(
                "reference_alias_contract_differs_by_rank",
                location,
                repr(sorted(alias_poststates)),
            )

    def validate_dispatch_records(
        self,
        allreduce: dict[str, Any],
        case: Case,
        location: str,
        *,
        expected_mode: Optional[str],
        expected_stream: Optional[str],
    ) -> dict[int, dict[str, str]]:
        records = allreduce.get("reference_dispatch_by_rank")
        by_rank: dict[int, dict[str, str]] = {}
        if not isinstance(records, list) or len(records) != 4:
            self.error("invalid_reference_dispatch", location, repr(records))
            return by_rank
        for index, record in enumerate(records):
            record_location = f"{location}:reference_dispatch_by_rank[{index}]"
            if not isinstance(record, dict) or not isinstance(record.get("rank"), int):
                self.error("invalid_reference_dispatch_record", record_location, repr(record))
                continue
            rank = record["rank"]
            if rank in by_rank:
                self.error("duplicate_reference_dispatch_rank", record_location, str(rank))
                continue
            expected = {
                "group_ranks": [0, 1, 2, 3],
                "world_size": 4,
                # GPU production leaves LOCAL_SIZE unset; it is a CPU shared-
                # memory hint, so GroupCoordinator records its zero default.
                "local_size": 0,
                "shape": [case.rows, 6144],
                "stride": [6144, 1],
                "dtype": "torch.bfloat16",
                "message_bytes": case.message_bytes,
            }
            for key, expected_value in expected.items():
                if record.get(key) != expected_value:
                    self.error(
                        "reference_dispatch_mismatch",
                        record_location,
                        f"{key}: expected {expected_value!r}, got {record.get(key)!r}",
                    )
            if expected_mode is not None and record.get("execution_mode") != expected_mode:
                self.error(
                    "reference_dispatch_mode_mismatch",
                    record_location,
                    repr(record.get("execution_mode")),
                )
            stream = record.get("stream")
            if not isinstance(stream, dict):
                self.error("missing_reference_dispatch_stream", record_location, repr(stream))
            else:
                if expected_stream is not None and stream.get("requested") != expected_stream:
                    self.error(
                        "reference_dispatch_stream_mismatch",
                        record_location,
                        repr(stream.get("requested")),
                    )
                if not isinstance(stream.get("cuda_stream"), int):
                    self.error(
                        "invalid_reference_dispatch_stream",
                        record_location,
                        repr(stream.get("cuda_stream")),
                    )
            backend = record.get("predicted_reference_backend")
            algorithm = record.get("predicted_reference_algorithm")
            if backend not in TRACE_GPU_BACKENDS or not isinstance(algorithm, str) or not algorithm:
                self.error(
                    "invalid_reference_dispatch_selection",
                    record_location,
                    f"backend={backend!r}, algorithm={algorithm!r}",
                )
            else:
                by_rank[rank] = {"backend": backend, "algorithm": algorithm}
        if set(by_rank) != set(range(4)):
            self.error(
                "reference_dispatch_rank_set_mismatch",
                location,
                f"expected [0, 1, 2, 3], got {sorted(by_rank)!r}",
            )
        if len({item["backend"] for item in by_rank.values()}) > 1:
            self.error(
                "reference_dispatch_backend_differs_by_rank",
                location,
                repr(by_rank),
            )
        if len({item["algorithm"] for item in by_rank.values()}) > 1:
            self.error(
                "reference_dispatch_algorithm_differs_by_rank",
                location,
                repr(by_rank),
            )
        return by_rank

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
        if result.get("schema_version") != 2:
            self.error(
                "result_schema_mismatch",
                location,
                f"expected 2, got {result.get('schema_version')!r}",
            )
        if result.get("reference_policy") != (
            "SGLANG_GLM52_OPT=0 production GroupCoordinator path"
        ):
            self.error(
                "reference_policy_mismatch",
                location,
                repr(result.get("reference_policy")),
            )
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
        else:
            expected_correctness = {
                "exact_full_checks": "before and after timing",
                "exact_full_tensor_consumer": "every warmup and measured sample",
                "exact_position_probe": "every warmup and measured sample",
                "alternating_input_variants": 2,
                "validation_errors_aggregated_over_tp_cpu_group": True,
            }
            for key, expected in expected_correctness.items():
                if correctness.get(key) != expected:
                    self.error(
                        "correctness_contract_mismatch",
                        location,
                        f"{key}: expected {expected!r}, got {correctness.get(key)!r}",
                    )
            candidate_present = isinstance(result.get("candidate"), dict)
            if correctness.get("candidate_state_guarded") is not candidate_present:
                self.error(
                    "candidate_guard_contract_mismatch",
                    location,
                    (
                        f"candidate_present={candidate_present}, guarded="
                        f"{correctness.get('candidate_state_guarded')!r}"
                    ),
                )
        gate = result.get("gate")
        if not isinstance(gate, dict) or gate.get("stock_fallback_active") is not True:
            self.error("stock_fallback_not_active", location, repr(gate))
        elif gate.get("candidate_present") is not isinstance(result.get("candidate"), dict):
            self.error(
                "gate_candidate_presence_mismatch",
                location,
                repr(gate.get("candidate_present")),
            )
        allreduce = result.get("allreduce")
        if not isinstance(allreduce, dict) or allreduce.get("message_bytes") != case.message_bytes:
            self.error(
                "message_bytes_mismatch",
                location,
                f"expected {case.message_bytes}, got {None if not isinstance(allreduce, dict) else allreduce.get('message_bytes')!r}",
            )
        elif isinstance(allreduce, dict):
            self.validate_reference_contracts(allreduce, case, location)
            self.validate_dispatch_records(
                allreduce,
                case,
                location,
                expected_mode=expected_mode,
                expected_stream=expected_stream,
            )
            reference_contracts = allreduce.get("reference_contract_by_rank")
            candidate_contracts = allreduce.get("candidate_contract_by_rank")
            candidate_present = isinstance(result.get("candidate"), dict)
            if candidate_present and candidate_contracts != reference_contracts:
                self.error(
                    "candidate_contract_differs_from_reference",
                    location,
                    f"reference={reference_contracts!r}, candidate={candidate_contracts!r}",
                )
            if not candidate_present and candidate_contracts is not None:
                self.error(
                    "unexpected_candidate_contract",
                    location,
                    repr(candidate_contracts),
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
            if not isinstance(execution.get("cuda_stream"), int):
                self.error(
                    "invalid_execution_stream_handle",
                    location,
                    repr(execution.get("cuda_stream")),
                )
        timing = result.get("timing_contract")
        expected_timing = {
            "gate_metric": "ready_region rank-max paired p50",
            "collective_only": "start event through event recorded immediately after run()",
            "ready_region": (
                "start event through full-tensor same-stream negation and "
                "position-sensitive gather"
            ),
            "restoration": (
                "variant source-to-local copy and consumer/probe poison occur "
                "before every start event"
            ),
            "rank_start_alignment": (
                "selected stream synchronized after restoration; TP CPU-group "
                "all-gather selects a common same-host monotonic deadline 5 ms "
                "after latest arrival; ranks busy-wait; host timestamps bracket "
                "every start-event record call; a host-only 500 us envelope "
                "predicate allows at most 10 physical attempts for the whole "
                "logical A/B pair; the exact input variant alternates on every "
                "physical pair"
            ),
            "alignment_retry_admission": (
                "scheduled arrivals, common targets, and start-record brackets "
                "only; collective and ready-region latency are excluded"
            ),
        }
        if not isinstance(timing, dict):
            self.error("missing_timing_contract", location, repr(timing))
        else:
            if set(timing) != set(expected_timing):
                self.error(
                    "timing_contract_mismatch",
                    location,
                    "expected exactly "
                    f"{sorted(expected_timing)!r}, got {sorted(timing)!r}",
                )
            for key, expected in expected_timing.items():
                if timing.get(key) != expected:
                    self.error(
                        "timing_contract_mismatch",
                        location,
                        f"{key}: expected {expected!r}, got {timing.get(key)!r}",
                    )
        readiness = result.get("readiness_probe")
        if not isinstance(readiness, dict):
            self.error("missing_readiness_probe", location, repr(readiness))
        else:
            positions = readiness.get("positions")
            if (
                not isinstance(positions, list)
                or not positions
                or readiness.get("num_elements") != len(positions)
                or any(
                    not isinstance(position, list)
                    or len(position) != 2
                    or any(not isinstance(value, int) for value in position)
                    for position in positions
                )
            ):
                self.error("invalid_readiness_probe_positions", location, repr(readiness))
            expected_readiness = {
                "dtype": "torch.bfloat16",
                "validation": "exact on every warmup and measured sample",
                "full_tensor_consumer": "preallocated BF16 negation output",
                "full_tensor_validation": "exact on every warmup and measured sample",
            }
            for key, expected in expected_readiness.items():
                if readiness.get(key) != expected:
                    self.error(
                        "readiness_probe_contract_mismatch",
                        location,
                        f"{key}: expected {expected!r}, got {readiness.get(key)!r}",
                    )
        trace = result.get("all_reduce_trace")
        if expected_trace is not None:
            if not isinstance(trace, dict) or trace.get("enabled") is not expected_trace:
                self.error(
                    "trace_enablement_mismatch",
                    location,
                    f"expected enabled={expected_trace}, got {trace!r}",
                )
            if isinstance(gate, dict) and gate.get("performance_eligible") is not (
                not expected_trace
            ):
                self.error(
                    "trace_performance_eligibility_mismatch",
                    location,
                    repr(gate.get("performance_eligible")),
                )
        environment = result.get("environment")
        if isinstance(environment, dict):
            env = environment.get("env")
            if not isinstance(env, dict):
                self.error("missing_environment_variables", location, repr(env))
            else:
                expected_env = {
                    "CUDA_VISIBLE_DEVICES": "0,1,2,3",
                    "SGLANG_GLM52_OPT": "0",
                    "SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2": None,
                    "NCCL_ALGO": None,
                    "NCCL_PROTO": None,
                }
                for key, expected in expected_env.items():
                    if env.get(key) != expected:
                        self.error(
                            "environment_variable_mismatch",
                            location,
                            f"{key}: expected {expected!r}, got {env.get(key)!r}",
                        )
            device_rows = environment.get("device_by_rank")
            if not isinstance(device_rows, list) or len(device_rows) != 4:
                self.error("invalid_device_rank_records", location, repr(device_rows))
            else:
                devices: dict[int, dict[str, Any]] = {}
                for row in device_rows:
                    if isinstance(row, dict) and isinstance(row.get("rank"), int):
                        devices[row["rank"]] = row
                    else:
                        self.error("invalid_device_rank_record", location, repr(row))
                if set(devices) != set(range(4)):
                    self.error(
                        "device_rank_set_mismatch",
                        location,
                        repr(sorted(devices)),
                    )
                for rank, row in devices.items():
                    if "B200" not in str(row.get("name")) or row.get("capability") != [10, 0]:
                        self.error(
                            "device_identity_mismatch",
                            location,
                            f"rank {rank}: {row!r}",
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
        allreduce = result.get("allreduce")
        expected_dispatch: dict[int, dict[str, str]] = {}
        reference_alias: dict[int, bool] = {}
        if isinstance(allreduce, dict):
            dispatch_rows = allreduce.get("reference_dispatch_by_rank")
            if isinstance(dispatch_rows, list):
                for row in dispatch_rows:
                    if isinstance(row, dict) and isinstance(row.get("rank"), int):
                        expected_dispatch[row["rank"]] = {
                            "backend": row.get("predicted_reference_backend"),
                            "algorithm": row.get("predicted_reference_algorithm"),
                        }
            contract_rows = allreduce.get("reference_contract_by_rank")
            if isinstance(contract_rows, list):
                for rank, contract in enumerate(contract_rows):
                    if isinstance(contract, dict) and isinstance(
                        contract.get("output_aliases_local"), bool
                    ):
                        reference_alias[rank] = contract["output_aliases_local"]
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

        all_rank_backends: dict[int, set[str]] = {}
        all_rank_algorithms: dict[int, set[str]] = {}
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
            algorithms: set[str] = set()
            backend_sources: set[str] = set()
            alias_observations: list[dict[str, Any]] = []
            callers: set[str] = set()
            for record_index, record in enumerate(records):
                record_location = f"{path}:record[{record_index}]"
                if record.get("schema_version") != 1:
                    self.error(
                        "trace_schema_mismatch",
                        record_location,
                        repr(record.get("schema_version")),
                    )
                if record.get("record_type") != "sglang_all_reduce_reachability":
                    self.error(
                        "trace_record_type_mismatch",
                        record_location,
                        repr(record.get("record_type")),
                    )
                process = record.get("process")
                if (
                    not isinstance(process, dict)
                    or not isinstance(process.get("pid"), int)
                    or process["pid"] <= 0
                ):
                    self.error("invalid_trace_process", record_location, repr(process))
                signature = record.get("signature")
                if not isinstance(signature, dict):
                    self.error("missing_trace_signature", record_location, repr(signature))
                else:
                    signature_id = signature.get("id")
                    if (
                        not isinstance(signature_id, str)
                        or len(signature_id) != 20
                        or any(character not in "0123456789abcdef" for character in signature_id)
                    ):
                        self.error(
                            "invalid_trace_signature",
                            record_location,
                            repr(signature_id),
                        )
                    if signature.get("dedup_scope") != "process":
                        self.error(
                            "trace_signature_scope_mismatch",
                            record_location,
                            repr(signature.get("dedup_scope")),
                        )
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
                caller = record.get("caller")
                if not isinstance(caller, dict):
                    self.error("missing_trace_caller", record_location, repr(caller))
                else:
                    caller_tag = caller.get("tag")
                    key_chain = caller.get("key_chain")
                    stack = caller.get("stack")
                    if (
                        not isinstance(caller_tag, str)
                        or not caller_tag.startswith(
                            "serving_native.runner:Runtime.reference:"
                        )
                    ):
                        self.error(
                            "trace_caller_mismatch",
                            record_location,
                            repr(caller_tag),
                        )
                    else:
                        callers.add(caller_tag)
                    if (
                        not isinstance(key_chain, list)
                        or not key_chain
                        or key_chain[0] != caller_tag
                        or any(not isinstance(item, str) or not item for item in key_chain)
                    ):
                        self.error(
                            "invalid_trace_caller_chain",
                            record_location,
                            repr(key_chain),
                        )
                    if not isinstance(stack, list) or not stack:
                        self.error("missing_trace_caller_stack", record_location, repr(stack))
                    elif not any(
                        isinstance(frame, dict)
                        and frame.get("module") == "serving_native.runner"
                        and frame.get("function") == "Runtime.reference"
                        for frame in stack
                    ):
                        self.error(
                            "trace_caller_stack_mismatch",
                            record_location,
                            repr(stack),
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
                        "device": f"cuda:{rank}",
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
                    if not isinstance(tensor.get("data_ptr"), int) or tensor["data_ptr"] <= 0:
                        self.error(
                            "invalid_trace_tensor_pointer",
                            record_location,
                            f"{tensor_key}: {tensor.get('data_ptr')!r}",
                        )
                graph = record.get("graph")
                if not isinstance(graph, dict):
                    self.error("missing_trace_graph", record_location, repr(graph))
                    capture_observations.append(None)
                else:
                    capture_state = graph.get("cuda_stream_capturing")
                    capture_observations.append(capture_state)
                    if not isinstance(capture_state, bool):
                        self.error(
                            "invalid_trace_capture_state",
                            record_location,
                            repr(capture_state),
                        )
                    if "probe_error" in graph:
                        self.error("trace_graph_probe_error", record_location, repr(graph))
                    if graph.get("cuda_graph_replay_observed") is not False:
                        self.error(
                            "trace_replay_claim_mismatch",
                            record_location,
                            repr(graph.get("cuda_graph_replay_observed")),
                        )
                    if graph.get("python_hook_observation") != "eager_or_capture_dispatch_only":
                        self.error(
                            "trace_python_observation_mismatch",
                            record_location,
                            repr(graph.get("python_hook_observation")),
                        )
                stream = record.get("stream")
                if not isinstance(stream, dict) or stream.get("present") is not True:
                    self.error("missing_trace_stream", record_location, repr(stream))
                elif "probe_error" in stream:
                    self.error("trace_stream_probe_error", record_location, repr(stream))
                elif (
                    stream.get("device") != f"cuda:{rank}"
                    or not isinstance(stream.get("cuda_stream"), int)
                    or stream["cuda_stream"] <= 0
                    or not isinstance(stream.get("priority"), int)
                ):
                    self.error("invalid_trace_stream", record_location, repr(stream))
                alias = record.get("alias")
                if not isinstance(alias, dict):
                    self.error("missing_trace_alias", record_location, repr(alias))
                else:
                    input_record = record.get("input")
                    output_record = record.get("output")
                    input_pointer = (
                        input_record.get("data_ptr")
                        if isinstance(input_record, dict)
                        else None
                    )
                    output_pointer = (
                        output_record.get("data_ptr")
                        if isinstance(output_record, dict)
                        else None
                    )
                    expected_same = (
                        input_pointer == output_pointer
                        if isinstance(input_pointer, int) and isinstance(output_pointer, int)
                        else None
                    )
                    if (
                        not isinstance(alias.get("same_data_ptr"), bool)
                        or alias.get("same_data_ptr") != expected_same
                        or alias.get("input_data_ptr") != input_pointer
                        or alias.get("output_data_ptr") != output_pointer
                        or not isinstance(alias.get("python_object_identity"), bool)
                        or (
                            alias.get("python_object_identity") is True
                            and alias.get("same_data_ptr") is not True
                        )
                    ):
                        self.error("invalid_trace_alias", record_location, repr(alias))
                    alias_observations.append(
                        {
                            "capturing": (
                                graph.get("cuda_stream_capturing")
                                if isinstance(graph, dict)
                                else None
                            ),
                            "same_data_ptr": alias.get("same_data_ptr"),
                            "python_object_identity": alias.get(
                                "python_object_identity"
                            ),
                        }
                    )
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
                    predicates = selection.get("predicates")
                    if not isinstance(predicates, dict):
                        self.error(
                            "missing_trace_predicates",
                            record_location,
                            repr(predicates),
                        )
                        predicates = {}
                    elif set(predicates) != TRACE_PREDICATE_KEYS:
                        self.error(
                            "trace_predicate_key_mismatch",
                            record_location,
                            (
                                f"missing={sorted(TRACE_PREDICATE_KEYS - set(predicates))!r}, "
                                f"extra={sorted(set(predicates) - TRACE_PREDICATE_KEYS)!r}"
                            ),
                        )
                    communicators = selection.get("communicators")
                    expected_communicator_keys = {
                        "device_group",
                        "hpu",
                        "xpu",
                        "npu",
                        "pynccl",
                        "custom",
                        "quick",
                        "pymscclpp",
                        "torch_symm_mem",
                    }
                    if (
                        not isinstance(communicators, dict)
                        or set(communicators) != expected_communicator_keys
                    ):
                        self.error(
                            "trace_communicator_map_mismatch",
                            record_location,
                            repr(communicators),
                        )
                    backend = selection.get("selected_backend")
                    if backend not in TRACE_GPU_BACKENDS:
                        self.error("missing_selected_backend", record_location, repr(backend))
                    else:
                        backends.add(backend)
                    predicted_backend = selection.get("predicted_backend")
                    derived_backend = selected_backend_from_predicates(predicates)
                    if predicted_backend != derived_backend:
                        self.error(
                            "trace_predicted_backend_mismatch",
                            record_location,
                            f"recorded={predicted_backend!r}, derived={derived_backend!r}",
                        )
                    backend_source = selection.get("selected_backend_source")
                    if (
                        backend_source
                        not in {
                            "observed_dispatch_branch",
                            "pre_dispatch_mirror_of_inplace_selector",
                        }
                        or selection.get("prediction_matches_selection") is not True
                        or predicted_backend != backend
                    ):
                        self.error(
                            "trace_backend_observation_mismatch",
                            record_location,
                            repr(selection),
                        )
                    elif isinstance(backend_source, str):
                        backend_sources.add(backend_source)
                    communicator = selection.get("selected_communicator")
                    algorithm = selection.get("selected_algorithm")
                    algorithm_source = selection.get("selected_algorithm_source")
                    if not isinstance(communicator, str) or not communicator:
                        self.error(
                            "missing_selected_communicator",
                            record_location,
                            repr(communicator),
                        )
                    if not isinstance(algorithm, str) or not algorithm:
                        self.error(
                            "missing_selected_algorithm",
                            record_location,
                            repr(algorithm),
                        )
                    else:
                        algorithms.add(algorithm)
                    if algorithm_source not in {
                        "backend_contract_constant",
                        "custom_private_selector_mirror_pre_dispatch",
                    }:
                        self.error(
                            "invalid_selected_algorithm_source",
                            record_location,
                            repr(algorithm_source),
                        )
                    if backend == "custom_all_reduce_outplace" and (
                        algorithm != selection.get("custom_algorithm")
                        or algorithm_source
                        != "custom_private_selector_mirror_pre_dispatch"
                        or selection.get("custom_algorithm_source")
                        != "custom_private_selector_mirror_pre_dispatch"
                    ):
                        self.error(
                            "custom_algorithm_evidence_mismatch",
                            record_location,
                            repr(selection),
                        )
                    record_capture = (
                        graph.get("cuda_stream_capturing")
                        if isinstance(graph, dict)
                        else None
                    )
                    dispatch = expected_dispatch.get(rank)
                    if (
                        dispatch is not None
                        and trace_record_matches_selected_execution(
                            case, record_capture
                        )
                        and (
                        backend != dispatch.get("backend")
                        or algorithm != dispatch.get("algorithm")
                        )
                    ):
                        self.error(
                            "trace_dispatch_prediction_mismatch",
                            record_location,
                            f"trace=({backend!r}, {algorithm!r}), dispatch={dispatch!r}",
                        )
                graph_capture = (
                    graph.get("cuda_stream_capturing")
                    if isinstance(graph, dict)
                    else None
                )
                if (
                    rank in reference_alias
                    and trace_record_matches_selected_execution(case, graph_capture)
                    and isinstance(alias, dict)
                    and alias.get("same_data_ptr") != reference_alias[rank]
                ):
                    self.error(
                        "trace_reference_alias_mismatch",
                        record_location,
                        (
                            f"trace={alias.get('same_data_ptr')!r}, "
                            f"reference_contract={reference_alias[rank]!r}"
                        ),
                    )
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
            if not backends:
                self.error("no_trace_backends", str(path), "no selected backend")
            if not algorithms:
                self.error("no_trace_algorithms", str(path), "no selected algorithm")
            all_rank_backends[rank] = backends
            all_rank_algorithms[rank] = algorithms
            summary["rank_files"].append(
                {
                    "rank": rank,
                    "path": str(path),
                    "records": len(records),
                    "cuda_stream_capturing_observations": capture_observations,
                    "selected_backends": sorted(backends),
                    "selected_algorithms": sorted(algorithms),
                    "selected_backend_sources": sorted(backend_sources),
                    "callers": sorted(callers),
                    "alias_observations": alias_observations,
                    "sha256": file_sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
        backend_sets = {tuple(sorted(value)) for value in all_rank_backends.values()}
        algorithm_sets = {tuple(sorted(value)) for value in all_rank_algorithms.values()}
        if len(backend_sets) != 1:
            self.error(
                "trace_backend_differs_by_rank",
                relative,
                repr({rank: sorted(value) for rank, value in all_rank_backends.items()}),
            )
        if len(algorithm_sets) != 1:
            self.error(
                "trace_algorithm_differs_by_rank",
                relative,
                repr({rank: sorted(value) for rank, value in all_rank_algorithms.items()}),
            )
        summary["trace_evidence_valid"] = trace.get("evidence_valid")
        return summary

    @staticmethod
    def _attempt_sample_audit(
        sample: Any,
        *,
        side: str,
        logical_pair_index: int,
        position: int,
        attempt_id: int,
        retry_ordinal: int,
        input_variant: int,
    ) -> dict[str, Any]:
        location = (
            f"raw_samples.measurement_attempts[{attempt_id}].samples.{side}"
        )
        if not isinstance(sample, dict):
            raise ValueError(f"{location} is not an object")
        expected_sample_keys = {
            "sample_index",
            "position",
            "variant",
            "pair_attempt_id",
            "pair_retry_ordinal",
            "scheduled_start_arrival_ns_by_rank",
            "scheduled_start_target_ns_by_rank",
            "start_record_bracket_ns_by_rank",
            "start_record_envelope_span_ns",
            "readiness_probe_exact",
            "collective_only",
            "ready_region",
        }
        if set(sample) != expected_sample_keys:
            raise ValueError(f"{location} schema mismatch: {sample!r}")
        expected_scalars = {
            "sample_index": logical_pair_index,
            "position": position,
            "variant": input_variant,
            "pair_attempt_id": attempt_id,
            "pair_retry_ordinal": retry_ordinal,
        }
        for key, expected in expected_scalars.items():
            observed = sample.get(key)
            if (
                not isinstance(observed, int)
                or isinstance(observed, bool)
                or observed != expected
            ):
                raise ValueError(
                    f"{location}.{key}: expected {expected!r}, got {observed!r}"
                )
        if sample.get("readiness_probe_exact") is not True:
            raise ValueError(f"{location}.readiness_probe_exact is not true")

        scheduled_arrivals = sample.get("scheduled_start_arrival_ns_by_rank")
        scheduled_targets = sample.get("scheduled_start_target_ns_by_rank")
        start_brackets = sample.get("start_record_bracket_ns_by_rank")
        start_envelope_span = sample.get("start_record_envelope_span_ns")
        if (
            not isinstance(scheduled_arrivals, list)
            or len(scheduled_arrivals) != 4
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
                for value in scheduled_arrivals
            )
            or not isinstance(scheduled_targets, list)
            or len(scheduled_targets) != 4
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
                for value in scheduled_targets
            )
            or len(set(scheduled_targets)) != 1
            or not isinstance(start_brackets, list)
            or len(start_brackets) != 4
            or any(
                not isinstance(bracket, list)
                or len(bracket) != 2
                or any(
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value <= 0
                    for value in bracket
                )
                or bracket[1] < bracket[0]
                for bracket in start_brackets
            )
            or not isinstance(start_envelope_span, int)
            or isinstance(start_envelope_span, bool)
            or start_envelope_span < 0
        ):
            raise ValueError(
                f"{location} has invalid scheduled-start targets or "
                "start-record brackets"
            )
        derived_target = max(scheduled_arrivals) + SCHEDULED_START_LEAD_NS
        if scheduled_targets[0] != derived_target:
            raise ValueError(
                f"{location} scheduled target {scheduled_targets[0]} != max "
                f"arrival + {SCHEDULED_START_LEAD_NS} ({derived_target})"
            )
        if any(
            bracket[0] < target
            for bracket, target in zip(start_brackets, scheduled_targets)
        ):
            raise ValueError(f"{location} records before its scheduled target")
        derived_start_envelope_span = max(
            bracket[1] for bracket in start_brackets
        ) - min(bracket[0] for bracket in start_brackets)
        if start_envelope_span != derived_start_envelope_span:
            raise ValueError(
                f"{location}.start_record_envelope_span_ns "
                f"{start_envelope_span} != {derived_start_envelope_span}"
            )

        parsed_metrics: dict[str, tuple[float, list[float]]] = {}
        for metric_name in ("collective_only", "ready_region"):
            metric_record = sample.get(metric_name)
            if not isinstance(metric_record, dict) or set(metric_record) != {
                "local_ms",
                "rank_ms",
                "rank_max_ms",
            }:
                raise ValueError(f"{location}.{metric_name} schema mismatch")
            rank_values = metric_record.get("rank_ms")
            if (
                not isinstance(rank_values, list)
                or len(rank_values) != 4
                or any(
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    for value in rank_values
                )
                or not isinstance(metric_record.get("local_ms"), (int, float))
                or isinstance(metric_record.get("local_ms"), bool)
                or not isinstance(metric_record.get("rank_max_ms"), (int, float))
                or isinstance(metric_record.get("rank_max_ms"), bool)
            ):
                raise ValueError(
                    f"{location}.{metric_name} values must be numeric with four ranks"
                )
            try:
                rank_ms = [float(value) for value in rank_values]
                local_ms = float(metric_record["local_ms"])
                rank_max_ms = float(metric_record["rank_max_ms"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{location}.{metric_name}: {exc}") from exc
            if any(not math.isfinite(value) or value <= 0 for value in rank_ms):
                raise ValueError(
                    f"{location}.{metric_name}.rank_ms must be finite and > 0"
                )
            if not math.isfinite(local_ms) or local_ms <= 0:
                raise ValueError(
                    f"{location}.{metric_name}.local_ms must be finite and > 0"
                )
            if not math.isclose(
                local_ms, rank_ms[0], rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(f"{location}.{metric_name}.local_ms != rank_ms[0]")
            if not math.isclose(
                rank_max_ms,
                max(rank_ms),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"{location}.{metric_name}.rank_max_ms != max(rank_ms)"
                )
            parsed_metrics[metric_name] = (rank_max_ms, rank_ms)
        collective_rank = parsed_metrics["collective_only"][1]
        ready_rank = parsed_metrics["ready_region"][1]
        if any(
            ready + 1e-9 < collective
            for collective, ready in zip(collective_rank, ready_rank)
        ):
            raise ValueError(f"{location} ready_region precedes collective_only")
        return {
            "start_record_envelope_span_ns": derived_start_envelope_span,
            "collective_only": parsed_metrics["collective_only"][0],
            "ready_region": parsed_metrics["ready_region"][0],
        }

    @classmethod
    def _measurement_attempt_audit(cls, result: dict[str, Any]) -> dict[str, Any]:
        raw = result.get("raw_samples")
        if not isinstance(raw, dict):
            raise ValueError("raw_samples is absent")
        expected_raw_keys = {
            "rank_order",
            "warmup_order",
            "warmup_variants",
            "measured_order",
            "reference",
            "candidate",
            "alignment_policy",
            "measurement_attempts",
        }
        if set(raw) != expected_raw_keys:
            raise ValueError(f"raw_samples schema mismatch: {raw!r}")
        if raw.get("rank_order") != [0, 1, 2, 3]:
            raise ValueError(
                "raw_samples.rank_order must be [0, 1, 2, 3], got "
                f"{raw.get('rank_order')!r}"
            )
        reference_projection = raw.get("reference")
        candidate_projection = raw.get("candidate")
        if not isinstance(reference_projection, list):
            raise ValueError("raw_samples.reference is absent")
        if candidate_projection is not None and not isinstance(
            candidate_projection, list
        ):
            raise ValueError("raw_samples.candidate must be a list or null")
        paired = isinstance(candidate_projection, list)
        sides = ("reference", "candidate") if paired else ("reference",)

        warmup_order = raw.get("warmup_order")
        warmup_variants = raw.get("warmup_variants")
        if not isinstance(warmup_order, list) or not isinstance(
            warmup_variants, list
        ):
            raise ValueError(
                "raw_samples warmup_order and warmup_variants must be lists"
            )
        expected_warmup_order = [
            (
                ["reference", "candidate"]
                if paired and index % 2 == 0
                else ["candidate", "reference"]
                if paired
                else ["reference"]
            )
            for index in range(len(warmup_order))
        ]
        expected_warmup_variants = [
            (1 + index) % 2 for index in range(len(warmup_order))
        ]
        if warmup_order != expected_warmup_order:
            raise ValueError(
                "raw_samples.warmup_order does not preserve the fixed A/B order"
            )
        if (
            any(
                not isinstance(variant, int) or isinstance(variant, bool)
                for variant in warmup_variants
            )
            or warmup_variants != expected_warmup_variants
        ):
            raise ValueError(
                "raw_samples.warmup_variants must alternate from physical variant 1"
            )

        policy = raw.get("alignment_policy")
        expected_policy_keys = {
            "start_record_envelope_limit_ns",
            "max_pair_attempts",
            "requested_logical_pairs",
            "accepted_pair_attempts",
            "rejected_pair_attempts",
            "total_pair_attempts",
            "initial_input_variant",
            "next_input_variant",
            "admission_fields",
            "latency_fields_excluded",
        }
        if not isinstance(policy, dict) or set(policy) != expected_policy_keys:
            raise ValueError(
                "raw_samples.alignment_policy schema mismatch: "
                f"{policy!r}"
            )
        if (
            policy.get("start_record_envelope_limit_ns")
            != START_RECORD_ENVELOPE_LIMIT_NS
            or policy.get("max_pair_attempts") != MAX_PAIR_ATTEMPTS
            or policy.get("admission_fields") != list(ALIGNMENT_ADMISSION_FIELDS)
            or policy.get("latency_fields_excluded") is not True
        ):
            raise ValueError(
                "raw_samples.alignment_policy contract mismatch: "
                f"{policy!r}"
            )
        initial_input_variant = policy.get("initial_input_variant")
        next_input_variant = policy.get("next_input_variant")
        expected_initial_input_variant = (1 + len(warmup_order)) % 2
        if (
            not isinstance(initial_input_variant, int)
            or isinstance(initial_input_variant, bool)
            or initial_input_variant != expected_initial_input_variant
        ):
            raise ValueError(
                "raw_samples.alignment_policy.initial_input_variant: expected "
                f"{expected_initial_input_variant}, got {initial_input_variant!r}"
            )
        if (
            not isinstance(next_input_variant, int)
            or isinstance(next_input_variant, bool)
            or next_input_variant not in (0, 1)
        ):
            raise ValueError(
                "raw_samples.alignment_policy.next_input_variant must be 0 or 1"
            )
        count_keys = (
            "requested_logical_pairs",
            "accepted_pair_attempts",
            "rejected_pair_attempts",
            "total_pair_attempts",
        )
        if any(
            not isinstance(policy.get(key), int)
            or isinstance(policy.get(key), bool)
            or policy[key] < 0
            for key in count_keys
        ):
            raise ValueError(
                "raw_samples.alignment_policy counts must be non-negative integers"
            )
        requested = policy["requested_logical_pairs"]
        if requested <= 0:
            raise ValueError(
                "raw_samples.alignment_policy.requested_logical_pairs must be > 0"
            )

        attempts = raw.get("measurement_attempts")
        if not isinstance(attempts, list):
            raise ValueError("raw_samples.measurement_attempts is absent")
        accepted_attempts: list[dict[str, Any]] = []
        rejected_attempts = 0
        expected_logical_pair_index = 0
        expected_retry_ordinal = 0
        values = {
            side: {"collective_only": [], "ready_region": []} for side in sides
        }
        expected_attempt_keys = {
            "attempt_id",
            "logical_pair_index",
            "retry_ordinal",
            "order",
            "variant",
            "accepted",
            "rejection_reasons",
            "samples",
        }
        for attempt_id, attempt in enumerate(attempts):
            location = f"raw_samples.measurement_attempts[{attempt_id}]"
            if not isinstance(attempt, dict) or set(attempt) != expected_attempt_keys:
                raise ValueError(f"{location} schema mismatch: {attempt!r}")
            recorded_attempt_id = attempt.get("attempt_id")
            if (
                not isinstance(recorded_attempt_id, int)
                or isinstance(recorded_attempt_id, bool)
                or recorded_attempt_id != attempt_id
            ):
                raise ValueError(
                    f"{location}.attempt_id must be sequential, got "
                    f"{recorded_attempt_id!r}"
                )
            logical_pair_index = attempt.get("logical_pair_index")
            retry_ordinal = attempt.get("retry_ordinal")
            if (
                not isinstance(logical_pair_index, int)
                or isinstance(logical_pair_index, bool)
                or logical_pair_index != expected_logical_pair_index
            ):
                raise ValueError(
                    f"{location}.logical_pair_index: expected "
                    f"{expected_logical_pair_index}, got {logical_pair_index!r}"
                )
            if (
                not isinstance(retry_ordinal, int)
                or isinstance(retry_ordinal, bool)
                or retry_ordinal != expected_retry_ordinal
            ):
                raise ValueError(
                    f"{location}.retry_ordinal: expected "
                    f"{expected_retry_ordinal}, got {retry_ordinal!r}"
                )
            if retry_ordinal >= MAX_PAIR_ATTEMPTS:
                raise ValueError(
                    f"{location}.retry_ordinal exceeds the bounded retry policy"
                )
            expected_order = (
                ["reference", "candidate"]
                if paired and logical_pair_index % 2 == 0
                else ["candidate", "reference"]
                if paired
                else ["reference"]
            )
            if attempt.get("order") != expected_order:
                raise ValueError(
                    f"{location}.order: expected {expected_order!r}, got "
                    f"{attempt.get('order')!r}"
                )
            variant = attempt.get("variant")
            expected_variant = (initial_input_variant + attempt_id) % 2
            if (
                not isinstance(variant, int)
                or isinstance(variant, bool)
                or variant != expected_variant
            ):
                raise ValueError(
                    f"{location}.variant: expected {expected_variant}, got "
                    f"{variant!r}"
                )
            accepted = attempt.get("accepted")
            reasons = attempt.get("rejection_reasons")
            samples = attempt.get("samples")
            if not isinstance(accepted, bool) or not isinstance(reasons, list):
                raise ValueError(f"{location} has invalid admission fields")
            if not isinstance(samples, dict) or set(samples) != set(sides):
                raise ValueError(
                    f"{location}.samples: expected sides {list(sides)!r}, got "
                    f"{samples!r}"
                )

            sample_audits: dict[str, dict[str, Any]] = {}
            for position, side in enumerate(expected_order):
                sample_audits[side] = cls._attempt_sample_audit(
                    samples[side],
                    side=side,
                    logical_pair_index=logical_pair_index,
                    position=position,
                    attempt_id=attempt_id,
                    retry_ordinal=retry_ordinal,
                    input_variant=variant,
                )
            expected_reasons = [
                {
                    "side": side,
                    "start_record_envelope_span_ns": sample_audits[side][
                        "start_record_envelope_span_ns"
                    ],
                    "limit_ns": START_RECORD_ENVELOPE_LIMIT_NS,
                }
                for side in expected_order
                if sample_audits[side]["start_record_envelope_span_ns"]
                > START_RECORD_ENVELOPE_LIMIT_NS
            ]
            if reasons != expected_reasons:
                raise ValueError(
                    f"{location}.rejection_reasons do not match independently "
                    f"derived envelopes: expected {expected_reasons!r}, got {reasons!r}"
                )
            if accepted:
                if expected_reasons:
                    raise ValueError(
                        f"{location} accepted an over-limit start-record envelope"
                    )
                accepted_attempts.append(attempt)
                for side in sides:
                    for metric_name in ("collective_only", "ready_region"):
                        values[side][metric_name].append(
                            sample_audits[side][metric_name]
                        )
                expected_logical_pair_index += 1
                expected_retry_ordinal = 0
            else:
                if not expected_reasons:
                    raise ValueError(
                        f"{location} was rejected without an over-limit envelope"
                    )
                rejected_attempts += 1
                expected_retry_ordinal += 1
                if expected_retry_ordinal >= MAX_PAIR_ATTEMPTS:
                    raise ValueError(
                        f"{location} exhausts the bounded retry policy without "
                        "an accepted logical pair"
                    )

        if expected_logical_pair_index != requested or expected_retry_ordinal != 0:
            raise ValueError(
                "raw_samples.measurement_attempts does not contain exactly one "
                f"accepted attempt for logical pairs 0..{requested - 1}"
            )
        accepted_count = len(accepted_attempts)
        total_count = len(attempts)
        expected_counts = {
            "accepted_pair_attempts": accepted_count,
            "rejected_pair_attempts": rejected_attempts,
            "total_pair_attempts": total_count,
        }
        for key, expected in expected_counts.items():
            if policy.get(key) != expected:
                raise ValueError(
                    f"raw_samples.alignment_policy.{key}: expected {expected}, "
                    f"got {policy.get(key)!r}"
                )
        if accepted_count != requested or total_count != accepted_count + rejected_attempts:
            raise ValueError("raw_samples.alignment_policy attempt counts are inconsistent")
        if total_count > requested * MAX_PAIR_ATTEMPTS:
            raise ValueError("raw_samples.measurement_attempts exceeds its retry bound")
        expected_next_input_variant = (initial_input_variant + total_count) % 2
        if next_input_variant != expected_next_input_variant:
            raise ValueError(
                "raw_samples.alignment_policy.next_input_variant: expected "
                f"{expected_next_input_variant}, got {next_input_variant!r}"
            )

        measured_order = raw.get("measured_order")
        expected_measured_order = [attempt["order"] for attempt in accepted_attempts]
        if measured_order != expected_measured_order:
            raise ValueError(
                "raw_samples.measured_order is not the exact accepted-attempt "
                "projection"
            )
        expected_reference = [
            attempt["samples"]["reference"] for attempt in accepted_attempts
        ]
        if reference_projection != expected_reference:
            raise ValueError(
                "raw_samples.reference is not the exact accepted-attempt projection"
            )
        if paired:
            expected_candidate = [
                attempt["samples"]["candidate"] for attempt in accepted_attempts
            ]
            if candidate_projection != expected_candidate:
                raise ValueError(
                    "raw_samples.candidate is not the exact accepted-attempt projection"
                )

        return {
            "requested_logical_pairs": requested,
            "accepted_pair_attempts": accepted_count,
            "rejected_pair_attempts": rejected_attempts,
            "total_pair_attempts": total_count,
            "total_sample_calls": sum(len(attempt["samples"]) for attempt in attempts),
            "values": values,
        }

    @classmethod
    def _sample_values(
        cls, result: dict[str, Any], side: str, metric: str
    ) -> list[float]:
        audit = cls._measurement_attempt_audit(result)
        try:
            return list(audit["values"][side][metric])
        except KeyError as exc:
            raise ValueError(f"raw_samples.{side}.{metric} is absent") from exc

    def validate_measurement_attempts(
        self, result: dict[str, Any], location: str
    ) -> dict[str, int]:
        try:
            audit = self._measurement_attempt_audit(result)
        except ValueError as exc:
            self.error("invalid_measurement_attempts", location, str(exc))
            return {}
        return {
            key: int(audit[key])
            for key in (
                "requested_logical_pairs",
                "accepted_pair_attempts",
                "rejected_pair_attempts",
                "total_pair_attempts",
                "total_sample_calls",
            )
        }

    def validate_sample_count(
        self,
        result: dict[str, Any],
        location: str,
        side: str,
        expected: int,
        metric: str = "ready_region",
    ) -> list[float]:
        try:
            values = self._sample_values(result, side, metric)
        except ValueError as exc:
            self.error("invalid_raw_samples", location, str(exc))
            return []
        if len(values) != expected:
            self.error(
                "sample_count_mismatch",
                location,
                f"raw_samples.{side}.{metric}: expected {expected}, got {len(values)}",
            )
        return values

    def validate_reported_median(
        self,
        result: dict[str, Any],
        location: str,
        side: str,
        values: list[float],
        metric: str = "ready_region",
    ) -> Optional[float]:
        if not values:
            return None
        try:
            reported = float(result[side][metric]["median_ms"])
        except (KeyError, TypeError, ValueError) as exc:
            self.error("missing_reported_median", location, f"{side}: {exc}")
            return None
        derived = statistics.median(values)
        if not math.isclose(reported, derived, rel_tol=1e-12, abs_tol=1e-12):
            self.error(
                "reported_median_mismatch",
                location,
                f"{side}.{metric}: reported={reported}, derived={derived}",
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
                attempt_summary = self.validate_measurement_attempts(result, relative)
                checks.append(
                    {
                        "task": case.task,
                        "mode": mode,
                        "stream": stream,
                        "samples": len(values),
                        "measurement_attempts": attempt_summary,
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
                attempt_summary = self.validate_measurement_attempts(result, relative)
                median = self.validate_reported_median(
                    result, relative, "reference", values
                )
                collective_values = self.validate_sample_count(
                    result,
                    relative,
                    "reference",
                    100,
                    metric="collective_only",
                )
                collective_median = self.validate_reported_median(
                    result,
                    relative,
                    "reference",
                    collective_values,
                    metric="collective_only",
                )
                overheads = [
                    ready - collective
                    for ready, collective in zip(values, collective_values)
                ]
                reported_overhead = result.get("reference", {}).get(
                    "readiness_probe_overhead_median_ms"
                )
                derived_overhead = statistics.median(overheads) if overheads else None
                if (
                    not isinstance(reported_overhead, (int, float))
                    or derived_overhead is None
                    or not math.isclose(
                        float(reported_overhead),
                        derived_overhead,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                ):
                    self.error(
                        "reported_readiness_overhead_mismatch",
                        relative,
                        f"reported={reported_overhead!r}, derived={derived_overhead!r}",
                    )
                if median is not None:
                    medians.append(median)
                run_summaries.append(
                    {
                        "run": run,
                        "ready_region_rank_max_median_ms": median,
                        "collective_only_rank_max_median_ms": collective_median,
                        "readiness_probe_overhead_median_ms": derived_overhead,
                        "samples": len(values),
                        "measurement_attempts": attempt_summary,
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
            if len(medians) == 3:
                relative_spread = baseline_relative_run_spread(medians)
                output[case.short_name]["relative_run_spread"] = relative_spread
                output[case.short_name]["relative_run_spread_definition"] = (
                    "max(run_medians) / min(run_medians) - 1"
                )
                output[case.short_name]["performance_evidence_eligible"] = (
                    relative_spread < 0.03
                )
                if relative_spread >= 0.03:
                    self.error(
                        "baseline_run_spread_exceeds_3pct",
                        f"baseline/{case.short_name}",
                        f"relative spread={relative_spread}",
                    )
        return output

    def validate_paired_result(
        self,
        result: dict[str, Any],
        case: Case,
        location: str,
        *,
        expected_entrypoint: str,
    ) -> dict[str, Any]:
        self.validate_common_result(
            result,
            case,
            location,
            expected_mode=case.execution_mode,
            expected_stream=case.stream,
            expected_trace=False,
        )
        attempt_summary = self.validate_measurement_attempts(result, location)
        reference_values = self.validate_sample_count(
            result, location, "reference", 100
        )
        candidate_values = self.validate_sample_count(
            result, location, "candidate", 100
        )
        reference_collective = self.validate_sample_count(
            result,
            location,
            "reference",
            100,
            metric="collective_only",
        )
        candidate_collective = self.validate_sample_count(
            result,
            location,
            "candidate",
            100,
            metric="collective_only",
        )
        reference_median = self.validate_reported_median(
            result, location, "reference", reference_values
        )
        candidate_median = self.validate_reported_median(
            result, location, "candidate", candidate_values
        )
        reference_collective_median = self.validate_reported_median(
            result,
            location,
            "reference",
            reference_collective,
            metric="collective_only",
        )
        candidate_collective_median = self.validate_reported_median(
            result,
            location,
            "candidate",
            candidate_collective,
            metric="collective_only",
        )
        candidate = result.get("candidate")
        if not isinstance(candidate, dict):
            self.error("missing_candidate_summary", location, repr(candidate))
            return {}
        manifest = candidate.get("manifest")
        if not isinstance(manifest, dict):
            self.error("missing_candidate_manifest", location, repr(manifest))
        else:
            entrypoint = Path(str(manifest.get("entrypoint", ""))).name
            if entrypoint != expected_entrypoint:
                self.error(
                    "candidate_entrypoint_mismatch",
                    location,
                    f"expected {expected_entrypoint!r}, got {entrypoint!r}",
                )
            harness_root = self.expected_roots.get("kernel_harness_git")
            if harness_root is not None:
                expected_path = str(
                    Path(harness_root)
                    / "serving_native"
                    / "candidates"
                    / expected_entrypoint
                )
                for key in ("entrypoint", "requested_path"):
                    if manifest.get(key) != expected_path:
                        self.error(
                            "candidate_manifest_path_mismatch",
                            location,
                            f"{key}: expected {expected_path!r}, got {manifest.get(key)!r}",
                        )
            for key in (
                "entrypoint_sha256_at_import",
                "manifest_sha256",
            ):
                digest = manifest.get(key)
                if (
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)
                ):
                    self.error(
                        "invalid_candidate_manifest_digest",
                        location,
                        f"{key}={digest!r}",
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
        if ratios:
            ordered_ratios = sorted(ratios)
            expected_p10 = ordered_ratios[
                min(len(ordered_ratios) - 1, int(0.1 * len(ordered_ratios)))
            ]
            expected_p90 = ordered_ratios[
                min(
                    len(ordered_ratios) - 1,
                    max(0, int(0.9 * len(ordered_ratios)) - 1),
                )
            ]
            for key, expected in (
                ("paired_p10_speedup", expected_p10),
                ("paired_p90_speedup", expected_p90),
            ):
                observed = candidate.get(key)
                if not isinstance(observed, (int, float)) or not math.isclose(
                    float(observed), expected, rel_tol=1e-12, abs_tol=1e-12
                ):
                    self.error(
                        "reported_speedup_quantile_mismatch",
                        location,
                        f"{key}: reported={observed!r}, derived={expected}",
                    )
        expected_gate = bool(derived_speedup is not None and derived_speedup >= 1.03)
        if candidate.get("passes_3pct_median_gate") is not expected_gate:
            self.error(
                "candidate_gate_summary_mismatch",
                location,
                repr(candidate.get("passes_3pct_median_gate")),
            )
        gate = result.get("gate")
        if not isinstance(gate, dict) or gate.get("passed") is not expected_gate:
            self.error(
                "candidate_gate_result_mismatch",
                location,
                repr(gate),
            )

        return {
            "reference_ready_region_rank_max_median_ms": reference_median,
            "candidate_ready_region_rank_max_median_ms": candidate_median,
            "reference_collective_only_rank_max_median_ms": reference_collective_median,
            "candidate_collective_only_rank_max_median_ms": candidate_collective_median,
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
            "measurement_attempts": attempt_summary,
            "gate_passed": result.get("gate", {}).get("passed"),
            "disposition": result.get("disposition"),
            "candidate_manifest_sha256": (
                manifest.get("manifest_sha256") if isinstance(manifest, dict) else None
            ),
            "candidate_entrypoint": (
                manifest.get("entrypoint") if isinstance(manifest, dict) else None
            ),
            "candidate_entrypoint_sha256": (
                manifest.get("entrypoint_sha256_at_import")
                if isinstance(manifest, dict)
                else None
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

    def validate_paired(
        self,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
        controls: dict[str, Any] = {}
        candidates: dict[str, Any] = {}
        selected_candidates: dict[str, dict[str, Any]] = {}
        failed_logs: list[dict[str, Any]] = []
        for case in CASES:
            control_relative = f"paired/{case.short_name}_reference_control.json"
            control = self.read_json(control_relative)
            expected_variant: Optional[str] = None
            if control is not None:
                controls[case.short_name] = self.validate_paired_result(
                    control,
                    case,
                    control_relative,
                    expected_entrypoint="reference.py",
                )
                control_speedup = controls[case.short_name].get(
                    "derived_paired_median_speedup"
                )
                control_eligible = bool(
                    isinstance(control_speedup, (int, float))
                    and abs(float(control_speedup) - 1.0) < 0.03
                )
                controls[case.short_name][
                    "performance_evidence_eligible"
                ] = control_eligible
                if not control_eligible:
                    self.error(
                        "reference_control_noise_exceeds_3pct",
                        control_relative,
                        f"paired median speedup={control_speedup!r}",
                    )
                contracts = control.get("allreduce", {}).get(
                    "reference_contract_by_rank"
                )
                first_contract = (
                    contracts[0]
                    if isinstance(contracts, list) and len(contracts) == 4
                    else None
                )
                if isinstance(first_contract, dict):
                    contract_key = (
                        first_contract.get("output_aliases_local"),
                        first_contract.get("local_poststate"),
                    )
                    expected_variant = {
                        (True, "reduced"): "inplace",
                        (False, "source"): "outplace",
                    }.get(contract_key)
                if expected_variant is None:
                    self.error(
                        "unsupported_reference_alias_contract",
                        control_relative,
                        repr(first_contract),
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
                    elif row["requirement"] != "attempt" or row["exit_code"] != 0:
                        self.error(
                            "persisted_attempt_failed_status",
                            relative,
                            (
                                f"requirement={row['requirement']!r}, "
                                f"status exit={row['exit_code']}"
                            ),
                        )
                    result = self.read_json(relative)
                    if result is not None:
                        summary = self.validate_paired_result(
                            result,
                            case,
                            relative,
                            expected_entrypoint=(
                                "allreduce_torch.py"
                                if variant == "inplace"
                                else "allreduce_torch_outplace.py"
                            ),
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
                        if row["requirement"] != "attempt":
                            self.error(
                                "misclassified_c10d_attempt",
                                f"status.tsv:{row['line_number']}",
                                f"requirement={row['requirement']!r}",
                            )
                        failure = self.failed_log_summary(row)
                        log_path = Path(str(failure.get("path", "")))
                        log_text = (
                            log_path.read_text(encoding="utf-8", errors="replace")
                            if log_path.is_file()
                            else ""
                        )
                        signature = (
                            "candidate destructive/alias ABI differs from reference:"
                        )
                        failure["abi_mismatch_signature_count"] = log_text.count(
                            signature
                        )
                        try:
                            canonical_failures = (
                                _canonical_abi_mismatch_failures(log_text)
                                if row["exit_code"] == 1
                                else []
                            )
                        except ValueError as exc:
                            canonical_failures = []
                            failure["classification_error"] = str(exc)
                        failure["canonical_failure_records"] = len(
                            canonical_failures
                        )
                        failure["classification"] = (
                            "expected_abi_contract_mismatch"
                            if canonical_failures
                            else "unexpected_attempt_failure"
                        )
                        if not canonical_failures:
                            self.error(
                                "unexpected_c10d_attempt_failure",
                                str(log_path),
                                (
                                    f"{step} exited {row['exit_code']}; "
                                    f"signature_count={log_text.count(signature)}; "
                                    f"classification_error={failure.get('classification_error')!r}"
                                ),
                            )
                        failed_logs.append(failure)
            resolution_step = f"paired/{case.short_name}_c10d_abi_resolution"
            resolution_row = self.status_rows.get(resolution_step)
            if (
                resolution_row is None
                or resolution_row["requirement"] != "required"
                or resolution_row["exit_code"] != 0
            ):
                self.error(
                    "c10d_abi_resolution_failed",
                    "status.tsv",
                    resolution_step,
                )
            receipt_relative = f"paired/{case.short_name}_c10d_selection.json"
            receipt = self.read_json(receipt_relative)
            harness_root = self.expected_roots.get("kernel_harness_git")
            if receipt is not None and harness_root is not None:
                try:
                    expected_receipt = resolve_c10d_abi(
                        self.root, case, Path(harness_root)
                    )
                except ValueError as exc:
                    self.error(
                        "c10d_abi_resolution_replay_failed",
                        receipt_relative,
                        str(exc),
                    )
                else:
                    expected_receipt["valid"] = True
                    expected_receipt["errors"] = []
                    if receipt != expected_receipt:
                        self.error(
                            "c10d_abi_resolution_receipt_mismatch",
                            receipt_relative,
                            "receipt differs from a fresh fail-closed replay",
                        )
            if len(persisted) != 1:
                self.error(
                    "c10d_abi_result_count_mismatch",
                    f"paired/{case.short_name}",
                    repr(persisted),
                )
            elif expected_variant is not None and persisted != [expected_variant]:
                self.error(
                    "c10d_abi_variant_mismatch",
                    f"paired/{case.short_name}",
                    f"expected={expected_variant!r}, persisted={persisted!r}",
                )
            if len(case_candidates) == 1:
                selected_candidates[case.short_name] = case_candidates[0]
            candidates[case.short_name] = {
                "expected_variant": expected_variant,
                "selection_receipt": file_summary(self.root / receipt_relative),
                "persisted_results": case_candidates,
                "failed_attempts": [
                    item
                    for item in failed_logs
                    if item["step"].startswith(f"paired/{case.short_name}_")
                ],
            }
        candidates["failed_attempt_logs"] = failed_logs
        return controls, candidates, selected_candidates

    def validate_profile_result(
        self,
        relative: str,
        case: Case,
        *,
        candidate: bool,
        candidate_expectation: Optional[dict[str, Any]] = None,
    ) -> dict[str, int]:
        result = self.read_json(relative)
        if result is None:
            return {}
        self.validate_common_result(
            result,
            case,
            relative,
            expected_mode=case.execution_mode,
            expected_stream=case.stream,
            expected_trace=False,
        )
        attempt_summary = self.validate_measurement_attempts(result, relative)
        reference_ready = self.validate_sample_count(
            result, relative, "reference", 20
        )
        reference_collective = self.validate_sample_count(
            result,
            relative,
            "reference",
            20,
            metric="collective_only",
        )
        self.validate_reported_median(
            result, relative, "reference", reference_ready
        )
        self.validate_reported_median(
            result,
            relative,
            "reference",
            reference_collective,
            metric="collective_only",
        )
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
            candidate_ready = self.validate_sample_count(
                result, relative, "candidate", 20
            )
            candidate_collective = self.validate_sample_count(
                result,
                relative,
                "candidate",
                20,
                metric="collective_only",
            )
            self.validate_reported_median(
                result, relative, "candidate", candidate_ready
            )
            self.validate_reported_median(
                result,
                relative,
                "candidate",
                candidate_collective,
                metric="collective_only",
            )
            candidate_summary = result.get("candidate")
            if not isinstance(candidate_summary, dict):
                self.error("missing_profile_candidate", relative, repr(result.get("candidate")))
            else:
                manifest = candidate_summary.get("manifest")
                if not isinstance(manifest, dict) or not isinstance(
                    candidate_expectation, dict
                ):
                    self.error(
                        "profile_candidate_manifest_missing",
                        relative,
                        repr(manifest),
                    )
                else:
                    expected_entrypoint = candidate_expectation.get(
                        "candidate_entrypoint"
                    )
                    expected_entrypoint_sha = candidate_expectation.get(
                        "candidate_entrypoint_sha256"
                    )
                    expected_manifest_sha = candidate_expectation.get(
                        "candidate_manifest_sha256"
                    )
                    for key, expected in (
                        ("entrypoint", expected_entrypoint),
                        ("requested_path", expected_entrypoint),
                        (
                            "entrypoint_sha256_at_import",
                            expected_entrypoint_sha,
                        ),
                        ("manifest_sha256", expected_manifest_sha),
                    ):
                        if manifest.get(key) != expected:
                            self.error(
                                "profile_candidate_manifest_mismatch",
                                relative,
                                f"{key}: expected {expected!r}, got {manifest.get(key)!r}",
                            )
                    entrypoint_path = Path(str(expected_entrypoint or ""))
                    if not entrypoint_path.is_file():
                        self.error(
                            "profile_candidate_source_missing",
                            relative,
                            str(entrypoint_path),
                        )
                    elif file_sha256(entrypoint_path) != expected_entrypoint_sha:
                        self.error(
                            "profile_candidate_source_sha_mismatch",
                            relative,
                            str(entrypoint_path),
                        )
        elif result.get("candidate") is not None:
            self.error("unexpected_profile_candidate", relative, "candidate is present")
        return attempt_summary

    def load_profile_selection(self) -> dict[str, dict[str, Any]]:
        selected: dict[str, dict[str, Any]] = {}
        for case in CASES:
            relative = f"paired/{case.short_name}_c10d_selection.json"
            receipt = self.read_json(relative)
            if receipt is None:
                continue
            record = receipt.get("selected")
            if not isinstance(record, dict):
                self.error("invalid_profile_selection", relative, repr(record))
                continue
            variant = record.get("variant")
            candidate_path = record.get("candidate_path")
            result_path = record.get("result_path")
            if variant not in {"inplace", "outplace"}:
                self.error("unknown_profile_candidate", relative, repr(record))
                continue
            expected_result = (
                self.root / f"paired/{case.short_name}_c10d_{variant}.json"
            ).resolve()
            if (
                not isinstance(result_path, str)
                or (self.root / result_path).resolve() != expected_result
            ):
                self.error(
                    "profile_selection_result_mismatch",
                    relative,
                    f"expected={expected_result}, got={result_path!r}",
                )
            harness_root = self.expected_roots.get("kernel_harness_git")
            expected_candidate = (
                (
                    Path(harness_root)
                    / "serving_native"
                    / "candidates"
                    / (
                        "allreduce_torch_outplace.py"
                        if variant == "outplace"
                        else "allreduce_torch.py"
                    )
                ).resolve()
                if harness_root is not None
                else None
            )
            if (
                expected_candidate is not None
                and (
                    not isinstance(candidate_path, str)
                    or Path(candidate_path).resolve() != expected_candidate
                )
            ):
                self.error(
                    "profile_selection_candidate_path_mismatch",
                    relative,
                    f"expected={expected_candidate}, got={candidate_path!r}",
                )
            selected[case.short_name] = dict(record)
        return selected

    def validate_stats_status(self, expected_names: set[str]) -> dict[str, Any]:
        path = self.root / "profile/stats_status.tsv"
        rows: dict[str, dict[str, Any]] = {}
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                if reader.fieldnames != ["report", "exit_code", "log"]:
                    self.error(
                        "invalid_stats_status_header",
                        "profile/stats_status.tsv",
                        repr(reader.fieldnames),
                    )
                    return rows
                for line_number, row in enumerate(reader, start=2):
                    name = row.get("report", "")
                    if not name or name in rows:
                        self.error(
                            "invalid_stats_status_report",
                            f"profile/stats_status.tsv:{line_number}",
                            repr(name),
                        )
                        continue
                    try:
                        exit_code = int(row.get("exit_code", ""))
                    except ValueError:
                        self.error(
                            "invalid_stats_status_exit",
                            f"profile/stats_status.tsv:{line_number}",
                            repr(row.get("exit_code")),
                        )
                        continue
                    expected_log = self.root / "profile" / f"{name}.postprocess.log"
                    if Path(str(row.get("log", ""))).name != expected_log.name:
                        self.error(
                            "stats_status_log_mismatch",
                            f"profile/stats_status.tsv:{line_number}",
                            repr(row.get("log")),
                        )
                    if exit_code != 0:
                        self.error(
                            "failed_nsys_stats",
                            f"profile/stats_status.tsv:{line_number}",
                            f"{name}: exit={exit_code}",
                        )
                    if not expected_log.is_file():
                        self.error(
                            "missing_nsys_postprocess_log",
                            f"profile/stats_status.tsv:{line_number}",
                            str(expected_log),
                        )
                    rows[name] = {
                        "exit_code": exit_code,
                        "postprocess_log": file_summary(expected_log),
                        "stats_log": file_summary(
                            self.root / "profile" / f"{name}.stats.log"
                        ),
                    }
        except FileNotFoundError:
            self.error(
                "missing_stats_status",
                "profile/stats_status.tsv",
                "postprocessing status is absent",
            )
        except OSError as exc:
            self.error(
                "stats_status_read_error",
                "profile/stats_status.tsv",
                f"{type(exc).__name__}: {exc}",
            )
        if set(rows) != expected_names:
            self.error(
                "stats_status_report_set_mismatch",
                "profile/stats_status.tsv",
                f"expected={sorted(expected_names)!r}, got={sorted(rows)!r}",
            )
        return rows

    def validate_profile_exports(
        self,
        name: str,
        expected_range: str,
        *,
        expected_collective_launches_per_device: int,
        expected_graph_launches_per_process: Optional[int],
    ) -> dict[str, Any]:
        profile = self.root / "profile"
        required = {
            "sqlite": profile / f"{name}.sqlite",
            "lifecycle_cuda_api": profile
            / f"{name}.lifecycle_cuda_api_trace.csv",
            "lifecycle_nvtx": profile
            / f"{name}.lifecycle_nvtx_pushpop_trace.csv",
            "measured_window": profile / f"{name}.measured_window.tsv",
            "measured_cuda_api": profile
            / f"{name}.measured_cuda_api_trace.csv",
            "measured_kernel_exec": profile
            / f"{name}.measured_cuda_kern_exec_trace.csv",
            "measured_gpu": profile / f"{name}.measured_cuda_gpu_trace.csv",
            "measured_nvtx": profile
            / f"{name}.measured_nvtx_pushpop_trace.csv",
        }
        for label, path in required.items():
            if not path.is_file() or path.stat().st_size == 0:
                self.error(
                    "missing_profile_export",
                    str(path.relative_to(self.root)),
                    label,
                )

        bounds: Optional[tuple[int, int]] = None
        lifecycle_nvtx_path = required["lifecycle_nvtx"]
        if lifecycle_nvtx_path.is_file():
            try:
                bounds = nvtx_bounds_from_csv(lifecycle_nvtx_path, expected_range)
            except ValueError as exc:
                self.error(
                    "profile_nvtx_bounds_mismatch",
                    str(lifecycle_nvtx_path.relative_to(self.root)),
                    str(exc),
                )
        window_path = required["measured_window"]
        if bounds is not None and window_path.is_file():
            expected_window = (
                "range\tstart_ns\tend_ns\n"
                f"{expected_range}\t{bounds[0]}\t{bounds[1]}\n"
            )
            try:
                observed_window = window_path.read_text(encoding="utf-8")
            except OSError as exc:
                self.error(
                    "profile_measured_window_read_error",
                    str(window_path.relative_to(self.root)),
                    f"{type(exc).__name__}: {exc}",
                )
            else:
                if observed_window != expected_window:
                    self.error(
                        "profile_measured_window_mismatch",
                        str(window_path.relative_to(self.root)),
                        f"expected={expected_window!r}, got={observed_window!r}",
                    )

        kernel_path = required["measured_kernel_exec"]
        kernel_rows: list[dict[str, str]] = []
        expected_kernel_header = [
            "API Start (ns)",
            "API Dur (ns)",
            "Queue Start (ns)",
            "Queue Dur (ns)",
            "Kernel Start (ns)",
            "Kernel Dur (ns)",
            "Total Dur (ns)",
            "PID",
            "TID",
            "DevId",
            "API Function",
            "GridXYZ",
            "BlockXYZ",
            "Kernel Name",
        ]
        if kernel_path.is_file():
            try:
                with kernel_path.open(newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    if reader.fieldnames != expected_kernel_header:
                        self.error(
                            "profile_kernel_export_header_mismatch",
                            str(kernel_path.relative_to(self.root)),
                            repr(reader.fieldnames),
                        )
                    kernel_rows = list(reader)
            except OSError as exc:
                self.error(
                    "profile_kernel_export_read_error",
                    str(kernel_path.relative_to(self.root)),
                    f"{type(exc).__name__}: {exc}",
                )
        kernels_by_device: dict[int, int] = {}
        collective_kernels_by_device: dict[int, int] = {}
        pids_by_device: dict[int, set[int]] = {}
        collective_kernel_pattern = re.compile(
            r"all.?reduce|cross_device_reduce_(?:1stage|2stage)",
            flags=re.IGNORECASE,
        )
        for row in kernel_rows:
            try:
                device = int(row["DevId"])
                pid = int(row["PID"])
                duration = int(row["Kernel Dur (ns)"])
                kernel_name = row["Kernel Name"]
            except (KeyError, TypeError, ValueError) as exc:
                self.error(
                    "profile_kernel_export_row_mismatch",
                    str(kernel_path.relative_to(self.root)),
                    f"{row!r}: {exc}",
                )
                continue
            if device not in range(4) or pid <= 0 or duration <= 0 or not kernel_name:
                self.error(
                    "profile_kernel_export_row_mismatch",
                    str(kernel_path.relative_to(self.root)),
                    repr(row),
                )
            if bounds is not None:
                try:
                    kernel_start = int(row["Kernel Start (ns)"])
                except (KeyError, TypeError, ValueError) as exc:
                    self.error(
                        "profile_kernel_export_timestamp_mismatch",
                        str(kernel_path.relative_to(self.root)),
                        f"{row!r}: {exc}",
                    )
                else:
                    if kernel_start >= bounds[1] or kernel_start + duration <= bounds[0]:
                        self.error(
                            "profile_kernel_outside_measured_window",
                            str(kernel_path.relative_to(self.root)),
                            repr(row),
                        )
            kernels_by_device[device] = kernels_by_device.get(device, 0) + 1
            pids_by_device.setdefault(device, set()).add(pid)
            if collective_kernel_pattern.search(kernel_name):
                collective_kernels_by_device[device] = (
                    collective_kernels_by_device.get(device, 0) + 1
                )
        if set(kernels_by_device) != set(range(4)) or any(
            count < 20 for count in kernels_by_device.values()
        ):
            self.error(
                "profile_kernel_export_device_coverage_mismatch",
                str(kernel_path.relative_to(self.root)),
                repr(kernels_by_device),
            )
        if set(collective_kernels_by_device) != set(range(4)) or any(
            count < expected_collective_launches_per_device
            for count in collective_kernels_by_device.values()
        ):
            self.error(
                "profile_collective_kernel_coverage_mismatch",
                str(kernel_path.relative_to(self.root)),
                (
                    f"expected at least {expected_collective_launches_per_device} "
                    f"per device, got {collective_kernels_by_device!r}"
                ),
            )
        if set(pids_by_device) != set(range(4)) or any(
            len(pids) != 1 for pids in pids_by_device.values()
        ) or len({next(iter(pids)) for pids in pids_by_device.values() if pids}) != 4:
            self.error(
                "profile_kernel_export_process_coverage_mismatch",
                str(kernel_path.relative_to(self.root)),
                repr(pids_by_device),
            )

        api_path = required["measured_cuda_api"]
        api_pids: set[int] = set()
        graph_launches_by_pid: dict[int, int] = {}
        if api_path.is_file():
            try:
                with api_path.open(newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    for row in reader:
                        try:
                            pid = int(row["Pid"])
                        except (KeyError, TypeError, ValueError):
                            self.error(
                                "profile_api_export_row_mismatch",
                                str(api_path.relative_to(self.root)),
                                repr(row),
                            )
                            continue
                        api_pids.add(pid)
                        if re.fullmatch(
                            r"cudaGraphLaunch(?:_v[0-9]+)?", row.get("Name", "")
                        ):
                            graph_launches_by_pid[pid] = (
                                graph_launches_by_pid.get(pid, 0) + 1
                            )
            except OSError as exc:
                self.error(
                    "profile_api_export_read_error",
                    str(api_path.relative_to(self.root)),
                    f"{type(exc).__name__}: {exc}",
                )
        kernel_pids = {
            pid for device_pids in pids_by_device.values() for pid in device_pids
        }
        if kernel_pids and not kernel_pids <= api_pids:
            self.error(
                "profile_api_export_process_coverage_mismatch",
                str(api_path.relative_to(self.root)),
                f"kernel_pids={sorted(kernel_pids)}, api_pids={sorted(api_pids)}",
            )
        if expected_graph_launches_per_process is not None:
            expected_graph_counts = {
                pid: expected_graph_launches_per_process for pid in kernel_pids
            }
            observed_graph_counts = {
                pid: graph_launches_by_pid.get(pid, 0) for pid in kernel_pids
            }
            if observed_graph_counts != expected_graph_counts:
                self.error(
                    "profile_graph_replay_count_mismatch",
                    str(api_path.relative_to(self.root)),
                    (
                        f"expected={expected_graph_counts!r}, "
                        f"got={observed_graph_counts!r}"
                    ),
                )

        nvtx_path = required["measured_nvtx"]
        if nvtx_path.is_file() and expected_range not in nvtx_path.read_text(
            encoding="utf-8", errors="replace"
        ):
            self.error(
                "profile_export_nvtx_range_missing",
                str(nvtx_path.relative_to(self.root)),
                expected_range,
            )
        return {
            **{label: file_summary(path) for label, path in required.items()},
            "expected_collective_launches_per_device": (
                expected_collective_launches_per_device
            ),
            "expected_graph_launches_per_process": (
                expected_graph_launches_per_process
            ),
            "measured_kernel_rows": len(kernel_rows),
            "measured_window_ns": list(bounds) if bounds is not None else None,
            "kernels_by_device": kernels_by_device,
            "collective_kernels_by_device": collective_kernels_by_device,
            "pids_by_device": {
                device: sorted(pids) for device, pids in pids_by_device.items()
            },
            "measured_cuda_api_pids": sorted(api_pids),
            "graph_launches_by_pid": graph_launches_by_pid,
        }

    def validate_profiles(
        self, expected_selected: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        expected_stats_names = {case.short_name for case in CASES}
        expected_stats_names.update(
            f"{case.short_name}_c10d" for case in CASES
        )
        stats_status = self.validate_stats_status(expected_stats_names)
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
            stock_measurements = self.validate_profile_result(
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
            stock_exports = self.validate_profile_exports(
                case.short_name,
                expected_stock_range,
                expected_collective_launches_per_device=stock_measurements.get(
                    "total_sample_calls", 0
                ),
                expected_graph_launches_per_process=(
                    stock_measurements.get("total_sample_calls", 0)
                    if case.execution_mode == "cuda_graph"
                    else None
                ),
            )
            item: dict[str, Any] = {
                "stock": report_summary,
                "stock_stats_log": file_summary(stock_stats),
                "stock_measurement_attempts": stock_measurements,
                "stock_exports": stock_exports,
                "candidate": None,
            }
            expectation = expected_selected.get(case.short_name)
            expected_variant = (
                expectation.get("variant") if isinstance(expectation, dict) else None
            )
            recorded = recorded_selected.get(case.short_name)
            recorded_variant = (
                recorded.get("variant") if isinstance(recorded, dict) else None
            )
            if expected_variant is None:
                self.error(
                    "missing_abi_compatible_candidate",
                    f"paired/{case.short_name}",
                    "exactly one candidate result is required",
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
                        f"paired/{case.short_name}_c10d_selection.json",
                        (
                            f"{case.short_name}: expected {expected_variant!r}, "
                            f"got {recorded_variant!r}"
                        ),
                    )
                if isinstance(recorded, dict):
                    expected_result_path = (
                        self.root / str(expectation.get("result_path", ""))
                    ).resolve()
                    comparisons = (
                        (
                            "candidate_path",
                            recorded.get("candidate_path"),
                            expectation.get("candidate_entrypoint"),
                        ),
                        (
                            "result_path",
                            str(
                                (
                                    self.root
                                    / str(recorded.get("result_path", ""))
                                ).resolve()
                            ),
                            str(expected_result_path),
                        ),
                        (
                            "candidate_entrypoint_sha256",
                            recorded.get("candidate_entrypoint_sha256"),
                            expectation.get("candidate_entrypoint_sha256"),
                        ),
                        (
                            "candidate_manifest_sha256",
                            recorded.get("candidate_manifest_sha256"),
                            expectation.get("candidate_manifest_sha256"),
                        ),
                    )
                    for field, observed, expected in comparisons:
                        if observed != expected:
                            self.error(
                                "candidate_profile_selection_provenance_mismatch",
                                f"paired/{case.short_name}_c10d_selection.json",
                                f"{field}: expected={expected!r}, got={observed!r}",
                            )
                candidate_report = self.root / f"profile/{case.short_name}_c10d.nsys-rep"
                candidate_summary = file_summary(candidate_report)
                if not candidate_summary["present"] or candidate_summary.get("bytes", 0) == 0:
                    self.error(
                        "missing_candidate_nsys_report",
                        f"profile/{case.short_name}_c10d.nsys-rep",
                        "candidate report absent or empty",
                    )
                candidate_measurements = self.validate_profile_result(
                    f"profile/{case.short_name}_c10d.result.json",
                    case,
                    candidate=True,
                    candidate_expectation=expectation,
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
                    "measurement_attempts": candidate_measurements,
                    "exports": self.validate_profile_exports(
                        f"{case.short_name}_c10d",
                        expected_candidate_range,
                        expected_collective_launches_per_device=(
                            candidate_measurements.get("total_sample_calls", 0)
                        ),
                        expected_graph_launches_per_process=(
                            candidate_measurements.get("total_sample_calls", 0)
                            if case.execution_mode == "cuda_graph"
                            else None
                        ),
                    ),
                }
            output[case.short_name] = item

        extra_recorded = set(recorded_selected) - {case.short_name for case in CASES}
        if extra_recorded:
            self.error(
                "unexpected_profile_selection_cases",
                "profile/c10d_profile_selection.tsv",
                repr(sorted(extra_recorded)),
            )
        output["stats_status"] = stats_status
        return output

    def validate_producer_abi(self) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for rows in (16, 32):
            task = f"linear_attn_o_decode_m{rows}"
            relative = f"producer_abi/{task}.json"
            result = self.read_json(relative)
            if result is None:
                continue
            if result.get("schema_version") != 1:
                self.error(
                    "producer_abi_schema_mismatch",
                    relative,
                    repr(result.get("schema_version")),
                )
            workload = result.get("workload")
            expected_workload = {
                "name": task,
                "family": "packed_fp8_gemm",
                "phase": "decode",
                "world_size": 1,
                "distributed": False,
                "source_symbol": (
                    "sglang.kernels.ops.quantization.fp8_kernel."
                    "w8a8_block_fp8_matmul_deepgemm"
                ),
                "params": {"m": rows, "n": 6144, "k": 16384},
            }
            if not isinstance(workload, dict):
                self.error("missing_producer_abi_workload", relative, repr(workload))
            else:
                for key, expected in expected_workload.items():
                    if workload.get(key) != expected:
                        self.error(
                            "producer_abi_workload_mismatch",
                            relative,
                            f"{key}: expected {expected!r}, got {workload.get(key)!r}",
                        )
            if result.get("reference_policy") != "SGLANG_GLM52_OPT=0 production path":
                self.error(
                    "producer_abi_reference_policy_mismatch",
                    relative,
                    repr(result.get("reference_policy")),
                )
            if result.get("execution_mode") != "eager_cuda_event":
                self.error(
                    "producer_abi_execution_mode_mismatch",
                    relative,
                    repr(result.get("execution_mode")),
                )
            if result.get("timing_contract") != "interleaved paired A/B; maximum CUDA-event latency across ranks":
                self.error(
                    "producer_abi_timing_contract_mismatch",
                    relative,
                    repr(result.get("timing_contract")),
                )
            paired = result.get("paired_measurements")
            reference_values: list[float] = []
            candidate_values: list[float] = []
            paired_valid = True
            if not isinstance(paired, dict):
                self.error(
                    "missing_producer_abi_paired_measurements",
                    relative,
                    repr(paired),
                )
                paired_valid = False
            else:
                if set(paired) != {"metric", "samples"}:
                    self.error(
                        "producer_abi_paired_schema_mismatch",
                        relative,
                        f"keys={sorted(paired)}",
                    )
                    paired_valid = False
                if paired.get("metric") != "rank_max_cuda_event_ms":
                    self.error(
                        "producer_abi_paired_metric_mismatch",
                        relative,
                        repr(paired.get("metric")),
                    )
                    paired_valid = False
                samples = paired.get("samples")
                if not isinstance(samples, list) or len(samples) != 100:
                    self.error(
                        "producer_abi_paired_sample_count_mismatch",
                        relative,
                        f"expected 100, got {len(samples) if isinstance(samples, list) else samples!r}",
                    )
                    paired_valid = False
                    samples = []
                expected_sample_keys = {
                    "sample_index",
                    "order",
                    "reference_ms",
                    "candidate_ms",
                }
                for index, sample in enumerate(samples):
                    expected_order = (
                        ["reference", "candidate"]
                        if index % 2 == 0
                        else ["candidate", "reference"]
                    )
                    if not isinstance(sample, dict):
                        self.error(
                            "producer_abi_paired_sample_schema_mismatch",
                            relative,
                            f"sample {index}: {sample!r}",
                        )
                        paired_valid = False
                        continue
                    if set(sample) != expected_sample_keys:
                        self.error(
                            "producer_abi_paired_sample_schema_mismatch",
                            relative,
                            f"sample {index}: keys={sorted(sample)}",
                        )
                        paired_valid = False
                    if sample.get("sample_index") != index:
                        self.error(
                            "producer_abi_paired_sample_index_mismatch",
                            relative,
                            f"sample {index}: {sample.get('sample_index')!r}",
                        )
                        paired_valid = False
                    if sample.get("order") != expected_order:
                        self.error(
                            "producer_abi_paired_order_mismatch",
                            relative,
                            f"sample {index}: {sample.get('order')!r}",
                        )
                        paired_valid = False
                    values: list[float] = []
                    for key in ("reference_ms", "candidate_ms"):
                        raw_value = sample.get(key)
                        if (
                            isinstance(raw_value, bool)
                            or not isinstance(raw_value, (int, float))
                            or not math.isfinite(float(raw_value))
                            or float(raw_value) <= 0
                        ):
                            self.error(
                                "invalid_producer_abi_paired_sample",
                                relative,
                                f"sample {index} {key}={raw_value!r}",
                            )
                            paired_valid = False
                            values = []
                            break
                        values.append(float(raw_value))
                    if len(values) == 2:
                        reference_values.append(values[0])
                        candidate_values.append(values[1])
            if len(reference_values) != 100 or len(candidate_values) != 100:
                paired_valid = False

            summaries: dict[str, dict[str, float]] = {}
            for side in ("reference", "candidate"):
                summary = result.get(side)
                if not isinstance(summary, dict):
                    self.error(
                        "missing_producer_abi_summary",
                        relative,
                        f"{side}={summary!r}",
                    )
                    continue
                try:
                    minimum = float(summary["min_ms"])
                    median = float(summary["median_ms"])
                    p95 = float(summary["p95_ms"])
                except (KeyError, TypeError, ValueError) as exc:
                    self.error(
                        "invalid_producer_abi_summary",
                        relative,
                        f"{side}: {exc}",
                    )
                    continue
                if (
                    any(not math.isfinite(value) or value <= 0 for value in (minimum, median, p95))
                    or minimum > median
                    or median > p95
                ):
                    self.error(
                        "invalid_producer_abi_distribution",
                        relative,
                        f"{side}: min={minimum}, median={median}, p95={p95}",
                    )
                summaries[side] = {
                    "min_ms": minimum,
                    "median_ms": median,
                    "p95_ms": p95,
                }
            derived_summaries: dict[str, dict[str, float]] = {}
            if paired_valid:
                for side, values in (
                    ("reference", reference_values),
                    ("candidate", candidate_values),
                ):
                    ordered = sorted(values)
                    p95_index = min(
                        len(ordered) - 1,
                        max(0, int(0.95 * len(ordered)) - 1),
                    )
                    derived = {
                        "min_ms": min(values),
                        "median_ms": statistics.median(values),
                        "p95_ms": ordered[p95_index],
                    }
                    derived_summaries[side] = derived
                    reported = summaries.get(side)
                    if reported is None:
                        continue
                    for key, expected in derived.items():
                        if not math.isclose(
                            reported[key],
                            expected,
                            rel_tol=1e-12,
                            abs_tol=1e-12,
                        ):
                            self.error(
                                "producer_abi_summary_mismatch",
                                relative,
                                (
                                    f"{side}.{key}: reported={reported[key]}, "
                                    f"derived={expected}"
                                ),
                            )
            candidate = result.get("candidate")
            expected_candidate = self.expected_roots.get("kernel_harness_git")
            if isinstance(candidate, dict) and expected_candidate is not None:
                expected_path = str(
                    Path(expected_candidate)
                    / "serving_native"
                    / "candidates"
                    / "reference.py"
                )
                if candidate.get("path") != expected_path:
                    self.error(
                        "producer_abi_candidate_path_mismatch",
                        relative,
                        f"expected {expected_path!r}, got {candidate.get('path')!r}",
                    )
            speedup = candidate.get("speedup") if isinstance(candidate, dict) else None
            ratios = (
                [
                    reference / candidate_value
                    for reference, candidate_value in zip(
                        reference_values, candidate_values
                    )
                ]
                if paired_valid
                else []
            )
            derived_speedup = statistics.median(ratios) if ratios else None
            if (
                isinstance(speedup, bool)
                or not isinstance(speedup, (int, float))
                or not math.isfinite(float(speedup))
            ):
                self.error("invalid_producer_abi_speedup", relative, repr(speedup))
            else:
                speedup_value = float(speedup)
                p10 = candidate.get("paired_p10_speedup")
                p90 = candidate.get("paired_p90_speedup")
                if (
                    not isinstance(p10, (int, float))
                    or not isinstance(p90, (int, float))
                    or not math.isfinite(float(p10))
                    or not math.isfinite(float(p90))
                    or float(p10) <= 0
                    or float(p90) <= 0
                    or float(p10) > speedup_value
                    or speedup_value > float(p90)
                ):
                    self.error(
                        "invalid_producer_abi_paired_quantiles",
                        relative,
                        f"p10={p10!r}, median={speedup_value}, p90={p90!r}",
                    )
                if derived_speedup is not None:
                    ordered_ratios = sorted(ratios)
                    expected_p10 = ordered_ratios[
                        min(
                            len(ordered_ratios) - 1,
                            int(0.1 * len(ordered_ratios)),
                        )
                    ]
                    expected_p90 = ordered_ratios[
                        min(
                            len(ordered_ratios) - 1,
                            max(0, int(0.9 * len(ordered_ratios)) - 1),
                        )
                    ]
                    for key, reported, expected in (
                        ("speedup", speedup_value, derived_speedup),
                        ("paired_p10_speedup", p10, expected_p10),
                        ("paired_p90_speedup", p90, expected_p90),
                    ):
                        if (
                            isinstance(reported, bool)
                            or not isinstance(reported, (int, float))
                            or not math.isclose(
                                float(reported),
                                expected,
                                rel_tol=1e-12,
                                abs_tol=1e-12,
                            )
                        ):
                            self.error(
                                "producer_abi_paired_summary_mismatch",
                                relative,
                                f"{key}: reported={reported!r}, derived={expected}",
                            )
                    expected_gate = derived_speedup >= 1.03
                    if abs(derived_speedup - 1.0) >= 0.03:
                        self.error(
                            "producer_abi_reference_control_noise_exceeds_3pct",
                            relative,
                            f"derived paired median speedup={derived_speedup}",
                        )
                    if candidate.get("passes_3pct_median_gate") is not expected_gate:
                        self.error(
                            "producer_abi_gate_mismatch",
                            relative,
                            repr(candidate.get("passes_3pct_median_gate")),
                        )
            output[task] = {
                "scope": (
                    "single-GPU packed-int32 UE8M0 producer ABI check only; "
                    "not a producer-AllReduce-consumer region"
                ),
                "summaries": summaries,
                "derived_summaries": derived_summaries,
                "reported_paired_median_speedup": speedup,
                "derived_paired_median_speedup": derived_speedup,
                "paired_p10_speedup": (
                    candidate.get("paired_p10_speedup")
                    if isinstance(candidate, dict)
                    else None
                ),
                "paired_p90_speedup": (
                    candidate.get("paired_p90_speedup")
                    if isinstance(candidate, dict)
                    else None
                ),
                "paired_samples": len(ratios),
                "result": file_summary(self.root / relative),
            }
        return output

    def validate_backend_scout(self) -> dict[str, Any]:
        step = "backend_scout/custom_allreduce"
        row = self.status_rows.get(step)
        if row is None:
            return {}
        log = self.resolve_log(str(row.get("log", "")), step=step)
        lines = (
            log.read_text(encoding="utf-8", errors="replace").splitlines()
            if log.is_file()
            else []
        )
        if row["requirement"] != "required" or row["exit_code"] != 0:
            self.error(
                "backend_scout_failed",
                f"status.tsv:{row['line_number']}",
                f"requirement={row['requirement']!r}, exit={row['exit_code']}",
            )
        parsed_rows: dict[int, dict[str, Any]] = {}
        header_rows = [
            tuple(line.replace("|", " ").split())
            for line in lines
            if "message_bytes" in line
        ]
        if header_rows != [BACKEND_SCOUT_HEADER]:
            self.error(
                "backend_scout_header_mismatch",
                str(log),
                f"expected={BACKEND_SCOUT_HEADER!r}, got={header_rows!r}",
            )
        for line in lines:
            fields = line.replace("|", " ").split()
            if len(fields) < 2 or not fields[0].isdigit() or not fields[1].isdigit():
                continue
            if len(fields) != 10:
                self.error("backend_scout_row_mismatch", str(log), line)
                continue
            try:
                values = [float(value) for value in fields[2:]]
            except ValueError as exc:
                self.error(
                    "backend_scout_row_mismatch", str(log), f"{line}: {exc}"
                )
                continue
            row_id = int(fields[0])
            message_bytes = int(fields[1])
            if message_bytes in parsed_rows:
                self.error(
                    "backend_scout_duplicate_message_size",
                    str(log),
                    str(message_bytes),
                )
            if any(not math.isfinite(value) or value <= 0 for value in values):
                self.error(
                    "backend_scout_invalid_measurement",
                    str(log),
                    line,
                )
            expected_row_id = len(parsed_rows)
            expected_message_bytes = (
                BACKEND_SCOUT_MESSAGE_SIZES[expected_row_id]
                if expected_row_id < len(BACKEND_SCOUT_MESSAGE_SIZES)
                else None
            )
            if row_id != expected_row_id or message_bytes != expected_message_bytes:
                self.error(
                    "backend_scout_row_order_mismatch",
                    str(log),
                    (
                        f"expected row {expected_row_id} message {expected_message_bytes}, "
                        f"got row {row_id} message {message_bytes}"
                    ),
                )
            parsed_rows[message_bytes] = {
                "row_id": row_id,
                "providers": {
                    provider: {
                        "latency_us": values[index],
                        "bandwidth_gb_s": values[index + len(BACKEND_SCOUT_PROVIDERS)],
                    }
                    for index, provider in enumerate(BACKEND_SCOUT_PROVIDERS)
                },
            }
        if tuple(parsed_rows) != BACKEND_SCOUT_MESSAGE_SIZES:
            self.error(
                "backend_scout_table_mismatch",
                str(log),
                f"sizes={tuple(parsed_rows)!r}",
            )
        return {
            "scope": "upstream custom-AllReduce performance-only scout",
            "exit_code": row["exit_code"],
            "successful": row["exit_code"] == 0,
            "measurements": parsed_rows,
            "message_sizes": list(parsed_rows),
            "log": file_summary(log),
            "tail": lines[-40:],
        }

    def analyze(self) -> dict[str, Any]:
        if not self.root.is_dir():
            self.error("missing_campaign_root", str(self.root), "directory is absent")
            return self.finish({})
        self.load_status()
        source_identity = self.load_source_identity()
        environment = self.validate_environment_evidence()
        traces = {case.short_name: self.validate_trace(case) for case in CASES}
        semantics = self.validate_semantics()
        baselines = self.validate_baselines()
        controls, candidates, selected = self.validate_paired()
        profiles = self.validate_profiles(selected)
        producer_abi = self.validate_producer_abi()
        backend_scout = self.validate_backend_scout()
        return self.finish(
            {
                "source_identity": source_identity,
                "environment_evidence": environment,
                "reachability": traces,
                "semantics": semantics,
                "baselines": baselines,
                "reference_control_noise": controls,
                "c10d_attempts": candidates,
                "backend_scout": backend_scout,
                "producer_abi": producer_abi,
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
    parser.add_argument(
        "--resolve-c10d", choices=[case.short_name for case in CASES]
    )
    parser.add_argument("--harness-root")
    return parser.parse_args(argv)


def render(result: dict[str, Any]) -> str:
    return json.dumps(result, indent=2, sort_keys=True) + "\n"


def main(argv: Optional[Iterable[str]] = None) -> int:
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        if exc.code == 0:
            return 0
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
    except ValueError as exc:
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

    if args.resolve_c10d is not None:
        if not args.harness_root:
            result = {
                "schema_version": 1,
                "scope": "tp4_allreduce_diagnostic_only",
                "valid": False,
                "errors": [
                    {
                        "code": "resolution_argument_error",
                        "location": "command_line",
                        "detail": "--harness-root is required with --resolve-c10d",
                    }
                ],
            }
        else:
            case = next(
                case for case in CASES if case.short_name == args.resolve_c10d
            )
            try:
                result = resolve_c10d_abi(
                    Path(args.campaign_root), case, Path(args.harness_root)
                )
            except ValueError as exc:
                result = {
                    "schema_version": 1,
                    "scope": "tp4_allreduce_diagnostic_only",
                    "short_name": case.short_name,
                    "valid": False,
                    "errors": [
                        {
                            "code": "c10d_abi_resolution_error",
                            "location": f"paired/{case.short_name}",
                            "detail": str(exc),
                        }
                    ],
                }
            else:
                result["valid"] = True
                result["errors"] = []
    else:
        if args.harness_root:
            result = {
                "schema_version": 1,
                "scope": "tp4_allreduce_diagnostic_only",
                "valid": False,
                "errors": [
                    {
                        "code": "argument_error",
                        "location": "command_line",
                        "detail": "--harness-root is only valid with --resolve-c10d",
                    }
                ],
            }
        else:
            result = Analyzer(Path(args.campaign_root)).analyze()
    payload = render(result)
    if args.output:
        output = Path(args.output).expanduser()
        temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(payload, encoding="utf-8")
            temporary.replace(output)
        except OSError as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
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
