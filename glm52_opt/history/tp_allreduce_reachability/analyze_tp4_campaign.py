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
            sections = re.split(r"^capability=([rwn])\s*$", text, flags=re.MULTILINE)
            observed_sections: dict[str, str] = {}
            for index in range(1, len(sections), 2):
                observed_sections[sections[index]] = sections[index + 1]
            if set(observed_sections) != {"r", "w", "n"}:
                self.error(
                    "p2p_capability_section_mismatch",
                    "environment/p2p_capability.log",
                    repr(sorted(observed_sections)),
                )
            for capability, section in observed_sections.items():
                matrix_rows = re.findall(
                    r"^\s*GPU[0-3]\s+(.+)$", section, flags=re.MULTILINE
                )
                ok_count = sum(
                    len(re.findall(r"\bOK\b", row)) for row in matrix_rows
                )
                p2p_counts[capability] = ok_count
                if len(matrix_rows) != 4 or ok_count != 12 or any(
                    f"GPU{rank}" not in section for rank in range(4)
                ):
                    self.error(
                        "p2p_capability_not_full_mesh",
                        "environment/p2p_capability.log",
                        f"capability={capability}, directed_ok_edges={ok_count}",
                    )
        except OSError as exc:
            self.error(
                "p2p_capability_read_error",
                "environment/p2p_capability.log",
                f"{type(exc).__name__}: {exc}",
            )
        return {
            "lock_receipt": file_summary(lock_path),
            "compute_process_snapshots": process_snapshots,
            "p2p_full_mesh_directed_ok_edges": p2p_counts,
            "p2p_capability": file_summary(p2p_path),
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
                "local_size": 4,
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
        }
        if not isinstance(timing, dict):
            self.error("missing_timing_contract", location, repr(timing))
        else:
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
    def _sample_values(result: dict[str, Any], side: str, metric: str) -> list[float]:
        raw = result.get("raw_samples")
        if not isinstance(raw, dict) or not isinstance(raw.get(side), list):
            raise ValueError(f"raw_samples.{side} is absent")
        if raw.get("rank_order") != [0, 1, 2, 3]:
            raise ValueError(
                f"raw_samples.rank_order must be [0, 1, 2, 3], got {raw.get('rank_order')!r}"
            )
        measured_order = raw.get("measured_order")
        if not isinstance(measured_order, list) or len(measured_order) != len(raw[side]):
            raise ValueError("raw_samples.measured_order length is invalid")
        paired = isinstance(raw.get("candidate"), list)
        values: list[float] = []
        for index, sample in enumerate(raw[side]):
            if not isinstance(sample, dict):
                raise ValueError(f"raw_samples.{side}[{index}] is not an object")
            if sample.get("sample_index") != index:
                raise ValueError(
                    f"raw_samples.{side}[{index}].sample_index="
                    f"{sample.get('sample_index')!r}"
                )
            if sample.get("variant") != index % 2:
                raise ValueError(
                    f"raw_samples.{side}[{index}].variant={sample.get('variant')!r}"
                )
            expected_order = (
                (
                    ["reference", "candidate"]
                    if index % 2 == 0
                    else ["candidate", "reference"]
                )
                if paired
                else ["reference"]
            )
            if measured_order[index] != expected_order:
                raise ValueError(
                    f"raw_samples.measured_order[{index}]={measured_order[index]!r}"
                )
            expected_position = (
                (index % 2 if side == "reference" else 1 - (index % 2))
                if paired
                else 0
            )
            if sample.get("position") != expected_position:
                raise ValueError(
                    f"raw_samples.{side}[{index}].position={sample.get('position')!r}"
                )
            if sample.get("readiness_probe_exact") is not True:
                raise ValueError(
                    f"raw_samples.{side}[{index}].readiness_probe_exact is not true"
                )
            parsed_metrics: dict[str, tuple[float, list[float]]] = {}
            for metric_name in ("collective_only", "ready_region"):
                metric_record = sample.get(metric_name)
                if not isinstance(metric_record, dict):
                    raise ValueError(
                        f"raw_samples.{side}[{index}].{metric_name} is absent"
                    )
                rank_values = metric_record.get("rank_ms")
                if not isinstance(rank_values, list) or len(rank_values) != 4:
                    raise ValueError(
                        f"raw_samples.{side}[{index}].{metric_name}.rank_ms must have four values"
                    )
                try:
                    rank_ms = [float(value) for value in rank_values]
                    local_ms = float(metric_record["local_ms"])
                    rank_max_ms = float(metric_record["rank_max_ms"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"raw_samples.{side}[{index}].{metric_name}: {exc}"
                    ) from exc
                if any(not math.isfinite(value) or value <= 0 for value in rank_ms):
                    raise ValueError(
                        f"raw_samples.{side}[{index}].{metric_name}.rank_ms must be finite and > 0"
                    )
                if not math.isfinite(local_ms) or local_ms <= 0:
                    raise ValueError(
                        f"raw_samples.{side}[{index}].{metric_name}.local_ms must be finite and > 0"
                    )
                if not math.isclose(local_ms, rank_ms[0], rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError(
                        f"raw_samples.{side}[{index}].{metric_name}.local_ms != rank_ms[0]"
                    )
                if not math.isclose(
                    rank_max_ms,
                    max(rank_ms),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        f"raw_samples.{side}[{index}].{metric_name}.rank_max_ms != max(rank_ms)"
                    )
                parsed_metrics[metric_name] = (rank_max_ms, rank_ms)
            collective_rank = parsed_metrics["collective_only"][1]
            ready_rank = parsed_metrics["ready_region"][1]
            if any(
                ready + 1e-9 < collective
                for collective, ready in zip(collective_rank, ready_rank)
            ):
                raise ValueError(
                    f"raw_samples.{side}[{index}] ready_region precedes collective_only"
                )
            try:
                value = parsed_metrics[metric][0]
            except KeyError as exc:
                raise ValueError(f"raw_samples.{side}[{index}].{metric}: {exc}") from exc
            values.append(value)
        return values

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
                        failed_logs.append(self.failed_log_summary(row))
            if persisted:
                if len(persisted) > 1:
                    self.error(
                        "multiple_abi_compatible_candidates",
                        f"paired/{case.short_name}",
                        repr(persisted),
                    )
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
        candidate_entrypoint: Optional[str] = None,
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
                observed_entrypoint = (
                    Path(str(manifest.get("entrypoint", ""))).name
                    if isinstance(manifest, dict)
                    else None
                )
                if (
                    candidate_entrypoint is None
                    or observed_entrypoint != candidate_entrypoint
                ):
                    self.error(
                        "profile_candidate_entrypoint_mismatch",
                        relative,
                        f"expected {candidate_entrypoint!r}, got {observed_entrypoint!r}",
                    )
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
                    if candidate_path.endswith("/allreduce_torch_outplace.py"):
                        variant = "outplace"
                    elif candidate_path.endswith("/allreduce_torch.py"):
                        variant = "inplace"
                    else:
                        self.error(
                            "unknown_profile_candidate",
                            f"profile/c10d_profile_selection.tsv:{line_number}",
                            candidate_path,
                        )
                        continue
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
                    expected_log = self.root / "profile" / f"{name}.stats.log"
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
                    if not expected_log.is_file() or expected_log.stat().st_size == 0:
                        self.error(
                            "missing_nsys_stats",
                            f"profile/stats_status.tsv:{line_number}",
                            str(expected_log),
                        )
                    rows[name] = {
                        "exit_code": exit_code,
                        "stats_log": file_summary(expected_log),
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

    def validate_profiles(self, expected_selected: dict[str, str]) -> dict[str, Any]:
        expected_stats_names = {case.short_name for case in CASES}
        expected_stats_names.update(
            f"{short_name}_c10d" for short_name in expected_selected
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
                    candidate_entrypoint=(
                        "allreduce_torch_outplace.py"
                        if expected_variant == "outplace"
                        else "allreduce_torch.py"
                    ),
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
            if not isinstance(speedup, (int, float)) or not math.isfinite(float(speedup)):
                self.error("invalid_producer_abi_speedup", relative, repr(speedup))
            output[task] = {
                "scope": (
                    "single-GPU packed-int32 UE8M0 producer ABI check only; "
                    "not a producer-AllReduce-consumer region"
                ),
                "summaries": summaries,
                "paired_median_speedup": speedup,
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
        return {
            "scope": "upstream custom-AllReduce performance-only scout",
            "exit_code": row["exit_code"],
            "successful": row["exit_code"] == 0,
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
