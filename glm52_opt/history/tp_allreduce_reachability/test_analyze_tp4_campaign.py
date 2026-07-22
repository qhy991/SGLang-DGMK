#!/usr/bin/env python3
"""CPU-only corruption tests for the TP4 campaign analyzer."""

from __future__ import annotations

import contextlib
import copy
import csv
import hashlib
import importlib.util
import io
import json
import shutil
import statistics
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "tp_allreduce_campaign_analyzer", HERE / "analyze_tp4_campaign.py"
)
assert SPEC is not None and SPEC.loader is not None
ANALYZER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ANALYZER
SPEC.loader.exec_module(ANALYZER)


def _sample(index: int, position: int) -> dict:
    return {
        "sample_index": index,
        "position": position,
        "variant": index % 2,
        "scheduled_start_arrival_ns_by_rank": [
            994_998_700,
            994_998_800,
            994_998_900,
            994_999_000,
        ],
        "scheduled_start_target_ns_by_rank": [999_999_000] * 4,
        "start_record_bracket_ns_by_rank": [
            [1_000_000_000, 1_000_000_010],
            [1_000_000_100, 1_000_000_110],
            [1_000_000_200, 1_000_000_210],
            [1_000_000_300, 1_000_000_310],
        ],
        "start_record_envelope_span_ns": 310,
        "readiness_probe_exact": True,
        "collective_only": {
            "local_ms": 1.0,
            "rank_ms": [1.0, 1.1, 1.2, 1.3],
            "rank_max_ms": 1.3,
        },
        "ready_region": {
            "local_ms": 1.1,
            "rank_ms": [1.1, 1.2, 1.3, 1.4],
            "rank_max_ms": 1.4,
        },
    }


def _paired_resolution_result(
    *,
    case,
    contracts: list[dict],
    manifest: dict,
    harness: Path,
    sglang: Path,
    harness_sha: str,
    sglang_sha: str,
) -> dict:
    measured_order = [
        ["reference", "candidate"]
        if index % 2 == 0
        else ["candidate", "reference"]
        for index in range(100)
    ]
    reference_samples = [_sample(index, index % 2) for index in range(100)]
    candidate_samples = [_sample(index, 1 - index % 2) for index in range(100)]
    metric_summary = {
        "collective_only": {"median_ms": 1.3},
        "ready_region": {"median_ms": 1.4},
    }
    dispatch = [
        {
            "rank": rank,
            "group_ranks": [0, 1, 2, 3],
            "world_size": 4,
            "local_size": 0,
            "shape": [case.rows, 6144],
            "stride": [6144, 1],
            "dtype": "torch.bfloat16",
            "message_bytes": case.message_bytes,
            "execution_mode": case.execution_mode,
            "stream": {"requested": case.stream, "cuda_stream": 1},
            "predicted_reference_backend": "custom_all_reduce_outplace",
            "predicted_reference_algorithm": "ONE_SHOT_PUSH",
        }
        for rank in range(4)
    ]
    git_environment = {
        "kernel_harness_git": {
            "dirty": False,
            "status": [],
            "sha": harness_sha,
            "root": str(harness.resolve()),
        },
        "sglang_git": {
            "dirty": False,
            "status": [],
            "sha": sglang_sha,
            "root": str(sglang.resolve()),
        },
    }
    return {
        "schema_version": 2,
        "scope": "tp4_allreduce_diagnostic_only",
        "reference_policy": "SGLANG_GLM52_OPT=0 production GroupCoordinator path",
        "workload": {
            "name": case.task,
            "family": "allreduce",
            "world_size": 4,
            "params": {
                "local_tokens": case.rows,
                "hidden": 6144,
                "dtype": "bfloat16",
            },
        },
        "correctness": {
            "passed": True,
            "exact_full_checks": "before and after timing",
            "exact_full_tensor_consumer": "every warmup and measured sample",
            "exact_position_probe": "every warmup and measured sample",
            "alternating_input_variants": 2,
            "validation_errors_aggregated_over_tp_cpu_group": True,
            "candidate_state_guarded": True,
        },
        "gate": {
            "candidate_present": True,
            "performance_eligible": True,
            "stock_fallback_active": True,
            "passed": False,
        },
        "allreduce": {
            "message_bytes": case.message_bytes,
            "reference_contract_by_rank": contracts,
            "candidate_contract_by_rank": contracts,
            "reference_dispatch_by_rank": dispatch,
        },
        "execution": {
            "mode": case.execution_mode,
            "stream": case.stream,
            "cuda_stream": 1,
        },
        "timing_contract": {
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
                "every start-event record call"
            ),
        },
        "readiness_probe": {
            "positions": [[0, 0]],
            "num_elements": 1,
            "dtype": "torch.bfloat16",
            "validation": "exact on every warmup and measured sample",
            "full_tensor_consumer": "preallocated BF16 negation output",
            "full_tensor_validation": "exact on every warmup and measured sample",
        },
        "all_reduce_trace": {"enabled": False},
        "environment": {
            "env": {
                "CUDA_VISIBLE_DEVICES": "0,1,2,3",
                "SGLANG_GLM52_OPT": "0",
                "SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2": None,
                "NCCL_ALGO": None,
                "NCCL_PROTO": None,
            },
            "device_by_rank": [
                {"rank": rank, "name": "NVIDIA B200", "capability": [10, 0]}
                for rank in range(4)
            ],
            **git_environment,
        },
        "reference": copy.deepcopy(metric_summary),
        "candidate": {
            **copy.deepcopy(metric_summary),
            "manifest": manifest,
            "speedup": 1.0,
            "paired_p10_speedup": 1.0,
            "paired_p90_speedup": 1.0,
            "passes_3pct_median_gate": False,
        },
        "raw_samples": {
            "rank_order": [0, 1, 2, 3],
            "measured_order": measured_order,
            "reference": reference_samples,
            "candidate": candidate_samples,
        },
        "disposition": "tp4_diagnostic_no_replacement",
    }


def _write_resolution_fixture(root: Path, harness: Path) -> None:
    case = ANALYZER.CASES[0]
    paired = root / "paired"
    candidates = harness / "serving_native/candidates"
    paired.mkdir(parents=True, exist_ok=True)
    candidates.mkdir(parents=True, exist_ok=True)
    candidate_path = candidates / "allreduce_torch.py"
    candidate_path.write_text("def run(inputs, runtime):\n    return inputs\n")
    reference_path = candidates / "reference.py"
    reference_path.write_text("def run(inputs, runtime):\n    return inputs\n")
    entrypoint_sha = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    reference_sha = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    harness_sha = "1" * 40
    sglang_sha = "2" * 40
    sglang = Path(ANALYZER.__file__).resolve().parents[3]
    environment = root / "environment"
    environment.mkdir(exist_ok=True)
    (environment / "source_identity.log").write_text(
        f"{harness_sha}\n{sglang_sha}\n", encoding="utf-8"
    )
    contracts = [
        {
            "output": {
                "shape": [16, 6144],
                "stride": [6144, 1],
                "dtype": "torch.bfloat16",
                "device": f"cuda:{rank}",
            },
            "output_aliases_local": True,
            "local_poststate": "reduced",
            "source_immutable": True,
            "exact_values": True,
        }
        for rank in range(4)
    ]
    reference_manifest = {
        "entrypoint": str(reference_path.resolve()),
        "requested_path": str(reference_path.resolve()),
        "entrypoint_sha256_at_import": reference_sha,
        "manifest_sha256": "b" * 64,
    }
    control = _paired_resolution_result(
        case=case,
        contracts=contracts,
        manifest=reference_manifest,
        harness=harness,
        sglang=sglang,
        harness_sha=harness_sha,
        sglang_sha=sglang_sha,
    )
    (paired / "m16_reference_control.json").write_text(
        json.dumps(control), encoding="utf-8"
    )
    candidate_manifest = {
        "entrypoint": str(candidate_path.resolve()),
        "requested_path": str(candidate_path.resolve()),
        "entrypoint_sha256_at_import": entrypoint_sha,
        "manifest_sha256": "a" * 64,
    }
    result = _paired_resolution_result(
        case=case,
        contracts=contracts,
        manifest=candidate_manifest,
        harness=harness,
        sglang=sglang,
        harness_sha=harness_sha,
        sglang_sha=sglang_sha,
    )
    (paired / "m16_c10d_inplace.json").write_text(
        json.dumps(result), encoding="utf-8"
    )
    failures = [
        {
            "rank": rank,
            "error": (
                "AssertionError: candidate destructive/alias ABI differs from "
                "reference: candidate != reference"
            ),
        }
        for rank in range(4)
    ]
    failure_log = paired / "m16_c10d_outplace.log"
    failure_log.write_text(
        "SharedValidationError: candidate correctness failed collectively: "
        + json.dumps(failures, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    success_log = paired / "m16_c10d_inplace.log"
    success_log.write_text("success\n", encoding="utf-8")
    control_log = paired / "m16_reference_control.log"
    control_log.write_text("success\n", encoding="utf-8")
    status = root / "status.tsv"
    status.write_text(
        "\t".join(ANALYZER.STATUS_FIELDS)
        + "\n"
        + "\t".join(
            (
                "required",
                "paired/m16_reference_control",
                "0",
                "2026-07-22T00:00:00Z",
                "2026-07-22T00:00:01Z",
                str(control_log),
            )
        )
        + "\n"
        + "\t".join(
            (
                "attempt",
                "paired/m16_c10d_inplace",
                "0",
                "2026-07-22T00:00:00Z",
                "2026-07-22T00:00:01Z",
                str(success_log),
            )
        )
        + "\n"
        + "\t".join(
            (
                "attempt",
                "paired/m16_c10d_outplace",
                "1",
                "2026-07-22T00:00:02Z",
                "2026-07-22T00:00:03Z",
                str(failure_log),
            )
        )
        + "\n",
        encoding="utf-8",
    )


def _producer_result(rows: int, harness: Path) -> dict:
    reference_values = [1.0 + index / 10_000 for index in range(100)]
    candidate_values = list(reference_values)

    def summary(values: list[float]) -> dict[str, float]:
        ordered = sorted(values)
        p95_index = min(len(ordered) - 1, max(0, int(0.95 * len(ordered)) - 1))
        return {
            "median_ms": statistics.median(values),
            "min_ms": min(values),
            "p95_ms": ordered[p95_index],
        }

    return {
        "schema_version": 1,
        "workload": {
            "name": f"linear_attn_o_decode_m{rows}",
            "family": "packed_fp8_gemm",
            "phase": "decode",
            "world_size": 1,
            "distributed": False,
            "source_symbol": (
                "sglang.kernels.ops.quantization.fp8_kernel."
                "w8a8_block_fp8_matmul_deepgemm"
            ),
            "params": {"m": rows, "n": 6144, "k": 16384},
        },
        "reference": summary(reference_values),
        "reference_policy": "SGLANG_GLM52_OPT=0 production path",
        "execution_mode": "eager_cuda_event",
        "timing_contract": (
            "interleaved paired A/B; maximum CUDA-event latency across ranks"
        ),
        "candidate": {
            **summary(candidate_values),
            "path": str(
                harness.resolve() / "serving_native/candidates/reference.py"
            ),
            "speedup": 1.0,
            "passes_3pct_median_gate": False,
            "paired_p10_speedup": 1.0,
            "paired_p90_speedup": 1.0,
        },
        "paired_measurements": {
            "metric": "rank_max_cuda_event_ms",
            "samples": [
                {
                    "sample_index": index,
                    "order": (
                        ["reference", "candidate"]
                        if index % 2 == 0
                        else ["candidate", "reference"]
                    ),
                    "reference_ms": reference_values[index],
                    "candidate_ms": candidate_values[index],
                }
                for index in range(100)
            ],
        },
    }


class TestCampaignAnalyzer(unittest.TestCase):
    def test_selector_priority_is_replayed_from_predicates(self):
        predicates = {key: False for key in ANALYZER.TRACE_PREDICATE_KEYS}
        predicates.update(custom_eligible=True, pynccl_inplace_eligible=True)
        self.assertEqual(
            ANALYZER.selected_backend_from_predicates(predicates),
            "custom_all_reduce_outplace",
        )
        predicates["pynccl_symmetric_eligible"] = True
        self.assertEqual(
            ANALYZER.selected_backend_from_predicates(predicates),
            "pynccl_symmetric_inplace",
        )

    def test_graph_dispatch_comparison_ignores_eager_prewarm(self):
        graph_case = ANALYZER.CASES[0]
        eager_case = ANALYZER.CASES[2]
        self.assertFalse(
            ANALYZER.trace_record_matches_selected_execution(graph_case, False)
        )
        self.assertTrue(
            ANALYZER.trace_record_matches_selected_execution(graph_case, True)
        )
        self.assertTrue(
            ANALYZER.trace_record_matches_selected_execution(eager_case, False)
        )
        self.assertFalse(
            ANALYZER.trace_record_matches_selected_execution(eager_case, True)
        )

    def test_rank_max_and_alternating_order_are_rederived(self):
        result = {
            "raw_samples": {
                "rank_order": [0, 1, 2, 3],
                "measured_order": [
                    ["reference", "candidate"],
                    ["candidate", "reference"],
                ],
                "reference": [_sample(0, 0), _sample(1, 1)],
                "candidate": [_sample(0, 1), _sample(1, 0)],
            }
        }
        self.assertEqual(
            ANALYZER.Analyzer._sample_values(
                result, "candidate", "collective_only"
            ),
            [1.3, 1.3],
        )

        corrupted = copy.deepcopy(result)
        corrupted["raw_samples"]["candidate"][0]["ready_region"][
            "rank_max_ms"
        ] = 1.3
        with self.assertRaisesRegex(ValueError, "rank_max_ms"):
            ANALYZER.Analyzer._sample_values(
                corrupted, "candidate", "ready_region"
            )

        mismatched_target = copy.deepcopy(result)
        mismatched_target["raw_samples"]["candidate"][0][
            "scheduled_start_target_ns_by_rank"
        ][3] += 1
        with self.assertRaisesRegex(ValueError, "scheduled-start targets"):
            ANALYZER.Analyzer._sample_values(
                mismatched_target, "candidate", "ready_region"
            )

        mismatched_arrival = copy.deepcopy(result)
        mismatched_arrival["raw_samples"]["candidate"][0][
            "scheduled_start_arrival_ns_by_rank"
        ][3] -= 1
        with self.assertRaisesRegex(ValueError, "max arrival"):
            ANALYZER.Analyzer._sample_values(
                mismatched_arrival, "candidate", "ready_region"
            )

    def test_single_sided_reference_order_is_accepted(self):
        result = {
            "raw_samples": {
                "rank_order": [0, 1, 2, 3],
                "measured_order": [["reference"]],
                "reference": [_sample(0, 0)],
                "candidate": None,
            }
        }
        self.assertEqual(
            ANALYZER.Analyzer._sample_values(
                result, "reference", "ready_region"
            ),
            [1.4],
        )

        misaligned = copy.deepcopy(result)
        misaligned["raw_samples"]["reference"][0][
            "start_record_bracket_ns_by_rank"
        ] = [
            [1_000_000_000, 1_000_000_010],
            [1_000_000_100, 1_000_000_110],
            [1_000_000_200, 1_000_000_210],
            [1_000_600_001, 1_000_600_011],
        ]
        misaligned["raw_samples"]["reference"][0][
            "start_record_envelope_span_ns"
        ] = 600_011
        with self.assertRaisesRegex(ValueError, "exceeds 500000"):
            ANALYZER.Analyzer._sample_values(
                misaligned, "reference", "ready_region"
            )

    def test_reference_alias_contract_must_match_all_ranks(self):
        case = ANALYZER.CASES[0]
        contracts = []
        for rank in range(4):
            contracts.append(
                {
                    "output": {
                        "shape": [16, 6144],
                        "stride": [6144, 1],
                        "dtype": "torch.bfloat16",
                        "device": f"cuda:{rank}",
                    },
                    "output_aliases_local": rank != 3,
                    "local_poststate": "reduced" if rank != 3 else "source",
                    "source_immutable": True,
                    "exact_values": True,
                }
            )
        analyzer = ANALYZER.Analyzer(Path("/tmp/unused-tp-campaign"))
        analyzer.validate_reference_contracts(
            {"reference_contract_by_rank": contracts}, case, "fixture"
        )
        self.assertIn(
            "reference_alias_contract_differs_by_rank",
            {error["code"] for error in analyzer.errors},
        )

    def test_profile_selection_receipt_rejects_unknown_variant(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "paired").mkdir()
            (root / "paired/m16_c10d_selection.json").write_text(
                json.dumps({"selected": {"variant": "unknown"}}),
                encoding="utf-8",
            )
            analyzer = ANALYZER.Analyzer(root)
            self.assertEqual(analyzer.load_profile_selection(), {})
            self.assertIn(
                "unknown_profile_candidate",
                {error["code"] for error in analyzer.errors},
            )

    def test_c10d_resolution_is_content_and_failure_exact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "campaign"
            harness = Path(tmpdir) / "harness"
            root.mkdir()
            _write_resolution_fixture(root, harness)
            receipt = ANALYZER.resolve_c10d_abi(root, ANALYZER.CASES[0], harness)
            self.assertEqual(receipt["expected_variant"], "inplace")
            self.assertEqual(receipt["selected"]["variant"], "inplace")
            self.assertEqual(receipt["failed"]["variant"], "outplace")
            receipt_path = root / "paired/m16_c10d_selection.json"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    ANALYZER.main(
                        [
                            str(root),
                            "--resolve-c10d",
                            "m16",
                            "--harness-root",
                            str(harness),
                            "--output",
                            str(receipt_path),
                        ]
                    ),
                    0,
                )
            persisted_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertTrue(persisted_receipt["valid"])
            self.assertEqual(persisted_receipt["selected"], receipt["selected"])

            archive = Path(tmpdir) / "archive"
            shutil.copytree(root, archive)
            archived_receipt = ANALYZER.resolve_c10d_abi(
                archive, ANALYZER.CASES[0], harness
            )
            self.assertEqual(archived_receipt, receipt)

            status = root / "status.tsv"
            status.write_text(
                status.read_text(encoding="utf-8").replace(
                    "paired/m16_c10d_outplace\t1\t",
                    "paired/m16_c10d_outplace\t124\t",
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unexpected failed-attempt state"):
                ANALYZER.resolve_c10d_abi(root, ANALYZER.CASES[0], harness)

            _write_resolution_fixture(root, harness)
            failure_log = root / "paired/m16_c10d_outplace.log"
            failure_log.write_text(
                failure_log.read_text(encoding="utf-8")
                + "SharedValidationError: candidate import failed collectively: []\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unexpected shared-validation"):
                ANALYZER.resolve_c10d_abi(root, ANALYZER.CASES[0], harness)

    def test_baseline_spread_uses_strict_max_over_min_boundary(self):
        self.assertLess(
            ANALYZER.baseline_relative_run_spread([1.0, 1.01, 1.029]), 0.03
        )
        self.assertGreaterEqual(
            ANALYZER.baseline_relative_run_spread([1.0, 1.01, 1.03]), 0.03
        )
        with self.assertRaises(ValueError):
            ANALYZER.baseline_relative_run_spread([1.0, 0.0, 1.01])

    def test_producer_abi_summaries_are_rederived_from_paired_samples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "campaign"
            harness = Path(tmpdir) / "harness"
            producer = root / "producer_abi"
            producer.mkdir(parents=True)
            for rows in (16, 32):
                (producer / f"linear_attn_o_decode_m{rows}.json").write_text(
                    json.dumps(_producer_result(rows, harness)),
                    encoding="utf-8",
                )
            analyzer = ANALYZER.Analyzer(root)
            analyzer.expected_roots["kernel_harness_git"] = str(harness.resolve())
            summary = analyzer.validate_producer_abi()
            self.assertEqual(analyzer.errors, [])
            self.assertEqual(
                summary["linear_attn_o_decode_m16"]["paired_samples"], 100
            )
            self.assertEqual(
                summary["linear_attn_o_decode_m32"][
                    "derived_paired_median_speedup"
                ],
                1.0,
            )

            corrupted_path = producer / "linear_attn_o_decode_m16.json"
            corrupted = json.loads(corrupted_path.read_text(encoding="utf-8"))
            corrupted["paired_measurements"]["samples"][0]["candidate_ms"] = 2.0
            corrupted["candidate"]["speedup"] = 1.01
            corrupted_path.write_text(json.dumps(corrupted), encoding="utf-8")
            corrupted_analyzer = ANALYZER.Analyzer(root)
            corrupted_analyzer.expected_roots["kernel_harness_git"] = str(
                harness.resolve()
            )
            corrupted_analyzer.validate_producer_abi()
            self.assertTrue(
                {
                    "producer_abi_summary_mismatch",
                    "producer_abi_paired_summary_mismatch",
                }
                <= {error["code"] for error in corrupted_analyzer.errors}
            )

    def test_backend_scout_requires_exact_table_schema_and_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            log = root / "backend_scout/custom_allreduce.log"
            log.parent.mkdir()
            header = "message_bytes | " + " ".join(
                [
                    *(f"{provider}(us)" for provider in ANALYZER.BACKEND_SCOUT_PROVIDERS),
                    *(f"{provider}(GB/s)" for provider in ANALYZER.BACKEND_SCOUT_PROVIDERS),
                ]
            )
            rows = [
                f"{row_id} {message_bytes} | 1 2 3 4 | 5 6 7 8"
                for row_id, message_bytes in enumerate(
                    ANALYZER.BACKEND_SCOUT_MESSAGE_SIZES
                )
            ]
            log.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
            analyzer = ANALYZER.Analyzer(root)
            analyzer.status_rows["backend_scout/custom_allreduce"] = {
                "requirement": "required",
                "exit_code": 0,
                "line_number": 2,
                "log": str(log),
            }
            summary = analyzer.validate_backend_scout()
            self.assertEqual(analyzer.errors, [])
            self.assertEqual(
                summary["measurements"][4096]["providers"]["nccl"]["latency_us"],
                1.0,
            )

            log.write_text(
                "\n".join([header.replace("nccl(us)", "jit(us)"), *rows]) + "\n",
                encoding="utf-8",
            )
            corrupted = ANALYZER.Analyzer(root)
            corrupted.status_rows["backend_scout/custom_allreduce"] = (
                analyzer.status_rows["backend_scout/custom_allreduce"]
            )
            corrupted.validate_backend_scout()
            self.assertIn(
                "backend_scout_header_mismatch",
                {error["code"] for error in corrupted.errors},
            )

    def test_profile_exports_require_four_device_processes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            profile = root / "profile"
            profile.mkdir()
            name = "m16"
            expected_range = (
                "serving_native/tp4_allreduce_decode_m16/"
                "cuda_graph/nondefault/reference"
            )
            for suffix in (
                ".sqlite",
                ".lifecycle_cuda_api_trace.csv",
                ".measured_cuda_gpu_trace.csv",
            ):
                (profile / f"{name}{suffix}").write_text("evidence\n")
            (profile / f"{name}.measured_nvtx_pushpop_trace.csv").write_text(
                f"Name\n{expected_range}\n", encoding="utf-8"
            )
            (profile / f"{name}.lifecycle_nvtx_pushpop_trace.csv").write_text(
                "Start (ns),End (ns),Name\n"
                f"1,100,{expected_range}\n",
                encoding="utf-8",
            )
            (profile / f"{name}.measured_window.tsv").write_text(
                "range\tstart_ns\tend_ns\n"
                f"{expected_range}\t1\t100\n",
                encoding="utf-8",
            )
            api_path = profile / f"{name}.measured_cuda_api_trace.csv"
            api_path.write_text(
                "Pid,Name\n"
                + "".join(
                    f"{1000 + rank},cudaGraphLaunch_v12000\n"
                    for rank in range(4)
                    for _ in range(20)
                ),
                encoding="utf-8",
            )
            kernel_path = profile / f"{name}.measured_cuda_kern_exec_trace.csv"
            header = [
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
            rows = [
                [
                    "1",
                    "1",
                    "",
                    "",
                    "2",
                    "3",
                    "4",
                    str(1000 + rank),
                    "1",
                    str(rank),
                    "cudaGraphLaunch",
                    "1 1 1",
                    "1 1 1",
                    "cross_device_reduce_1stage",
                ]
                for rank in range(4)
                for _ in range(20)
            ]
            with kernel_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(header)
                writer.writerows(rows)
            analyzer = ANALYZER.Analyzer(root)
            summary = analyzer.validate_profile_exports(
                name,
                expected_range,
                expected_collective_launches_per_device=20,
                expected_graph_launches_per_process=20,
            )
            self.assertEqual(analyzer.errors, [])
            self.assertEqual(summary["kernels_by_device"], {0: 20, 1: 20, 2: 20, 3: 20})

            rows[20][7] = "1000"
            with kernel_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(header)
                writer.writerows(rows)
            corrupted = ANALYZER.Analyzer(root)
            corrupted.validate_profile_exports(
                name,
                expected_range,
                expected_collective_launches_per_device=20,
                expected_graph_launches_per_process=20,
            )
            self.assertIn(
                "profile_kernel_export_process_coverage_mismatch",
                {error["code"] for error in corrupted.errors},
            )

            for row in rows:
                row[-1] = "generic_kernel"
            with kernel_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(header)
                writer.writerows(rows)
            missing_collective = ANALYZER.Analyzer(root)
            missing_collective.validate_profile_exports(
                name,
                expected_range,
                expected_collective_launches_per_device=20,
                expected_graph_launches_per_process=20,
            )
            self.assertIn(
                "profile_collective_kernel_coverage_mismatch",
                {error["code"] for error in missing_collective.errors},
            )

    def test_lock_process_and_p2p_receipts_are_fail_closed(self):
        matrix = """\
 GPU0 GPU1 GPU2 GPU3
 GPU0 X OK OK OK
 GPU1 OK X OK OK
 GPU2 OK OK X OK
 GPU3 OK OK OK X
Legend:\n  OK = Status Ok
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            environment = root / "environment"
            environment.mkdir()
            lock_lines = ["CUDA_VISIBLE_DEVICES=0,1,2,3"]
            for rank in range(4):
                path = f"/home/qinhaiyan/glm52-goal-runs/locks/gpu{rank}.lock"
                lock_lines.append(
                    f"fd={9 + rank} expected={path} actual={path}"
                )
            (environment / "lock_receipt.log").write_text(
                "\n".join(lock_lines) + "\n", encoding="utf-8"
            )
            process_header = (
                "timestamp, gpu_uuid, pid, process_name, used_gpu_memory [MiB]\n"
            )
            for phase in ("before", "after"):
                (environment / f"compute_processes_{phase}.log").write_text(
                    process_header, encoding="utf-8"
                )
            (environment / "check_env.log").write_text(
                "\n".join(
                    (
                        "python:      /tmp/python",
                        "venv:        /tmp/venv",
                        "cuda home:   /usr/local/cuda",
                        "sglang:      checkout /tmp/sglang (123456789)",
                        "gpu:         NVIDIA B200 sm_100",
                        "visible gpus:4  CUDA_VISIBLE_DEVICES=0,1,2,3",
                        "torch/cuda:  2.11.0+cu130 / 13.0",
                        "M3 kernels:  present in SGLANG_DIR",
                        "Environment check passed.",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            uuids = [
                "GPU-30b619de-87f2-1862-0d07-a595da8fe417",
                "GPU-5b9be10b-5bfc-b658-9b31-f7ae8516dc54",
                "GPU-df8b1d78-b06c-39a2-54f0-66b9fabf3a99",
                "GPU-705cc39e-af1d-c4c0-d97d-4155237646af",
            ]
            inventory_header = (
                "index, uuid, name, pci.bus_id, clocks.current.sm [MHz], "
                "clocks.current.memory [MHz], power.draw [W], temperature.gpu"
            )
            inventory_rows = [
                f"{rank}, {uuid}, NVIDIA B200, 00000000:0{rank + 5}:00.0, "
                "120 MHz, 3996 MHz, 100.0 W, 30"
                for rank, uuid in enumerate(uuids)
            ]
            (environment / "nvidia_smi.log").write_text(
                "\n".join([inventory_header, *inventory_rows]) + "\n",
                encoding="utf-8",
            )
            final_header = (
                "index, uuid, clocks.current.sm [MHz], clocks.current.memory [MHz], "
                "power.draw [W], temperature.gpu"
            )
            final_rows = [
                f"{rank}, {uuid}, 120 MHz, 3996 MHz, 100.0 W, 30"
                for rank, uuid in enumerate(uuids)
            ]
            (environment / "nvidia_smi_after.log").write_text(
                "\n".join([final_header, *final_rows]) + "\n",
                encoding="utf-8",
            )
            (environment / "topology.log").write_text(
                "GPU0 GPU1 GPU2 GPU3 CPU Affinity NUMA Affinity GPU NUMA ID\n"
                "GPU0 X NV18 NV18 NV18 0-119 0 N/A\n"
                "GPU1 NV18 X NV18 NV18 0-119 0 N/A\n"
                "GPU2 NV18 NV18 X NV18 0-119 0 N/A\n"
                "GPU3 NV18 NV18 NV18 X 0-119 0 N/A\n",
                encoding="utf-8",
            )
            status_lines = []
            before_lines = []
            after_lines = []
            for rank, uuid in enumerate(uuids):
                gpu_header = f"GPU {rank}: NVIDIA B200 (UUID: {uuid})"
                status_lines.append(gpu_header)
                before_lines.append(gpu_header)
                after_lines.append(gpu_header)
                for link in range(18):
                    status_lines.append(f"Link {link}: 53.125 GB/s")
                    before_lines.extend(
                        (
                            f"Link {link}: Data Tx: {1000 + link} KiB",
                            f"Link {link}: Data Rx: {2000 + link} KiB",
                        )
                    )
                    after_lines.extend(
                        (
                            f"Link {link}: Data Tx: {1100 + link} KiB",
                            f"Link {link}: Data Rx: {2100 + link} KiB",
                        )
                    )
            (environment / "nvlink_status.log").write_text(
                "\n".join(status_lines) + "\n", encoding="utf-8"
            )
            (environment / "nvlink_throughput_before.log").write_text(
                "\n".join(before_lines) + "\n", encoding="utf-8"
            )
            (environment / "nvlink_throughput_after.log").write_text(
                "\n".join(after_lines) + "\n", encoding="utf-8"
            )
            p2p_text = "".join(
                f"capability={capability}\n{matrix}" for capability in "rwn"
            )
            (environment / "p2p_capability.log").write_text(
                p2p_text, encoding="utf-8"
            )
            analyzer = ANALYZER.Analyzer(root)
            summary = analyzer.validate_environment_evidence()
            self.assertEqual(analyzer.errors, [])
            self.assertEqual(
                summary["p2p_full_mesh_directed_ok_edges"],
                {"r": 12, "w": 12, "n": 12},
            )

            with (environment / "compute_processes_after.log").open(
                "a", encoding="utf-8"
            ) as output:
                output.write(
                    "2026/07/22 00:00:00.000, GPU-test, 123, python, 1 MiB\n"
                )
            corrupted = ANALYZER.Analyzer(root)
            corrupted.validate_environment_evidence()
            self.assertIn(
                "unexpected_compute_process",
                {error["code"] for error in corrupted.errors},
            )

            (environment / "compute_processes_after.log").write_text(
                process_header, encoding="utf-8"
            )
            stalled_after = list(after_lines)
            stalled_after[1:37] = before_lines[1:37]
            (environment / "nvlink_throughput_after.log").write_text(
                "\n".join(stalled_after) + "\n", encoding="utf-8"
            )
            stalled_rank = ANALYZER.Analyzer(root)
            stalled_rank.validate_environment_evidence()
            self.assertIn(
                "nvlink_counter_delta_missing",
                {error["code"] for error in stalled_rank.errors},
            )
            (environment / "nvlink_throughput_after.log").write_text(
                "\n".join(after_lines) + "\n", encoding="utf-8"
            )

            duplicate_status_lines = list(status_lines)
            duplicate_status_lines.insert(2, "Link 0: 53.125 GB/s")
            (environment / "nvlink_status.log").write_text(
                "\n".join(duplicate_status_lines) + "\n", encoding="utf-8"
            )
            duplicate_status = ANALYZER.Analyzer(root)
            duplicate_status.validate_environment_evidence()
            self.assertIn(
                "nvlink_duplicate_status",
                {error["code"] for error in duplicate_status.errors},
            )
            (environment / "nvlink_status.log").write_text(
                "\n".join(status_lines) + "\n", encoding="utf-8"
            )

            (environment / "nvidia_smi_after.log").write_text(
                "\n".join(
                    [final_header, final_rows[0].replace("100.0 W", "nan W"), *final_rows[1:]]
                )
                + "\n",
                encoding="utf-8",
            )
            nonfinite_power = ANALYZER.Analyzer(root)
            nonfinite_power.validate_environment_evidence()
            self.assertIn(
                "gpu_inventory_contract_mismatch",
                {error["code"] for error in nonfinite_power.errors},
            )
            (environment / "nvidia_smi_after.log").write_text(
                "\n".join([final_header, *final_rows]) + "\n",
                encoding="utf-8",
            )

            (environment / "p2p_capability.log").write_text(
                p2p_text.replace("GPU0 X OK OK OK", "GPU0 X NS OK OK", 1),
                encoding="utf-8",
            )
            disabled_edge = ANALYZER.Analyzer(root)
            disabled_edge.validate_environment_evidence()
            self.assertIn(
                "p2p_capability_not_full_mesh",
                {error["code"] for error in disabled_edge.errors},
            )

            (environment / "p2p_capability.log").write_text(
                p2p_text + "GPU0 X OK OK OK\n", encoding="utf-8"
            )
            duplicate_rank = ANALYZER.Analyzer(root)
            duplicate_rank.validate_environment_evidence()
            self.assertIn(
                "p2p_capability_duplicate_rank",
                {error["code"] for error in duplicate_rank.errors},
            )

            (environment / "p2p_capability.log").write_text(
                p2p_text.replace(" GPU0 GPU1 GPU2 GPU3\n", "", 1),
                encoding="utf-8",
            )
            missing_header = ANALYZER.Analyzer(root)
            missing_header.validate_environment_evidence()
            self.assertIn(
                "p2p_capability_header_mismatch",
                {error["code"] for error in missing_header.errors},
            )

            (environment / "p2p_capability.log").write_text(
                p2p_text + f"capability=n\n{matrix}", encoding="utf-8"
            )
            duplicate_capability = ANALYZER.Analyzer(root)
            duplicate_capability.validate_environment_evidence()
            self.assertIn(
                "p2p_capability_section_mismatch",
                {error["code"] for error in duplicate_capability.errors},
            )

            (environment / "p2p_capability.log").write_text(
                p2p_text.replace(
                    "GPU0 X OK OK OK", "GPU0 X OK OK OK EXTRA", 1
                ),
                encoding="utf-8",
            )
            malformed_row = ANALYZER.Analyzer(root)
            malformed_row.validate_environment_evidence()
            self.assertIn(
                "p2p_capability_malformed_rank_row",
                {error["code"] for error in malformed_row.errors},
            )

            (environment / "p2p_capability.log").write_text(
                p2p_text + f"capability=x\n{matrix}", encoding="utf-8"
            )
            unknown_capability = ANALYZER.Analyzer(root)
            unknown_capability.validate_environment_evidence()
            self.assertIn(
                "p2p_capability_section_mismatch",
                {error["code"] for error in unknown_capability.errors},
            )

    def test_campaign_freezes_gpu_local_size_default_and_checks_idle_first(self):
        source = (HERE / "run_locked_tp4_campaign.sh").read_text(encoding="utf-8")
        self.assertNotIn("export LOCAL_SIZE", source)
        self.assertIn("unset \\\n  LOCAL_SIZE", source)
        self.assertIn("physical GPUs are busy despite a valid scheduler lock receipt", source)
        self.assertLess(
            source.index("compute_snapshot="),
            source.index("mkdir -p"),
        )
        self.assertLess(
            source.index("environment/compute_processes_before"),
            source.index("environment/check_env"),
        )

    def test_help_is_successful(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(ANALYZER.main(["--help"]), 0)
        self.assertIn("campaign_root", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
