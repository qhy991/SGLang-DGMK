#!/usr/bin/env python3
"""CPU-only corruption tests for the TP4 campaign analyzer."""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
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

    def test_unknown_profile_candidate_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "profile").mkdir()
            (root / "profile/c10d_profile_selection.tsv").write_text(
                "m16\t/tmp/not-a-candidate.py\t/tmp/m16_c10d_inplace.json\n",
                encoding="utf-8",
            )
            analyzer = ANALYZER.Analyzer(root)
            self.assertEqual(analyzer.load_profile_selection(), {})
            self.assertIn(
                "unknown_profile_candidate",
                {error["code"] for error in analyzer.errors},
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
