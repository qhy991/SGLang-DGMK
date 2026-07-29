import sys
import unittest
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "benchmark"
sys.path.insert(0, str(BENCHMARK_DIR))

from assert_flashmla_glm52_nsys_dispatch import (  # noqa: E402
    FLAT_KERNEL,
    analyze_dispatch,
)
from flashmla_glm52_contract import (  # noqa: E402
    DISPATCH_BUILD_DISABLED,
    DISPATCH_HIT,
    DISPATCH_MISS,
    expected_dispatch_state,
    flat_token_byte_offset,
    generic_token_byte_offset,
    production_abi_failures,
)


def dispatch_inputs(**overrides):
    inputs = {
        "original_h_q": 64,
        "kernel_h_q": 64,
        "s_q": 1,
        "d_qk": 576,
        "topk": 2048,
        "page_block_size": 64,
        "stride_kv_row": 656,
        "stride_kv_block": 64 * 656,
        "extra_topk": 0,
        "has_topk_length": False,
        "has_extra_topk_length": False,
        "has_attn_sink": False,
    }
    inputs.update(overrides)
    return inputs


def production_abi(**overrides):
    batch_size = 16
    valid_lengths = [
        512 + round(index * 63 / (batch_size - 1))
        for index in range(batch_size)
    ]
    record = {
        "batch_size": batch_size,
        "local_heads_q": 8,
        "padded_heads_q": 64,
        "page_size": 64,
        "topk": 2048,
        "head_dim_qk": 576,
        "head_dim_v": 512,
        "packed_row_bytes": 656,
        "zero_padded_heads_all_zero": True,
        "topk_length_present": False,
        "extra_topk_length_present": False,
        "attn_sink_present": False,
        "extra_kv_present": False,
        "invalid_indices_all_minus_one": True,
        "physical_indices_within_batch_rows": True,
        "valid_lengths": valid_lengths,
        "cache_seqlens_values": valid_lengths,
        "valid_index_count": sum(valid_lengths),
        "q": {
            "shape": [batch_size, 1, 64, 576],
            "dtype": "torch.bfloat16",
            "stride": [64 * 576, 64 * 576, 576, 1],
        },
        "packed_kv": {
            "shape": [144, 64, 1, 656],
            "dtype": "torch.float8_e4m3fn",
            "stride": [64 * 656, 656, 656, 1],
        },
        "indices": {
            "shape": [batch_size, 1, 2048],
            "dtype": "torch.int32",
            "stride": [2048, 2048, 1],
        },
        "block_table": {
            "shape": [batch_size, 0],
            "dtype": "torch.int32",
            "stride": [1, 1],
        },
        "cache_seqlens": {
            "shape": [batch_size],
            "dtype": "torch.int32",
            "stride": [1],
        },
    }
    record.update(overrides)
    return record


class DispatchContractTest(unittest.TestCase):
    def test_flat_offset_matches_generic_beyond_int32_product_boundary(self):
        for token_index in (0, 63, 64, 3_273_603, 3_273_604, 2_147_483_647):
            with self.subTest(token_index=token_index):
                generic = generic_token_byte_offset(token_index)
                flat = flat_token_byte_offset(token_index)
                self.assertEqual(flat, generic)
                if token_index >= 3_273_604:
                    self.assertGreater(flat, 2_147_483_647)

    def test_native_h64_exact_production_shape_hits(self):
        self.assertEqual(
            expected_dispatch_state(build_enabled=True, **dispatch_inputs()),
            DISPATCH_HIT,
        )

    def test_original_h128_adapter_misses_after_kernel_h_q_rewrite(self):
        self.assertEqual(
            expected_dispatch_state(
                build_enabled=True,
                **dispatch_inputs(original_h_q=128, kernel_h_q=64),
            ),
            DISPATCH_MISS,
        )

    def test_build_disabled_and_each_guard_miss(self):
        self.assertEqual(
            expected_dispatch_state(build_enabled=False, **dispatch_inputs()),
            DISPATCH_BUILD_DISABLED,
        )
        misses = (
            {"s_q": 2},
            {"d_qk": 512},
            {"topk": 128},
            {"page_block_size": 128},
            {"stride_kv_row": 657},
            {"stride_kv_block": 65 * 656},
            {"extra_topk": 64},
            {"has_topk_length": True},
            {"has_extra_topk_length": True},
            {"has_attn_sink": True},
        )
        for override in misses:
            with self.subTest(override=override):
                self.assertEqual(
                    expected_dispatch_state(
                        build_enabled=True, **dispatch_inputs(**override)
                    ),
                    DISPATCH_MISS,
                )


class ProductionAbiContractTest(unittest.TestCase):
    def test_exact_tp8_production_abi_passes(self):
        self.assertEqual(production_abi_failures(production_abi()), [])

    def test_nonempty_block_table_is_rejected(self):
        record = production_abi()
        record["block_table"] = {
            "shape": [16, 9],
            "dtype": "torch.int32",
            "stride": [9, 1],
        }
        self.assertIn(
            "block_table.shape: [16, 9] != [16, 0]",
            production_abi_failures(record),
        )

    def test_unpadded_heads_and_wrong_kv_stride_are_rejected(self):
        record = production_abi(
            local_heads_q=64,
            zero_padded_heads_all_zero=False,
        )
        record["packed_kv"] = {
            "shape": [144, 64, 1, 656],
            "dtype": "torch.float8_e4m3fn",
            "stride": [65 * 656, 656, 656, 1],
        }
        failures = production_abi_failures(record)
        self.assertIn("local_heads_q: 64 != 8", failures)
        self.assertIn("zero_padded_heads_all_zero: False != True", failures)
        self.assertIn(
            f"packed_kv.stride: {[65 * 656, 656, 656, 1]!r} != "
            f"{[64 * 656, 656, 656, 1]!r}",
            failures,
        )


class NsysDispatchEvidenceTest(unittest.TestCase):
    def test_native_h64_named_flat_launches(self):
        evidence = analyze_dispatch(
            input_record={
                "dispatch": {"state": "hit"},
                "shape": {"h_q": 64},
                "profile": {"region": "graph", "iterations": 5},
            },
            csv_text=(
                '"Time (%)","Instances","Name"\n'
                f'"100.0","5","void {FLAT_KERNEL}<types>()"\n'
            ),
            expected_state="hit",
            expected_h_q=64,
            expected_region="graph",
            expected_iterations=5,
            expected_flat=5,
            expected_generic=0,
        )
        self.assertEqual(evidence["verdict"], "PASS")

    def test_original_h128_only_launches_two_generic_halves(self):
        evidence = analyze_dispatch(
            input_record={
                "dispatch": {"state": "miss"},
                "shape": {"h_q": 128},
                "profile": {"region": "operator", "iterations": 5},
            },
            csv_text=(
                '"Time (%)","Instances","Name"\n'
                '"100.0","10","void '
                'flash_fwd_splitkv_mla_fp8_sparse_kernel<types>()"\n'
            ),
            expected_state="miss",
            expected_h_q=128,
            expected_region="operator",
            expected_iterations=5,
            expected_flat=0,
            expected_generic=10,
        )
        self.assertEqual(evidence["verdict"], "PASS")

    def test_launch_count_mismatch_fails(self):
        evidence = analyze_dispatch(
            input_record={
                "dispatch": {"state": "miss"},
                "shape": {"h_q": 128},
                "profile": {"region": "operator", "iterations": 5},
            },
            csv_text=(
                '"Instances","Name"\n'
                '"5","flash_fwd_splitkv_mla_fp8_sparse_kernel<types>()"\n'
            ),
            expected_state="miss",
            expected_h_q=128,
            expected_region="operator",
            expected_iterations=5,
            expected_flat=0,
            expected_generic=10,
        )
        self.assertEqual(evidence["verdict"], "FAIL")
        self.assertIn("generic_instances: 5 != 10", evidence["failures"])

    def test_profile_region_mismatch_fails(self):
        evidence = analyze_dispatch(
            input_record={
                "dispatch": {"state": "hit"},
                "shape": {"h_q": 64},
                "profile": {"region": "containing", "iterations": 5},
            },
            csv_text=(
                '"Instances","Name"\n'
                f'"5","{FLAT_KERNEL}<types>()"\n'
            ),
            expected_state="hit",
            expected_h_q=64,
            expected_region="graph",
            expected_iterations=5,
            expected_flat=5,
            expected_generic=0,
        )
        self.assertEqual(evidence["verdict"], "FAIL")
        self.assertIn(
            "profile_region: 'containing' != 'graph'",
            evidence["failures"],
        )


if __name__ == "__main__":
    unittest.main()
