"""External GLM-5.2 attention-O fixed-N/K acceptance gate.

This test intentionally covers non-speculative decode only.  Task 27 does not
authorize TARGET_VERIFY, so no EAGLE/MTP arguments belong in this lane.

The global benchmark batches are 128 and 256 because DP8 must produce local
decode M=16 and M=32.  The generated decode traces must be retained and audited
to confirm those local buckets and the fixed N=6144/K=16384 candidate kernel.
"""

import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings

GLM_52_FP8_MODEL_PATH = "zai-org/GLM-5.2-FP8"

COMMON_ARGS = [
    "--trust-remote-code",
    "--reasoning-parser=glm45",
    "--tool-call-parser=glm47",
    "--mem-fraction-static=0.85",
    "--enable-metrics",
    "--dp=8",
    "--ep=8",
    "--enable-dp-attention",
    "--max-running-requests=256",
    "--cuda-graph-bs-decode",
    "16",
    "32",
]

COMMON_ENV = {
    "SGLANG_ENABLE_JIT_DEEPGEMM": "1",
    "SGLANG_GLM52_OPT": "0",
    "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "1",
}


def _model(*, candidate: bool, variant: str) -> ModelLaunchSettings:
    return ModelLaunchSettings(
        GLM_52_FP8_MODEL_PATH,
        tp_size=8,
        extra_args=COMMON_ARGS,
        env={
            **COMMON_ENV,
            "SGLANG_OPT_GLM52_ATTN_O_DECODE_DIRECT_NK": (
                "1" if candidate else "0"
            ),
        },
        variant=variant,
    )


class TestGlm52AttnODecodeDirectNk(unittest.TestCase):
    """Checkpoint-backed TP8/DP8/EP8 gate; production remains off until pass."""

    def test_a_non_mtp_accuracy(self):
        result = run_combined_tests(
            models=[
                _model(candidate=False, variant="stock-accuracy"),
                _model(candidate=True, variant="candidate-accuracy"),
            ],
            test_name="GLM-5.2-FP8-attn-o-direct-nk-non-mtp-accuracy",
            accuracy_params=AccuracyTestParams(
                dataset="gsm8k",
                baseline_accuracy=0.92,
            ),
        )
        self.assertTrue(result["all_passed"], result)

    def test_b_non_mtp_alternating_performance(self):
        # Three independent server-level AB/BA series on the same eight-GPU
        # node.  Each launch profiles global B128/B256, corresponding to local
        # DP8 decode M16/M32.
        result = run_combined_tests(
            models=[
                _model(candidate=False, variant="stock-s1"),
                _model(candidate=True, variant="candidate-s1"),
                _model(candidate=True, variant="candidate-s2"),
                _model(candidate=False, variant="stock-s2"),
                _model(candidate=False, variant="stock-s3"),
                _model(candidate=True, variant="candidate-s3"),
            ],
            test_name="GLM-5.2-FP8-attn-o-direct-nk-non-mtp-performance",
            performance_params=PerformanceTestParams(
                batch_sizes=[128, 256],
                input_lens=(8192,),
                output_lens=(512,),
                profile_dir="performance_profiles_glm_52_attn_o_direct_nk",
            ),
        )
        self.assertTrue(result["all_passed"], result)

        by_variant = {entry["variant"]: entry for entry in result["results"]}
        for series in range(1, 4):
            stock = by_variant[f"stock-s{series}"]["perf_result"]
            candidate = by_variant[f"candidate-s{series}"]["perf_result"]
            stock_by_batch = {
                row.batch_size: row for row in stock.benchmark_results
            }
            candidate_by_batch = {
                row.batch_size: row for row in candidate.benchmark_results
            }
            self.assertEqual(stock_by_batch.keys(), candidate_by_batch.keys())
            for batch_size in (128, 256):
                ratio = (
                    candidate_by_batch[batch_size].overall_throughput
                    / stock_by_batch[batch_size].overall_throughput
                )
                self.assertGreaterEqual(
                    ratio,
                    1.0,
                    (
                        f"server regression in series {series}, "
                        f"global batch {batch_size}: {ratio:.6f}x"
                    ),
                )


if __name__ == "__main__":
    unittest.main()
