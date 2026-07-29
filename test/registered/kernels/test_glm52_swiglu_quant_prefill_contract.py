"""CPU-only contract tests for the default-off Task-29 prefill route."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.layers.glm52_opt import swiglu_quant_prefill


def _fake_tensor(shape, stride, dtype, device="cuda:0"):
    return SimpleNamespace(
        is_cuda=True,
        dtype=dtype,
        shape=shape,
        stride=lambda: stride,
        storage_offset=lambda: 0,
        device=torch.device(device),
    )


def _dispatch_kwargs():
    return {
        "group_size": 128,
        "swiglu_limit": None,
        "swizzle": False,
        "gemm1_alpha": None,
        "gemm1_clamp_limit": None,
        "column_major_scales": True,
        "scale_tma_aligned": True,
        "scale_ue8m0": True,
        "pdl": True,
    }


class TestGlm52SwigluQuantPrefillContract(unittest.TestCase):
    def setUp(self):
        self.old_state = swiglu_quant_prefill._STATE

    def tearDown(self):
        swiglu_quant_prefill._STATE = self.old_state

    def test_import_is_cpu_only_and_default_off(self):
        script = r"""
import os
import torch

def forbidden(*args, **kwargs):
    raise AssertionError("CUDA queried during Task-29 import")

torch.cuda.current_device = forbidden
torch.cuda.get_device_capability = forbidden
before = dict(os.environ)
import sglang.srt.layers.glm52_opt.swiglu_quant_prefill as module
assert module.dispatch_state()["reason"] == "default_off"
assert before == dict(os.environ)
"""
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = ""
        subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            env=environment,
        )

    def test_fixed_abi_and_bounded_portfolio(self):
        self.assertEqual(swiglu_quant_prefill.EXPERTS, 32)
        self.assertEqual(swiglu_quant_prefill.ALIGNED_M, 35200)
        self.assertEqual(swiglu_quant_prefill.VALID_M, 32982)
        self.assertEqual(swiglu_quant_prefill.GATE_UP, 4096)
        self.assertEqual(swiglu_quant_prefill.HIDDEN, 2048)
        self.assertEqual(swiglu_quant_prefill.GROUP_SIZE, 128)
        self.assertEqual(swiglu_quant_prefill.PACKED_WORDS, 4)
        self.assertEqual(
            swiglu_quant_prefill.TRITON_VARIANTS,
            {
                "endpoint512_w4": (512, 4),
                "endpoint1024_w8": (1024, 8),
                "endpoint2048_w8": (2048, 8),
            },
        )
        self.assertEqual(
            swiglu_quant_prefill.CUDA_VARIANTS,
            {
                "cuda_s8_v16_b128",
                "cuda_s16_v8_b256",
                "cuda_s8_v16_b128_cgld",
            },
        )

    def test_disabled_dispatch_returns_before_contract_or_launch(self):
        swiglu_quant_prefill._STATE = swiglu_quant_prefill._DispatchState(
            False, "test_default_off"
        )
        malformed = SimpleNamespace(shape=(35200, 4096))
        with patch.object(
            swiglu_quant_prefill,
            "_eligibility_error",
            side_effect=AssertionError("disabled dispatch inspected ABI"),
        ), patch.object(
            swiglu_quant_prefill,
            "silu_mul_quant_packed_explicit",
            side_effect=AssertionError("disabled dispatch launched"),
        ):
            self.assertIsNone(
                swiglu_quant_prefill.maybe_silu_mul_quant_packed(
                    malformed,
                    None,
                    None,
                    **_dispatch_kwargs(),
                )
            )

    def test_armed_decode_and_unrelated_contiguous_shapes_remain_stock(self):
        swiglu_quant_prefill._STATE = swiglu_quant_prefill._DispatchState(
            True,
            "ready",
            variant="cuda_s8_v16_b128_cgld",
            gpu_id=0,
            graph_mode=False,
        )
        with patch.object(
            swiglu_quant_prefill,
            "_eligibility_error",
            side_effect=AssertionError("non-target shape inspected ABI"),
        ), patch.object(
            swiglu_quant_prefill,
            "silu_mul_quant_packed_explicit",
            side_effect=AssertionError("non-target shape launched"),
        ):
            for shape in ((32, 4096), (1024, 4096)):
                gateup = _fake_tensor(
                    shape,
                    (4096, 1),
                    torch.bfloat16,
                )
                self.assertIsNone(
                    swiglu_quant_prefill.maybe_silu_mul_quant_packed(
                        gateup,
                        None,
                        None,
                        **_dispatch_kwargs(),
                    )
                )

    def test_selected_exact_bucket_abi_error_aborts_without_stock_retry(self):
        swiglu_quant_prefill._STATE = swiglu_quant_prefill._DispatchState(
            True,
            "ready",
            variant="cuda_s8_v16_b128_cgld",
            gpu_id=0,
            graph_mode=False,
        )
        gateup = _fake_tensor(
            (35200, 4096),
            (4096, 1),
            torch.bfloat16,
        )
        rowmap = _fake_tensor((35200,), (1,), torch.int32)
        endpoint = _fake_tensor((32,), (1,), torch.int32)
        with patch.object(
            swiglu_quant_prefill,
            "_eligibility_error",
            return_value="forced exact-bucket ABI failure",
        ), patch.object(
            swiglu_quant_prefill,
            "silu_mul_quant_packed_explicit",
            side_effect=AssertionError("failed exact bucket launched"),
        ), self.assertRaisesRegex(
            RuntimeError, "forced exact-bucket ABI failure"
        ):
            swiglu_quant_prefill.maybe_silu_mul_quant_packed(
                gateup,
                rowmap,
                endpoint,
                **_dispatch_kwargs(),
            )

    def test_selected_exact_bucket_missing_metadata_aborts(self):
        swiglu_quant_prefill._STATE = swiglu_quant_prefill._DispatchState(
            True,
            "ready",
            variant="cuda_s8_v16_b128_cgld",
            gpu_id=0,
            graph_mode=False,
        )
        gateup = _fake_tensor(
            (35200, 4096),
            (4096, 1),
            torch.bfloat16,
        )
        endpoint = _fake_tensor((32,), (1,), torch.int32)
        with self.assertRaisesRegex(RuntimeError, "no production row map"):
            swiglu_quant_prefill.maybe_silu_mul_quant_packed(
                gateup,
                None,
                endpoint,
                **_dispatch_kwargs(),
            )

    def test_startup_requires_exact_topology_and_eager_prefill(self):
        exact = SimpleNamespace(**swiglu_quant_prefill.REQUIRED_TOPOLOGY)
        topology, graph_mode = (
            swiglu_quant_prefill._validate_startup_topology(exact)
        )
        self.assertEqual(topology, swiglu_quant_prefill.REQUIRED_TOPOLOGY)
        self.assertFalse(graph_mode)

        wrong = dict(swiglu_quant_prefill.REQUIRED_TOPOLOGY)
        wrong["ep_size"] = 4
        with self.assertRaisesRegex(RuntimeError, "exact TP8/DP8/EP8"):
            swiglu_quant_prefill._validate_startup_topology(
                SimpleNamespace(**wrong)
            )

        graph = SimpleNamespace(
            **swiglu_quant_prefill.REQUIRED_TOPOLOGY,
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend="piecewise")
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "frozen as eager"):
            swiglu_quant_prefill._validate_startup_topology(graph)

    def test_runner_substitutes_only_activation_between_w13_and_w2(self):
        source = (
            Path(__file__).parents[3]
            / "python/sglang/srt/layers/moe/moe_runner/deep_gemm.py"
        ).read_text()
        contiguous = source[source.index("    def _run_contiguous_gemm(") :]
        contiguous = contiguous[
            : contiguous.index("    def _run_bf16_contiguous_gemm(")
        ]
        task29_call = contiguous.index("maybe_silu_mul_quant_packed(")
        stock_activation = contiguous.index(
            "silu_and_mul_contig_post_quant("
        )
        w2_call = contiguous.index("try_dispatch_w2_prefill_stock(")
        self.assertLess(task29_call, stock_activation)
        self.assertLess(stock_activation, w2_call)
        self.assertIn("runner_input.expert_start_loc", contiguous)
        self.assertNotIn(".cpu()", contiguous)
        self.assertNotIn(".tolist()", contiguous)


if __name__ == "__main__":
    unittest.main()
