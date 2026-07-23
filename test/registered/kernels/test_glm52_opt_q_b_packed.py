"""CPU-only routing tests for the GLM-5.2 q_b packed-scale path."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.glm52_opt import fp8_gemm


def _inputs(*, mixed_scales: bool = False):
    x = torch.empty((16, 8), dtype=torch.uint8)
    w = torch.empty((16, 8), dtype=torch.uint8)
    x_scale = torch.empty((16, 1), dtype=torch.int32)
    w_dtype = torch.float32 if mixed_scales else torch.int32
    w_scale = torch.empty((16, 1), dtype=w_dtype)
    out = torch.empty((16, 16), dtype=torch.bfloat16)
    return x, w, x_scale, w_scale, out


class QBScaleRoutingTest(unittest.TestCase):
    def test_q_b_packed_pair_uses_native_entry(self):
        calls = []
        with (
            patch.object(fp8_gemm, "has_packed_warp_fp8_gemm_nt", return_value=True),
            patch.object(
                fp8_gemm,
                "_run_q_b_packed_warp",
                side_effect=lambda *args: calls.append(args),
            ),
        ):
            ok, path = fp8_gemm.run_fp8_gemm(
                "q_b_proj", *_inputs(), [128, 128], "unused", phase="decode"
            )

        self.assertTrue(ok)
        self.assertEqual(path, "native_packed_warp")
        self.assertEqual(len(calls), 1)

    def test_q_b_missing_overlay_falls_back_to_stock(self):
        with patch.object(fp8_gemm, "has_packed_warp_fp8_gemm_nt", return_value=False):
            ok, path = fp8_gemm.run_fp8_gemm(
                "q_b_proj", *_inputs(), [128, 128], "unused", phase="decode"
            )

        self.assertFalse(ok)
        self.assertEqual(path, "packed_warp_unavailable")

    def test_q_b_mixed_scale_abi_falls_back_to_stock(self):
        ok, path = fp8_gemm.run_fp8_gemm(
            "q_b_proj",
            *_inputs(mixed_scales=True),
            [128, 128],
            "unused",
            phase="decode",
        )

        self.assertFalse(ok)
        self.assertEqual(path, "packed_abi_mixed_dtypes")

    def test_q_b_overlay_error_falls_back_to_stock(self):
        with (
            patch.object(fp8_gemm, "has_packed_warp_fp8_gemm_nt", return_value=True),
            patch.object(
                fp8_gemm,
                "_run_q_b_packed_warp",
                side_effect=RuntimeError("test failure"),
            ),
        ):
            ok, path = fp8_gemm.run_fp8_gemm(
                "q_b_proj", *_inputs(), [128, 128], "unused", phase="decode"
            )

        self.assertFalse(ok)
        self.assertEqual(path, "packed_warp_error")


if __name__ == "__main__":
    unittest.main()
