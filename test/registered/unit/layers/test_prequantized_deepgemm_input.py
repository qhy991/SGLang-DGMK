import types
import unittest
from unittest import mock

import torch

import sglang.srt.layers.quantization.fp8 as fp8_module
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.srt.layers.quantization.fp8_utils import (
    deepgemm_w8a8_block_fp8_linear_with_fallback,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _packed_scale(rows: int, groups: int) -> torch.Tensor:
    aligned_rows = (rows + 3) // 4 * 4
    aligned_groups = (groups + 3) // 4 * 4
    return torch.empty(
        (aligned_groups // 4, aligned_rows), dtype=torch.int32
    ).transpose(0, 1)[:rows]


class TestPrequantizedDeepGemmInput(CustomTestCase):
    @staticmethod
    def _method(callable_):
        method = object.__new__(Fp8LinearMethod)
        method.use_marlin = False
        method.use_mxfp8 = False
        method.block_quant = True
        method.weight_block_size = [128, 128]
        method.w8a8_block_fp8_linear = callable_
        return method

    @staticmethod
    def _layer():
        return types.SimpleNamespace(
            prefix="model.layers.1.self_attn.fused_qkv_a_proj_with_mqa",
            weight=torch.empty((64, 128), dtype=torch.float8_e4m3fn),
            weight_scale_inv=torch.empty((64, 1), dtype=torch.int32),
        )

    def test_marked_tuple_uses_prequantized_deepgemm_abi(self):
        calls = []

        def fake_deepgemm(**kwargs):
            calls.append(kwargs)
            return torch.empty((2, 64), dtype=torch.bfloat16)

        method = self._method(fake_deepgemm)
        xq = torch.empty((2, 128), dtype=torch.float8_e4m3fn)
        xs = _packed_scale(2, 1)
        bf16 = torch.empty((2, 128), dtype=torch.bfloat16)
        setattr(bf16, "_sglang_dsa_bf16_passthrough", True)

        with mock.patch.object(
            fp8_module,
            "deepgemm_w8a8_block_fp8_linear_with_fallback",
            fake_deepgemm,
        ):
            method.apply(self._layer(), (xq, xs, bf16))

        self.assertIs(calls[0]["input"], xq)
        self.assertIs(calls[0]["input_scale"], xs)

    def test_marked_tuple_falls_back_to_exact_bf16_for_other_backend(self):
        calls = []

        def fake_backend(**kwargs):
            calls.append(kwargs)
            return torch.empty((2, 64), dtype=torch.bfloat16)

        method = self._method(fake_backend)
        xq = torch.empty((2, 128), dtype=torch.float8_e4m3fn)
        xs = _packed_scale(2, 1)
        bf16 = torch.empty((2, 128), dtype=torch.bfloat16)
        setattr(bf16, "_sglang_dsa_bf16_passthrough", True)

        method.apply(self._layer(), (xq, xs, bf16))

        self.assertIs(calls[0]["input"], bf16)
        self.assertIsNone(calls[0]["input_scale"])

    def test_wrapper_skips_quant_for_valid_packed_input(self):
        captured = []
        xq = torch.empty((2, 128), dtype=torch.float8_e4m3fn)
        xs = _packed_scale(2, 1)
        weight = torch.empty((64, 128), dtype=torch.float8_e4m3fn)
        weight_scale = torch.empty((64, 1), dtype=torch.int32)
        expected = torch.arange(128, dtype=torch.bfloat16).view(2, 64)

        def fake_dispatch(*args, **kwargs):
            captured.append((args, kwargs))
            return expected

        with mock.patch(
            "sglang.srt.layers.glm52_opt.dispatch.try_dispatch_fp8_gemm",
            side_effect=fake_dispatch,
        ):
            output = deepgemm_w8a8_block_fp8_linear_with_fallback(
                xq,
                weight,
                [128, 128],
                weight_scale,
                input_scale=xs,
            )

        self.assertTrue(torch.equal(output, expected))
        self.assertEqual(output.data_ptr(), expected.data_ptr())
        dispatched_input = captured[0][0][0]
        dispatched_scale = captured[0][0][2]
        self.assertEqual(dispatched_input.data_ptr(), xq.data_ptr())
        self.assertEqual(dispatched_input.shape, xq.shape)
        self.assertIs(dispatched_input.dtype, xq.dtype)
        self.assertEqual(dispatched_scale.data_ptr(), xs.data_ptr())
        self.assertEqual(dispatched_scale.shape, xs.shape)
        self.assertEqual(dispatched_scale.stride(), xs.stride())
        self.assertIs(captured[0][0][5], torch.bfloat16)

    def test_wrapper_rejects_compacted_packed_scale(self):
        xq = torch.empty((2, 128), dtype=torch.float8_e4m3fn)
        compact = torch.empty((2, 1), dtype=torch.int32)
        weight = torch.empty((64, 128), dtype=torch.float8_e4m3fn)
        weight_scale = torch.empty((64, 1), dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "packed MN-major"):
            deepgemm_w8a8_block_fp8_linear_with_fallback(
                xq,
                weight,
                [128, 128],
                weight_scale,
                input_scale=compact,
            )


if __name__ == "__main__":
    unittest.main()
