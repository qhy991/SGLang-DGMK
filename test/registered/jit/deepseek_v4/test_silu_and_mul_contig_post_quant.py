from __future__ import annotations

import unittest

import torch
from sgl_kernel import silu_and_mul

from sglang.jit_kernel.dsv4 import silu_and_mul_contig_post_quant
from sglang.kernels.ops.quantization.fp8_kernel import (
    create_per_token_group_quant_fp8_output_scale,
    sglang_per_token_group_quant_fp8,
)
from sglang.test.ci.ci_register import register_cuda_ci


register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")

GROUP_SIZE = 128
HIDDEN_SIZE = 2048


def _allocate_output(m: int) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (m, HIDDEN_SIZE)
    output = torch.empty(shape, device="cuda", dtype=torch.float8_e4m3fn)
    output_scale = create_per_token_group_quant_fp8_output_scale(
        x_shape=shape,
        device="cuda",
        group_size=GROUP_SIZE,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )
    return output, output_scale


class TestSiluAndMulContigPostQuant(unittest.TestCase):
    def test_bf16_round_matches_production_two_kernel_path(self) -> None:
        """The GLM-5.2 path must preserve its observable intermediate BF16 round."""
        for m in (1, 2, 4, 8, 12, 16):
            with self.subTest(m=m):
                torch.manual_seed(20260802 + m)
                gate_up = torch.randn(
                    (m, HIDDEN_SIZE * 2), device="cuda", dtype=torch.bfloat16
                )

                activated = torch.empty(
                    (m, HIDDEN_SIZE), device="cuda", dtype=torch.bfloat16
                )
                silu_and_mul(gate_up, activated)
                reference, reference_scale = sglang_per_token_group_quant_fp8(
                    activated,
                    group_size=GROUP_SIZE,
                    column_major_scales=True,
                    scale_tma_aligned=True,
                    scale_ue8m0=True,
                    enable_v2=True,
                )

                candidate, candidate_scale = _allocate_output(m)
                silu_and_mul_contig_post_quant(
                    input=gate_up,
                    output=candidate,
                    output_scale=candidate_scale,
                    quant_group_size=GROUP_SIZE,
                    scale_ue8m0=True,
                    transposed=True,
                    round_to_bf16=True,
                )
                torch.cuda.synchronize()

                self.assertTrue(
                    torch.equal(candidate.view(torch.int8), reference.view(torch.int8)),
                    "FP8 codes differ from the production two-kernel path",
                )
                self.assertTrue(
                    torch.equal(candidate_scale, reference_scale),
                    "packed UE8M0 scales differ from the production two-kernel path",
                )


if __name__ == "__main__":
    unittest.main()
