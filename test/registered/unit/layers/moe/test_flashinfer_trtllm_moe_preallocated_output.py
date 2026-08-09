"""Contracts for FlashInfer TRT-LLM MoE preallocated outputs."""

import unittest
from unittest import mock

import torch

from sglang.srt.layers.moe.flashinfer_trtllm_moe import (
    trtllm_fp8_block_scale_moe_wrapper,
    trtllm_fp8_block_scale_routed_moe_wrapper,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


class TestFlashinferTrtllmMoePreallocatedOutput(unittest.TestCase):
    @staticmethod
    def _common_inputs(device="meta"):
        device = torch.device(device)
        return {
            "routing_bias": None,
            "hidden_states": torch.empty(
                (2, 8), device=device, dtype=torch.float8_e4m3fn
            ),
            "hidden_states_scale": torch.empty((1, 2), device=device),
            "gemm1_weights": torch.empty((1,), device=device, dtype=torch.uint8),
            "gemm1_weights_scale": torch.empty((1,), device=device),
            "gemm2_weights": torch.empty((1,), device=device, dtype=torch.uint8),
            "gemm2_weights_scale": torch.empty((1,), device=device),
            "output": torch.empty((2, 8), device=device, dtype=torch.bfloat16),
            "num_experts": 16,
            "top_k": 2,
            "n_group": None,
            "topk_group": None,
            "intermediate_size": 4,
            "local_expert_offset": 0,
            "local_num_experts": 16,
            "routed_scaling_factor": None,
        }

    def test_meta_wrappers_accept_the_preallocated_output(self):
        standard = self._common_inputs()
        standard["routing_logits"] = torch.empty((2, 16), device="meta")
        routed = self._common_inputs()
        routed["topk_ids"] = torch.empty((2, 2), device="meta", dtype=torch.int32)

        self.assertIsNone(trtllm_fp8_block_scale_moe_wrapper(**standard))
        self.assertIsNone(trtllm_fp8_block_scale_routed_moe_wrapper(**routed))

    def test_custom_ops_declare_output_mutation(self):
        for name in (
            "trtllm_fp8_block_scale_moe_wrapper",
            "trtllm_fp8_block_scale_routed_moe_wrapper",
        ):
            schema = str(getattr(torch.ops.sglang, name).default._schema)
            schema_prefix = schema.split(" output", maxsplit=1)[0]
            output_arg = schema_prefix.rsplit(",", maxsplit=1)[-1]
            self.assertIn("Tensor(", output_arg)
            self.assertIn("!)", output_arg)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_wrappers_mutate_output_without_return_alias_in_custom_op(self):
        cases = (
            (
                trtllm_fp8_block_scale_moe_wrapper,
                "flashinfer.fused_moe.trtllm_fp8_block_scale_moe",
                {"routing_logits": torch.empty((2, 16), device="cuda")},
            ),
            (
                trtllm_fp8_block_scale_routed_moe_wrapper,
                "flashinfer.fused_moe.trtllm_fp8_block_scale_routed_moe",
                {"topk_ids": torch.empty((2, 2), device="cuda", dtype=torch.int32)},
            ),
        )

        def fake_flashinfer(**kwargs):
            kwargs["output"].fill_(3)
            return kwargs["output"]

        for wrapper, patch_target, routing_input in cases:
            with self.subTest(wrapper=wrapper.__name__):
                inputs = self._common_inputs("cuda")
                output = inputs.pop("output")
                inputs.update(routing_input)

                def invoke(out, wrapper=wrapper, inputs=inputs):
                    wrapper(**inputs, output=out)
                    return out

                with mock.patch(patch_target, fake_flashinfer):
                    eager = invoke(output)
                    compiled = torch.compile(invoke, fullgraph=True, backend="eager")
                    compiled_output = torch.zeros_like(output)
                    returned = compiled(compiled_output)

                self.assertIs(eager, output)
                self.assertIs(returned, compiled_output)
                self.assertTrue(torch.equal(eager, returned))
                self.assertTrue(torch.equal(returned, torch.full_like(returned, 3)))


if __name__ == "__main__":
    unittest.main(verbosity=3)
