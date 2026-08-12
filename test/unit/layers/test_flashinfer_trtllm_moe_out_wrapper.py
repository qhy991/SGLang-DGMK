import unittest

import torch

from sglang.srt.layers.moe.flashinfer_trtllm_moe import (
    _require_preserved_output,
    trtllm_fp8_block_scale_moe_out_wrapper,
    trtllm_fp8_block_scale_routed_moe_out_wrapper,
)


def _arguments(output: torch.Tensor) -> dict:
    device = output.device
    hidden_states = torch.zeros((2, 4), dtype=torch.float8_e4m3fn, device=device)
    return {
        "routing_bias": None,
        "hidden_states": hidden_states,
        "hidden_states_scale": torch.ones(
            (1, 2), dtype=torch.float32, device=device
        ),
        "gemm1_weights": torch.zeros(
            (2, 4, 4), dtype=torch.float8_e4m3fn, device=device
        ),
        "gemm1_weights_scale": torch.ones(
            (2, 1, 1), dtype=torch.float32, device=device
        ),
        "gemm2_weights": torch.zeros(
            (2, 2, 4), dtype=torch.float8_e4m3fn, device=device
        ),
        "gemm2_weights_scale": torch.ones(
            (2, 1, 1), dtype=torch.float32, device=device
        ),
        "num_experts": 2,
        "top_k": 1,
        "n_group": None,
        "topk_group": None,
        "intermediate_size": 2,
        "local_expert_offset": 0,
        "local_num_experts": 2,
        "routed_scaling_factor": 1.0,
        "output": output,
    }


class TestFlashInferTrtllmMoeOutWrapper(unittest.TestCase):
    def test_out_wrapper_schema_and_fullgraph_compile(self) -> None:
        for routed in (False, True):
            with self.subTest(routed=routed):
                output = torch.empty((2, 4), dtype=torch.bfloat16, device="meta")
                kwargs = _arguments(output)
                if routed:
                    op = trtllm_fp8_block_scale_routed_moe_out_wrapper
                    kwargs["topk_ids"] = torch.zeros(
                        (2, 1), dtype=torch.int32, device="meta"
                    )
                else:
                    op = trtllm_fp8_block_scale_moe_out_wrapper
                    kwargs["routing_logits"] = torch.zeros(
                        (2, 2), dtype=torch.float32, device="meta"
                    )

                def invoke(owner_output: torch.Tensor) -> torch.Tensor:
                    op(**{**kwargs, "output": owner_output})
                    return owner_output

                compiled = torch.compile(invoke, backend="eager", fullgraph=True)
                observed = compiled(output)

                self.assertEqual(observed.shape, output.shape)
                self.assertEqual(observed.dtype, output.dtype)
                self.assertEqual(observed.device.type, "meta")
                schema = str(op.default._schema)
                self.assertRegex(schema, r"Tensor\(a\d+!\) output")
                self.assertTrue(schema.endswith("-> ()"))

    def test_output_ownership_validation(self) -> None:
        output = torch.empty((2, 4), dtype=torch.bfloat16)
        _require_preserved_output(output, output, "test")
        with self.assertRaisesRegex(RuntimeError, "did not preserve"):
            _require_preserved_output(torch.empty_like(output), output, "test")


if __name__ == "__main__":
    unittest.main()
