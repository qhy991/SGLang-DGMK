"""CPU contract tests for the FlashInfer TRT-LLM block-FP8 MoE wrapper."""

import unittest
from contextlib import nullcontext
from unittest import mock

import torch

from sglang.srt.layers.moe.flashinfer_trtllm_moe import (
    trtllm_fp8_block_scale_moe_wrapper,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class TestFlashinferTrtllmMoeWrapper(unittest.TestCase):
    @staticmethod
    def _inputs(device="meta"):
        device = torch.device(device)
        return {
            "routing_logits": torch.empty((2, 16), device=device),
            "routing_bias": None,
            "hidden_states": torch.empty(
                (2, 8), device=device, dtype=torch.float8_e4m3fn
            ),
            "hidden_states_scale": torch.empty((2, 1), device=device),
            "gemm1_weights": torch.empty((1,), device=device, dtype=torch.uint8),
            "gemm1_weights_scale": torch.empty((1,), device=device),
            "gemm2_weights": torch.empty((1,), device=device, dtype=torch.uint8),
            "gemm2_weights_scale": torch.empty((1,), device=device),
            "num_experts": 16,
            "top_k": 2,
            "n_group": None,
            "topk_group": None,
            "intermediate_size": 4,
            "local_expert_offset": 0,
            "local_num_experts": 16,
            "routed_scaling_factor": None,
        }

    def test_optional_routing_replay_fake_contract(self):
        inputs = self._inputs()

        for routing_replay_out in (
            None,
            torch.empty((2, 2), device="meta", dtype=torch.int16),
        ):
            output = trtllm_fp8_block_scale_moe_wrapper(
                **inputs, routing_replay_out=routing_replay_out
            )
            self.assertEqual(output.shape, (2, 8))
            self.assertEqual(output.dtype, torch.bfloat16)
            self.assertEqual(output.device.type, "meta")

    def test_routing_replay_is_declared_mutable(self):
        schema = str(
            torch.ops.sglang.trtllm_fp8_block_scale_moe_wrapper.default._schema
        )
        replay_argument = schema.split("routing_replay_out=None", maxsplit=1)[0]
        replay_argument = replay_argument.rsplit(",", maxsplit=1)[-1]
        self.assertIn("Tensor(", replay_argument)
        self.assertIn("!)?", replay_argument)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_fullgraph_preserves_routing_replay_mutation(self):
        inputs = self._inputs("cuda")
        expected = torch.tensor([[3, 7], [5, 11]], device="cuda", dtype=torch.int16)

        def fake_flashinfer(**kwargs):
            kwargs["routing_replay_out"].copy_(expected)
            hidden_states = kwargs["hidden_states"]
            return torch.full(
                hidden_states.shape,
                2.0,
                device=hidden_states.device,
                dtype=torch.bfloat16,
            )

        def invoke(replay):
            output = trtllm_fp8_block_scale_moe_wrapper(
                **inputs, routing_replay_out=replay
            )
            return output, replay

        with mock.patch(
            "flashinfer.fused_moe.trtllm_fp8_block_scale_moe", fake_flashinfer
        ):
            eager_replay = torch.zeros_like(expected)
            eager_output, eager_mutated = invoke(eager_replay)

            compiled = torch.compile(invoke, fullgraph=True, backend="eager")
            compiled_replay = torch.zeros_like(expected)
            compiled_output, compiled_mutated = compiled(compiled_replay)
            torch.cuda.synchronize()

        self.assertTrue(torch.equal(eager_mutated, expected))
        self.assertTrue(torch.equal(compiled_mutated, expected))
        self.assertTrue(torch.equal(compiled_replay, expected))
        self.assertTrue(torch.equal(eager_output, compiled_output))
        self.assertEqual(compiled_output.float().mean().item(), 2.0)

    def test_capture_cache_accepts_int16_replay(self):
        from sglang.srt.state_capturer.base import BaseDeviceCache

        cache = BaseDeviceCache(
            max_batch_size=2,
            num_layers=8,
            topk_size=2,
            device="cpu",
            name="flashinfer-replay-test",
        )
        replay = torch.tensor([[3, 7], [5, 11]], dtype=torch.int16)

        cache.capture(layer_id=4, topk_indices=replay)

        self.assertEqual(cache.buffer.dtype, torch.int32)
        self.assertEqual(cache.buffer[:, 4].tolist(), replay.tolist())

    def test_bypassed_runner_captures_flashinfer_replay_buffer(self):
        from sglang.srt.layers.moe import topk as topk_module
        from sglang.srt.layers.moe.moe_runner import flashinfer_trtllm as runner
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardDispatchOutput,
        )
        from sglang.srt.layers.moe.topk import BypassedTopKOutput, TopKConfig
        from sglang.srt.layers.moe.utils import RoutingMethodType
        from sglang.srt.state_capturer import routed_experts as routed_experts_module

        hidden_states = torch.randn((2, 8))
        topk_config = TopKConfig(top_k=2)
        dispatch_output = StandardDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_output=BypassedTopKOutput(
                hidden_states=hidden_states,
                router_logits=torch.randn((2, 16)),
                topk_config=topk_config,
            ),
        )
        quant_info = runner.FlashInferTrtllmFp8MoeQuantInfo(
            w13_weight=torch.empty((16, 8, 8)),
            w2_weight=torch.empty((16, 8, 4)),
            global_num_experts=16,
            local_expert_offset=0,
            local_num_experts=16,
            intermediate_size=4,
            routing_method_type=RoutingMethodType.DeepSeekV3,
            block_quant=True,
            weight_block_k=128,
            w13_weight_scale_inv=torch.ones((16, 1, 1)),
            w2_weight_scale_inv=torch.ones((16, 1, 1)),
        )
        runner_config = MoeRunnerConfig(layer_id=7, activation="silu")
        replay_value = torch.tensor([[3, 7], [5, 11]], dtype=torch.int16)
        replay_buffers = []

        def fake_moe(**kwargs):
            replay = kwargs["routing_replay_out"]
            replay_buffers.append(replay)
            if replay is not None:
                replay.copy_(replay_value)
            return torch.zeros_like(hidden_states)

        common_patches = (
            mock.patch.object(
                runner,
                "per_token_group_quant_fp8",
                return_value=(hidden_states, torch.ones((2, 1))),
            ),
            mock.patch.object(runner, "trtllm_fp8_block_scale_moe_wrapper", fake_moe),
            mock.patch.object(
                runner, "use_symmetric_memory", return_value=nullcontext()
            ),
            mock.patch.object(runner, "get_tp_group", return_value=None),
            mock.patch.object(runner, "is_allocation_symmetric", return_value=False),
            mock.patch.object(topk_module, "capture_routed_experts_if_allowed"),
        )

        with (
            common_patches[0],
            common_patches[1],
            common_patches[2],
            common_patches[3],
            common_patches[4],
            common_patches[5] as capture_mock,
        ):
            with mock.patch.object(
                routed_experts_module,
                "get_global_experts_capturer",
                return_value=object(),
            ):
                runner.fused_experts_none_to_flashinfer_trtllm_fp8(
                    dispatch_output, quant_info, runner_config
                )

            self.assertEqual(replay_buffers[-1].dtype, torch.int16)
            self.assertEqual(replay_buffers[-1].tolist(), replay_value.tolist())
            capture_mock.assert_called_once_with(
                topk_config, runner_config.layer_id, replay_buffers[-1]
            )

            capture_mock.reset_mock()
            with mock.patch.object(
                routed_experts_module,
                "get_global_experts_capturer",
                return_value=None,
            ):
                runner.fused_experts_none_to_flashinfer_trtllm_fp8(
                    dispatch_output, quant_info, runner_config
                )

            self.assertIsNone(replay_buffers[-1])
            capture_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=3)
