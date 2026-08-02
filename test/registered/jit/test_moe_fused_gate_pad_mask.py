from __future__ import annotations

import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestMoEFusedGatePadMask(unittest.TestCase):
    def test_fused_mask_matches_production_two_kernel_path(self) -> None:
        from sglang.jit_kernel.dsv4 import mask_topk_ids
        from sglang.jit_kernel.moe_fused_gate import moe_fused_gate

        torch.manual_seed(20260802)
        scores = torch.randn(16, 256, dtype=torch.float32, device="cuda")
        bias = 0.05 * torch.randn(256, dtype=torch.float32, device="cuda")

        for num_valid in (0, 1, 2, 7, 12, 15, 16):
            with self.subTest(num_valid=num_valid):
                ntn = torch.tensor(num_valid, dtype=torch.int32, device="cuda")
                ref_weights, ref_ids = moe_fused_gate(
                    scores,
                    bias,
                    8,
                    scoring_func="sigmoid",
                    renormalize=True,
                    routed_scaling_factor=2.5,
                )
                mask_topk_ids(ref_ids, ntn)
                fused_weights, fused_ids = moe_fused_gate(
                    scores,
                    bias,
                    8,
                    scoring_func="sigmoid",
                    renormalize=True,
                    routed_scaling_factor=2.5,
                    num_token_non_padded=ntn,
                )
                torch.cuda.synchronize()

                self.assertTrue(torch.equal(fused_weights, ref_weights))
                self.assertTrue(torch.equal(fused_ids, ref_ids))
                if num_valid < scores.shape[0]:
                    self.assertTrue(bool(torch.all(fused_ids[num_valid:] == -1)))

    def test_select_experts_flag_matches_stock_postprocess(self) -> None:
        from sglang.srt.environ import envs
        from sglang.srt.layers.moe.topk import TopKConfig, select_experts

        torch.manual_seed(20260802)
        hidden_states = torch.randn(16, 1, dtype=torch.bfloat16, device="cuda")
        router_logits = torch.randn(16, 256, dtype=torch.float32, device="cuda")
        correction_bias = 0.05 * torch.randn(
            256, dtype=torch.float32, device="cuda"
        )
        num_token_non_padded = torch.tensor(15, dtype=torch.int32, device="cuda")
        config = TopKConfig(
            top_k=8,
            use_grouped_topk=True,
            topk_group=1,
            num_expert_group=1,
            renormalize=True,
            correction_bias=correction_bias,
            routed_scaling_factor=2.5,
            scoring_func="sigmoid",
        )

        with envs.SGLANG_GLM52_ROUTER_PAD_MASK_FUSION.override(False):
            reference = select_experts(
                hidden_states,
                router_logits,
                config,
                layer_id=0,
                num_token_non_padded=num_token_non_padded,
            )
        with envs.SGLANG_GLM52_ROUTER_PAD_MASK_FUSION.override(True):
            candidate = select_experts(
                hidden_states,
                router_logits,
                config,
                layer_id=0,
                num_token_non_padded=num_token_non_padded,
            )
        torch.cuda.synchronize()

        self.assertTrue(torch.equal(candidate.topk_weights, reference.topk_weights))
        self.assertTrue(torch.equal(candidate.topk_ids, reference.topk_ids))
        self.assertTrue(bool(torch.all(candidate.topk_ids[15:] == -1)))


if __name__ == "__main__":
    unittest.main()
