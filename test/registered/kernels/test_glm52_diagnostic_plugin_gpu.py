"""Opt-in B200 ABI/reachability tests for new diagnostic provider call sites."""

from __future__ import annotations

import gc
import os
import unittest
from unittest.mock import patch

import torch


@unittest.skipUnless(
    os.getenv("SGLANG_RUN_GLM52_DIAGNOSTIC_PLUGIN_GPU_TEST") == "1",
    "set SGLANG_RUN_GLM52_DIAGNOSTIC_PLUGIN_GPU_TEST=1 for the B200 test",
)
class TestGlm52DiagnosticPluginGpu(unittest.TestCase):
    def setUp(self):
        managed = (
            "SGLANG_GLM52_OPT",
            "SGLANG_GLM52_OPT_PROFILE",
            "SGLANG_GLM52_OPT_OPS",
            "SGLANG_GLM52_OPT_HIT_FILE",
        )
        self.saved_env = {name: os.environ.get(name) for name in managed}
        os.environ["SGLANG_GLM52_OPT"] = "1"
        os.environ["SGLANG_GLM52_OPT_PROFILE"] = "diagnostic_all"
        os.environ["SGLANG_GLM52_OPT_HIT_FILE"] = (
            "/tmp/glm52-diagnostic-plugin-hits.json"
        )

        from sglang.srt.layers.glm52_opt.context import set_forward_mode
        from sglang.srt.utils import is_sm100_supported

        if not torch.cuda.is_available() or not is_sm100_supported():
            self.skipTest("requires an SM100 GPU")
        self.set_forward_mode = set_forward_mode

    def tearDown(self):
        self.set_forward_mode(None)
        for name, value in self.saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        gc.collect()
        torch.cuda.empty_cache()

    def test_bf16_projection_and_router_boundaries(self):
        from sglang.srt.layers.glm52_opt.dispatch import (
            try_dispatch_index_wk_weights_proj,
            try_dispatch_router_logit_gemm,
            try_dispatch_router_sigmoid_topk,
        )
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        self.set_forward_mode(ForwardMode.DECODE, 16)

        os.environ["SGLANG_GLM52_OPT_OPS"] = "index_wk_weights_proj"
        x = torch.randn((16, 6144), device="cuda", dtype=torch.bfloat16)
        wk_weight = torch.randn((160, 6144), device="cuda", dtype=torch.bfloat16)
        wk_out = torch.empty((16, 160), device="cuda", dtype=torch.bfloat16)
        with patch(
            "sglang.srt.layers.glm52_opt.dispatch.run_index_wk_weights_proj",
            return_value=wk_out,
        ) as candidate:
            self.assertIs(
                try_dispatch_index_wk_weights_proj(x, wk_weight),
                wk_out,
            )
        candidate.assert_called_once()

        os.environ["SGLANG_GLM52_OPT_OPS"] = "router_logit_gemm"
        router_weight = torch.randn(
            (256, 6144), device="cuda", dtype=torch.bfloat16
        )
        logits = torch.empty((16, 256), device="cuda", dtype=torch.float32)
        with patch(
            "sglang.srt.layers.glm52_opt.dispatch.run_router_logit_gemm",
            return_value=logits,
        ) as candidate:
            self.assertIs(try_dispatch_router_logit_gemm(x, router_weight), logits)
        candidate.assert_called_once()

        os.environ["SGLANG_GLM52_OPT_OPS"] = "router_sigmoid_topk"
        bias = torch.randn((256,), device="cuda", dtype=torch.float32)
        weights = torch.empty((16, 8), device="cuda", dtype=torch.float32)
        ids = torch.empty((16, 8), device="cuda", dtype=torch.int32)
        with patch(
            "sglang.srt.layers.glm52_opt.dispatch.run_router_sigmoid_topk",
            return_value=(weights, ids),
        ) as candidate:
            result = try_dispatch_router_sigmoid_topk(
                scores=logits,
                bias=bias,
                topk=8,
                scoring_func="sigmoid",
                num_fused_shared_experts=0,
                renormalize=True,
                routed_scaling_factor=2.5,
                apply_routed_scaling_factor_on_output=False,
                moe_softcapping=0.0,
                num_expert_group=1,
                topk_group=1,
            )
        self.assertEqual(result, (weights, ids))
        candidate.assert_called_once()

        with (
            patch(
                "sglang.srt.layers.glm52_opt.dispatch.run_router_sigmoid_topk",
                side_effect=RuntimeError("selected provider failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "selected provider failed"),
        ):
            try_dispatch_router_sigmoid_topk(
                scores=logits,
                bias=bias,
                topk=8,
                scoring_func="sigmoid",
                num_fused_shared_experts=0,
                renormalize=True,
                routed_scaling_factor=2.5,
                apply_routed_scaling_factor_on_output=False,
                moe_softcapping=0.0,
                num_expert_group=1,
                topk_group=1,
            )

    def test_decode_and_prefill_swiglu_quant_boundaries(self):
        from sglang.srt.layers.glm52_opt.dispatch import (
            try_dispatch_moe_swiglu_quant_decode,
            try_dispatch_moe_swiglu_quant_prefill,
        )
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        os.environ["SGLANG_GLM52_OPT_OPS"] = "moe_swiglu_quant"
        self.set_forward_mode(ForwardMode.DECODE, 16)
        decode_in = torch.empty(
            (32, 1024, 4096), device="cuda", dtype=torch.bfloat16
        )
        masked_m = torch.full((32,), 4, device="cuda", dtype=torch.int32)
        decode_out = torch.empty(
            (32, 1024, 2048), device="cuda", dtype=torch.float8_e4m3fn
        )
        decode_scale = torch.empty(
            (32, 4, 1024), device="cuda", dtype=torch.int32
        ).transpose(1, 2)
        with patch(
            "sglang.srt.layers.glm52_opt.dispatch.run_moe_swiglu_quant",
            return_value=(decode_out, decode_scale),
        ) as candidate:
            result = try_dispatch_moe_swiglu_quant_decode(
                decode_in,
                masked_m,
                group_size=128,
                topk=8,
                swiglu_limit=None,
                swizzle=False,
                gemm1_alpha=None,
                gemm1_clamp_limit=None,
                num_real_tokens=16,
            )
        self.assertEqual(result, (decode_out, decode_scale))
        candidate.assert_called_once()

        del decode_in, decode_out, decode_scale, masked_m
        torch.cuda.empty_cache()

        self.set_forward_mode(ForwardMode.EXTEND, 4096)
        prefill_in = torch.empty(
            (35200, 4096), device="cuda", dtype=torch.bfloat16
        )
        m_indices = torch.empty((35200,), device="cuda", dtype=torch.int32)
        endpoint = torch.empty((32,), device="cuda", dtype=torch.int32)
        prefill_out = torch.empty(
            (35200, 2048), device="cuda", dtype=torch.float8_e4m3fn
        )
        prefill_scale = torch.empty(
            (4, 35200), device="cuda", dtype=torch.int32
        ).transpose(0, 1)
        with patch(
            "sglang.srt.layers.glm52_opt.dispatch.run_moe_swiglu_quant",
            return_value=(prefill_out, prefill_scale),
        ) as candidate:
            result = try_dispatch_moe_swiglu_quant_prefill(
                prefill_in,
                m_indices,
                endpoint,
                group_size=128,
                swiglu_limit=None,
                swizzle=False,
                gemm1_alpha=None,
                gemm1_clamp_limit=None,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=True,
                pdl=True,
            )
        self.assertEqual(result, (prefill_out, prefill_scale))
        candidate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
