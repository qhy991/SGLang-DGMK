"""Opt-in B200 validation for GLM-5.2 ``infini_kernel`` fixed-N/K paths.

Run directly:

  CUDA_VISIBLE_DEVICES=0 \
  SGLANG_RUN_GLM52_INFINI_FIXED_NK_GPU_TEST=1 \
  PYTHONPATH=python \
  python test/registered/kernels/test_glm52_infini_fixed_nk.py -q
"""

from __future__ import annotations

import gc
import os
import unittest

import torch


@unittest.skipUnless(
    os.getenv("SGLANG_RUN_GLM52_INFINI_FIXED_NK_GPU_TEST") == "1",
    "set SGLANG_RUN_GLM52_INFINI_FIXED_NK_GPU_TEST=1 for the B200 test",
)
class TestGlm52InfiniFixedNk(unittest.TestCase):
    def test_production_apply_matches_stock_and_graph(self):
        managed_env = (
            "SGLANG_GLM52_OPT",
            "SGLANG_GLM52_OPT_PROFILE",
            "SGLANG_GLM52_OPT_OPS",
            "SGLANG_GLM52_OPT_M_BUCKETS",
            "SGLANG_GLM52_ALLOW_ABI_ADAPTER",
            "SGLANG_GLM52_OPT_HIT_FILE",
            "SGLANG_GLM52_O_PROJ_GRAPH_ONLY",
        )
        old_env = {key: os.environ.get(key) for key in managed_env}

        def restore_env():
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.addCleanup(restore_env)
        os.environ["SGLANG_GLM52_OPT"] = "1"
        os.environ["SGLANG_GLM52_OPT_PROFILE"] = "e2e_candidates"
        os.environ["SGLANG_GLM52_ALLOW_ABI_ADAPTER"] = "0"
        os.environ.setdefault(
            "SGLANG_GLM52_OPT_HIT_FILE",
            "/tmp/sglang-glm52-infini-fixed-nk-hits.json",
        )

        from sglang.srt.layers.deep_gemm_wrapper import compile_utils
        from sglang.srt.layers.glm52_opt.context import (
            op_context,
            set_forward_mode,
        )
        from sglang.srt.layers.glm52_opt.dispatch import _HIT_COUNTS
        from sglang.srt.layers.glm52_opt.registry import lookup
        from sglang.srt.layers.quantization.fp8_utils import (
            deepgemm_w8a8_block_fp8_linear_with_fallback,
        )
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.utils import is_sm100_supported

        if not torch.cuda.is_available() or not is_sm100_supported():
            self.skipTest("requires an SM100 GPU")

        # Compile only the five audited shapes on demand. Server-wide
        # precompile enumeration is a separate startup contract.
        old_precompile = compile_utils._ENABLE_JIT_DEEPGEMM_PRECOMPILE
        self.addCleanup(
            setattr,
            compile_utils,
            "_ENABLE_JIT_DEEPGEMM_PRECOMPILE",
            old_precompile,
        )
        compile_utils._ENABLE_JIT_DEEPGEMM_PRECOMPILE = False
        self.addCleanup(set_forward_mode, None)

        cases = (
            ("index_q_upproj", ForwardMode.DECODE, 16, 4096, 2048),
            ("index_q_upproj", ForwardMode.DECODE, 32, 4096, 2048),
            ("o_proj", ForwardMode.DECODE, 16, 6144, 16384),
            ("o_proj", ForwardMode.DECODE, 32, 6144, 16384),
            ("fused_qkv_a_proj", ForwardMode.EXTEND, 4096, 2624, 6144),
        )

        torch.manual_seed(0)
        for op, mode, m, n, k in cases:
            with self.subTest(op=op, m=m, n=n, k=k):
                os.environ["SGLANG_GLM52_OPT_OPS"] = op
                os.environ["SGLANG_GLM52_OPT_M_BUCKETS"] = f"{op}:{m}"
                set_forward_mode(mode, m)
                phase = "decode" if mode is ForwardMode.DECODE else "prefill"
                spec = lookup(op, phase, m=m)
                self.assertIsNotNone(spec)
                graph_only_op = bool(spec.graph_only)

                weight = torch.randn(
                    (n, k), device="cuda", dtype=torch.bfloat16
                ).to(torch.float8_e4m3fn)
                weight_scale = torch.full(
                    (k // 128 // 4, n),
                    0x7F7F7F7F,
                    device="cuda",
                    dtype=torch.int32,
                ).T
                x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)

                os.environ["SGLANG_GLM52_OPT"] = "0"
                with op_context(op):
                    stock = deepgemm_w8a8_block_fp8_linear_with_fallback(
                        x, weight, [128, 128], weight_scale
                    )

                os.environ["SGLANG_GLM52_OPT"] = "1"
                hit_key = f"fp8_gemm/fixed_nk:{op}:{phase}:m{m}"

                if graph_only_op:
                    # Production setting (graph-only on): eager decode must
                    # decline to stock, take no fixed-N/K hit, and match stock
                    # bit-for-bit.  Only o_proj is graph_only today.
                    os.environ["SGLANG_GLM52_O_PROJ_GRAPH_ONLY"] = "1"
                    before = _HIT_COUNTS.get(hit_key, 0)
                    with op_context(op):
                        declined = deepgemm_w8a8_block_fp8_linear_with_fallback(
                            x, weight, [128, 128], weight_scale
                        )
                    self.assertEqual(_HIT_COUNTS.get(hit_key, 0), before)
                    torch.testing.assert_close(declined, stock, rtol=0, atol=0)
                    # Diagnostic eager (graph-only off): candidate is selected;
                    # this also warms the fixed-N/K JIT before graph capture.
                    os.environ["SGLANG_GLM52_O_PROJ_GRAPH_ONLY"] = "0"

                with op_context(op):
                    candidate = deepgemm_w8a8_block_fp8_linear_with_fallback(
                        x, weight, [128, 128], weight_scale
                    )
                self.assertGreater(_HIT_COUNTS.get(hit_key, 0), 0)
                torch.testing.assert_close(candidate, stock, rtol=0, atol=0)

                # Warm fixed-N/K JIT before graph capture (eager selection).
                with op_context(op):
                    deepgemm_w8a8_block_fp8_linear_with_fallback(
                        x, weight, [128, 128], weight_scale
                    )
                # Graph capture selects the fixed-N/K candidate even under the
                # production graph-only setting: capture overrides the decline.
                if graph_only_op:
                    os.environ["SGLANG_GLM52_O_PROJ_GRAPH_ONLY"] = "1"
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph), op_context(op):
                    graph_out = deepgemm_w8a8_block_fp8_linear_with_fallback(
                        x, weight, [128, 128], weight_scale
                    )
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(graph_out, stock, rtol=0, atol=0)

                del graph_out, graph, candidate, stock
                del x, weight_scale, weight
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
