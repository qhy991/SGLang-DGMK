import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.deep_gemm_wrapper import entrypoint
from sglang.srt.layers.glm52_opt import dispatch


class _FakeDeepGemm:
    def __init__(self, return_value, events):
        self.return_value = return_value
        self.events = events
        self.calls = []

    def fp8_m_grouped_gemm_nt_masked(self, *args, **kwargs):
        self.events.append("stock_call")
        self.calls.append((args, kwargs))
        return self.return_value


class _FakeConfiguredDeepGemm:
    def __init__(self):
        self.pdl = False
        self.num_sms = 0
        self.tc_util = 0

    def set_pdl(self, value):
        self.pdl = bool(value)

    def get_pdl(self):
        return self.pdl

    def set_num_sms(self, value):
        self.num_sms = int(value)

    def get_num_sms(self):
        return self.num_sms

    def set_tc_util(self, value):
        self.tc_util = int(value)

    def get_tc_util(self):
        return self.tc_util


def _inputs():
    lhs = (torch.empty((2, 16, 8)), torch.empty((2, 16, 1), dtype=torch.int32))
    rhs = (torch.empty((2, 4, 8)), torch.empty((2, 4, 1), dtype=torch.int32))
    out = torch.empty((2, 16, 4), dtype=torch.bfloat16)
    masked_m = torch.tensor([1, 2], dtype=torch.int32)
    return lhs, rhs, out, masked_m


class TestGlm52MoeOverlapContract(unittest.TestCase):
    def _run_wrapper(
        self,
        *,
        replacement_result,
        stock_return,
        w13_result=False,
        exact_w13_tensors=False,
        overlap_args=None,
        recipe_a=None,
        recipe_b=None,
        max_block_n=256,
    ):
        events = []
        fake_deep_gemm = _FakeDeepGemm(stock_return, events)
        replacement_calls = []
        configured_sms = []

        @contextmanager
        def configure_num_sms(num_sms):
            configured_sms.append(num_sms)
            events.append("sms_enter")
            try:
                yield
            finally:
                events.append("sms_exit")

        def replacement(*args, **kwargs):
            events.append("replacement_call")
            replacement_calls.append((args, kwargs))
            return replacement_result

        lhs, rhs, out, masked_m = _inputs()
        with (
            patch.object(entrypoint, "deep_gemm", fake_deep_gemm, create=True),
            patch.object(entrypoint, "_ensure_cuda", side_effect=lambda value: value),
            patch.object(entrypoint, "_sanity_check_input"),
            patch.object(
                entrypoint,
                "is_exact_w13_tensor_call",
                return_value=exact_w13_tensors,
            ),
            patch.object(
                entrypoint.compile_utils,
                "deep_gemm_execution_hook",
                side_effect=lambda *args, **kwargs: nullcontext(),
            ),
            patch.object(
                entrypoint,
                "configure_deep_gemm_num_sms",
                side_effect=configure_num_sms,
            ),
            patch.object(
                entrypoint,
                "try_dispatch_w13_decode",
                side_effect=(
                    w13_result
                    if isinstance(w13_result, BaseException)
                    else lambda *args, **kwargs: w13_result
                ),
            ),
            patch.object(dispatch, "try_dispatch_moe_masked", side_effect=replacement),
        ):
            result = entrypoint.grouped_gemm_nt_f8f8bf16_masked(
                lhs,
                rhs,
                out,
                masked_m,
                expected_m=4,
                overlap_args=overlap_args,
                max_block_n=max_block_n,
                recipe_a=recipe_a,
                recipe_b=recipe_b,
            )

        return SimpleNamespace(
            result=result,
            lhs=lhs,
            rhs=rhs,
            out=out,
            masked_m=masked_m,
            events=events,
            configured_sms=configured_sms,
            replacement_calls=replacement_calls,
            stock_calls=fake_deep_gemm.calls,
        )

    def _assert_stock_positional_abi(self, call, run):
        args, _ = call
        self.assertIs(args[0], run.lhs)
        self.assertIs(args[1], run.rhs)
        self.assertIs(args[2], run.out)
        self.assertIs(args[3], run.masked_m)
        self.assertEqual(args[4], 4)

    def test_overlap_bypasses_replacement_and_preserves_return_and_sms_scope(self):
        sentinel = object()
        signal = object()
        overlap_args = SimpleNamespace(num_sms=116, signal=signal)
        run = self._run_wrapper(
            replacement_result=True,
            stock_return=sentinel,
            overlap_args=overlap_args,
            max_block_n=160,
        )

        self.assertIs(run.result, sentinel)
        self.assertEqual(run.replacement_calls, [])
        self.assertEqual(run.configured_sms, [116])
        self.assertEqual(run.events, ["sms_enter", "stock_call", "sms_exit"])
        self.assertEqual(len(run.stock_calls), 1)
        self._assert_stock_positional_abi(run.stock_calls[0], run)
        self.assertEqual(
            run.stock_calls[0][1],
            {"enable_overlap": True, "max_block_n": 160, "signal": signal},
        )

    def test_recipe_bypasses_replacement_and_reaches_stock(self):
        run = self._run_wrapper(
            replacement_result=True,
            stock_return=None,
            recipe_a=(1, 128),
            recipe_b=(128, 128),
        )

        self.assertIsNone(run.result)
        self.assertEqual(run.replacement_calls, [])
        self.assertEqual(run.configured_sms, [None])
        self.assertEqual(len(run.stock_calls), 1)
        self._assert_stock_positional_abi(run.stock_calls[0], run)
        self.assertEqual(
            run.stock_calls[0][1],
            {"recipe_a": (1, 128), "recipe_b": (128, 128)},
        )

    def test_eligible_replacement_returns_output_without_stock_call(self):
        run = self._run_wrapper(replacement_result=True, stock_return=object())

        self.assertIs(run.result, run.out)
        self.assertEqual(len(run.replacement_calls), 1)
        self.assertEqual(run.stock_calls, [])
        self.assertEqual(run.configured_sms, [None])
        self.assertEqual(run.events, ["sms_enter", "replacement_call", "sms_exit"])

    def test_eligible_replacement_decline_has_no_overlap_keywords(self):
        sentinel = object()
        run = self._run_wrapper(replacement_result=False, stock_return=sentinel)

        self.assertIs(run.result, sentinel)
        self.assertEqual(len(run.replacement_calls), 1)
        self.assertEqual(len(run.stock_calls), 1)
        self._assert_stock_positional_abi(run.stock_calls[0], run)
        self.assertEqual(run.stock_calls[0][1], {})
        self.assertEqual(run.configured_sms, [None])

    def test_dedicated_w13_candidate_precedes_generic_dispatch(self):
        run = self._run_wrapper(
            replacement_result=True,
            stock_return=object(),
            w13_result=True,
            exact_w13_tensors=True,
        )

        self.assertIsNone(run.result)
        self.assertEqual(run.replacement_calls, [])
        self.assertEqual(run.stock_calls, [])
        self.assertEqual(run.configured_sms, [])
        self.assertEqual(run.events, [])

    def test_dedicated_w13_launch_error_propagates_without_stock_retry(self):
        with self.assertRaisesRegex(RuntimeError, "candidate launch failed"):
            self._run_wrapper(
                replacement_result=True,
                stock_return=object(),
                w13_result=RuntimeError("candidate launch failed"),
                exact_w13_tensors=True,
            )

    def test_exact_w13_decline_bypasses_generic_overlay_and_preserves_stock_none(self):
        run = self._run_wrapper(
            replacement_result=True,
            stock_return=None,
            w13_result=False,
            exact_w13_tensors=True,
        )

        self.assertIsNone(run.result)
        self.assertEqual(run.replacement_calls, [])
        self.assertEqual(len(run.stock_calls), 1)
        self.assertEqual(run.configured_sms, [None])

    def test_post_assignment_updater_sets_installed_state_before_w13_initializer(self):
        fake_deep_gemm = _FakeConfiguredDeepGemm()
        calls = []

        def initialize(gpu_id, server_args, *, compile_utils_loader):
            calls.append(
                (
                    "initialize",
                    gpu_id,
                    server_args,
                    fake_deep_gemm.get_pdl(),
                    fake_deep_gemm.get_num_sms(),
                    fake_deep_gemm.get_tc_util(),
                )
            )
            self.assertIs(compile_utils_loader.__self__, entrypoint.compile_utils)
            self.assertIs(
                compile_utils_loader.__func__,
                entrypoint.compile_utils.load.__func__,
            )
            return True

        with (
            patch.object(entrypoint, "deep_gemm", fake_deep_gemm, create=True),
            patch.object(entrypoint, "initialization_requested", return_value=True),
            patch.object(
                entrypoint,
                "initialize_w13_decode_after_assignment",
                side_effect=initialize,
            ),
            patch.object(
                entrypoint,
                "dispatch_state",
                return_value={"enabled": True},
            ),
            patch.object(
                entrypoint.compile_utils,
                "update_deep_gemm_config",
                side_effect=AssertionError("normal updater must not run twice"),
            ),
            patch.object(
                entrypoint.envs.SGLANG_DEEPGEMM_PDL, "get", return_value=False
            ),
        ):
            entrypoint.update_deep_gemm_config(3, "server-args")

        self.assertEqual(
            calls,
            [("initialize", 3, "server-args", True, 148, 100)],
        )

    def test_failed_w13_initializer_restores_installed_stock_runtime(self):
        fake_deep_gemm = _FakeConfiguredDeepGemm()
        fake_deep_gemm.set_pdl(False)
        fake_deep_gemm.set_num_sms(117)
        fake_deep_gemm.set_tc_util(93)
        with (
            patch.object(entrypoint, "deep_gemm", fake_deep_gemm, create=True),
            patch.object(entrypoint, "initialization_requested", return_value=True),
            patch.object(
                entrypoint,
                "initialize_w13_decode_after_assignment",
                side_effect=RuntimeError("requested candidate initialization failed"),
            ),
            patch.object(
                entrypoint,
                "dispatch_state",
                return_value={"enabled": False},
            ),
            patch.object(
                entrypoint.compile_utils,
                "update_deep_gemm_config",
                side_effect=AssertionError(
                    "normal updater must not hide explicit candidate failure"
                ),
            ),
            patch.object(
                entrypoint.envs.SGLANG_DEEPGEMM_PDL,
                "get",
                return_value=False,
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "requested candidate initialization failed",
            ):
                entrypoint.update_deep_gemm_config(2, "server-args")

        self.assertFalse(fake_deep_gemm.get_pdl())
        self.assertEqual(fake_deep_gemm.get_num_sms(), 117)
        self.assertEqual(fake_deep_gemm.get_tc_util(), 93)


if __name__ == "__main__":
    unittest.main()
