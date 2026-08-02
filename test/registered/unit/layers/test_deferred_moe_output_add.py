import unittest

import torch

from sglang.srt.layers.communicator import (
    CommunicateSummableTensorPairFn,
    LayerCommunicator,
    LayerScatterModes,
    ScatterMode,
    defer_moe_output_add,
    has_deferred_moe_output_add,
    materialize_deferred_moe_output_add,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDeferredMoeOutputAdd(CustomTestCase):
    def test_materialize_restores_exact_stock_bf16_add(self):
        shared = torch.tensor(
            [[1.0, 0.333984375, 128.0, -0.5]], dtype=torch.bfloat16
        )
        routed = torch.tensor(
            [[0.0078125, -0.25, 1.0, 0.125]], dtype=torch.bfloat16
        )
        expected = shared.clone()
        expected.add_(routed)

        carried = defer_moe_output_add(shared, routed)
        self.assertIs(carried, shared)
        self.assertTrue(has_deferred_moe_output_add(carried))

        materialized = materialize_deferred_moe_output_add(carried)
        self.assertIs(materialized, shared)
        self.assertFalse(has_deferred_moe_output_add(materialized))
        self.assertTrue(torch.equal(materialized, expected))

    def test_duplicate_defer_fails_closed(self):
        shared = torch.zeros((1, 4), dtype=torch.bfloat16)
        routed = torch.ones_like(shared)
        defer_moe_output_add(shared, routed)
        with self.assertRaisesRegex(RuntimeError, "already attached"):
            defer_moe_output_add(shared, routed)

    def test_only_nonfinal_trivial_scattered_seam_is_admitted(self):
        communicator = object.__new__(LayerCommunicator)
        communicator.is_last_layer = False
        communicator.layer_scatter_modes = LayerScatterModes(
            layer_input_mode=ScatterMode.SCATTERED,
            attn_mode=ScatterMode.TP_ATTN_FULL,
            mlp_mode=ScatterMode.SCATTERED,
            middle_residual_mode=ScatterMode.SCATTERED,
            layer_output_mode=ScatterMode.SCATTERED,
        )
        communicator._communicate_summable_tensor_pair_fn = (
            CommunicateSummableTensorPairFn._trivial
        )
        self.assertTrue(communicator.can_defer_moe_output_add())

        communicator.is_last_layer = True
        self.assertFalse(communicator.can_defer_moe_output_add())

        communicator.is_last_layer = False
        communicator._communicate_summable_tensor_pair_fn = lambda **_: None
        self.assertFalse(communicator.can_defer_moe_output_add())


if __name__ == "__main__":
    unittest.main()
