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
    select_unquantized_bf16_input,
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
        expected.add_(routed, alpha=2.5)

        carried = defer_moe_output_add(shared, routed, 2.5)
        self.assertIs(carried.shared, shared)
        self.assertIs(carried.routed, routed)
        self.assertEqual(carried.routed_scale, 2.5)
        self.assertTrue(has_deferred_moe_output_add(carried))

        materialized = materialize_deferred_moe_output_add(carried)
        self.assertIs(materialized, shared)
        self.assertFalse(has_deferred_moe_output_add(materialized))
        self.assertTrue(torch.equal(materialized, expected))

    def test_duplicate_defer_fails_closed(self):
        shared = torch.zeros((1, 4), dtype=torch.bfloat16)
        routed = torch.ones_like(shared)
        carried = defer_moe_output_add(shared, routed, 2.5)
        with self.assertRaisesRegex(RuntimeError, "already attached"):
            defer_moe_output_add(carried, routed, 2.5)

    def test_deferred_carrier_survives_fullgraph_compile(self):
        def compiled_boundary(shared, routed):
            return materialize_deferred_moe_output_add(
                defer_moe_output_add(shared.clone(), routed, 2.5)
            )

        compiled = torch.compile(compiled_boundary, backend="eager", fullgraph=True)
        shared = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.bfloat16)
        routed = torch.tensor([[0.5, -0.5, 1.0, -1.0]], dtype=torch.bfloat16)
        expected = torch.add(shared, routed, alpha=2.5)
        self.assertTrue(torch.equal(compiled(shared, routed), expected))

    def test_marked_bf16_side_output_is_selected_for_tensor_only_consumer(self):
        quantized = torch.zeros((1, 4), dtype=torch.float32)
        scale = torch.ones((1, 1), dtype=torch.int32)
        normalized = torch.ones((1, 4), dtype=torch.bfloat16)
        setattr(normalized, "_sglang_dsa_bf16_passthrough", True)
        carried = (quantized, scale, normalized)

        self.assertIs(select_unquantized_bf16_input(carried), normalized)

        unmarked = normalized.clone()
        ordinary_tuple = (quantized, scale, unmarked)
        self.assertIs(select_unquantized_bf16_input(ordinary_tuple), ordinary_tuple)

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
