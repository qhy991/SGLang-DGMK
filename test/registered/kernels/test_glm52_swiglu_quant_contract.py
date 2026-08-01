"""CPU-only contract tests for the bounded masked SwiGLU quant dispatcher."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.glm52_opt import swiglu_quant


def _fake_inputs(
    *,
    capability: tuple[int, int] = (10, 3),
    slab: int = 8192,
):
    device = torch.device("cuda:0")
    gateup = SimpleNamespace(
        is_cuda=True,
        device=device,
        dtype=torch.bfloat16,
        ndim=3,
        shape=(32, slab, 4096),
        is_contiguous=lambda: True,
    )
    masked_m = SimpleNamespace(
        is_cuda=True,
        device=device,
        dtype=torch.int32,
        shape=(32,),
        is_contiguous=lambda: True,
    )
    return capability, gateup, masked_m


def _eligibility_kwargs(**overrides):
    kwargs = {
        "group_size": 128,
        "topk": 8,
        "swiglu_limit": None,
        "swizzle": False,
        "gemm1_alpha": None,
        "gemm1_clamp_limit": None,
        "num_real_tokens": 16,
        "variant": "cuda_valid_cta",
    }
    kwargs.update(overrides)
    return kwargs


class SwigluQuantContractTest(unittest.TestCase):
    def test_exact_b300_contract_is_eligible(self):
        capability, gateup, masked_m = _fake_inputs()
        with patch.object(
            swiglu_quant, "_device_capability", return_value=capability
        ):
            self.assertIsNone(
                swiglu_quant._eligibility_error(
                    gateup, masked_m, **_eligibility_kwargs()
                )
            )

    def test_hardware_slab_pair_is_not_interchangeable(self):
        capability, gateup, masked_m = _fake_inputs(
            capability=(10, 0), slab=8192
        )
        with patch.object(
            swiglu_quant, "_device_capability", return_value=capability
        ):
            error = swiglu_quant._eligibility_error(
                gateup, masked_m, **_eligibility_kwargs()
            )
        self.assertIn("unsupported GPU/expert-slab contract", error)

    def test_unsupported_semantics_fall_back_before_allocation(self):
        capability, gateup, masked_m = _fake_inputs()
        with (
            patch.object(
                swiglu_quant, "_device_capability", return_value=capability
            ),
            patch.object(swiglu_quant, "_allocate_outputs") as allocate,
        ):
            result = swiglu_quant.maybe_silu_mul_quant_packed(
                gateup,
                masked_m,
                **_eligibility_kwargs(swiglu_limit=1.0),
            )
        self.assertIsNone(result)
        allocate.assert_not_called()

    def test_decode_bucket_is_bounded(self):
        capability, gateup, masked_m = _fake_inputs()
        with patch.object(
            swiglu_quant, "_device_capability", return_value=capability
        ):
            error = swiglu_quant._eligibility_error(
                gateup,
                masked_m,
                **_eligibility_kwargs(num_real_tokens=64),
            )
        self.assertIn("M16 or M32", error)


if __name__ == "__main__":
    unittest.main()
