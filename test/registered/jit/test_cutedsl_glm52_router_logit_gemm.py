"""Correctness probes for the bounded GLM-5.2 router-logit portfolio."""

from __future__ import annotations

import pytest
import torch
from sglang.jit_kernel.utils import get_jit_cuda_arch, is_hip_runtime
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from sglang.jit_kernel.cutedsl_glm52_router_logit_gemm import (
    cutedsl_glm52_router_logit_gemm,
)


def _inputs(m: int, pattern: str) -> tuple[torch.Tensor, torch.Tensor]:
    if pattern == "random":
        generator = torch.Generator(device="cuda").manual_seed(20260731 + m)
        x = torch.randn(
            (m, 6144),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        weight = torch.randn(
            (256, 6144),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        ) * (6144**-0.5)
    elif pattern == "zero":
        x = torch.zeros((m, 6144), dtype=torch.bfloat16, device="cuda")
        weight = torch.zeros((256, 6144), dtype=torch.bfloat16, device="cuda")
    elif pattern == "ramp":
        x = torch.linspace(-0.25, 0.25, m * 6144, device="cuda").reshape(m, 6144)
        weight = torch.linspace(
            -0.125,
            0.125,
            256 * 6144,
            device="cuda",
        ).reshape(256, 6144)
        x, weight = x.bfloat16(), weight.bfloat16()
    elif pattern == "alternating":
        x = torch.ones((m, 6144), dtype=torch.bfloat16, device="cuda")
        weight = torch.ones((256, 6144), dtype=torch.bfloat16, device="cuda")
        x[:, 1::2] = -1
        weight[:, ::2] = -1
    elif pattern == "small_large":
        x = torch.full((m, 6144), 2**-8, dtype=torch.bfloat16, device="cuda")
        weight = torch.full(
            (256, 6144),
            2**4,
            dtype=torch.bfloat16,
            device="cuda",
        )
        weight[1::2].mul_(-1)
    else:
        raise ValueError(pattern)
    return x.contiguous(), weight.contiguous()


@pytest.mark.parametrize("m", [16, 32])
@pytest.mark.parametrize("tactic", ["A", "B", "C", "D"])
@pytest.mark.parametrize(
    "pattern",
    ["random", "zero", "ramp", "alternating", "small_large"],
)
def test_glm52_router_logit_gemm(m: int, tactic: str, pattern: str) -> None:
    if is_hip_runtime() or get_jit_cuda_arch().major != 10:
        pytest.skip("SM100 required")
    x, weight = _inputs(m, pattern)
    out = cutedsl_glm52_router_logit_gemm(x, weight, tactic=tactic)
    stock = torch.mm(x, weight.t(), out_dtype=torch.float32)
    oracle = x.double() @ weight.double().t()

    assert out.dtype == torch.float32
    assert out.shape == (m, 256)
    assert out.stride() == (256, 1)
    torch.testing.assert_close(out, stock, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(out.double(), oracle, rtol=2e-3, atol=2e-3)


def test_exact_ties_preserve_stock_topk_ids() -> None:
    if is_hip_runtime() or get_jit_cuda_arch().major != 10:
        pytest.skip("SM100 required")
    from sglang.srt.layers.moe.topk import fused_topk

    x = torch.ones((16, 6144), dtype=torch.bfloat16, device="cuda")
    weight = torch.zeros((256, 6144), dtype=torch.bfloat16, device="cuda")
    candidate = cutedsl_glm52_router_logit_gemm(x, weight, tactic="A")
    stock = torch.mm(x, weight.t(), out_dtype=torch.float32)
    bias = torch.zeros(256, dtype=torch.float32, device="cuda")
    stock_weights, stock_ids = fused_topk(
        x,
        stock,
        8,
        True,
        correction_bias=bias,
        scoring_func="sigmoid",
    )
    candidate_weights, candidate_ids = fused_topk(
        x,
        candidate,
        8,
        True,
        correction_bias=bias,
        scoring_func="sigmoid",
    )
    expected_ids = torch.arange(8, dtype=torch.int32, device="cuda").expand(16, 8)
    torch.testing.assert_close(stock_ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(candidate_weights, stock_weights, rtol=0, atol=0)
    torch.testing.assert_close(candidate_ids, stock_ids, rtol=0, atol=0)


def test_near_ties_preserve_stock_topk_ids() -> None:
    if is_hip_runtime() or get_jit_cuda_arch().major != 10:
        pytest.skip("SM100 required")
    from sglang.srt.layers.moe.topk import fused_topk

    x = torch.zeros((32, 6144), dtype=torch.bfloat16, device="cuda")
    x[:, 0] = 1
    weight = torch.zeros((256, 6144), dtype=torch.bfloat16, device="cuda")
    weight[:9, 0] = torch.arange(9, dtype=torch.float32, device="cuda") * 2**-10
    candidate = cutedsl_glm52_router_logit_gemm(x, weight, tactic="C")
    stock = torch.mm(x, weight.t(), out_dtype=torch.float32)
    bias = torch.zeros(256, dtype=torch.float32, device="cuda")
    stock_weights, stock_ids = fused_topk(
        x,
        stock,
        8,
        True,
        correction_bias=bias,
        scoring_func="sigmoid",
    )
    candidate_weights, candidate_ids = fused_topk(
        x,
        candidate,
        8,
        True,
        correction_bias=bias,
        scoring_func="sigmoid",
    )
    torch.testing.assert_close(candidate, stock, rtol=0, atol=0)
    torch.testing.assert_close(candidate_weights, stock_weights, rtol=0, atol=0)
    torch.testing.assert_close(candidate_ids, stock_ids, rtol=0, atol=0)


def test_invalid_shape_is_fail_closed() -> None:
    x = torch.empty((8, 6144), dtype=torch.bfloat16, device="cuda")
    weight = torch.empty((256, 6144), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="exactly"):
        cutedsl_glm52_router_logit_gemm(x, weight, tactic="A")
