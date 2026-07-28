# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Exact GLM-5.2 decode router-logit GEMM for Blackwell.

This experiment is deliberately narrower than ``cutedsl_bf16_gemm``:

* A is contiguous BF16 ``[M, 6144]`` for M in {16, 32};
* stored B is contiguous BF16 ``[256, 6144]``;
* C is newly allocated contiguous FP32 ``[M, 256]``;
* the kernel is one-SM tcgen05/TMEM/TMA, has no bias and does not use PDL.

The generic TGV kernel body already parameterizes its TMEM-to-RMEM copy and
global store by C's element type.  This module supplies a dedicated FP32 C
compile path and an intentionally bounded tactic portfolio.  It must remain
default-off until the Task 31 correctness, graph, and latency gates pass.
"""

from __future__ import annotations

from typing import NamedTuple

import cuda.bindings.driver as cuda
import cutlass
import torch
from cutlass.cute import experimental as cute_ext
from cutlass.cute.runtime import make_fake_stream

from sglang.jit_kernel.cutedsl_bf16_gemm import (
    TgvGemmCuteExtKernel,
    _bmm_no_bias,
    _to_cute_swap,
)
from sglang.srt.utils import get_device_sm
from sglang.srt.utils.common import direct_register_custom_op


class RouterTactic(NamedTuple):
    cta_m: int
    cta_n: int
    cta_k: int
    num_ab_stage: int


ROUTER_LOGIT_BUILD_IDENTITY = "glm52_router_logit_fp32_v1"
ROUTER_LOGIT_TACTICS: tuple[RouterTactic, ...] = (
    RouterTactic(64, 8, 128, 8),  # A: M16=8 CTAs,  M32=16 CTAs
    RouterTactic(64, 16, 128, 8),  # B: M16=4 CTAs,  M32=8 CTAs
    RouterTactic(64, 32, 128, 6),  # C: M16=4 CTAs,  M32=4 CTAs
    RouterTactic(128, 8, 128, 6),  # D: M16=4 CTAs,  M32=8 CTAs
)
ROUTER_LOGIT_TACTIC_NAMES = ("A", "B", "C", "D")
_COMPILE_CACHE: dict[tuple[object, ...], object] = {}


def resolve_router_tactic(tactic: str) -> tuple[int, RouterTactic]:
    if tactic not in ROUTER_LOGIT_TACTIC_NAMES:
        raise ValueError(
            f"unknown GLM-5.2 router tactic {tactic!r}; "
            f"choose from {ROUTER_LOGIT_TACTIC_NAMES}"
        )
    index = ROUTER_LOGIT_TACTIC_NAMES.index(tactic)
    return index, ROUTER_LOGIT_TACTICS[index]


def _validate_exact_abi(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    tactic_id: int,
) -> None:
    if get_device_sm() != 100:
        raise RuntimeError("Task 31 router GEMM requires SM100")
    if tactic_id < 0 or tactic_id >= len(ROUTER_LOGIT_TACTICS):
        raise ValueError(f"router tactic id {tactic_id} is outside [0, 4)")
    if hidden_states.device != router_weight.device or not hidden_states.is_cuda:
        raise ValueError("router inputs must share one CUDA device")
    if hidden_states.dtype != torch.bfloat16 or router_weight.dtype != torch.bfloat16:
        raise TypeError("router inputs must both be BF16")
    if tuple(hidden_states.shape) not in ((16, 6144), (32, 6144)):
        raise ValueError("hidden_states must be exactly [16|32, 6144]")
    if tuple(router_weight.shape) != (256, 6144):
        raise ValueError("router_weight must be exactly [256, 6144]")
    if tuple(hidden_states.stride()) != (6144, 1):
        raise ValueError("hidden_states must have stride (6144, 1)")
    if tuple(router_weight.stride()) != (6144, 1):
        raise ValueError("router_weight must have stride (6144, 1)")
    if hidden_states.storage_offset() != 0 or router_weight.storage_offset() != 0:
        raise ValueError("router inputs must have zero storage offset")
    if hidden_states.data_ptr() % 32 or router_weight.data_ptr() % 32:
        raise ValueError("router inputs must be at least 32-byte aligned")


def _compiled_router_kernel(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    out: torch.Tensor,
    tactic_id: int,
):
    tactic = ROUTER_LOGIT_TACTICS[tactic_id]
    a_, b_, c_, bias_, layout = _to_cute_swap(
        hidden_states,
        router_weight.t(),
        out,
        None,
    )
    assert bias_ is None
    key = (
        ROUTER_LOGIT_BUILD_IDENTITY,
        get_device_sm(),
        str(hidden_states.dtype),
        str(router_weight.dtype),
        str(out.dtype),
        tuple(hidden_states.shape),
        tuple(router_weight.shape),
        tuple(hidden_states.stride()),
        tuple(router_weight.stride()),
        tuple(out.stride()),
        tactic_id,
        tuple(tactic),
        layout,
        False,  # use_2cta
        False,  # use_pdl
        False,  # has_bias
    )
    compiled = _COMPILE_CACHE.get(key)
    if compiled is None:
        gemm = TgvGemmCuteExtKernel(
            acc_dtype=cutlass.Float32,
            cta_m=tactic.cta_m,
            cta_n=tactic.cta_n,
            cta_k=tactic.cta_k,
            num_ab_stage=tactic.num_ab_stage,
            use_2cta=False,
            use_pdl=False,
            pdl_launch=False,
            has_bias=False,
        )
        compiled = cute_ext.compile(
            _bmm_no_bias,
            gemm,
            a_,
            b_,
            c_,
            make_fake_stream(),
        )
        _COMPILE_CACHE[key] = compiled
    return compiled, a_, b_, c_


def _router_logit_gemm_run(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    tactic_id: int,
) -> torch.Tensor:
    _validate_exact_abi(hidden_states, router_weight, tactic_id)
    out = torch.empty(
        (hidden_states.shape[0], 256),
        dtype=torch.float32,
        device=hidden_states.device,
    )
    compiled, a_, b_, c_ = _compiled_router_kernel(
        hidden_states,
        router_weight,
        out,
        tactic_id,
    )
    stream = cuda.CUstream(torch.cuda.current_stream(hidden_states.device).cuda_stream)
    compiled(a_, b_, c_, stream)
    return out


def _router_logit_gemm_fake(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    tactic_id: int,
) -> torch.Tensor:
    if hidden_states.dim() != 2 or router_weight.dim() != 2:
        raise ValueError("router tensors must be rank two")
    return hidden_states.new_empty(
        (hidden_states.shape[0], router_weight.shape[0]),
        dtype=torch.float32,
    )


direct_register_custom_op(
    op_name="cutedsl_glm52_router_logit_gemm",
    op_func=_router_logit_gemm_run,
    mutates_args=[],
    fake_impl=_router_logit_gemm_fake,
)


def cutedsl_glm52_router_logit_gemm(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    *,
    tactic: str,
) -> torch.Tensor:
    """Run one explicitly selected exact Task 31 tactic.

    Import, compilation, layout, and launch failures intentionally propagate;
    this API never executes the stock denominator after candidate selection.
    """

    tactic_id, _ = resolve_router_tactic(tactic)
    return torch.ops.sglang.cutedsl_glm52_router_logit_gemm(
        hidden_states,
        router_weight,
        tactic_id,
    )
