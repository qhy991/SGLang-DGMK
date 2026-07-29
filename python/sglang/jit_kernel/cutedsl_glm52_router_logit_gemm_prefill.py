# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Exact GLM-5.2 M4096 router-logit GEMM for Blackwell.

This default-off experiment preserves the production ABI:

* contiguous BF16 hidden states ``[4096, 6144]``;
* contiguous stored BF16 router weight ``[256, 6144]``;
* newly allocated contiguous FP32 logits ``[4096, 256]``;
* no bias, PDL, split-K, helper kernel, or fallback.

After the production A/B swap the CuTe kernel sees M=256, N=4096, K=6144.
The bounded A-D portfolio uses two-CTA tcgen05 MMA, TMEM accumulation, TMA
loads, and a direct FP32 global store.
"""

from __future__ import annotations

import os
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
from sglang.jit_kernel.cutedsl_glm52_router_logit_gemm import (
    ROUTER_LOGIT_CUTLASS_DSL_VERSION,
    ROUTER_LOGIT_DISTRIBUTION_TREES,
)
from sglang.srt.utils import get_device_sm
from sglang.srt.utils.common import direct_register_custom_op


class RouterPrefillTactic(NamedTuple):
    cta_m: int
    cta_n: int
    cta_k: int
    num_ab_stage: int


ROUTER_LOGIT_PREFILL_BUILD_IDENTITY = "glm52_router_logit_prefill_fp32_v2"
ROUTER_LOGIT_PREFILL_TACTICS: tuple[RouterPrefillTactic, ...] = (
    RouterPrefillTactic(64, 64, 128, 6),
    RouterPrefillTactic(64, 128, 128, 7),
    RouterPrefillTactic(128, 64, 128, 5),
    RouterPrefillTactic(128, 128, 128, 4),
)
ROUTER_LOGIT_PREFILL_TACTIC_NAMES = ("A", "B", "C", "D")
_COMPILE_CACHE: dict[tuple[object, ...], object] = {}
_LAUNCH_BACKEND = os.environ.get("SGLANG_GLM52_ROUTER_PREFILL_LAUNCH", "driver")
if _LAUNCH_BACKEND not in ("driver", "tvmffi"):
    raise ValueError(
        "SGLANG_GLM52_ROUTER_PREFILL_LAUNCH must be driver or tvmffi"
    )
_USE_TVM_FFI = _LAUNCH_BACKEND == "tvmffi"
_EXPECTED_TVM_FFI = "1" if _USE_TVM_FFI else "0"
for _name in (
    "CUTE_DSL_ENABLE_TVM_FFI",
    "CUTE_EXPERIMENTAL_DSL_ENABLE_TVM_FFI",
):
    if os.environ.get(_name) != _EXPECTED_TVM_FFI:
        raise RuntimeError(
            f"{_LAUNCH_BACKEND} launch requires {_name}={_EXPECTED_TVM_FFI}"
        )
if _USE_TVM_FFI:
    # CUTLASS DSL 4.5.2 attaches the TVM-FFI argument converter to the
    # stable CuTe DSL singleton but not yet to CuteExperimentalDSL.  The
    # TGV kernel is experimental while its ABI uses the same cute.Tensor
    # and EnvStream argument types, so bind that shipped converter here.
    from cutlass.cute import _tvm_ffi_args_spec_converter
    from cutlass.cutlass_dsl.tvm_ffi_provider import TVMFFIJitCompiledFunction

    _tvm_ffi_args_spec_converter.attach_args_spec_converter(
        cute_ext._dsl.CuteExperimentalDSL._get_dsl()
    )
    # CuteExperimentalDSL 4.5.2 unconditionally reassigns the compiled
    # function's class after compilation.  Its normal driver class has an
    # incompatible Python layout with TVMFFIJitCompiledFunction.  This exact
    # GEMM has no compiler-added workspace arguments, so retain the FFI class
    # at that final assignment instead of applying the driver-only checker.
    cute_ext._dsl.CuteExperimentalDSL.JitCompiledFunction = (
        TVMFFIJitCompiledFunction
    )


def resolve_router_prefill_tactic(
    tactic: str,
) -> tuple[int, RouterPrefillTactic]:
    if tactic not in ROUTER_LOGIT_PREFILL_TACTIC_NAMES:
        raise ValueError(
            f"unknown GLM-5.2 prefill router tactic {tactic!r}; "
            f"choose from {ROUTER_LOGIT_PREFILL_TACTIC_NAMES}"
        )
    index = ROUTER_LOGIT_PREFILL_TACTIC_NAMES.index(tactic)
    return index, ROUTER_LOGIT_PREFILL_TACTICS[index]


def _validate_exact_prefill_abi(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    tactic_id: int,
) -> None:
    if get_device_sm() != 100:
        raise RuntimeError("Task 33 router GEMM requires SM100")
    if tactic_id < 0 or tactic_id >= len(ROUTER_LOGIT_PREFILL_TACTICS):
        raise ValueError(f"router prefill tactic id {tactic_id} is outside [0, 4)")
    if hidden_states.device != router_weight.device or not hidden_states.is_cuda:
        raise ValueError("router inputs must share one CUDA device")
    if hidden_states.dtype != torch.bfloat16 or router_weight.dtype != torch.bfloat16:
        raise TypeError("router inputs must both be BF16")
    if tuple(hidden_states.shape) != (4096, 6144):
        raise ValueError("hidden_states must be exactly [4096, 6144]")
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


def _compiled_router_prefill_kernel(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    out: torch.Tensor,
    tactic_id: int,
):
    tactic = ROUTER_LOGIT_PREFILL_TACTICS[tactic_id]
    a_, b_, c_, bias_, layout = _to_cute_swap(
        hidden_states,
        router_weight.t(),
        out,
        None,
    )
    assert bias_ is None
    key = (
        ROUTER_LOGIT_PREFILL_BUILD_IDENTITY,
        ROUTER_LOGIT_CUTLASS_DSL_VERSION,
        tuple(
            (
                name,
                record["version"],
                record["tree_sha256"],
            )
            for name, record in sorted(ROUTER_LOGIT_DISTRIBUTION_TREES.items())
        ),
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
        True,  # use_2cta
        False,  # use_pdl
        False,  # has_bias
        _LAUNCH_BACKEND,
    )
    compiled = _COMPILE_CACHE.get(key)
    if compiled is None:
        gemm = TgvGemmCuteExtKernel(
            acc_dtype=cutlass.Float32,
            cta_m=tactic.cta_m,
            cta_n=tactic.cta_n,
            cta_k=tactic.cta_k,
            num_ab_stage=tactic.num_ab_stage,
            use_2cta=True,
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


def _router_logit_gemm_prefill_run(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    tactic_id: int,
) -> torch.Tensor:
    _validate_exact_prefill_abi(hidden_states, router_weight, tactic_id)
    out = torch.empty(
        (4096, 256),
        dtype=torch.float32,
        device=hidden_states.device,
    )
    compiled, a_, b_, c_ = _compiled_router_prefill_kernel(
        hidden_states,
        router_weight,
        out,
        tactic_id,
    )
    # Keep the stream explicit for both launch backends. TVM-FFI's
    # environment-stream shortcut launches outside torch's active capture
    # stream when reached through a Python CUDA custom-op implementation.
    stream = cuda.CUstream(
        torch.cuda.current_stream(hidden_states.device).cuda_stream
    )
    compiled(a_, b_, c_, stream)
    return out


def _router_logit_gemm_prefill_fake(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    tactic_id: int,
) -> torch.Tensor:
    del tactic_id
    if hidden_states.dim() != 2 or router_weight.dim() != 2:
        raise ValueError("router tensors must be rank two")
    return hidden_states.new_empty(
        (hidden_states.shape[0], router_weight.shape[0]),
        dtype=torch.float32,
    )


direct_register_custom_op(
    op_name="cutedsl_glm52_router_logit_gemm_prefill",
    op_func=_router_logit_gemm_prefill_run,
    mutates_args=[],
    fake_impl=_router_logit_gemm_prefill_fake,
)


def cutedsl_glm52_router_logit_gemm_prefill(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    *,
    tactic: str,
) -> torch.Tensor:
    """Run one explicitly selected exact Task 33 tactic without fallback."""

    tactic_id, _ = resolve_router_prefill_tactic(tactic)
    return torch.ops.sglang.cutedsl_glm52_router_logit_gemm_prefill(
        hidden_states,
        router_weight,
        tactic_id,
    )
