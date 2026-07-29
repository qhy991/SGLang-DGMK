"""GLM-5.2 decode SwiGLU plus direct packed-UE8M0 Triton candidates.

This module is deliberately narrow.  The production dispatcher may select it
only for the audited GLM-5.2 DeepEP-low-latency ABI; explicit benchmark callers
use :func:`silu_mul_quant_packed_explicit`, which raises instead of silently
falling back when any part of that ABI is not satisfied.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Final, Optional

import torch
import triton
import triton.language as tl


_EXPERTS: Final = 32
_EXPERT_SLAB: Final = 1024
_GATE_UP: Final = 4096
_HIDDEN: Final = 2048
_GROUP_SIZE: Final = 128
_TOPK: Final = 8
_PACKED_GROUPS: Final = _HIDDEN // (_GROUP_SIZE * 4)
_TRITON_BUILD: Final = "3.6.0"

# The names are persisted in result files and form part of the JIT cache key.
_TRITON_VARIANTS: Final = {
    "row2048_w8": (2048, 8),
    "split1024_w4": (1024, 4),
    "group512_w1": (512, 1),
}
_CUDA_VARIANT: Final = "cuda_valid_cta"
_VARIANT_NAMES: Final = frozenset((*_TRITON_VARIANTS, _CUDA_VARIANT))


@triton.jit
def _silu_mul_quant_packed_kernel(
    gateup_ptr,
    output_ptr,
    scale_ptr,
    masked_m_ptr,
    stride_input_e,
    stride_input_m,
    stride_output_e,
    stride_output_m,
    stride_scale_e,
    stride_scale_g4,
    stride_scale_m,
    NUM_EXPERTS: tl.constexpr,
    HIDDEN: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Map one live row/split, compute FP32 SwiGLU, and write packed scales."""

    work_id = tl.program_id(0)
    split_id = tl.program_id(1)

    # The only source of row ownership is the device-resident expert-count
    # tensor.  The host supplies only a conservative number of routed slots.
    expert_offsets = tl.arange(0, NUM_EXPERTS)
    masked_m = tl.load(masked_m_ptr + expert_offsets)
    inclusive = tl.cumsum(masked_m)
    total = tl.sum(masked_m)
    if work_id >= total:
        return
    exclusive = inclusive - masked_m
    owner = (exclusive <= work_id) & (work_id < inclusive)
    expert_id = tl.sum(tl.where(owner, expert_offsets, 0))
    token_id = work_id - tl.sum(tl.where(owner, exclusive, 0))

    stride_input_e = tl.cast(stride_input_e, tl.int64)
    stride_input_m = tl.cast(stride_input_m, tl.int64)
    stride_output_e = tl.cast(stride_output_e, tl.int64)
    stride_output_m = tl.cast(stride_output_m, tl.int64)
    stride_scale_e = tl.cast(stride_scale_e, tl.int64)
    stride_scale_g4 = tl.cast(stride_scale_g4, tl.int64)
    stride_scale_m = tl.cast(stride_scale_m, tl.int64)

    offsets = split_id * BLOCK_N + tl.arange(0, BLOCK_N)
    input_base = (
        gateup_ptr + expert_id * stride_input_e + token_id * stride_input_m
    )
    output_base = (
        output_ptr + expert_id * stride_output_e + token_id * stride_output_m
    )

    gate = tl.load(input_base + offsets).to(tl.float32)
    up = tl.load(input_base + HIDDEN + offsets).to(tl.float32)
    # Stock uses FP32 g / (1 + __expf(-g)) followed by an FP32 multiply.
    values = (gate / (1.0 + tl.exp(-gate))) * up

    groups_per_program: tl.constexpr = BLOCK_N // GROUP_SIZE
    packed_words_per_program: tl.constexpr = groups_per_program // 4
    values_2d = tl.reshape(values, (groups_per_program, GROUP_SIZE))

    # CUDA fmaxf ignores a single NaN operand.  Replacing NaN magnitudes by
    # zero gives the same reduction identity as stock's local_max=0 sequence.
    magnitudes = tl.abs(values_2d)
    magnitudes = tl.where(values_2d == values_2d, magnitudes, 0.0)
    absmax = tl.max(magnitudes, axis=1)
    absmax = tl.maximum(absmax, 1.0e-10)
    raw_scale = absmax / 448.0

    # Exact stock cast_to_ue8m0: retain the FP32 exponent and round upward when
    # any mantissa bit is set.  This avoids log2/ceil boundary drift.
    raw_bits = raw_scale.to(tl.int32, bitcast=True)
    exponent = (raw_bits >> 23) & 0xFF
    exponent += (raw_bits & 0x7FFFFF) != 0
    scale_bits = exponent << 23
    scale = scale_bits.to(tl.float32, bitcast=True)

    inverse_scale = tl.reshape(
        1.0 / scale, (groups_per_program, 1)
    )
    quantized = tl.reshape(
        tl.clamp(values_2d * inverse_scale, -448.0, 448.0),
        (BLOCK_N,),
    ).to(output_ptr.dtype.element_ty)
    tl.store(output_base + offsets, quantized)

    # Four consecutive exponent bytes form one little-endian int32.  Physical
    # storage is [E, 4, 1024], while callers receive its [E, 1024, 4] view.
    exponent_2d = tl.reshape(exponent, (packed_words_per_program, 4))
    byte_shifts = tl.arange(0, 4)[None, :] * 8
    packed = tl.sum(exponent_2d << byte_shifts, axis=1)
    word_offsets = (
        split_id * packed_words_per_program
        + tl.arange(0, packed_words_per_program)
    )
    scale_offsets = (
        expert_id * stride_scale_e
        + word_offsets * stride_scale_g4
        + token_id * stride_scale_m
    )
    tl.store(scale_ptr + scale_offsets, packed)


def variant_names() -> tuple[str, ...]:
    return (*_TRITON_VARIANTS, _CUDA_VARIANT)


@lru_cache(maxsize=None)
def _device_capability(device_index: int) -> tuple[int, int]:
    # The first query occurs during benchmark correctness/warmup, never during
    # an audited timing or graph-capture phase.
    return torch.cuda.get_device_capability(device_index)


def _eligibility_error(
    gateup_output: torch.Tensor,
    masked_m: torch.Tensor,
    *,
    group_size: int,
    topk: int,
    swiglu_limit: Optional[float],
    swizzle: bool,
    gemm1_alpha: Optional[float],
    gemm1_clamp_limit: Optional[float],
    num_real_tokens: Optional[int],
    variant: str,
) -> str | None:
    if variant not in _VARIANT_NAMES:
        return f"unknown Task-25 variant {variant!r}"
    if variant in _TRITON_VARIANTS and triton.__version__ != _TRITON_BUILD:
        return (
            f"Task-25 candidate requires Triton {_TRITON_BUILD}, "
            f"found {triton.__version__}"
        )
    if not gateup_output.is_cuda or not masked_m.is_cuda:
        return "Task-25 candidate requires CUDA tensors"
    if _device_capability(gateup_output.device.index or 0) != (10, 0):
        return "Task-25 candidate requires NVIDIA sm_100"
    if gateup_output.dtype != torch.bfloat16:
        return "gateup_output must be bfloat16"
    if tuple(gateup_output.shape) != (_EXPERTS, _EXPERT_SLAB, _GATE_UP):
        return (
            "gateup_output must have exact shape "
            f"{(_EXPERTS, _EXPERT_SLAB, _GATE_UP)}"
        )
    if not gateup_output.is_contiguous():
        return "gateup_output must be contiguous"
    if masked_m.dtype != torch.int32 or tuple(masked_m.shape) != (_EXPERTS,):
        return "masked_m must be int32[32]"
    if not masked_m.is_contiguous():
        return "masked_m must be contiguous"
    if masked_m.device != gateup_output.device:
        return "gateup_output and masked_m must share a device"
    if group_size != _GROUP_SIZE or topk != _TOPK:
        return "Task-25 candidate requires group_size=128 and topk=8"
    if swiglu_limit is not None:
        return "Task-25 candidate supports only unclamped GLM-5.2 SwiGLU"
    if swizzle:
        return "Task-25 candidate does not support swizzled gate/up storage"
    if gemm1_alpha is not None or gemm1_clamp_limit is not None:
        return "Task-25 candidate supports ordinary SiLU, not gemm1_alpha mode"
    if num_real_tokens not in (16, 32):
        return "Task-25 candidate requires a host-known M16 or M32 decode bucket"
    return None


def _allocate_outputs(
    gateup_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty(
        (_EXPERTS, _EXPERT_SLAB, _HIDDEN),
        device=gateup_output.device,
        dtype=torch.float8_e4m3fn,
    )
    scale_storage = torch.empty(
        (_EXPERTS, _PACKED_GROUPS, _EXPERT_SLAB),
        device=gateup_output.device,
        dtype=torch.int32,
    )
    return output, scale_storage


def silu_mul_quant_packed_into(
    gateup_output: torch.Tensor,
    output: torch.Tensor,
    scale_storage: torch.Tensor,
    masked_m: torch.Tensor,
    *,
    num_real_tokens: int,
    variant: str,
) -> None:
    """Lower-level explicit launcher used by poison and ownership tests."""

    if tuple(output.shape) != (_EXPERTS, _EXPERT_SLAB, _HIDDEN):
        raise ValueError("output must have shape [32, 1024, 2048]")
    if output.dtype != torch.float8_e4m3fn or not output.is_contiguous():
        raise ValueError("output must be contiguous float8_e4m3fn")
    if output.device != gateup_output.device:
        raise ValueError("output must share gateup_output's device")
    if tuple(scale_storage.shape) != (
        _EXPERTS,
        _PACKED_GROUPS,
        _EXPERT_SLAB,
    ):
        raise ValueError("scale_storage must have physical shape [32, 4, 1024]")
    if scale_storage.dtype != torch.int32 or not scale_storage.is_contiguous():
        raise ValueError("scale_storage must be contiguous int32")
    if scale_storage.device != gateup_output.device:
        raise ValueError("scale_storage must share gateup_output's device")

    if variant == _CUDA_VARIANT:
        from sglang.srt.layers.glm52_opt.swiglu_quant_cuda import launch_into

        launch_into(
            gateup_output,
            output,
            scale_storage,
            masked_m,
            num_real_tokens=num_real_tokens,
        )
        return

    block_n, num_warps = _TRITON_VARIANTS[variant]
    split_count = _HIDDEN // block_n
    routed_slots = max(1, num_real_tokens * _TOPK)
    _silu_mul_quant_packed_kernel[(routed_slots, split_count)](
        gateup_output,
        output,
        scale_storage,
        masked_m,
        gateup_output.stride(0),
        gateup_output.stride(1),
        output.stride(0),
        output.stride(1),
        scale_storage.stride(0),
        scale_storage.stride(1),
        scale_storage.stride(2),
        NUM_EXPERTS=_EXPERTS,
        HIDDEN=_HIDDEN,
        GROUP_SIZE=_GROUP_SIZE,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=1,
    )


def silu_mul_quant_packed_explicit(
    gateup_output: torch.Tensor,
    masked_m: torch.Tensor,
    *,
    group_size: int,
    topk: int,
    num_real_tokens: Optional[int],
    variant: str,
    swiglu_limit: Optional[float] = None,
    swizzle: bool = False,
    gemm1_alpha: Optional[float] = None,
    gemm1_clamp_limit: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run an explicitly requested candidate, raising on every ineligible ABI."""

    error = _eligibility_error(
        gateup_output,
        masked_m,
        group_size=group_size,
        topk=topk,
        swiglu_limit=swiglu_limit,
        swizzle=swizzle,
        gemm1_alpha=gemm1_alpha,
        gemm1_clamp_limit=gemm1_clamp_limit,
        num_real_tokens=num_real_tokens,
        variant=variant,
    )
    if error is not None:
        raise RuntimeError(error)
    assert num_real_tokens is not None
    output, scale_storage = _allocate_outputs(gateup_output)
    silu_mul_quant_packed_into(
        gateup_output,
        output,
        scale_storage,
        masked_m,
        num_real_tokens=num_real_tokens,
        variant=variant,
    )
    return output, scale_storage.transpose(1, 2)


def maybe_silu_mul_quant_packed(
    gateup_output: torch.Tensor,
    masked_m: torch.Tensor,
    *,
    group_size: int,
    topk: int,
    num_real_tokens: Optional[int],
    variant: str,
    swiglu_limit: Optional[float] = None,
    swizzle: bool = False,
    gemm1_alpha: Optional[float] = None,
    gemm1_clamp_limit: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Fail closed to stock before launch for an automatic production request."""

    error = _eligibility_error(
        gateup_output,
        masked_m,
        group_size=group_size,
        topk=topk,
        swiglu_limit=swiglu_limit,
        swizzle=swizzle,
        gemm1_alpha=gemm1_alpha,
        gemm1_clamp_limit=gemm1_clamp_limit,
        num_real_tokens=num_real_tokens,
        variant=variant,
    )
    if error is not None:
        return None
    return silu_mul_quant_packed_explicit(
        gateup_output,
        masked_m,
        group_size=group_size,
        topk=topk,
        num_real_tokens=num_real_tokens,
        variant=variant,
        swiglu_limit=swiglu_limit,
        swizzle=swizzle,
        gemm1_alpha=gemm1_alpha,
        gemm1_clamp_limit=gemm1_clamp_limit,
    )
