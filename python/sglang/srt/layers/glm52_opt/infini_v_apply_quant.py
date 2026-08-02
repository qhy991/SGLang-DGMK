"""B300 GLM-5.2 fused BF16 V-apply BMM + packed o_proj quant.

The measured decode graph executes two nodes after attention: a BF16 V-apply
BMM that writes directly into batch-major flattened storage, then group-128 FP8
quantization. This kernel writes o_proj's packed FP8/UE8M0 ABI directly while
preserving the stock BF16 rounding boundary without spilling that activation.

Default off. Admission is fail-closed to B300 decode CUDA-graph buckets, the
exact GLM-5.2 tensor layouts, the packed DeepGEMM o_proj backend, and no KV-B
LoRA correction.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs

_HEADS = 64
_BMM_K = 512
_V_DIM = 256
_GROUP = 128
_HIDDEN = _HEADS * _V_DIM
_OUTPUT = 6144


@triton.jit
def _infini_v_apply_quant_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    scale_bytes_ptr,
    stride_a_h: tl.constexpr,
    stride_a_m: tl.constexpr,
    stride_a_k: tl.constexpr,
    stride_b_h: tl.constexpr,
    stride_b_k: tl.constexpr,
    stride_b_n: tl.constexpr,
    rows: tl.constexpr,
    aligned_rows: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    group_id = tl.program_id(0)
    head_id = group_id // 2
    half_id = group_id % 2

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = half_id * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in tl.static_range(0, 512, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        a = tl.load(
            a_ptr
            + head_id * stride_a_h
            + offs_m[:, None] * stride_a_m
            + offs_k[None, :] * stride_a_k,
            mask=offs_m[:, None] < rows,
            other=0.0,
        )
        b = tl.load(
            b_ptr
            + head_id * stride_b_h
            + offs_k[:, None] * stride_b_k
            + offs_n[None, :] * stride_b_n
        )
        acc = tl.dot(a, b, acc)

    # torch.bmm stores BF16 before the standalone quantizer reads it. Keep this
    # exact observable rounding point even though the intermediate never spills.
    rounded = acc.to(tl.bfloat16).to(tl.float32)
    magnitudes = tl.abs(rounded)
    magnitudes = tl.where(rounded == rounded, magnitudes, 0.0)
    absmax = tl.maximum(tl.max(magnitudes, axis=1), 1.0e-10)
    raw_scale = absmax / 448.0
    raw_bits = raw_scale.to(tl.int32, bitcast=True)
    exponent = (raw_bits >> 23) & 0xFF
    exponent += (raw_bits & 0x7FFFFF) != 0
    scale_bits = exponent << 23
    scale = scale_bits.to(tl.float32, bitcast=True)

    quantized = tl.clamp(
        rounded / scale[:, None], -448.0, 448.0
    ).to(tl.float8e4nv)
    flat_col = group_id * 128 + tl.arange(0, BLOCK_N)
    tl.store(
        out_ptr + offs_m[:, None] * 16384 + flat_col[None, :],
        quantized,
        mask=offs_m[:, None] < rows,
    )

    byte_offset = (
        ((group_id // 4) * aligned_rows + offs_m) * 4 + group_id % 4
    )
    tl.store(
        scale_bytes_ptr + byte_offset,
        exponent.to(tl.uint8),
        mask=offs_m < rows,
    )


def enabled() -> bool:
    return envs.SGLANG_INFINI_V_APPLY_QUANT.get()


def _ceil_align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _o_proj_is_compatible(o_proj, device: torch.device) -> bool:
    weight = getattr(o_proj, "weight", None)
    quant_method = getattr(o_proj, "quant_method", None)
    weight_scale = getattr(o_proj, "weight_scale_inv", None)
    block_size = tuple(getattr(quant_method, "weight_block_size", ()) or ())
    if not (
        torch.is_tensor(weight)
        and weight.is_cuda
        and weight.device == device
        and weight.dtype is torch.float8_e4m3fn
        and weight.shape == (_OUTPUT, _HIDDEN)
        and weight.is_contiguous()
        and torch.is_tensor(weight_scale)
        and weight_scale.is_cuda
        and weight_scale.device == device
        and weight_scale.dtype is torch.int32
        and weight_scale.shape == (_OUTPUT, _HIDDEN // _GROUP // 4)
        and block_size == (_GROUP, _GROUP)
        and getattr(quant_method, "block_quant", False)
    ):
        return False

    from sglang.srt.layers.quantization.fp8_utils import (
        deepgemm_w8a8_block_fp8_linear_with_fallback,
    )

    return (
        getattr(quant_method, "w8a8_block_fp8_linear", None)
        is deepgemm_w8a8_block_fp8_linear_with_fallback
    )


def is_available(
    a_bf16: torch.Tensor,
    b_bf16: torch.Tensor,
    o_proj,
    *,
    decode_or_idle: bool,
    capture_mode: bool,
    lora_active: bool,
) -> bool:
    """Return True only for the exact measured production boundary."""
    if not enabled() or not decode_or_idle or not capture_mode or lora_active:
        return False
    rows = a_bf16.shape[1] if a_bf16.ndim == 3 else 0
    if not (
        a_bf16.is_cuda
        and a_bf16.dtype is torch.bfloat16
        and a_bf16.shape == (_HEADS, rows, _BMM_K)
        and 0 < rows <= 16
        and a_bf16.stride() == (_BMM_K, _HEADS * _BMM_K, 1)
        and b_bf16.is_cuda
        and b_bf16.device == a_bf16.device
        and b_bf16.dtype is torch.bfloat16
        and b_bf16.shape == (_HEADS, _BMM_K, _V_DIM)
        and b_bf16.stride() == (_BMM_K * _V_DIM, 1, _BMM_K)
    ):
        return False
    if torch.cuda.get_device_capability(a_bf16.device) != (10, 3):
        return False
    return _o_proj_is_compatible(o_proj, a_bf16.device)


def fused_v_apply_quant(
    a_bf16: torch.Tensor,
    b_bf16: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Emit the exact prequantized tuple consumed by block-FP8 o_proj."""
    rows = a_bf16.shape[1]
    aligned_rows = _ceil_align(rows, 4)
    values = torch.empty(
        (rows, _HIDDEN), device=a_bf16.device, dtype=torch.float8_e4m3fn
    )
    scale_storage = torch.empty(
        (_HIDDEN // _GROUP // 4, aligned_rows),
        device=a_bf16.device,
        dtype=torch.int32,
    )
    _infini_v_apply_quant_kernel[(_HEADS * 2,)](
        a_bf16,
        b_bf16,
        values,
        scale_storage.view(torch.uint8),
        stride_a_h=a_bf16.stride(0),
        stride_a_m=a_bf16.stride(1),
        stride_a_k=a_bf16.stride(2),
        stride_b_h=b_bf16.stride(0),
        stride_b_k=b_bf16.stride(1),
        stride_b_n=b_bf16.stride(2),
        rows=rows,
        aligned_rows=aligned_rows,
        BLOCK_M=16,
        BLOCK_N=128,
        BLOCK_K=128,
        # Keep the least-regressive audited identity for reproducibility. W8
        # was neutral in isolation but slowed the production chain by 4.6%; a
        # W4 resource experiment worsened that chain by 11.8%. The feature is
        # default-off and retained only as negative-result evidence.
        num_warps=8,
        num_stages=2,
    )
    return values, scale_storage.transpose(-1, -2)[:rows, :]
