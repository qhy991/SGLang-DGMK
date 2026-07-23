"""FP8 GEMM optimized paths for GLM-5.2.

Routing policy (aligned with llm_flops DECODE/PREFILL winners):

- ``q_b_proj`` (decode): DeepGEMM-GLM52 ``fp8_gemm_nt_packed_warp`` for the
  production int32 packed-UE8M0 ABI; historical f32 inputs retain the fused path
- ``o_proj`` (decode): native packed-UE8M0 + ``fp8_gemm_nt`` (matches o_proj_decode_hbm35)
- All other registered ``fp8_gemm`` ops: load archive ``candidate.run`` (Triton / pack)

Archive candidates expect **float32 block scales**. When SGLang has already packed
UE8M0 int32 scales, we unpack before calling ``run()``.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import deep_gemm

from sglang.srt.layers.glm52_opt.experimental_deepgemm import (
    get_experimental_deep_gemm,
    has_fused_fp8_gemm_nt,
    has_packed_warp_fp8_gemm_nt,
)
from sglang.srt.layers.glm52_opt.config import allow_abi_adapter
from sglang.srt.layers.glm52_opt.kernels.scale_pack import pack_scales

logger = logging.getLogger(__name__)

# Ops whose archive winner is reproduced by a native path (no archive file needed).
_NATIVE_PACKED_OPS = frozenset({"o_proj"})
_NATIVE_FORK_OPS = frozenset({"q_b_proj"})


def _unpack_weight_scale_f32(
    w_scale: torch.Tensor,
    weight_shape: Tuple[int, int],
    block_size: List[int],
) -> torch.Tensor:
    if w_scale.dtype != torch.int32:
        return w_scale
    from sglang.srt.layers.quantization.fp8_utils import _unpack_ue8m0_scale_for_triton

    return _unpack_ue8m0_scale_for_triton(w_scale, weight_shape, block_size)


def _unpack_act_scale_f32(
    x_scale: torch.Tensor, m: int, k: int, block_k: int
) -> torch.Tensor:
    """Unpack per-token UE8M0 int32 scales to f32 [M, K//block_k] for Triton candidates."""
    if x_scale.dtype != torch.int32:
        return x_scale
    # Packed layout: (M, K//block_k//4) int32, 4 exponents per int32 little-endian.
    k_groups = (k + block_k - 1) // block_k
    if x_scale.shape[0] != m:
        # Some layouts are TMA-aligned / transposed — make contiguous MN-major view.
        x_scale = x_scale.reshape(m, -1) if x_scale.numel() >= m else x_scale
    sf_u8 = x_scale.contiguous().view(torch.uint8).view(m, -1)
    sf_fp32 = (sf_u8.to(torch.int32) << 23).view(torch.float32)
    return sf_fp32[:, :k_groups].contiguous()


def ensure_f32_block_scales(
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    *,
    m: int,
    weight_shape: Tuple[int, int],
    block_size: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    block_n, block_k = block_size[0], block_size[1]
    n, k = weight_shape
    x_f = _unpack_act_scale_f32(x_scale, m, k, block_k)
    w_f = _unpack_weight_scale_f32(w_scale, weight_shape, block_size)
    return x_f, w_f


def _run_q_b_fused(
    x_fp8: torch.Tensor,
    w_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    fork = get_experimental_deep_gemm()
    if fork is not None and hasattr(fork, "fp8_gemm_nt_fused"):
        # Fused entry accepts f32 UE8M0-valued scales (packs inside the kernel).
        if x_scale.dtype == torch.int32 or w_scale.dtype == torch.int32:
            x_scale, w_scale = ensure_f32_block_scales(
                x_scale,
                w_scale,
                m=x_fp8.shape[0],
                weight_shape=tuple(w_fp8.shape),
                block_size=[128, 128],
            )
        fork.fp8_gemm_nt_fused((x_fp8, x_scale), (w_fp8, w_scale), out)
        return
    if x_scale.dtype == torch.int32 and w_scale.dtype == torch.int32:
        deep_gemm.fp8_gemm_nt(
            (x_fp8, x_scale), (w_fp8, w_scale), out, compiled_dims="nk"
        )
        return
    x_packed, w_packed = pack_scales(x_scale, w_scale)
    deep_gemm.fp8_gemm_nt((x_fp8, x_packed), (w_fp8, w_packed), out, compiled_dims="nk")


def _run_q_b_packed_warp(
    x_fp8: torch.Tensor,
    w_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Run the exact production packed-scale ABI without an adapter launch."""
    fork = get_experimental_deep_gemm()
    if fork is None or not hasattr(fork, "fp8_gemm_nt_packed_warp"):
        raise RuntimeError("experimental DeepGEMM packed-warp entry is unavailable")
    fork.fp8_gemm_nt_packed_warp(
        (x_fp8, x_scale),
        (w_fp8, w_scale),
        out,
        # Match the stock production call first; fixed-N/K specialization is
        # evaluated as a separate experiment and must earn its own oracle.
        compiled_dims="",
    )


def _run_packed_fp8_gemm(
    x_fp8: torch.Tensor,
    w_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    out: torch.Tensor,
    *,
    compiled_dims: Optional[str] = "nk",
) -> None:
    if w_scale.dtype == torch.int32 and x_scale.dtype == torch.int32:
        deep_gemm.fp8_gemm_nt(
            (x_fp8, x_scale),
            (w_fp8, w_scale),
            out,
            compiled_dims=compiled_dims,
        )
        return
    x_packed, w_packed = pack_scales(x_scale, w_scale)
    deep_gemm.fp8_gemm_nt(
        (x_fp8, x_packed),
        (w_fp8, w_packed),
        out,
        compiled_dims=compiled_dims,
    )


def _run_archive_candidate(
    archive_ref: str,
    x_fp8: torch.Tensor,
    w_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    out: torch.Tensor,
    block_size: List[int],
) -> None:
    from sglang.srt.layers.glm52_opt.archive_loader import load_run_fn

    m = x_fp8.shape[0]
    n, k = w_fp8.shape[0], w_fp8.shape[1]
    x_f, w_f = ensure_f32_block_scales(
        x_scale,
        w_scale,
        m=m,
        weight_shape=(n, k),
        block_size=block_size,
    )
    run_fn = load_run_fn(archive_ref)
    inputs = {
        "x_fp8": x_fp8,
        "w_fp8": w_fp8,
        "x_scale": x_f,
        "w_scale": w_f,
        "out": out,
        # Prefill flat candidates (e.g. fused_qkv_a_prefill.py) need these.
        "rows": m,
        "N": n,
        "M": m,
    }
    result = run_fn(inputs)
    if result is not None and result is not out:
        # Some Triton candidates allocate their own out and return it.
        if result.shape == out.shape:
            out.copy_(result)
        else:
            out.copy_(result.view_as(out))


def run_fp8_gemm(
    op_name: str,
    x_fp8: torch.Tensor,
    w_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    out: torch.Tensor,
    block_size: List[int],
    archive_ref: str,
    phase: str = "decode",
) -> Tuple[bool, str]:
    """Run optimized FP8 GEMM. Returns (ok, path) where path is
    native_packed_warp | native_fork | native_packed | archive |
    packed_fallback | packed_default.
    """
    packed_scales = x_scale.dtype == torch.int32 or w_scale.dtype == torch.int32

    # 1) q_b decode only. Production supplies BOTH scales in DeepGEMM's
    # non-contiguous int32 MN-major packed layout. The packed-warp entry consumes
    # that ABI directly and is fail-closed: missing overlay, mixed dtypes, or a
    # rejected shape returns to the caller, which invokes stock DeepGEMM.
    if op_name in _NATIVE_FORK_OPS and phase == "decode":
        packed_pair = x_scale.dtype == torch.int32 and w_scale.dtype == torch.int32
        if packed_pair:
            try:
                if not has_packed_warp_fp8_gemm_nt():
                    return False, "packed_warp_unavailable"
                _run_q_b_packed_warp(x_fp8, w_fp8, x_scale, w_scale, out)
                return True, "native_packed_warp"
            except Exception as exc:
                logger.warning(
                    "glm52_opt packed q_b path failed (%s); falling back to stock",
                    exc,
                )
                return False, "packed_warp_error"
        if packed_scales:
            return False, "packed_abi_mixed_dtypes"

        # Historical raw-f32 experiment remains available for offline replay,
        # but production never reaches it through an unpack adapter.
        try:
            if has_fused_fp8_gemm_nt():
                _run_q_b_fused(x_fp8, w_fp8, x_scale, w_scale, out)
                return True, "native_fork"
        except Exception as exc:
            logger.warning(
                "glm52_opt f32 q_b path failed (%s); falling back to stock", exc
            )
            return False, "native_fork_error"

    # 2) o_proj: packed UE8M0 (decode + prefill; matches o_proj_decode_hbm35)
    if op_name in _NATIVE_PACKED_OPS:
        _run_packed_fp8_gemm(x_fp8, w_fp8, x_scale, w_scale, out)
        return True, "native_packed"

    # 3) Everything else with an archive_ref: real candidate (PR5 Triton, pack, etc.)
    if archive_ref:
        if packed_scales and not allow_abi_adapter():
            return False, "packed_abi_requires_adapter"
        try:
            _run_archive_candidate(
                archive_ref, x_fp8, w_fp8, x_scale, w_scale, out, block_size
            )
            return True, "archive"
        except Exception as exc:
            logger.warning(
                "glm52_opt archive candidate failed for %s/%s (%s): %s; "
                "falling back to packed DeepGEMM",
                op_name,
                phase,
                archive_ref,
                exc,
            )
            _run_packed_fp8_gemm(x_fp8, w_fp8, x_scale, w_scale, out)
            return True, "packed_fallback"

    _run_packed_fp8_gemm(x_fp8, w_fp8, x_scale, w_scale, out)
    return True, "packed_default"
