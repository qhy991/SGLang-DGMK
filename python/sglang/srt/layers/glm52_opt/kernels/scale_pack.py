"""Shared CUDA scale-pack extension for packed-UE8M0 GEMM candidates."""

from __future__ import annotations

import os
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch

_KERNELS_DIR = Path(__file__).resolve().parent / "kernels" / "cuda"
_SCALE_PACK_CU = _KERNELS_DIR / "scale_pack.cu"


@lru_cache(maxsize=1)
def _load_pack_ext():
    if not _SCALE_PACK_CU.is_file():
        return None
    try:
        from torch.utils.cpp_extension import load as load_ext

        return load_ext(
            name="sglang_glm52_scale_pack",
            sources=[str(_SCALE_PACK_CU)],
            verbose=False,
        )
    except Exception as exc:  # pragma: no cover
        warnings.warn(
            f"glm52_opt: scale-pack CUDA build failed ({exc}); using torch fallback",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def pack_scales_torch(
    x_scale: torch.Tensor, w_scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    from deep_gemm import get_mn_major_tma_aligned_packed_ue8m0_tensor as packt

    n = w_scale.shape[0] * 128
    row_of_block = torch.arange(n, device=w_scale.device) // 128
    return packt(x_scale), packt(w_scale.index_select(-2, row_of_block))


def pack_scales(
    x_scale: torch.Tensor, w_scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    ext = _load_pack_ext()
    if ext is not None:
        return ext.pack_scales(x_scale, w_scale)
    return pack_scales_torch(x_scale, w_scale)


def unpack_f32_block_scale_if_packed(
    weight_scale: torch.Tensor,
    weight_shape: tuple[int, int],
    block_size: list[int],
) -> torch.Tensor:
    if weight_scale.dtype != torch.int32:
        return weight_scale
    from sglang.srt.layers.quantization.fp8_utils import _unpack_ue8m0_scale_for_triton

    return _unpack_ue8m0_scale_for_triton(weight_scale, weight_shape, block_size)
