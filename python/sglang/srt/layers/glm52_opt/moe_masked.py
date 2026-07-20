"""MoE masked grouped GEMM optimized paths."""

from __future__ import annotations

import deep_gemm
import torch

from sglang.srt.layers.glm52_opt.kernels.scale_pack import pack_scales


def _get_pdl():
    try:
        return deep_gemm.get_pdl()
    except Exception:
        return None


def _set_pdl(value):
    if value is None:
        return
    try:
        deep_gemm.set_pdl(value)
    except Exception:
        pass


def run_moe_masked(
    x_fp8: torch.Tensor,
    w_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    out: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
) -> None:
    if x_scale.dtype == torch.int32 and w_scale.dtype == torch.int32:
        x_packed, w_packed = x_scale, w_scale
    else:
        x_packed, w_packed = pack_scales(x_scale, w_scale)
    saved = _get_pdl()
    try:
        _set_pdl(True)
        deep_gemm.fp8_m_grouped_gemm_nt_masked(
            (x_fp8, x_packed),
            (w_fp8, w_packed),
            out,
            masked_m,
            expected_m,
            disable_ue8m0_cast=True,
        )
    finally:
        _set_pdl(saved)
