"""MoE masked grouped GEMM optimized paths."""

from __future__ import annotations

import deep_gemm
import torch

from sglang.srt.layers.glm52_opt.infini_moe_align import infini_mk_alignment
from sglang.srt.layers.glm52_opt.kernels.scale_pack import pack_scales


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
    # Respect SGLang's process-wide PDL policy: do not touch anything else global.
    #
    # The M-tile alignment IS now scoped per call, deliberately. The earlier note
    # here warned that a per-call get/set diverges from the production baseline;
    # that is exactly the point, and it is now measured rather than assumed:
    # DeepGEMM's masked branch hard-codes a 128-row M tile while decode has only a
    # handful of live rows per expert, so picking the tile from expected_m is
    # bit-exact (calc_diff == 0.0) and worth up to 1.14x. The alignment is restored
    # in a finally, so a stock prefill GEMM falling through try_dispatch_moe_masked
    # always sees 128 -- leaving a small tile installed would cost prefill ~3x.
    # Buckets, crossovers and the regression evidence: infini_moe_align.py.
    with infini_mk_alignment(expected_m):
        deep_gemm.fp8_m_grouped_gemm_nt_masked(
            (x_fp8, x_packed),
            (w_fp8, w_packed),
            out,
            masked_m,
            expected_m,
            disable_ue8m0_cast=True,
        )
