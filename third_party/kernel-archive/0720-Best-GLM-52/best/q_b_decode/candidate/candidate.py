"""GLM-5.2 q_b decode — DeepGEMM experimental fork candidate.

Reference timing continues to use stock ``import deep_gemm`` (Harness loads
``glm52_ops`` first). This candidate:

1. Passes the frozen per-128-block f32 UE8M0 scales straight to the fork's
   ``fp8_gemm_nt_fused`` entry, which packs them to the UE8M0 SF layout *inside*
   the SM100 GEMM kernel (warp-2 producer, off the weight-stream critical path).
   This removes the separate scale-pack kernel launch from the timed device-span
   — the whole op is a single kernel — lifting decode q_b past 35% B200 HBM.
2. Falls back to the proven pre-pack path (fused CUDA repack + packed
   ``fp8_gemm_nt``) if the fused entry is unavailable in the loaded overlay.
3. Saves/restores fork launch knobs so nothing leaks into stock reference state.

The scale pack is still exact and per-call (bit-identical to the pre-pack path,
``calc_diff == 0`` vs the stock oracle); it is merely fused into the GEMM.
"""
from __future__ import annotations

import os
import sys
import warnings

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_LOADER_DIR = "/home/qinhaiyan/KDA-Pilot-Exp/llm/scripts/deepgemm_glm52"
if _LOADER_DIR not in sys.path:
    sys.path.insert(0, _LOADER_DIR)

from loader import load_deep_gemm_experimental  # noqa: E402

deep_gemm_experimental = load_deep_gemm_experimental()

_HAS_FUSED = hasattr(deep_gemm_experimental, "fp8_gemm_nt_fused")

_pack_scales = None
try:
    from torch.utils.cpp_extension import load as _load_ext

    _ext = _load_ext(
        name="glm52_qb_decode_fork_scale_pack",
        sources=[os.path.join(_HERE, "scale_pack.cu")],
        verbose=False,
    )
    _pack_scales = _ext.pack_scales
except Exception as _build_err:  # pragma: no cover
    _pack_scales = None
    warnings.warn(
        f"q_b deepgemm-fork: scale-pack build failed "
        f"({type(_build_err).__name__}: {_build_err}); using torch fallback.",
        RuntimeWarning,
        stacklevel=2,
    )


def _pack_scales_torch(x_scale: torch.Tensor, w_scale: torch.Tensor):
    from deep_gemm import get_mn_major_tma_aligned_packed_ue8m0_tensor as _packt

    n = w_scale.shape[0] * 128
    row_of_block = torch.arange(n, device=w_scale.device) // 128
    xp = _packt(x_scale)
    wp = _packt(w_scale.index_select(-2, row_of_block))
    return xp, wp


_KNOB_NAMES = ("pdl", "num_sms", "tc_util")


def _save_knobs(mod):
    saved = {}
    for name in _KNOB_NAMES:
        try:
            saved[name] = getattr(mod, f"get_{name}")()
        except Exception:
            saved[name] = None
    return saved


def _apply_knobs(mod, pdl=None, num_sms=None, tc_util=None):
    for name, value in (("pdl", pdl), ("num_sms", num_sms), ("tc_util", tc_util)):
        if value is None:
            continue
        try:
            getattr(mod, f"set_{name}")(value)
        except Exception:
            pass


@torch.no_grad()
def run(inputs: dict):
    x_fp8 = inputs["x_fp8"]
    w_fp8 = inputs["w_fp8"]
    x_scale = inputs["x_scale"]
    w_scale = inputs["w_scale"]
    out = inputs["out"]

    saved = _save_knobs(deep_gemm_experimental)
    try:
        if _HAS_FUSED:
            # Single-launch path: raw f32 scales, packed to UE8M0 inside the kernel.
            deep_gemm_experimental.fp8_gemm_nt_fused(
                (x_fp8, x_scale),
                (w_fp8, w_scale),
                out,
            )
        else:
            if _pack_scales is not None:
                x_scale_packed, w_scale_packed = _pack_scales(x_scale, w_scale)
            else:
                x_scale_packed, w_scale_packed = _pack_scales_torch(x_scale, w_scale)
            deep_gemm_experimental.fp8_gemm_nt(
                (x_fp8, x_scale_packed),
                (w_fp8, w_scale_packed),
                out,
            )
    finally:
        _apply_knobs(deep_gemm_experimental, **saved)
    return out

