"""GLM-5.2 o_proj decode FP8 GEMM — packed-UE8M0 weight-streaming candidate.

The frozen reference (``deep_gemm.fp8_gemm_nt`` with float32 block scales) streams the
~100 MB fp8 weight but consumes float32 scaling factors, which forces DeepGEMM's
slower float32-scale path (~52 us, ~24% HBM on B200). SGLang's production dispatch
hands DeepGEMM *packed int32 UE8M0* scales instead and reaches ~33 us (~38% HBM) — the
same kernel, only the scale representation differs.

This candidate reproduces that packed dispatch **statelessly and per call**: it losslessly
repacks the supplied float32 UE8M0-valued scales (which are exact powers of two) into
DeepGEMM's packed-int32 layout and calls the same ``fp8_gemm_nt``. The repack is a single
fused CUDA kernel (built once at import, outside the timed window) so its host-launch cost
stays small enough that both frozen decode shapes clear 35% HBM with margin. Nothing is
cached across calls: every invocation reads the frozen inputs and rebuilds the packed
scales into a freshly allocated buffer.

Byte accounting: the ~100 MB ``w_fp8`` weight is passed through untouched and streamed once.
The only added per-call materialization is the packed scale buffer — the source float32
scale side-bands read are ~32 KB, and the packed int32 output (with the weight block scale
expanded to per-row, as the packed layout requires) is ~0.77 MB, i.e. <0.8% of the weight
stream.
"""
from __future__ import annotations

import os
import warnings

import torch
import deep_gemm

_HERE = os.path.dirname(os.path.abspath(__file__))

# ── Fused scale-pack CUDA kernel, compiled once at import (outside the timed call) ──
# A pure-torch path (via DeepGEMM's own helpers) is kept as a correctness fallback so the
# candidate always produces the right answer even if the extension cannot be built; the
# fast path is the compiled kernel. A build failure is surfaced loudly (not silent) because
# it would drop the candidate onto the slow path and quietly miss the HBM target.
_pack_scales = None
try:
    from torch.utils.cpp_extension import load as _load_ext

    _ext = _load_ext(
        name="glm52_oproj_decode_scale_pack",
        sources=[os.path.join(_HERE, "scale_pack.cu")],
        verbose=False,
    )
    _pack_scales = _ext.pack_scales
except Exception as _build_err:  # pragma: no cover - build environment fallback only
    _pack_scales = None
    warnings.warn(
        f"glm52 o_proj decode: scale-pack CUDA extension failed to build "
        f"({type(_build_err).__name__}: {_build_err}); falling back to the slower "
        f"pure-torch packing path — the HBM target will likely be missed.",
        RuntimeWarning,
        stacklevel=2,
    )


def _pack_scales_torch(x_scale: torch.Tensor, w_scale: torch.Tensor):
    """Correctness fallback: pack via DeepGEMM's reference helper (slower, not on the
    fast path). Weight block scales are expanded to per-row before packing, exactly as
    SGLang's production weight-requant does."""
    from deep_gemm import get_mn_major_tma_aligned_packed_ue8m0_tensor as _packt

    n = w_scale.shape[0] * 128
    row_of_block = torch.arange(n, device=w_scale.device) // 128
    xp = _packt(x_scale)
    wp = _packt(w_scale.index_select(-2, row_of_block))
    return xp, wp


@torch.no_grad()
def run(inputs: dict):
    x_fp8 = inputs["x_fp8"]
    w_fp8 = inputs["w_fp8"]
    x_scale = inputs["x_scale"]
    w_scale = inputs["w_scale"]
    out = inputs["out"]

    # Repack the float32 UE8M0 scales into DeepGEMM's packed-int32 layout, then run the
    # production-class packed dispatch. compiled_dims='nk' selects the faster codegen for
    # this decode (small-M, large-N/K) shape.
    if _pack_scales is not None:
        x_scale_packed, w_scale_packed = _pack_scales(x_scale, w_scale)
    else:
        x_scale_packed, w_scale_packed = _pack_scales_torch(x_scale, w_scale)

    deep_gemm.fp8_gemm_nt(
        (x_fp8, x_scale_packed),
        (w_fp8, w_scale_packed),
        out,
        compiled_dims="nk",
    )
    return out
