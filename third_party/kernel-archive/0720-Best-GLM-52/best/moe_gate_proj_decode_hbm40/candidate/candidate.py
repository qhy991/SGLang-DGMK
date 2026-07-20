"""GLM-5.2 MoE Gate Projection (decode) — packed-UE8M0 masked-grouped candidate.

The frozen reference (``deep_gemm.fp8_m_grouped_gemm_nt_masked`` with float32 block
scales) hands DeepGEMM unpacked f32 UE8M0 scales, which forces its slower f32-scale
path (~47 us, ~27% HBM on B200). SGLang's production dispatch instead hands DeepGEMM
*packed int32 UE8M0* scales and reaches the same kernel far faster — the same math,
only the scale representation differs (see glm52_ops BASELINE_CAVEAT). The same-family
moe_up/moe_down/o_proj campaigns confirmed this packed dispatch clears the 40% HBM bar
(moe_gate shares moe_up's exact shapes, scales and masks, so the mechanism ports 1:1).

This candidate reproduces that packed dispatch **statelessly and per call**:

1. It losslessly repacks the supplied float32 UE8M0 scales (exact powers of two) into
   DeepGEMM's packed-int32 MN-major layout via a SINGLE fused CUDA kernel (built once
   at import, outside the timed window). The weight scale is expanded to per-N-row
   inline (the masked-grouped ``disable_ue8m0_cast=True`` path asserts a per-row scale),
   the activation scale stays per-token — both operands, all E experts, one launch,
   one allocation, to keep the CUPTI device span tight.
2. It calls the identical ``fp8_m_grouped_gemm_nt_masked`` with ``disable_ue8m0_cast=True``
   so DeepGEMM consumes the pre-packed scales directly (skipping its internal cast).
3. It enables DeepGEMM's Programmatic Dependent Launch (PDL) around the GEMM. NCU shows
   the pack is tiny (~0.5 MB traffic) but the pack->GEMM dependency otherwise costs a
   host launch gap inside the timed CUPTI span; PDL lets the GEMM begin as the pack
   drains, shaving ~0.5-0.7 us (a measured knob cross-product found PDL the only knob
   with a material, robust effect — compiled_dims/num_sms differences were noise). The
   PDL global is saved and RESTORED around the call so nothing leaks into the harness's
   separate reference timing (no timing manipulation of the denominator).

Nothing is cached across calls: every invocation reads the frozen inputs and rebuilds
the packed scales into a freshly allocated buffer. A pure-torch pack (via DeepGEMM's own
helper) is kept as a correctness fallback so the answer is always right even if the
extension cannot build; the fast path is the compiled fused kernel and a build failure
is surfaced loudly (it would silently drop onto the slow path and miss the HBM target).

Byte accounting: the ~100 MB ``w_fp8`` weight is passed through untouched and streamed
once. The only added per-call materialization is the packed scale buffer — the source
f32 scales (~0.8 MB across x+w) read, and a packed int32 output (x: 8*128*12, w
N-expanded: 8*2048*12 int32 ~= 0.8 MB), well under 1% of the weight stream.
"""
from __future__ import annotations

import os
import warnings

import torch
import deep_gemm

_HERE = os.path.dirname(os.path.abspath(__file__))

# ── Fused scale-pack CUDA kernel, compiled once at import (outside the timed call) ──
_pack_scales = None
try:
    from torch.utils.cpp_extension import load as _load_ext

    _ext = _load_ext(
        name="glm52_moe_gate_decode_scale_pack",
        sources=[os.path.join(_HERE, "scale_pack.cu")],
        verbose=False,
    )
    _pack_scales = _ext.pack_scales
except Exception as _build_err:  # pragma: no cover - build environment fallback only
    _pack_scales = None
    warnings.warn(
        f"glm52 moe_gate decode: scale-pack CUDA extension failed to build "
        f"({type(_build_err).__name__}: {_build_err}); falling back to the slower "
        f"pure-torch packing path — the HBM target will likely be missed.",
        RuntimeWarning,
        stacklevel=2,
    )


def _pack_scales_torch(x_scale: torch.Tensor, w_scale: torch.Tensor):
    """Correctness fallback: pack via DeepGEMM's reference helper (slower, multi-launch,
    not on the fast path). The weight block scale is expanded to per-N-row before packing,
    exactly as the masked-grouped packed path requires."""
    from deep_gemm import get_mn_major_tma_aligned_packed_ue8m0_tensor as _packt

    n = w_scale.shape[1] * 128
    row_of_block = torch.arange(n, device=w_scale.device) // 128
    xp = _packt(x_scale)
    wp = _packt(w_scale.index_select(-2, row_of_block))
    return xp, wp


def _get_pdl():
    """Snapshot the global PDL knob (None if this deep_gemm lacks the getter)."""
    try:
        return deep_gemm.get_pdl()
    except Exception:  # pragma: no cover - older/newer deep_gemm without the getter
        return None


def _set_pdl(value):
    """Set the global PDL knob; a None value (or a failing setter) is a no-op."""
    if value is None:
        return
    try:
        deep_gemm.set_pdl(value)
    except Exception:  # pragma: no cover
        pass


@torch.no_grad()
def run(inputs: dict):
    x_fp8 = inputs["x_fp8"]
    w_fp8 = inputs["w_fp8"]
    x_scale = inputs["x_scale"]
    w_scale = inputs["w_scale"]
    out = inputs["out"]

    # Repack the float32 UE8M0 scales into DeepGEMM's packed-int32 MN-major layout, then
    # run the production-class packed dispatch (disable_ue8m0_cast=True) on the same inputs.
    if _pack_scales is not None:
        x_scale_packed, w_scale_packed = _pack_scales(x_scale, w_scale)
    else:
        x_scale_packed, w_scale_packed = _pack_scales_torch(x_scale, w_scale)

    # Enable PDL so the GEMM overlaps the tail of the pack (tightens the CUPTI span);
    # save/restore the global so the harness's separate reference timing is unaffected.
    saved_pdl = _get_pdl()
    try:
        _set_pdl(True)
        deep_gemm.fp8_m_grouped_gemm_nt_masked(
            (x_fp8, x_scale_packed),
            (w_fp8, w_scale_packed),
            out, inputs["masked_m"], inputs["expected_m"],
            disable_ue8m0_cast=True,
        )
    finally:
        _set_pdl(saved_pdl)
    return out
