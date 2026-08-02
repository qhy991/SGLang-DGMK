"""infini — fused residual-add + RMSNorm + per-token-group UE8M0 FP8 quant (sm100).

Fills the NVIDIA hole in an abstraction SGLang already has: `communicator.py`
dispatches a fused RMSNorm + FP8 *group* quant for ROCm gfx95 via
`aiter.ops.triton.fused_fp8_quant.fused_rms_fp8_group_quant`, and falls through
to the unfused two-kernel path on NVIDIA. This module supplies the sm100 backend.

It replaces exactly these two stock kernels:

    sgl_kernel.fused_add_rmsnorm                (flashinfer CuTe-DSL kernel)
    sglang_per_token_group_quant_fp8(ue8m0)     (per_token_group_quant_8bit_v2)

and returns the SAME `(x_fp8, packed-int32-UE8M0 x_scale)` pair the stock path
hands to `deep_gemm.fp8_gemm_nt`. No GEMM is touched. The saving is one kernel
launch plus the HBM round-trip of the normalized [M,K] bf16 activation, which
nothing outside the region consumes.

Numerics: **bit-exact** against the stock pair — byte-identical residual, x_fp8
and packed scale. That is mandatory, not nice-to-have: the reduction order is
pinned to flashinfer's CuTe-DSL kernel (flat-order FMA contracted to FMA, then an
ascending-offset warp butterfly, with `mean = sum_sq/H` as an IEEE div.rn). Four
wrong bytes out of 6291456 produced 147 elementwise failures in validation.

Measured (B200, CUDA-graph containing region, paired interleaved, two physical
GPUs, worst of {pooled, AB-median, BA-median, order-balanced}):

    post-attention -> MoE gate site   M=1024/2048/4096   1.153x / 1.228x / 1.180x

Default OFF. Enable with SGLANG_INFINI_FUSED_NORM_QUANT=1 (registered in srt/environ.py).

Two-input limitation and the GLM-5.2 three-input sibling
---------------------------------------------------------
Both `communicator.py` call sites pack the *unquantized* bf16 activation as a
third tuple element when `get_attn_tp_context().is_dsa` is true, because the DSA
indexer consumes it. This kernel does not emit that bf16 (not writing it is
precisely where the traffic saving comes from), so `is_available()` refuses the
DSA case and the caller falls through to the stock path. GLM-5.2 IS a DSA model,
so the original two-input provider remains inert for it.

The separately gated three-input sibling preserves the earlier observable
`bf16(shared+routed)` round and emits normalized bf16 for DSA together with the
residual and packed FP8 ABI. It is selected only when the model proves that the
shared/routed pair crosses a trivial SCATTERED layer boundary. Its B300 result is
measured independently; the two-input numbers above do not apply to it.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Optional, Tuple

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

INFINI_PROVIDER = "infini"
INFINI_OP_NAME = "infini_glm52_fused_norm_quant"

# ROWS=2, SMEM=0 is the measured optimum: a 6-config sweep
# (ROWS in {1,2,4} x SMEM in {0,1}) won 6/6 shapes across both prefill regions.
_SMEM_STAGE = 0

_HIDDEN = 6144          # the K this kernel is compiled for
_GROUP = 128
_SRC = Path(__file__).resolve().parent / "csrc" / "infini_fused_norm_quant.cu"

_mod = None
_load_error: Optional[str] = None


def enabled() -> bool:
    return envs.SGLANG_INFINI_FUSED_NORM_QUANT.get()


def shared_add_enabled() -> bool:
    return envs.SGLANG_INFINI_FUSED_SHARED_ADD_NORM_QUANT.get()


def _rows_per_block() -> int:
    return envs.SGLANG_INFINI_FUSED_NQ_ROWS.get()


def _shared_rows_per_block() -> int:
    return envs.SGLANG_INFINI_FUSED_SHARED_NQ_ROWS.get()


def _load():
    """Build once, lazily. Any failure is recorded and disables the provider —
    it must never raise into the model forward."""
    global _mod, _load_error
    if _mod is not None or _load_error is not None:
        return _mod
    try:
        from torch.utils.cpp_extension import load

        # TORCH_EXTENSIONS_DIR is upstream-owned, so os.getenv is correct here
        # (see .claude/skills/env-var-conventions: raw upstream vars stay out of Envs).
        build_dir = Path(
            os.getenv("TORCH_EXTENSIONS_DIR", str(Path.home() / ".cache"))
        ) / "infini_fused_nq"
        build_dir.mkdir(parents=True, exist_ok=True)
        _mod = load(
            name="infini_glm52_fused_norm_quant",
            sources=[str(_SRC)],
            extra_cuda_cflags=[
                "-O3",
                "-lineinfo",
                # NO -use_fast_math. `mean = sum_sq/H` must stay IEEE div.rn.f32;
                # under fast-math it becomes div.approx and costs 1-2 fp8 bytes,
                # which breaks bit-exactness (validated: 4 wrong bytes -> 147
                # elementwise failures).
                "--expt-relaxed-constexpr",
                "-gencode=arch=compute_100,code=sm_100",
            ],
            build_directory=str(build_dir),
            verbose=False,
        )
        logger.info("[infini] fused norm+quant loaded from %s", _SRC)
    except Exception as e:  # noqa: BLE001 - must not escape into forward
        _load_error = f"{type(e).__name__}: {e}"
        logger.warning("[infini] fused norm+quant unavailable: %s", _load_error)
    return _mod


def is_available(
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
    weight: torch.Tensor,
    needs_unquantized_bf16: bool,
) -> bool:
    """Fail-closed admission check. Every condition the kernel assumes is
    verified here; anything unrecognised returns False and the caller keeps the
    stock path. This must never be the reason a wrong number is produced."""
    if not enabled():
        return False
    if needs_unquantized_bf16:
        # DSA indexer needs the bf16 this kernel does not emit. See module docstring.
        return False
    if residual is None:
        return False  # only the residual-add form is implemented
    if not (hidden_states.is_cuda and residual.is_cuda and weight.is_cuda):
        return False
    if torch.cuda.get_device_capability(hidden_states.device)[0] != 10:
        return False  # sm100 only
    if hidden_states.dtype is not torch.bfloat16 or residual.dtype is not torch.bfloat16:
        return False
    if weight.dtype is not torch.bfloat16:
        return False
    if hidden_states.dim() != 2 or hidden_states.shape[-1] != _HIDDEN:
        return False
    if residual.shape != hidden_states.shape or weight.shape != (_HIDDEN,):
        return False
    if not (hidden_states.is_contiguous() and residual.is_contiguous()):
        return False
    return _load() is not None


def _ceil_align(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def _alloc_quant_outputs(M: int, K: int, device):
    """Byte-for-byte what sglang's own wrapper allocates.

    Mirrors fp8_kernel.create_per_token_group_quant_fp8_output_scale with
    column_major_scales=True, scale_tma_aligned=True, scale_ue8m0=True: an int32
    (aligned_k//4, aligned_mn) buffer viewed transposed, i.e. logical
    (M, K//512) with stride (1, aligned_mn).
    """
    x_q = torch.empty((M, K), device=device, dtype=torch.float8_e4m3fn)
    s_mn, s_k = M, K // _GROUP
    aligned_mn, aligned_k = _ceil_align(s_mn, 4), _ceil_align(s_k, 4)
    x_s = torch.empty(
        (aligned_k // 4, aligned_mn), device=device, dtype=torch.int32
    ).transpose(-1, -2)[:s_mn, :]
    return x_q, x_s


def infini_fused_add_rmsnorm_quant(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """residual += hidden ; h = RMSNorm(residual)*weight ; quant(h) -> (fp8, packed scale)

    Returns ((x_fp8, x_scale), residual) matching the stock path's contract.
    Call only after is_available() has returned True.
    """
    mod = _load()
    if mod is None:
        raise RuntimeError("[infini] fused norm+quant called while unavailable")

    M, K = hidden_states.shape
    xq, xs = _alloc_quant_outputs(M, K, hidden_states.device)

    mod.infini_fused_add_rmsnorm_quant_ue8m0(
        hidden_states, residual, weight, xq, xs, eps, _rows_per_block(), _SMEM_STAGE
    )
    return (xq, xs), residual


def infini_fused_shared_add_rmsnorm_quant(
    shared: torch.Tensor,
    routed: torch.Tensor,
    routed_scale: float,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse the source-proven post-DeepEP inter-layer boundary.

    This computes the production order exactly::

        combined = bf16(shared + routed_scale * routed)
        residual_out = combined + residual
        normed = RMSNorm(residual_out) * weight
        x_fp8, packed_scale = group_quant(normed)

    The explicit BF16 ``combined`` round is load-bearing. The returned tuple is
    ``(normed_bf16, residual_out, x_fp8, packed_ue8m0_scale)`` so the DSA
    indexer and FP8 projection consumers retain their complete ABI.

    The model selects this only behind a default-off, fail-closed DeepEP /
    SCATTERED-boundary admission check.
    """
    mod = _load()
    if mod is None:
        raise RuntimeError("[infini] fused shared-add norm+quant unavailable")
    tensors = (shared, routed, residual)
    if not all(
        tensor.is_cuda
        and tensor.dtype is torch.bfloat16
        and tensor.is_contiguous()
        and tensor.shape == shared.shape
        for tensor in tensors
    ):
        raise ValueError("shared/routed/residual must be contiguous CUDA BF16 peers")
    if shared.dim() != 2 or shared.shape[1] != _HIDDEN:
        raise ValueError(f"expected [M,{_HIDDEN}] shared input, got {tuple(shared.shape)}")
    if not (
        weight.is_cuda
        and weight.dtype is torch.bfloat16
        and weight.shape == (_HIDDEN,)
        and weight.is_contiguous()
    ):
        raise ValueError(f"weight must be CUDA BF16 [{_HIDDEN}]")

    M, K = shared.shape
    normed = torch.empty_like(shared)
    xq, xs = _alloc_quant_outputs(M, K, shared.device)
    mod.infini_fused_shared_add_rmsnorm_quant_ue8m0(
        shared,
        routed,
        routed_scale,
        residual,
        weight,
        normed,
        xq,
        xs,
        eps,
        _shared_rows_per_block(),
    )
    return normed, residual, xq, xs


def is_shared_add_available(
    shared: torch.Tensor,
    routed: torch.Tensor,
    routed_scale: float,
    residual: Optional[torch.Tensor],
    weight: torch.Tensor,
) -> bool:
    """Fail-closed admission for the DSA-capable three-input sibling."""
    if not shared_add_enabled() or residual is None:
        return False
    if not isinstance(routed_scale, (int, float)) or not math.isfinite(routed_scale):
        return False
    tensors = (shared, routed, residual)
    if not all(
        tensor.is_cuda
        and tensor.dtype is torch.bfloat16
        and tensor.is_contiguous()
        and tensor.shape == shared.shape
        for tensor in tensors
    ):
        return False
    if torch.cuda.get_device_capability(shared.device) != (10, 3):
        return False  # measured B300 contract; B200 keeps stock until re-gated
    if shared.dim() != 2 or shared.shape[1] != _HIDDEN:
        return False
    if not (
        weight.is_cuda
        and weight.dtype is torch.bfloat16
        and weight.shape == (_HIDDEN,)
        and weight.is_contiguous()
    ):
        return False
    if _shared_rows_per_block() not in (1, 2, 4):
        return False
    return _load() is not None
