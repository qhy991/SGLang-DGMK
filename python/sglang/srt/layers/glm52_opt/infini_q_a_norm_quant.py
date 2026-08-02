"""B300 GLM-5.2 q_a RMSNorm + packed-UE8M0 FP8 quant producer.

The production Q-LoRA activation is the non-contiguous ``[:, :2048]`` view of
the fused 2624-wide QKV-A projection. The normalized BF16 tensor branches to the
DSA indexer, while q_b projection immediately requantizes the same tensor. This
provider performs RMSNorm and quantization once, emits both consumer ABIs, and
leaves the q_b DeepGEMM unchanged.

It deliberately does not move this producer into a conventional q_b N-tile
prologue: N=16384 has many output tiles, so that schedule would repeat the same
row reduction and quantization. A later true megakernel needs an explicit
once-per-row producer/consumer handoff.

Default off. It is admitted only for B300 decode CUDA-graph buckets, the exact
strided input ABI, and the prequantized DeepGEMM linear backend.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_HIDDEN = 2048
_INPUT_ROW_STRIDE = 2624
_GROUP = 128
_SRC = Path(__file__).resolve().parent / "csrc" / "infini_q_a_norm_quant.cu"

_mod = None
_load_error: Optional[str] = None


def enabled() -> bool:
    return envs.SGLANG_INFINI_FUSED_QA_NORM_QUANT.get()


def _load():
    global _mod, _load_error
    if _mod is not None or _load_error is not None:
        return _mod
    try:
        from torch.utils.cpp_extension import load

        build_dir = Path(
            os.getenv("TORCH_EXTENSIONS_DIR", str(Path.home() / ".cache"))
        ) / "infini_q_a_norm_quant"
        build_dir.mkdir(parents=True, exist_ok=True)
        _mod = load(
            name="infini_glm52_q_a_norm_quant",
            sources=[str(_SRC)],
            extra_cuda_cflags=[
                "-O3",
                "-lineinfo",
                "--expt-relaxed-constexpr",
                "-gencode=arch=compute_100,code=sm_100",
            ],
            build_directory=str(build_dir),
            verbose=False,
        )
        logger.info("[infini] q_a RMSNorm+quant loaded from %s", _SRC)
    except Exception as exc:  # noqa: BLE001 - fail closed into the stock path
        _load_error = f"{type(exc).__name__}: {exc}"
        logger.warning("[infini] q_a RMSNorm+quant unavailable: %s", _load_error)
    return _mod


def _ceil_align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _alloc_quant_outputs(rows: int, device):
    values = torch.empty(
        (rows, _HIDDEN), device=device, dtype=torch.float8_e4m3fn
    )
    groups = _HIDDEN // _GROUP
    aligned_rows = _ceil_align(rows, 4)
    aligned_groups = _ceil_align(groups, 4)
    scales = torch.empty(
        (aligned_groups // 4, aligned_rows),
        device=device,
        dtype=torch.int32,
    ).transpose(-1, -2)[:rows, :]
    return values, scales


def is_available(
    q_lora: torch.Tensor,
    norm_weight: torch.Tensor,
    q_b_proj,
) -> bool:
    """Fail-closed admission for the exact B300 Q-LoRA producer boundary."""
    if not enabled():
        return False
    if not (
        q_lora.is_cuda
        and q_lora.dtype is torch.bfloat16
        and q_lora.dim() == 2
        and q_lora.shape[1] == _HIDDEN
        and 0 < q_lora.shape[0] <= 16
        and q_lora.stride() == (_INPUT_ROW_STRIDE, 1)
    ):
        return False
    if torch.cuda.get_device_capability(q_lora.device) != (10, 3):
        return False
    if not (
        norm_weight.is_cuda
        and norm_weight.device == q_lora.device
        and norm_weight.dtype is torch.bfloat16
        and norm_weight.shape == (_HIDDEN,)
        and norm_weight.is_contiguous()
    ):
        return False

    weight = getattr(q_b_proj, "weight", None)
    quant_method = getattr(q_b_proj, "quant_method", None)
    if not (
        torch.is_tensor(weight)
        and weight.is_cuda
        and weight.dtype is torch.float8_e4m3fn
        and weight.dim() == 2
        and weight.shape[1] == _HIDDEN
        and weight.shape[0] % 64 == 0
        and getattr(quant_method, "block_quant", False)
    ):
        return False
    from sglang.srt.layers.quantization.fp8_utils import (
        deepgemm_w8a8_block_fp8_linear_with_fallback,
    )

    if (
        getattr(quant_method, "w8a8_block_fp8_linear", None)
        is not deepgemm_w8a8_block_fp8_linear_with_fallback
    ):
        return False
    return _load() is not None


def infini_fused_q_a_rmsnorm_quant(
    q_lora: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return q_b's prequant tuple and the exact BF16 DSA side output."""
    mod = _load()
    if mod is None:
        raise RuntimeError("[infini] q_a RMSNorm+quant called while unavailable")
    rows = q_lora.shape[0]
    normed = torch.empty(
        (rows, _HIDDEN), device=q_lora.device, dtype=torch.bfloat16
    )
    values, scales = _alloc_quant_outputs(rows, q_lora.device)
    mod.fused_q_a_rmsnorm_quant_ue8m0(
        q_lora,
        norm_weight,
        normed,
        values,
        scales,
        eps,
    )
    setattr(normed, "_sglang_dsa_bf16_passthrough", True)
    return (values, scales, normed), normed
