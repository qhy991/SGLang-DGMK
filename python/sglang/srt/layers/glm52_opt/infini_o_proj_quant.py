"""B300 GLM-5.2 attention-output quant producer for o_proj.

The profiled decode chain is ``V-apply BMM -> BF16 [M,16384] -> packed
group-128 UE8M0 quant -> fixed-N/K o_proj DeepGEMM``.  This provider replaces
only the quant producer; the projection, tensor-parallel reduction, and all
attention/LoRA semantics remain unchanged.

Default off.  Admission is intentionally restricted to decode CUDA-graph
buckets on B300, the exact GLM-5.2 o_proj weight ABI, and no active KV-B LoRA
correction.  Any mismatch falls through to SGLang's stock quantization path.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_HIDDEN = 16384
_OUTPUT = 6144
_GROUP = 128
_SRC = Path(__file__).resolve().parent / "csrc" / "infini_o_proj_quant.cu"

_mod = None
_load_error: Optional[str] = None


def enabled() -> bool:
    return envs.SGLANG_INFINI_O_PROJ_QUANT.get()


def _load():
    global _mod, _load_error
    if _mod is not None or _load_error is not None:
        return _mod
    try:
        from torch.utils.cpp_extension import load

        build_dir = Path(
            os.getenv("TORCH_EXTENSIONS_DIR", str(Path.home() / ".cache"))
        ) / "infini_o_proj_quant"
        build_dir.mkdir(parents=True, exist_ok=True)
        _mod = load(
            name="infini_glm52_o_proj_quant",
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
        logger.info("[infini] o_proj quant loaded from %s", _SRC)
    except Exception as exc:  # noqa: BLE001 - fail closed into the stock path
        _load_error = f"{type(exc).__name__}: {exc}"
        logger.warning("[infini] o_proj quant unavailable: %s", _load_error)
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
    hidden: torch.Tensor,
    o_proj,
    *,
    decode_or_idle: bool,
    capture_mode: bool,
    lora_active: bool,
) -> bool:
    """Fail-closed admission for the exact B300 V-apply/o_proj boundary."""
    if not enabled() or not decode_or_idle or not capture_mode or lora_active:
        return False
    if not (
        hidden.is_cuda
        and hidden.dtype is torch.bfloat16
        and hidden.dim() == 2
        and hidden.shape[1] == _HIDDEN
        and 0 < hidden.shape[0] <= 16
        and hidden.is_contiguous()
        and hidden.stride() == (_HIDDEN, 1)
    ):
        return False
    if torch.cuda.get_device_capability(hidden.device) != (10, 3):
        return False

    weight = getattr(o_proj, "weight", None)
    quant_method = getattr(o_proj, "quant_method", None)
    weight_scale = getattr(o_proj, "weight_scale_inv", None)
    block_size = tuple(getattr(quant_method, "weight_block_size", ()) or ())
    if not (
        torch.is_tensor(weight)
        and weight.is_cuda
        and weight.device == hidden.device
        and weight.dtype is torch.float8_e4m3fn
        and weight.shape == (_OUTPUT, _HIDDEN)
        and weight.is_contiguous()
        and torch.is_tensor(weight_scale)
        and weight_scale.is_cuda
        and weight_scale.device == hidden.device
        and weight_scale.dtype is torch.int32
        and weight_scale.shape == (_OUTPUT, _HIDDEN // _GROUP // 4)
        and block_size == (_GROUP, _GROUP)
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


def quantize(hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the exact prequantized tuple consumed by block-FP8 o_proj."""
    mod = _load()
    if mod is None:
        raise RuntimeError("[infini] o_proj quant called while unavailable")
    values, scales = _alloc_quant_outputs(hidden.shape[0], hidden.device)
    mod.o_proj_quant_ue8m0(hidden, values, scales)
    return values, scales
