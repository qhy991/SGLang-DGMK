"""Fail-closed dispatch for the pinned GLM-5.2 DeepGEMM W13 overlay."""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import torch

from sglang.srt.layers.glm52_opt import config
from sglang.srt.layers.glm52_opt.context import (
    get_forward_m,
    get_forward_mode,
    get_op_name,
)
from sglang.srt.layers.glm52_opt.experimental_deepgemm import (
    get_experimental_deep_gemm,
)
from sglang.srt.layers.glm52_opt.phase import infer_glm52_phase
from sglang.srt.layers.glm52_opt.registry import lookup

logger = logging.getLogger(__name__)

_PINNED_VARIANT = "w13-bm32-a674bcf69"
_EXPECTED_M_BY_BUCKET = {16: frozenset((4, 5)), 32: frozenset((8, 9))}


def w13_overlay_abi(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    expected_m: int,
    overlap_args: Optional[Any],
    recipe_a: Optional[Tuple[int, int]],
    recipe_b: Optional[Tuple[int, int]],
) -> Optional[str]:
    """Return the exact supported ABI name, otherwise fail closed with None."""

    if overlap_args is not None:
        return None
    if tuple(lhs[0].shape) != (32, 1024, 6144):
        return None
    if tuple(lhs[1].shape) != (32, 1024, 12):
        return None
    if tuple(out.shape) != (32, 1024, 4096):
        return None
    if lhs[0].dtype != torch.float8_e4m3fn:
        return None
    if lhs[1].dtype != torch.int32 or rhs[1].dtype != torch.int32:
        return None
    if not any(
        expected_m in values for values in _EXPECTED_M_BY_BUCKET.values()
    ):
        return None

    if (
        tuple(rhs[0].shape) == (32, 4096, 6144)
        and tuple(rhs[1].shape) == (32, 4096, 12)
        and rhs[0].dtype == torch.float8_e4m3fn
        and recipe_a is None
        and recipe_b is None
    ):
        return "fp8"
    if (
        tuple(rhs[0].shape) == (32, 4096, 3072)
        and tuple(rhs[1].shape) == (32, 4096, 48)
        and rhs[0].dtype == torch.int8
        and recipe_a == (1, 128)
        and recipe_b == (1, 32)
    ):
        return "nvfp4"
    return None


def try_dispatch_w13(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    overlap_args: Optional[Any],
    recipe_a: Optional[Tuple[int, int]],
    recipe_b: Optional[Tuple[int, int]],
) -> tuple[bool, Any]:
    """Return ``(handled, result)`` while preserving DeepGEMM's return value."""

    if not config.is_enabled():
        return False, None
    if config.deepgemm_variant() != _PINNED_VARIANT:
        return False, None
    abi = w13_overlay_abi(
        lhs, rhs, out, expected_m, overlap_args, recipe_a, recipe_b
    )
    if abi is None:
        return False, None

    forward_m = get_forward_m()
    if (
        forward_m not in _EXPECTED_M_BY_BUCKET
        or expected_m not in _EXPECTED_M_BY_BUCKET[forward_m]
    ):
        return False, None
    phase = infer_glm52_phase(get_forward_mode(), forward_m)
    op = get_op_name()
    spec = lookup(op, phase, m=forward_m)
    if (
        phase != "decode"
        or op != "moe_gate_proj"
        or spec is None
        or spec.kind != "moe_masked"
    ):
        return False, None

    try:
        deep_gemm = get_experimental_deep_gemm()
        if deep_gemm is None:
            return False, None
        kwargs = {
            "compiled_dims": "nk",
            "disable_ue8m0_cast": True,
        }
        if abi == "nvfp4":
            kwargs["recipe_a"] = recipe_a
            kwargs["recipe_b"] = recipe_b
        result = deep_gemm.fp8_m_grouped_gemm_nt_masked(
            lhs, rhs, out, masked_m, expected_m, **kwargs
        )
    except Exception as exc:
        logger.warning(
            "GLM-5.2 %s W13 overlay failed; using stock: %s", abi, exc
        )
        return False, None
    return True, result
