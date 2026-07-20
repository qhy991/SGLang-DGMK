"""Central dispatch for GLM-5.2 optimized kernels."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch

from sglang.srt.layers.glm52_opt import config
from sglang.srt.layers.glm52_opt.context import get_forward_mode, get_op_name
from sglang.srt.layers.glm52_opt.fp8_gemm import run_fp8_gemm
from sglang.srt.layers.glm52_opt.moe_masked import run_moe_masked
from sglang.srt.layers.glm52_opt.phase import infer_glm52_phase
from sglang.srt.layers.glm52_opt.registry import lookup

logger = logging.getLogger(__name__)


def _current_phase(token_num: int) -> str:
    return infer_glm52_phase(get_forward_mode(), token_num)


def try_dispatch_fp8_gemm(
    input_2d: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    block_size: List[int],
    output_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    if not config.is_enabled():
        return None
    op = get_op_name()
    phase = _current_phase(input_2d.shape[0])
    spec = lookup(op, phase)
    if spec is None or spec.kind != "fp8_gemm":
        return None
    out = input_2d.new_empty(input_2d.shape[0], weight.shape[0], dtype=output_dtype)
    ok = run_fp8_gemm(
        op,
        input_2d,
        weight,
        x_scale,
        weight_scale,
        out,
        block_size,
        spec.archive_ref,
        phase=phase,
    )
    if not ok:
        return None
    if bias is not None:
        out = out + bias
    return out.view(*input_2d.shape[:-1], weight.shape[0])


def try_dispatch_moe_masked(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
) -> bool:
    if not config.is_enabled():
        return False
    op = get_op_name()
    # w13 is fused gate+up in SGLang; either tag should enable decode pack path.
    phase = _current_phase(lhs[0].shape[1] if lhs[0].ndim >= 2 else 1)
    spec = lookup(op, phase)
    if spec is None and op in ("moe_gate_proj", "moe_up_proj"):
        # Prefer gate's registered decode pack if only one is present.
        spec = lookup("moe_gate_proj", phase) or lookup("moe_up_proj", phase)
    if spec is None or spec.kind != "moe_masked":
        return False
    # Prefill moe_gate: Graph regresses at large M — never swap.
    if phase == "prefill" and op == "moe_gate_proj":
        return False
    x_fp8, x_scale = lhs
    w_fp8, w_scale = rhs
    run_moe_masked(x_fp8, w_fp8, x_scale, w_scale, out, masked_m, expected_m)
    return True
