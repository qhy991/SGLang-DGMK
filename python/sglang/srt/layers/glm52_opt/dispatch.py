"""Central dispatch for GLM-5.2 optimized kernels."""

from __future__ import annotations

import json
import logging
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from sglang.srt.layers.glm52_opt import config
from sglang.srt.layers.glm52_opt.context import (
    get_forward_m,
    get_forward_mode,
    get_op_name,
)
from sglang.srt.layers.glm52_opt.fp8_gemm import run_fp8_gemm
from sglang.srt.layers.glm52_opt.moe_masked import run_moe_masked
from sglang.srt.layers.glm52_opt.phase import infer_glm52_phase
from sglang.srt.layers.glm52_opt.registry import KernelSpec, lookup

logger = logging.getLogger(__name__)

_HIT_LOCK = threading.Lock()
_HIT_COUNTS: dict[str, int] = {}
_MISS_COUNTS: dict[str, int] = {}
_HIT_FILE = Path(
    os.environ.get("SGLANG_GLM52_OPT_HIT_FILE", "/home/ubuntu/wwxq/cache/sglang/glm52_opt_hits.json")
)


def _flush_stats() -> None:
    try:
        _HIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        payload = {"hits": dict(_HIT_COUNTS), "misses": dict(_MISS_COUNTS)}
        _HIT_FILE.write_text(json.dumps(payload, indent=2, sort_keys=True))
    except Exception as exc:
        print(f"[glm52_opt] hit-file write failed: {exc}", flush=True)


def _record_hit(
    kind: str, op: Optional[str], phase: str, m: Optional[int] = None
) -> None:
    """Count successful glm52_opt dispatches; log first hit per key."""
    key = f"{kind}:{op or 'untagged'}:{phase}"
    if m is not None:
        key += f":m{m}"
    with _HIT_LOCK:
        n = _HIT_COUNTS.get(key, 0) + 1
        _HIT_COUNTS[key] = n
        should_flush = n in (1, 10, 100) or n % 500 == 0
        if should_flush:
            _flush_stats()
    if n == 1:
        msg = f"glm52_opt HIT {key} (first)"
        logger.warning(msg)
        print(msg, flush=True)


def _record_miss(
    reason: str, op: Optional[str], phase: str, m: Optional[int] = None
) -> None:
    key = f"{reason}:{op or 'untagged'}:{phase}"
    if m is not None:
        key += f":m{m}"
    with _HIT_LOCK:
        n = _MISS_COUNTS.get(key, 0) + 1
        _MISS_COUNTS[key] = n
        should_log = n == 1
        should_flush = n in (1, 10, 100) or n % 500 == 0
        if should_flush:
            _flush_stats()
    if should_log:
        msg = f"glm52_opt MISS {key} (first)"
        logger.warning(msg)
        print(msg, flush=True)


def _current_phase(token_num: int) -> str:
    return infer_glm52_phase(get_forward_mode(), token_num)


def _nvtx_range(name: str):
    """Default-off profiler range; authoritative A/B emits no NVTX events."""

    @contextmanager
    def _cm():
        if not config.emit_infini_kernel_nvtx():
            yield
            return
        pushed = False
        try:
            torch.cuda.nvtx.range_push(name)
            pushed = True
        except Exception:
            pass
        try:
            yield
        finally:
            if pushed:
                try:
                    torch.cuda.nvtx.range_pop()
                except Exception:
                    pass

    return _cm()


def _profiler_range_name(spec: KernelSpec, m: int) -> str:
    name = spec.profiler_name or (
        f"infini_kernel_glm52_{spec.op}_{spec.phase}_{spec.implementation}"
    )
    if spec.n is not None and spec.k is not None:
        return f"{name}[M={m},N={spec.n},K={spec.k}]"
    return f"{name}[M={m}]"


def _fixed_nk_forward_mode_matches(spec: KernelSpec) -> bool:
    if spec.implementation != "fixed_nk":
        return True
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    mode = get_forward_mode()
    if spec.phase == "decode":
        return mode is ForwardMode.DECODE
    if spec.phase == "prefill":
        return mode is ForwardMode.EXTEND
    return False


def _fixed_nk_abi_matches(
    spec: KernelSpec,
    input_2d: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    block_size: List[int],
    output_dtype: torch.dtype,
    bias: Optional[torch.Tensor],
) -> bool:
    if spec.implementation != "fixed_nk":
        return True
    if spec.n is None or spec.k is None:
        return False
    m = int(input_2d.shape[0]) if input_2d.ndim == 2 else -1
    return (
        _fixed_nk_forward_mode_matches(spec)
        and input_2d.ndim == 2
        and tuple(input_2d.shape) == (m, spec.k)
        and tuple(weight.shape) == (spec.n, spec.k)
        and tuple(block_size) == (128, 128)
        and input_2d.dtype == torch.float8_e4m3fn
        and weight.dtype == torch.float8_e4m3fn
        and input_2d.is_cuda
        and weight.is_cuda
        and input_2d.is_contiguous()
        and weight.is_contiguous()
        and input_2d.device == weight.device
        and x_scale.dtype == torch.int32
        and weight_scale.dtype == torch.int32
        and x_scale.is_cuda
        and weight_scale.is_cuda
        and x_scale.device == input_2d.device
        and weight_scale.device == input_2d.device
        and tuple(x_scale.shape) == (m, spec.k // 128 // 4)
        and tuple(weight_scale.shape) == (spec.n, spec.k // 128 // 4)
        and tuple(x_scale.stride()) == (1, m)
        and tuple(weight_scale.stride()) == (1, spec.n)
        and output_dtype == torch.bfloat16
        and bias is None
    )


def record_psum_hit(op: Optional[str], m: Optional[int] = None) -> None:
    """Count contig PSUM layout applications (goals 08/09)."""
    phase = "prefill"
    try:
        phase = _current_phase(int(m) if m is not None else 1)
    except Exception:
        pass
    _record_hit("moe_contig_psum", op, phase, m=m)


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
    m = int(input_2d.shape[0])
    phase = _current_phase(m)
    spec = lookup(op, phase, m=m)
    if spec is None or spec.kind != "fp8_gemm":
        _record_miss("no_spec", op, phase, m=m)
        return None
    if not _fixed_nk_abi_matches(
        spec,
        input_2d,
        weight,
        x_scale,
        weight_scale,
        block_size,
        output_dtype,
        bias,
    ):
        _record_miss("fixed_nk_abi", op, phase, m=m)
        return None
    out = input_2d.new_empty(input_2d.shape[0], weight.shape[0], dtype=output_dtype)
    with _nvtx_range(_profiler_range_name(spec, m)):
        ok, path = run_fp8_gemm(
            op,
            input_2d,
            weight,
            x_scale,
            weight_scale,
            out,
            block_size,
            spec.archive_ref,
            phase=phase,
            implementation=spec.implementation,
        )
    if not ok:
        _record_miss(f"run_skipped:{path}", op, phase, m=m)
        return None
    _record_hit(f"fp8_gemm/{path}", op, phase, m=m)
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
    # The grouped input's second dimension is the fixed expert slab (1024),
    # not the model-forward token bucket.  Use launch-time ForwardBatch M so
    # M16 and M32 can independently select a replacement.
    forward_m = get_forward_m()
    spec = lookup(op, phase, m=forward_m)
    if spec is None and op in ("moe_gate_proj", "moe_up_proj"):
        # Prefer gate's registered decode pack if only one is present.
        spec = lookup("moe_gate_proj", phase, m=forward_m) or lookup(
            "moe_up_proj", phase, m=forward_m
        )
    if spec is None or spec.kind != "moe_masked":
        _record_miss("moe_no_spec", op, phase, m=forward_m)
        return False
    # Prefill moe_gate: Graph regresses at large M — never swap.
    if phase == "prefill" and op == "moe_gate_proj":
        _record_miss("moe_prefill_skip", op, phase, m=forward_m)
        return False
    x_fp8, x_scale = lhs
    w_fp8, w_scale = rhs
    try:
        range_name = (
            f"infini_kernel_glm52_{op or spec.op}_{phase}_moe_masked"
            f"[M={forward_m if forward_m is not None else 'unknown'}]"
        )
        with _nvtx_range(range_name):
            run_moe_masked(x_fp8, w_fp8, x_scale, w_scale, out, masked_m, expected_m)
    except Exception as exc:
        _record_miss(
            f"moe_run_fail:{type(exc).__name__}",
            op or (spec.op if spec else None),
            phase,
            m=forward_m,
        )
        logger.warning("glm52_opt moe_masked failed: %s", exc)
        return False
    _record_hit("moe_masked", op or spec.op, phase, m=forward_m)
    return True
