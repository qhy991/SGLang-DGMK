"""Central dispatch for GLM-5.2 optimized kernels."""

from __future__ import annotations

import json
import logging
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
from sglang.srt.layers.glm52_opt import config
from sglang.srt.layers.glm52_opt.context import (
    get_forward_m,
    get_forward_mode,
    get_op_name,
)
from sglang.srt.layers.glm52_opt.fp8_gemm import run_fp8_gemm
from sglang.srt.layers.glm52_opt.hotspot_provider import run_flashmla_sparse_decode
from sglang.srt.layers.glm52_opt.hotspot_provider import (
    run_moe_masked as run_hotspot_moe_masked,
)
from sglang.srt.layers.glm52_opt.moe_masked import run_moe_masked
from sglang.srt.layers.glm52_opt.phase import infer_glm52_phase
from sglang.srt.layers.glm52_opt.registry import KernelSpec, lookup

logger = logging.getLogger(__name__)

_HIT_LOCK = threading.Lock()
_HIT_COUNTS: dict[str, int] = {}
_MISS_COUNTS: dict[str, int] = {}
_HIT_FILE = Path(
    os.environ.get(
        "SGLANG_GLM52_OPT_HIT_FILE",
        "/home/ubuntu/wwxq/cache/sglang/glm52_opt_hits.json",
    )
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


def _tensor_contract(
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device | None = None,
) -> bool:
    return bool(
        tensor.is_cuda
        and tensor.dtype == dtype
        and tuple(tensor.shape) == shape
        and tuple(tensor.stride()) == stride
        and tensor.storage_offset() == 0
        and (device is None or tensor.device == device)
    )


def _moe_hotspot_abi_matches(
    spec: KernelSpec,
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    forward_m: Optional[int],
) -> bool:
    if (
        spec.implementation != "hotspot_plugin"
        or spec.kind != "moe_masked"
        or spec.n is None
        or spec.k is None
        or spec.num_groups is None
        or spec.slab_m is None
        or forward_m not in (16, 32)
    ):
        return False
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    if get_forward_mode() is not ForwardMode.DECODE:
        return False
    allowed_expected_m = {16: (4, 5), 32: (8, 9)}[int(forward_m)]
    if (
        expected_m not in allowed_expected_m
        or spec.expected_m_values is None
        or expected_m not in spec.expected_m_values
    ):
        return False

    x, x_scale = lhs
    weight, weight_scale = rhs
    groups, slab_m, n, k = (
        spec.num_groups,
        spec.slab_m,
        spec.n,
        spec.k,
    )
    scale_k = k // 512
    device = x.device
    return bool(
        _tensor_contract(
            x,
            shape=(groups, slab_m, k),
            stride=(slab_m * k, k, 1),
            dtype=torch.float8_e4m3fn,
        )
        and _tensor_contract(
            x_scale,
            shape=(groups, slab_m, scale_k),
            stride=(slab_m * scale_k, 1, slab_m),
            dtype=torch.int32,
            device=device,
        )
        and _tensor_contract(
            weight,
            shape=(groups, n, k),
            stride=(n * k, k, 1),
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        and _tensor_contract(
            weight_scale,
            shape=(groups, n, scale_k),
            stride=(n * scale_k, 1, n),
            dtype=torch.int32,
            device=device,
        )
        and _tensor_contract(
            out,
            shape=(groups, slab_m, n),
            stride=(slab_m * n, n, 1),
            dtype=torch.bfloat16,
            device=device,
        )
        and _tensor_contract(
            masked_m,
            shape=(groups,),
            stride=(1,),
            dtype=torch.int32,
            device=device,
        )
    )


def _flashmla_hotspot_abi_matches(
    spec: KernelSpec,
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: float,
    indices: torch.Tensor,
    block_table: torch.Tensor,
    is_fp8_kvcache: bool,
) -> bool:
    if (
        spec.implementation != "hotspot_plugin"
        or spec.kind != "dsa"
        or None
        in (
            spec.topk,
            spec.q_heads,
            spec.qk_dim,
            spec.v_dim,
            spec.page_size,
            spec.kv_dim,
        )
    ):
        return False
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    if get_forward_mode() is not ForwardMode.DECODE:
        return False
    m = int(q.shape[0]) if q.ndim == 4 else -1
    device = q.device
    return bool(
        m in (16, 32)
        and head_dim_v == spec.v_dim
        and is_fp8_kvcache is True
        and float(softmax_scale) == 0.0625
        and _tensor_contract(
            q,
            shape=(m, 1, int(spec.q_heads), int(spec.qk_dim)),
            stride=(
                int(spec.q_heads) * int(spec.qk_dim),
                int(spec.q_heads) * int(spec.qk_dim),
                int(spec.qk_dim),
                1,
            ),
            dtype=torch.bfloat16,
        )
        and k_cache.is_cuda
        and k_cache.dtype == torch.float8_e4m3fn
        and tuple(k_cache.shape[1:]) == (int(spec.page_size), 1, int(spec.kv_dim))
        and k_cache.is_contiguous()
        and k_cache.storage_offset() == 0
        and k_cache.device == device
        and _tensor_contract(
            cache_seqlens,
            shape=(m,),
            stride=(1,),
            dtype=torch.int32,
            device=device,
        )
        and _tensor_contract(
            tile_scheduler_metadata,
            shape=(148, 8),
            stride=(8, 1),
            dtype=torch.int32,
            device=device,
        )
        and _tensor_contract(
            num_splits,
            shape=(m + 1,),
            stride=(1,),
            dtype=torch.int32,
            device=device,
        )
        and _tensor_contract(
            indices,
            shape=(m, 1, int(spec.topk)),
            stride=(int(spec.topk), int(spec.topk), 1),
            dtype=torch.int32,
            device=device,
        )
        and block_table.is_cuda
        and block_table.dtype == torch.int32
        and tuple(block_table.shape) == (m, 0)
        and block_table.device == device
    )


def try_dispatch_flashmla_sparse_decode(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: float,
    indices: torch.Tensor,
    block_table: torch.Tensor,
    is_fp8_kvcache: bool,
) -> Optional[torch.Tensor]:
    """Run one exact FlashMLA provider call or return ``None`` before launch."""
    if not config.is_enabled():
        return None
    m = int(q.shape[0]) if q.ndim == 4 else -1
    phase = _current_phase(m)
    spec = lookup("dsa_decode_attn", phase, m=m)
    if spec is None or spec.kind != "dsa":
        _record_miss("flashmla_no_spec", "dsa_decode_attn", phase, m=m)
        return None
    if not _flashmla_hotspot_abi_matches(
        spec,
        q=q,
        k_cache=k_cache,
        cache_seqlens=cache_seqlens,
        head_dim_v=head_dim_v,
        tile_scheduler_metadata=tile_scheduler_metadata,
        num_splits=num_splits,
        softmax_scale=softmax_scale,
        indices=indices,
        block_table=block_table,
        is_fp8_kvcache=is_fp8_kvcache,
    ):
        _record_miss("flashmla_abi", spec.op, phase, m=m)
        return None

    with _nvtx_range(_profiler_range_name(spec, m)):
        result = run_flashmla_sparse_decode(
            q=q,
            k_cache=k_cache,
            cache_seqlens=cache_seqlens,
            head_dim_v=head_dim_v,
            tile_scheduler_metadata=tile_scheduler_metadata,
            num_splits=num_splits,
            softmax_scale=softmax_scale,
            indices=indices,
            block_table=block_table,
            is_fp8_kvcache=is_fp8_kvcache,
        )
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise RuntimeError(
            "FlashMLA hotspot provider must return the stock (output, lse) pair"
        )
    candidate_out = result[0]
    if not isinstance(candidate_out, torch.Tensor) or not _tensor_contract(
        candidate_out,
        shape=(m, 1, int(spec.q_heads), int(spec.v_dim)),
        stride=(
            int(spec.q_heads) * int(spec.v_dim),
            int(spec.q_heads) * int(spec.v_dim),
            int(spec.v_dim),
            1,
        ),
        dtype=torch.bfloat16,
        device=q.device,
    ):
        raise RuntimeError("FlashMLA hotspot provider returned an invalid output")
    _record_hit("hotspot_plugin/flashmla_sparse_decode", spec.op, phase, m=m)
    return candidate_out


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


def try_dispatch_moe_w2_bm16(
    contract: Any,
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    *,
    callsite_eligible: bool,
) -> bool:
    """Return False before launch or True after one committed launch."""
    forward_state = (
        contract.current_forward_state() if contract is not None else None
    )
    if (
        contract is None
        or not callsite_eligible
        or get_op_name() != "moe_down_proj"
        or forward_state is None
        or not forward_state[0].is_decode()
        or not (
            (
                int(forward_state[1]) == 16
                and int(expected_m) in (4, 5)
            )
            or (
                int(forward_state[1]) == 32
                and int(expected_m) in (8, 9)
            )
        )
    ):
        return False

    # Crossing this line commits to one candidate launch. Exceptions and return
    # contract violations propagate; stock must never execute afterward.
    result = contract.launch(
        lhs,
        rhs,
        out,
        masked_m,
        int(expected_m),
        masked_block_m_override=16,
    )
    if result is not None:
        raise RuntimeError(
            "W2/BM16 candidate violated the stock non-overlap None return contract"
        )
    return True


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
    if spec.implementation == "hotspot_plugin":
        if not _moe_hotspot_abi_matches(
            spec,
            lhs,
            rhs,
            out,
            masked_m,
            expected_m,
            forward_m,
        ):
            _record_miss("moe_hotspot_abi", spec.op, phase, m=forward_m)
            return False
        with _nvtx_range(
            _profiler_range_name(
                spec,
                int(forward_m) if forward_m is not None else -1,
            )
        ):
            returned = run_hotspot_moe_masked(
                spec.op,
                lhs=lhs,
                rhs=rhs,
                out=out,
                masked_m=masked_m,
                expected_m=expected_m,
            )
        if returned is not None:
            raise RuntimeError(
                f"{spec.op} hotspot provider violated the stock None return contract"
            )
        _record_hit("hotspot_plugin", spec.op, phase, m=forward_m)
        return True

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
