"""Exact GLM-5.2 prefill SwiGLU plus direct packed-UE8M0 candidates.

The implementation is intentionally restricted to the audited normal-DeepEP
contiguous handoff:

* BF16 gate/up input ``[35200, 4096]`` (gate first),
* device row map ``int32[35200]`` plus retained scatter endpoints
  ``int32[32]``,
* FP8-E4M3 output ``[35200, 2048]``, and
* physical packed-scale storage ``int32[4, 35200]`` returned as the
  ``[35200, 4]`` stride-``[1, 35200]`` production view.

Automatic production selection is default-off and fail-closed. Explicit
benchmark entry points raise on every eligibility failure and never execute
stock after a partial candidate launch.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from functools import cache
from typing import Any, Final

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc
from triton.language.extra.cuda import libdevice as cuda_libdevice

logger = logging.getLogger(__name__)

EXPERTS: Final = 32
ALIGNED_M: Final = 35200
VALID_M: Final = 32982
GATE_UP: Final = 4096
HIDDEN: Final = 2048
GROUP_SIZE: Final = 128
PACKED_WORDS: Final = HIDDEN // (GROUP_SIZE * 4)
TRITON_BUILD: Final = "3.6.0"

REQUIRED_TOPOLOGY: Final = {
    "tp_size": 8,
    "dp_size": 8,
    "ep_size": 8,
    "pp_size": 1,
    "moe_dp_size": 1,
    "enable_dp_attention": True,
    "moe_a2a_backend": "deepep",
    "deepep_mode": "auto",
    "moe_runner_backend": "deep_gemm",
    "ep_num_redundant_experts": 0,
}

# Exactly the three bounded mappings required by Task 29. The tuple is
# (BLOCK_N, num_warps); split count is HIDDEN // BLOCK_N.
TRITON_VARIANTS: Final = {
    "endpoint512_w4": (512, 4),
    "endpoint1024_w8": (1024, 8),
    "endpoint2048_w8": (2048, 8),
}
CUDA_VARIANTS: Final = frozenset(
    {
        "cuda_s8_v16_b128",
        "cuda_s16_v8_b256",
        "cuda_s8_v16_b128_cgld",
    }
)
VARIANT_NAMES: Final = frozenset(TRITON_VARIANTS) | CUDA_VARIANTS

# ``1.0f / 448.0f`` from the stock quant kernel, represented as the exact
# binary32 constant. Keeping multiplication (rather than division) reproduces
# stock's ``amax * MAX_8BIT_INV`` rounding.
FP8_MAX_INV: Final = 0.0022321429569274187


@triton.jit
def _silu_mul_quant_prefill_kernel(
    gateup_ptr,
    output_ptr,
    scale_storage_ptr,
    m_indices_ptr,
    endpoint_ptr,
    ALIGNED_ROWS: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    GATE_UP_WIDTH: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    QUANT_GROUP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FP8_MAX_INV_VALUE: tl.constexpr,
    INPUT_ROW_STRIDE: tl.constexpr,
    OUTPUT_ROW_STRIDE: tl.constexpr,
    SCALE_WORD_STRIDE: tl.constexpr,
    SCALE_ROW_STRIDE: tl.constexpr,
    GATE_FIRST: tl.constexpr,
    PRECISE_EXPF: tl.constexpr,
    BF16_ROUND_TRIP: tl.constexpr,
    ENDPOINT_ALIGNED_EXCLUSIVE_PLUS_VALID: tl.constexpr,
    USE_PDL: tl.constexpr,
    GRAPH_MODE: tl.constexpr,
):
    """Own one physical row/split and directly emit FP8 plus packed scales."""

    tl.static_assert(ALIGNED_ROWS == 35200)
    tl.static_assert(NUM_EXPERTS == 32)
    tl.static_assert(GATE_UP_WIDTH == 4096)
    tl.static_assert(OUTPUT_WIDTH == 2048)
    tl.static_assert(QUANT_GROUP == 128)
    tl.static_assert(GATE_FIRST)
    tl.static_assert(PRECISE_EXPF)
    tl.static_assert(BF16_ROUND_TRIP)
    tl.static_assert(ENDPOINT_ALIGNED_EXCLUSIVE_PLUS_VALID)
    # GRAPH_MODE is deliberately a constexpr even though the arithmetic is
    # identical. It separates eager and capture-specialized JIT cache keys.
    tl.static_assert(GRAPH_MODE or not GRAPH_MODE)  # noqa: SIM221

    row = tl.program_id(0)
    split = tl.program_id(1)

    if USE_PDL:
        gdc.gdc_wait()

    # ep_scatter fills m_indices over every aligned slab. It then atomically
    # advances endpoint[expert] from the aligned exclusive start to
    # aligned_start + actual_valid_count. This two-load test skips every gap
    # without a host scan, predicate buffer, or endpoint reconstruction.
    expert = tl.load(m_indices_ptr + row)
    live_end = tl.load(endpoint_ptr + expert)
    if row >= live_end:
        if USE_PDL:
            gdc.gdc_launch_dependents()
        return

    offsets = split * BLOCK_N + tl.arange(0, BLOCK_N)
    row64 = row.to(tl.int64)
    input_base = gateup_ptr + row64 * INPUT_ROW_STRIDE
    output_base = output_ptr + row64 * OUTPUT_ROW_STRIDE

    gate = tl.load(input_base + offsets).to(tl.float32)
    up = tl.load(input_base + OUTPUT_WIDTH + offsets).to(tl.float32)

    # Match stock activation.cuh compiled without --use_fast_math on SM100:
    # precise libdevice expf, round-to-nearest FP32 division and multiplication,
    # followed by the observable BF16 materialization.
    exp_neg_gate = cuda_libdevice.exp(-gate)
    silu = cuda_libdevice.div_rn(gate, 1.0 + exp_neg_gate)
    values_f32 = silu * up
    values_bf16 = values_f32.to(tl.bfloat16)
    values = values_bf16.to(tl.float32)

    groups_per_program: tl.constexpr = BLOCK_N // QUANT_GROUP
    packed_words_per_program: tl.constexpr = groups_per_program // 4
    values_2d = tl.reshape(values, (groups_per_program, QUANT_GROUP))

    # Stock starts every reduction at 1e-10 and uses fmaxf. fmaxf ignores one
    # NaN operand, so NaN magnitudes contribute the reduction identity.
    magnitudes = tl.abs(values_2d)
    magnitudes = tl.where(
        values_2d == values_2d, magnitudes, 1.0e-10  # noqa: PLR0124
    )
    absmax = tl.max(magnitudes, axis=1)
    absmax = tl.maximum(absmax, 1.0e-10)

    # Reproduce calculate_fp8_scales<true>: ceil the binary32 exponent when
    # any mantissa bit is present, construct both powers of two directly, and
    # store the biased scale exponent byte.
    raw_scale = absmax * FP8_MAX_INV_VALUE
    raw_bits = raw_scale.to(tl.int32, bitcast=True)
    raw_exponent = (raw_bits >> 23) & 0xFF
    raw_mantissa = raw_bits & 0x7FFFFF
    scale_exp_unbiased = raw_exponent - 127 + (raw_mantissa != 0)
    stored_exponent = scale_exp_unbiased + 127
    quant_scale_bits = (127 - scale_exp_unbiased) << 23
    quant_scale = quant_scale_bits.to(tl.float32, bitcast=True)

    quant_scale_2d = tl.reshape(
        quant_scale, (groups_per_program, 1)
    )
    scaled = values_2d * quant_scale_2d
    # CUDA fmaxf/fminf maps NaN through the lower clamp endpoint.
    scaled = tl.where(scaled == scaled, scaled, -448.0)  # noqa: PLR0124
    scaled = tl.minimum(tl.maximum(scaled, -448.0), 448.0)
    quantized = tl.reshape(scaled, (BLOCK_N,)).to(
        output_ptr.dtype.element_ty,
        fp_downcast_rounding="rtne",
    )
    tl.store(output_base + offsets, quantized)

    # Four consecutive scale bytes form one little-endian int32. Physical
    # storage is [4, 35200], so every program owns disjoint words and no
    # atomics, transpose, or post-pack launch is required.
    exponent_2d = tl.reshape(stored_exponent, (packed_words_per_program, 4))
    byte_shifts = tl.arange(0, 4)[None, :] * 8
    packed = tl.sum(exponent_2d << byte_shifts, axis=1)
    word_offsets = (
        split * packed_words_per_program
        + tl.arange(0, packed_words_per_program)
    )
    scale_offsets = (
        word_offsets.to(tl.int64) * SCALE_WORD_STRIDE
        + row64 * SCALE_ROW_STRIDE
    )
    tl.store(scale_storage_ptr + scale_offsets, packed)

    if USE_PDL:
        gdc.gdc_launch_dependents()


def variant_names() -> tuple[str, ...]:
    return (*TRITON_VARIANTS, *sorted(CUDA_VARIANTS))


@cache
def _device_capability(device_index: int) -> tuple[int, int]:
    # The query is cached during explicit warmup/startup and never repeats in
    # the timed production hot path.
    return torch.cuda.get_device_capability(device_index)


def _tensor_contract(
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    dtype: torch.dtype,
) -> bool:
    return bool(
        tensor.is_cuda
        and tensor.dtype == dtype
        and tuple(tensor.shape) == shape
        and tuple(tensor.stride()) == stride
        and tensor.storage_offset() == 0
    )


def _eligibility_error(
    gateup_output: torch.Tensor,
    m_indices: torch.Tensor,
    endpoint: torch.Tensor,
    *,
    group_size: int,
    variant: str,
    swiglu_limit: float | None,
    swizzle: bool,
    gemm1_alpha: float | None,
    gemm1_clamp_limit: float | None,
    column_major_scales: bool,
    scale_tma_aligned: bool,
    scale_ue8m0: bool,
    pdl: bool,
    graph_mode: bool,
) -> str | None:
    if variant not in VARIANT_NAMES:
        return f"unknown Task-29 variant {variant!r}"
    if triton.__version__ != TRITON_BUILD:
        return (
            f"Task-29 candidate requires Triton {TRITON_BUILD}, "
            f"found {triton.__version__}"
        )
    if not _tensor_contract(
        gateup_output,
        shape=(ALIGNED_M, GATE_UP),
        stride=(GATE_UP, 1),
        dtype=torch.bfloat16,
    ):
        return "gateup_output must be exact contiguous BF16[35200,4096]"
    if not _tensor_contract(
        m_indices,
        shape=(ALIGNED_M,),
        stride=(1,),
        dtype=torch.int32,
    ):
        return "m_indices must be exact contiguous int32[35200]"
    if not _tensor_contract(
        endpoint,
        shape=(EXPERTS,),
        stride=(1,),
        dtype=torch.int32,
    ):
        return "endpoint must be exact contiguous int32[32]"
    if not (
        gateup_output.device == m_indices.device == endpoint.device
    ):
        return "gateup_output, m_indices and endpoint must share a device"
    if _device_capability(gateup_output.device.index or 0) != (10, 0):
        return "Task-29 candidate requires NVIDIA sm_100"
    if group_size != GROUP_SIZE:
        return "Task-29 candidate requires group_size=128"
    if swiglu_limit is not None:
        return "Task-29 candidate supports only unclamped GLM-5.2 SwiGLU"
    if swizzle:
        return "Task-29 candidate requires gate-first contiguous storage"
    if gemm1_alpha is not None or gemm1_clamp_limit is not None:
        return "Task-29 candidate does not support gemm1_alpha/clamp mode"
    if not column_major_scales or not scale_tma_aligned or not scale_ue8m0:
        return (
            "Task-29 candidate requires direct column-major, TMA-aligned "
            "packed UE8M0 scales"
        )
    if not pdl:
        return "Task-29 candidate requires production PDL"
    if not isinstance(graph_mode, bool):
        return "graph_mode must be an explicit bool"
    return None


def _validate_output_contract(
    gateup_output: torch.Tensor,
    output: torch.Tensor,
    scale_storage: torch.Tensor,
) -> None:
    if not _tensor_contract(
        output,
        shape=(ALIGNED_M, HIDDEN),
        stride=(HIDDEN, 1),
        dtype=torch.float8_e4m3fn,
    ):
        raise ValueError("output must be exact contiguous FP8[35200,2048]")
    if not _tensor_contract(
        scale_storage,
        shape=(PACKED_WORDS, ALIGNED_M),
        stride=(ALIGNED_M, 1),
        dtype=torch.int32,
    ):
        raise ValueError("scale_storage must be exact contiguous int32[4,35200]")
    if not (
        gateup_output.device == output.device == scale_storage.device
    ):
        raise ValueError("all Task-29 data tensors must share a device")


def allocate_outputs(
    gateup_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty(
        (ALIGNED_M, HIDDEN),
        device=gateup_output.device,
        dtype=torch.float8_e4m3fn,
    )
    scale_storage = torch.empty(
        (PACKED_WORDS, ALIGNED_M),
        device=gateup_output.device,
        dtype=torch.int32,
    )
    return output, scale_storage


def silu_mul_quant_packed_into(
    gateup_output: torch.Tensor,
    output: torch.Tensor,
    scale_storage: torch.Tensor,
    m_indices: torch.Tensor,
    endpoint: torch.Tensor,
    *,
    variant: str,
    graph_mode: bool = False,
    pdl: bool = True,
) -> None:
    """Launch one explicitly selected exact kernel into caller-owned outputs."""

    error = _eligibility_error(
        gateup_output,
        m_indices,
        endpoint,
        group_size=GROUP_SIZE,
        variant=variant,
        swiglu_limit=None,
        swizzle=False,
        gemm1_alpha=None,
        gemm1_clamp_limit=None,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
        pdl=pdl,
        graph_mode=graph_mode,
    )
    if error is not None:
        raise RuntimeError(error)
    _validate_output_contract(gateup_output, output, scale_storage)

    if variant in CUDA_VARIANTS:
        from sglang.srt.layers.glm52_opt.swiglu_quant_prefill_cuda import (
            CUDA_VARIANTS as MATERIALIZED_CUDA_VARIANTS,
        )
        from sglang.srt.layers.glm52_opt.swiglu_quant_prefill_cuda import (
            run_into,
        )

        if variant not in MATERIALIZED_CUDA_VARIANTS:
            raise RuntimeError(f"unmaterialized Task-29 CUDA variant {variant!r}")
        run_into(
            gateup_output,
            output,
            scale_storage,
            m_indices,
            endpoint,
            variant=variant,
        )
        return

    block_n, num_warps = TRITON_VARIANTS[variant]
    split_count = HIDDEN // block_n
    _silu_mul_quant_prefill_kernel[(ALIGNED_M, split_count)](
        gateup_output,
        output,
        scale_storage,
        m_indices,
        endpoint,
        ALIGNED_ROWS=ALIGNED_M,
        NUM_EXPERTS=EXPERTS,
        GATE_UP_WIDTH=GATE_UP,
        OUTPUT_WIDTH=HIDDEN,
        QUANT_GROUP=GROUP_SIZE,
        BLOCK_N=block_n,
        FP8_MAX_INV_VALUE=FP8_MAX_INV,
        INPUT_ROW_STRIDE=GATE_UP,
        OUTPUT_ROW_STRIDE=HIDDEN,
        SCALE_WORD_STRIDE=ALIGNED_M,
        SCALE_ROW_STRIDE=1,
        GATE_FIRST=True,
        PRECISE_EXPF=True,
        BF16_ROUND_TRIP=True,
        ENDPOINT_ALIGNED_EXCLUSIVE_PLUS_VALID=True,
        USE_PDL=pdl,
        GRAPH_MODE=graph_mode,
        num_warps=num_warps,
        num_stages=1,
        launch_pdl=pdl,
    )


def silu_mul_quant_packed_explicit(
    gateup_output: torch.Tensor,
    m_indices: torch.Tensor,
    endpoint: torch.Tensor,
    *,
    group_size: int,
    variant: str,
    swiglu_limit: float | None = None,
    swizzle: bool = False,
    gemm1_alpha: float | None = None,
    gemm1_clamp_limit: float | None = None,
    column_major_scales: bool = True,
    scale_tma_aligned: bool = True,
    scale_ue8m0: bool = True,
    graph_mode: bool = False,
    pdl: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate and run an explicit candidate; eligibility failures raise."""

    error = _eligibility_error(
        gateup_output,
        m_indices,
        endpoint,
        group_size=group_size,
        variant=variant,
        swiglu_limit=swiglu_limit,
        swizzle=swizzle,
        gemm1_alpha=gemm1_alpha,
        gemm1_clamp_limit=gemm1_clamp_limit,
        column_major_scales=column_major_scales,
        scale_tma_aligned=scale_tma_aligned,
        scale_ue8m0=scale_ue8m0,
        pdl=pdl,
        graph_mode=graph_mode,
    )
    if error is not None:
        raise RuntimeError(error)
    output, scale_storage = allocate_outputs(gateup_output)
    silu_mul_quant_packed_into(
        gateup_output,
        output,
        scale_storage,
        m_indices,
        endpoint,
        variant=variant,
        graph_mode=graph_mode,
        pdl=pdl,
    )
    return output, scale_storage.transpose(0, 1)


@dataclass
class _DispatchState:
    enabled: bool = False
    reason: str = "default_off"
    variant: str = ""
    gpu_id: int | None = None
    graph_mode: bool = False
    topology: dict[str, Any] = field(default_factory=dict)
    triton_version: str = ""


_STATE = _DispatchState()
_INITIALIZE_LOCK = threading.Lock()


def requested_variant() -> str:
    """Return the explicitly armed variant through the central config gate."""

    from sglang.srt.layers.glm52_opt.config import swiglu_quant_prefill_variant

    return swiglu_quant_prefill_variant() or ""


def initialization_requested() -> bool:
    return bool(requested_variant())


def _validate_startup_topology(server_args: Any) -> tuple[dict[str, Any], bool]:
    actual = {
        name: getattr(server_args, name, None) for name in REQUIRED_TOPOLOGY
    }
    if actual != REQUIRED_TOPOLOGY:
        raise RuntimeError(
            "selected Task-29 candidate requires exact TP8/DP8/EP8 "
            f"AUTO-normal DeepEP topology: actual={actual}, "
            f"required={REQUIRED_TOPOLOGY}"
        )
    prefill_config = getattr(
        getattr(server_args, "cuda_graph_config", None), "prefill", None
    )
    backend = str(getattr(prefill_config, "backend", "")).lower()
    graph_mode = backend not in ("", "disabled", "backend.disabled")
    if graph_mode:
        raise RuntimeError(
            "Task-29 promotion bucket was frozen as eager, but the resolved "
            f"prefill CUDA-graph backend is {backend!r}"
        )
    return actual, graph_mode


def initialize_after_assignment(gpu_id: int, server_args: Any) -> bool:
    """Validate topology and warm the exact Triton specialization at startup."""

    global _STATE
    variant = requested_variant()
    if not variant:
        _STATE = _DispatchState(False, "default_off")
        return False
    try:
        topology, graph_mode = _validate_startup_topology(server_args)
    except RuntimeError:
        _STATE = _DispatchState(
            False,
            "topology_or_mode_mismatch",
            variant=variant,
            gpu_id=int(gpu_id),
        )
        raise

    with _INITIALIZE_LOCK:
        if _STATE.enabled:
            if _STATE.gpu_id != int(gpu_id) or _STATE.variant != variant:
                raise RuntimeError(
                    "Task-29 candidate was initialized for another worker"
                )
            return True
        if int(torch.cuda.current_device()) != int(gpu_id):
            raise RuntimeError("Task-29 initialized before GPU assignment")
        if torch.cuda.get_device_capability(gpu_id) != (10, 0):
            raise RuntimeError("Task-29 candidate requires an sm_100 B200")
        if triton.__version__ != TRITON_BUILD:
            raise RuntimeError(
                f"Task-29 requires Triton {TRITON_BUILD}, "
                f"found {triton.__version__}"
            )

        tensors: tuple[torch.Tensor, ...] | None = None
        try:
            device = torch.device("cuda", gpu_id)
            gateup = torch.empty(
                (ALIGNED_M, GATE_UP), device=device, dtype=torch.bfloat16
            )
            output, scale_storage = allocate_outputs(gateup)
            m_indices = torch.arange(
                ALIGNED_M, device=device, dtype=torch.int32
            ).remainder_(EXPERTS)
            endpoint = torch.full(
                (EXPERTS,), ALIGNED_M, device=device, dtype=torch.int32
            )
            tensors = gateup, output, scale_storage, m_indices, endpoint
            silu_mul_quant_packed_into(
                gateup,
                output,
                scale_storage,
                m_indices,
                endpoint,
                variant=variant,
                graph_mode=graph_mode,
                pdl=True,
            )
            torch.cuda.synchronize(gpu_id)
            _STATE = _DispatchState(
                True,
                "ready",
                variant=variant,
                gpu_id=int(gpu_id),
                graph_mode=graph_mode,
                topology=topology,
                triton_version=triton.__version__,
            )
            logger.info(
                "GLM-5.2 Task-29 candidate ready: variant=%s gpu=%d",
                variant,
                gpu_id,
            )
            return True
        except Exception as exc:
            _STATE = _DispatchState(
                False,
                f"initialization_failed:{type(exc).__name__}",
                variant=variant,
                gpu_id=int(gpu_id),
                graph_mode=graph_mode,
                topology=topology,
                triton_version=triton.__version__,
            )
            raise RuntimeError(
                "explicit Task-29 candidate failed startup materialization"
            ) from exc
        finally:
            if tensors is not None:
                del tensors
                torch.cuda.empty_cache()


def maybe_silu_mul_quant_packed(
    gateup_output: torch.Tensor,
    m_indices: torch.Tensor | None,
    endpoint: torch.Tensor | None,
    *,
    group_size: int,
    swiglu_limit: float | None,
    swizzle: bool,
    gemm1_alpha: float | None = None,
    gemm1_clamp_limit: float | None = None,
    column_major_scales: bool,
    scale_tma_aligned: bool,
    scale_ue8m0: bool,
    pdl: bool,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Dispatch only the armed exact bucket; unsupported automatic calls fall back."""

    state = _STATE
    if not state.enabled:
        return None
    # Once selected and shape-targeted, malformed ABI is an error rather than
    # a silent stock timing. Unrelated shapes remain stock.
    if tuple(gateup_output.shape) != (ALIGNED_M, GATE_UP):
        return None
    if m_indices is None:
        raise RuntimeError("selected Task-29 call has no production row map")
    if endpoint is None:
        raise RuntimeError("selected Task-29 call has no scatter endpoint")
    error = _eligibility_error(
        gateup_output,
        m_indices,
        endpoint,
        group_size=group_size,
        variant=state.variant,
        swiglu_limit=swiglu_limit,
        swizzle=swizzle,
        gemm1_alpha=gemm1_alpha,
        gemm1_clamp_limit=gemm1_clamp_limit,
        column_major_scales=column_major_scales,
        scale_tma_aligned=scale_tma_aligned,
        scale_ue8m0=scale_ue8m0,
        pdl=pdl,
        graph_mode=state.graph_mode,
    )
    if error is not None:
        raise RuntimeError(f"selected Task-29 exact bucket violates ABI: {error}")
    if gateup_output.device.index != state.gpu_id:
        raise RuntimeError("selected Task-29 call reached the wrong GPU")
    return silu_mul_quant_packed_explicit(
        gateup_output,
        m_indices,
        endpoint,
        group_size=group_size,
        variant=state.variant,
        swiglu_limit=swiglu_limit,
        swizzle=swizzle,
        gemm1_alpha=gemm1_alpha,
        gemm1_clamp_limit=gemm1_clamp_limit,
        column_major_scales=column_major_scales,
        scale_tma_aligned=scale_tma_aligned,
        scale_ue8m0=scale_ue8m0,
        graph_mode=state.graph_mode,
        pdl=pdl,
    )


def dispatch_state() -> dict[str, Any]:
    state = _STATE
    return {
        "enabled": state.enabled,
        "reason": state.reason,
        "variant": state.variant,
        "gpu_id": state.gpu_id,
        "graph_mode": state.graph_mode,
        "topology": state.topology,
        "triton_version": state.triton_version,
    }
