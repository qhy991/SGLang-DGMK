import importlib
import json
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial
from typing import Any, Callable, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.deep_gemm_wrapper.configurer import (  # noqa: F401
    DEEPGEMM_BLACKWELL,
    DEEPGEMM_NEED_TMA_ALIGNED_SCALES,
    DEEPGEMM_SCALE_UE8M0,
    ENABLE_JIT_DEEPGEMM,
)
from sglang.srt.layers.glm52_opt.w13_decode import (
    REQUIRED_NUM_SMS,
    REQUIRED_PDL,
    REQUIRED_TC_UTIL,
    dispatch_enabled as w13_dispatch_enabled,
    dispatch_state,
    initialization_requested,
    initialize_w13_decode_after_assignment,
    is_exact_w13_tensor_call,
    is_w13_tensor_shape_family,
    try_dispatch_w13_decode,
)
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class _LazyCompileUtils:
    """Delay compile_utils' cache-environment rewrite until worker setup."""

    def __init__(self):
        self._module = None

    def load(self):
        if self._module is None:
            self._module = importlib.import_module(
                "sglang.srt.layers.deep_gemm_wrapper.compile_utils"
            )
        return self._module

    def __getattr__(self, name):
        return getattr(self.load(), name)


# Keep the established attribute for tests/callers while making it lazy.
compile_utils = _LazyCompileUtils()


if ENABLE_JIT_DEEPGEMM:
    import deep_gemm
    from deep_gemm.utils.layout import get_mn_major_tma_aligned_tensor  # noqa: F401

_SANITY_CHECK = envs.SGLANG_DEEPGEMM_SANITY_CHECK.get()
_NUM_SMS_OVERRIDE_ACTIVE: ContextVar[bool] = ContextVar(
    "deep_gemm_num_sms_override_active", default=False
)
_W2_BM16_PROFILE_REQUESTED = False
_W2_BM16_PREPARED_CONTRACT: Any = None
_W2_BM16_DISPATCH: Optional[Callable[..., bool]] = None
_W2_BM16_CALLSITE_PREPARE: Optional[Callable[..., bool]] = None


def w2_bm16_profile_requested() -> bool:
    """Cheap setup-time flag; default-off callers never import candidate code."""
    return _W2_BM16_PROFILE_REQUESTED


def _glm52_moe_dispatch_compatible(
    overlap_args: Optional[Any],
    recipe_a: Optional[Tuple[int, int]],
    recipe_b: Optional[Tuple[int, int]],
) -> bool:
    return overlap_args is None and recipe_a is None and recipe_b is None


# TODO maybe rename these functions
def grouped_gemm_nt_f8f8bf16_masked(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    overlap_args: Optional[Any] = None,
    max_block_n: int = 256,
    recipe_a: Optional[Tuple[int, int]] = None,
    recipe_b: Optional[Tuple[int, int]] = None,
):
    num_groups, _, k = lhs[0].shape
    _, n, _ = rhs[0].shape

    # The selected exact W13 route contains no generic hook, precompile,
    # statistics, lock, file write, scale adapter, or stock retry. The optional
    # cached profiler flag adds only the explicitly requested NVTX range.
    exact_w13_tensors = False
    w13_tensor_family = False
    if w13_dispatch_enabled():
        exact_w13_tensors = is_exact_w13_tensor_call(lhs, rhs, out, masked_m)
        w13_tensor_family = exact_w13_tensors or is_w13_tensor_shape_family(
            lhs, rhs, out, masked_m
        )
    if w13_tensor_family and try_dispatch_w13_decode(
        lhs,
        rhs,
        out,
        masked_m,
        expected_m,
        overlap_args=overlap_args,
        max_block_n=max_block_n,
        recipe_a=recipe_a,
        recipe_b=recipe_b,
    ):
        return None

    _sanity_check_input(lhs)
    _sanity_check_input(rhs)

    lhs = _ensure_cuda(lhs)
    rhs = _ensure_cuda(rhs)
    kernel_type = compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED

    with compile_utils.deep_gemm_execution_hook(
        expected_m, n, k, num_groups, kernel_type
    ):
        with configure_deep_gemm_num_sms(
            overlap_args.num_sms if overlap_args is not None else None
        ):
            from sglang.srt.layers.glm52_opt.dispatch import try_dispatch_moe_masked

            # The glm52 replacement does not implement DeepEP/TBO overlap or
            # recipe-aware FP4/MXFP8 calls.  Taking it here would drop
            # enable_overlap/signal and change the overlap return contract,
            # which can regress or break the full MoE pipeline even if the
            # isolated GEMM is faster.
            glm52_compatible = _glm52_moe_dispatch_compatible(
                overlap_args, recipe_a, recipe_b
            )
            if (
                not exact_w13_tensors
                and glm52_compatible
                and try_dispatch_moe_masked(lhs, rhs, out, masked_m, expected_m)
            ):
                return out

            fp4_kwargs = {}
            if recipe_a is not None:
                fp4_kwargs["recipe_a"] = recipe_a
            if recipe_b is not None:
                fp4_kwargs["recipe_b"] = recipe_b

            return deep_gemm.fp8_m_grouped_gemm_nt_masked(
                lhs,
                rhs,
                out,
                masked_m,
                expected_m,
                **fp4_kwargs,
                **(
                    dict(
                        enable_overlap=True,
                        max_block_n=max_block_n,
                        signal=overlap_args.signal,
                    )
                    if overlap_args is not None
                    else {}
                ),
            )


def _grouped_gemm_nt_f8f8bf16_masked_w2_bm16(
    layer_contract: Any,
    runtime_contract: Any,
    callsite_prepare: Optional[Callable[..., bool]],
    candidate_dispatch: Optional[Callable[..., bool]],
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    overlap_args: Optional[Any] = None,
    max_block_n: int = 256,
    recipe_a: Optional[Tuple[int, int]] = None,
    recipe_b: Optional[Tuple[int, int]] = None,
):
    """Armed W2 down-GEMM callable; never installed on the default runner."""
    num_groups, _, k = lhs[0].shape
    _, n, _ = rhs[0].shape
    kernel_type = compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED

    _sanity_check_input(lhs)
    _sanity_check_input(rhs)

    lhs = _ensure_cuda(lhs)
    rhs = _ensure_cuda(rhs)

    with compile_utils.deep_gemm_execution_hook(
        expected_m, n, k, num_groups, kernel_type
    ):
        with configure_deep_gemm_num_sms(
            overlap_args.num_sms if overlap_args is not None else None
        ):
            compatible = (
                overlap_args is None
                and recipe_a is None
                and recipe_b is None
                and not _NUM_SMS_OVERRIDE_ACTIVE.get()
            )
            current_forward_state = (
                getattr(runtime_contract, "current_forward_state", None)
                if compatible
                else None
            )
            forward_state = (
                current_forward_state()
                if callable(current_forward_state)
                else None
            )
            selected_bucket = bool(
                forward_state is not None
                and forward_state[0].is_decode()
                and (
                    (
                        int(forward_state[1]) == 16
                        and int(expected_m) in (4, 5)
                    )
                    or (
                        int(forward_state[1]) == 32
                        and int(expected_m) in (8, 9)
                    )
                )
            )
            if selected_bucket:
                if (
                    layer_contract is None
                    or runtime_contract is None
                    or callsite_prepare is None
                    or candidate_dispatch is None
                ):
                    raise RuntimeError(
                        "selected W2/BM16 decode bucket has no prepared "
                        "runtime/layer contract"
                    )
                if layer_contract.callsite_checked:
                    callsite_eligible = layer_contract.callsite_eligible
                else:
                    callsite_eligible = callsite_prepare(
                        layer_contract,
                        lhs,
                        rhs,
                        out,
                        masked_m,
                        expected_m=expected_m,
                        recipe_a=recipe_a,
                        recipe_b=recipe_b,
                        overlap_args=overlap_args,
                    )
                if not callsite_eligible:
                    raise RuntimeError(
                        "selected W2/BM16 decode call no longer matches the "
                        f"prepared ABI: {layer_contract.callsite_reason}"
                    )
                if candidate_dispatch(
                    runtime_contract,
                    lhs,
                    rhs,
                    out,
                    masked_m,
                    expected_m,
                    callsite_eligible=True,
                ):
                    # The stock no-overlap ABI writes `out` and returns None.
                    return None
                raise RuntimeError(
                    "selected W2/BM16 candidate unexpectedly declined before launch"
                )

            # Unrelated prefill/speculative modes and unregistered buckets use
            # authoritative stock and suppress the legacy generic replacement.
            fp4_kwargs = {}
            if recipe_a is not None:
                fp4_kwargs["recipe_a"] = recipe_a
            if recipe_b is not None:
                fp4_kwargs["recipe_b"] = recipe_b
            return deep_gemm.fp8_m_grouped_gemm_nt_masked(
                lhs,
                rhs,
                out,
                masked_m,
                expected_m,
                **fp4_kwargs,
                **(
                    dict(
                        enable_overlap=True,
                        max_block_n=max_block_n,
                        signal=overlap_args.signal,
                    )
                    if overlap_args is not None
                    else {}
                ),
            )


def configure_w2_bm16_masked_down_gemm(
    runner_core: Any,
    *,
    w2_weight: torch.Tensor,
    w2_scale: torch.Tensor,
    block_shape: Optional[list[int]],
    deep_gemm_backend: bool,
    is_fp4_experts: bool,
    use_mxfp8: bool,
) -> None:
    """Bind an armed-only per-runner callable after immutable weights exist."""
    if not _W2_BM16_PROFILE_REQUESTED:
        return

    if (
        _W2_BM16_PREPARED_CONTRACT is None
        or _W2_BM16_CALLSITE_PREPARE is None
        or _W2_BM16_DISPATCH is None
    ):
        raise RuntimeError(
            "requested W2/BM16 profile has no prepared runtime contract"
        )
    from sglang.srt.layers.glm52_opt.experimental_deepgemm import (
        create_w2_bm16_layer_contract,
    )

    layer_contract = create_w2_bm16_layer_contract(
        w2_weight=w2_weight,
        w2_scale=w2_scale,
        block_shape=block_shape,
        deep_gemm_backend=deep_gemm_backend,
        is_fp4_experts=is_fp4_experts,
        use_mxfp8=use_mxfp8,
    )
    if layer_contract is None or not layer_contract.static_eligible:
        reason = (
            "missing-layer-contract"
            if layer_contract is None
            else layer_contract.static_reason
        )
        raise RuntimeError(
            "requested W2/BM16 profile cannot bind the GLM-5.2 W2 layer: "
            f"{reason}"
        )

    runner_core.set_masked_down_gemm(
        partial(
            _grouped_gemm_nt_f8f8bf16_masked_w2_bm16,
            layer_contract,
            _W2_BM16_PREPARED_CONTRACT,
            _W2_BM16_CALLSITE_PREPARE,
            _W2_BM16_DISPATCH,
        )
    )


def _ensure_cuda(
    pair: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        pair[0].cuda() if not pair[0].is_cuda else pair[0],
        pair[1].cuda() if not pair[1].is_cuda else pair[1],
    )


def grouped_gemm_nt_bf16_masked(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
):
    num_groups, _, k = a.shape
    _, n, _ = b.shape
    kernel_type = compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_BF16_MASKED

    with compile_utils.deep_gemm_execution_hook(
        expected_m, n, k, num_groups, kernel_type
    ):
        return deep_gemm.m_grouped_bf16_gemm_nt_masked(
            a,
            b,
            d,
            masked_m,
            expected_m,
        )


def grouped_gemm_nt_f8f8bf16_contig(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    m_indices: torch.Tensor,
    recipe_a: Optional[Tuple[int, int]] = None,
    recipe_b: Optional[Tuple[int, int]] = None,
    *,
    compiled_dims: str = "nk",
    use_psum_layout: bool = False,
    ensure_zero_padding: bool = True,
    expected_m_for_psum_layout: Optional[int] = None,
):
    m, k = lhs[0].shape
    num_groups, n, _ = rhs[0].shape
    kernel_type = compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG

    if m == 0:
        return

    _sanity_check_input(lhs)
    _sanity_check_input(rhs)

    fp4_kwargs = {}
    if recipe_a is not None:
        fp4_kwargs["recipe_a"] = recipe_a
    if recipe_b is not None:
        fp4_kwargs["recipe_b"] = recipe_b
    if compiled_dims != "nk":
        fp4_kwargs["compiled_dims"] = compiled_dims
    if use_psum_layout:
        fp4_kwargs.update(
            use_psum_layout=True,
            ensure_zero_padding=ensure_zero_padding,
            expected_m_for_psum_layout=expected_m_for_psum_layout,
        )

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            lhs, rhs, out, m_indices, **fp4_kwargs
        )


def grouped_gemm_nt_bf16_contig(
    a: torch.Tensor, b: torch.Tensor, d: torch.Tensor, m_indices: torch.Tensor
):
    m, k = a.shape
    num_groups, n, _ = b.shape
    kernel_type = compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_BF16_CONTIG

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b, d, m_indices)


def gemm_nt_f8f8bf16(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
):
    m, k = lhs[0].shape
    n, _ = rhs[0].shape
    num_groups = 1
    kernel_type = compile_utils.DeepGemmKernelType.GEMM_NT_F8F8BF16

    _sanity_check_input(lhs)
    _sanity_check_input(rhs)

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.fp8_gemm_nt(
            lhs,
            rhs,
            out,
        )


def gemm_nt_mxfp8_f8f8bf16(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
):
    m, k = lhs[0].shape
    n, _ = rhs[0].shape
    num_groups = 1
    kernel_type = compile_utils.DeepGemmKernelType.GEMM_NT_F8F8BF16

    _sanity_check_input(lhs)
    _sanity_check_input(rhs)

    disable_cast = lhs[1].dtype == torch.int and rhs[1].dtype == torch.int

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.fp8_fp4_gemm_nt(
            lhs,
            rhs,
            out,
            recipe_a=(1, 32),
            recipe_b=(1, 32),
            disable_ue8m0_cast=disable_cast,
        )


def gemm_nt_bf16bf16f32(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    out: torch.Tensor,
):
    m, k = lhs.shape
    n, _ = rhs.shape
    num_groups = 1
    kernel_type = compile_utils.DeepGemmKernelType.GEMM_NT_BF16BF16F32

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.bf16_gemm_nt(lhs, rhs, out)


def tf32_hc_prenorm_gemm(
    x: torch.Tensor,
    fn: torch.Tensor,
    out: torch.Tensor,
    sqrsum: torch.Tensor,
    num_splits: Optional[int],
):
    if x.shape[0] == 0:
        return
    deep_gemm.tf32_hc_prenorm_gemm(x, fn, out, sqrsum, num_splits=num_splits)


def update_deep_gemm_config(gpu_id: int, server_args: ServerArgs):
    global _W2_BM16_PROFILE_REQUESTED
    global _W2_BM16_PREPARED_CONTRACT
    global _W2_BM16_DISPATCH
    global _W2_BM16_CALLSITE_PREPARE

    _W2_BM16_PROFILE_REQUESTED = False
    _W2_BM16_PREPARED_CONTRACT = None
    _W2_BM16_DISPATCH = None
    _W2_BM16_CALLSITE_PREPARE = None
    w2_forward_context = None

    # deep_gemm.set_pdl can initialize CUDA state, so run it only after the
    # scheduler/TP worker has been forked and assigned a GPU.
    if envs.SGLANG_DEEPGEMM_PDL.get() and hasattr(deep_gemm, "set_pdl"):
        deep_gemm.set_pdl(True)

    compile_utils_configured = False
    if initialization_requested():
        # This bounded experiment fixes the denominator explicitly rather than
        # inheriting DeviceRuntime defaults or a pre-worker state.
        original_installed_state = {
            "pdl": bool(deep_gemm.get_pdl()),
            "num_sms": int(deep_gemm.get_num_sms()),
            "tc_util": int(deep_gemm.get_tc_util()),
        }
        initialized = False
        try:
            deep_gemm.set_pdl(REQUIRED_PDL)
            deep_gemm.set_num_sms(REQUIRED_NUM_SMS)
            deep_gemm.set_tc_util(REQUIRED_TC_UTIL)
            installed_state = {
                "pdl": bool(deep_gemm.get_pdl()),
                "num_sms": int(deep_gemm.get_num_sms()),
                "tc_util": int(deep_gemm.get_tc_util()),
            }
            required_state = {
                "pdl": REQUIRED_PDL,
                "num_sms": REQUIRED_NUM_SMS,
                "tc_util": REQUIRED_TC_UTIL,
            }
            if installed_state != required_state:
                raise RuntimeError(
                    "installed DeepGEMM W13 startup state mismatch: "
                    f"actual={installed_state}, required={required_state}"
                )
            compile_utils_configured = initialize_w13_decode_after_assignment(
                gpu_id,
                server_args,
                compile_utils_loader=compile_utils.load,
            )
            initialized = bool(dispatch_state()["enabled"])
        finally:
            # An invalid variant/manifest or failed DSO/JIT setup must return
            # to the exact stock runtime state; opt-in must not perturb its
            # fallback denominator.
            if not initialized:
                deep_gemm.set_pdl(original_installed_state["pdl"])
                deep_gemm.set_num_sms(original_installed_state["num_sms"])
                deep_gemm.set_tc_util(original_installed_state["tc_util"])

    if not compile_utils_configured:
        compile_utils.update_deep_gemm_config(gpu_id, server_args)

    # Opt-in GLM-5.2 experimental DeepGEMM overlay (does not replace stock import).
    try:
        from sglang.srt.layers.glm52_opt.config import (
            deepgemm_variant,
            is_enabled,
            w2_bm16_enabled,
        )

        if w2_bm16_enabled():
            # Arm before any fallible candidate import/preparation. If setup
            # fails, all W2 calls explicitly choose stock and do not enter a
            # legacy experimental dispatcher.
            _W2_BM16_PROFILE_REQUESTED = True
            from sglang.srt.layers.glm52_opt.experimental_deepgemm import (
                prepare_w2_bm16_callsite_contract,
                prepare_w2_bm16_deep_gemm,
            )
            from sglang.srt.layers.glm52_opt.dispatch import (
                try_dispatch_moe_w2_bm16,
            )

            contract = prepare_w2_bm16_deep_gemm(gpu_id)
            _W2_BM16_PREPARED_CONTRACT = contract
            _W2_BM16_DISPATCH = try_dispatch_moe_w2_bm16
            _W2_BM16_CALLSITE_PREPARE = prepare_w2_bm16_callsite_contract
            w2_forward_context = contract.forward_context
            logger.info(
                "GLM-5.2 W2/BM16 DeepGEMM runtime prepared: %s",
                json.dumps(contract.evidence(), sort_keys=True),
            )
        elif (
            not initialization_requested()
            and is_enabled()
            and deepgemm_variant()
        ):
            from sglang.srt.layers.glm52_opt.experimental_deepgemm import (
                get_experimental_deep_gemm,
            )

            get_experimental_deep_gemm()
            logger.info(
                "GLM-5.2 experimental DeepGEMM overlay loaded (variant=%s)",
                deepgemm_variant(),
            )
    except Exception as exc:
        if _W2_BM16_PROFILE_REQUESTED:
            raise RuntimeError(
                "requested GLM-5.2 W2/BM16 runtime preparation failed"
            ) from exc
        logger.warning("GLM-5.2 DeepGEMM overlay load skipped: %s", exc)
    return w2_forward_context


@contextmanager
def configure_deep_gemm_num_sms(num_sms):
    if num_sms is None or not ENABLE_JIT_DEEPGEMM:
        yield
    else:
        original_num_sms = deep_gemm.get_num_sms()
        token = _NUM_SMS_OVERRIDE_ACTIVE.set(True)
        deep_gemm.set_num_sms(num_sms)
        try:
            yield
        finally:
            deep_gemm.set_num_sms(original_num_sms)
            _NUM_SMS_OVERRIDE_ACTIVE.reset(token)


def _sanity_check_input(x_fp8: Tuple[torch.Tensor, torch.Tensor]):
    if not _SANITY_CHECK:
        return

    x, x_scale = x_fp8

    if x_scale.dtype == torch.int:
        return

    from sglang.srt.layers.quantization.fp8_utils import ceil_to_ue8m0

    x_scale_ceil = ceil_to_ue8m0(x_scale)
    assert torch.all(x_scale == x_scale_ceil), f"{x_scale=} {x_scale_ceil=}"
