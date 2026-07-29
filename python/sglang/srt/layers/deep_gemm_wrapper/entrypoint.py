import importlib
import logging
from contextlib import contextmanager
from typing import Any, Optional, Tuple

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
    dispatch_state,
    initialization_requested,
    initialize_w13_decode_after_assignment,
    is_exact_w13_tensor_call,
    try_dispatch_w13_decode,
)
from sglang.srt.layers.glm52_opt.w13_prefill import (
    dispatch_state as w13_prefill_dispatch_state,
    initialization_requested as w13_prefill_initialization_requested,
    initialize_w13_prefill_after_assignment,
)
from sglang.srt.layers.glm52_opt.swiglu_quant_prefill import (
    initialization_requested as swiglu_quant_prefill_initialization_requested,
    initialize_after_assignment as initialize_swiglu_quant_prefill_after_assignment,
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

    # The selected exact W13 route must contain no generic hook, precompile,
    # statistics, lock, file write, NVTX range, scale adapter, or stock retry.
    # Its launcher checks every shape/stride/dtype/metadata guard again.
    exact_w13_tensors = is_exact_w13_tensor_call(lhs, rhs, out, masked_m)
    if exact_w13_tensors and try_dispatch_w13_decode(
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
    # deep_gemm.set_pdl can initialize CUDA state, so run it only after the
    # scheduler/TP worker has been forked and assigned a GPU.
    if envs.SGLANG_DEEPGEMM_PDL.get() and hasattr(deep_gemm, "set_pdl"):
        deep_gemm.set_pdl(True)

    decode_requested = initialization_requested()
    prefill_requested = w13_prefill_initialization_requested()
    swiglu_quant_requested = swiglu_quant_prefill_initialization_requested()
    if sum((decode_requested, prefill_requested, swiglu_quant_requested)) > 1:
        raise RuntimeError(
            "W13 decode, W13 prefill and Task-29 activation experiments "
            "cannot be selected together"
        )

    compile_utils_configured = False
    if decode_requested or prefill_requested:
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
            if decode_requested:
                compile_utils_configured = initialize_w13_decode_after_assignment(
                    gpu_id,
                    server_args,
                    compile_utils_loader=compile_utils.load,
                )
                initialized = bool(dispatch_state()["enabled"])
            else:
                compile_utils_configured = initialize_w13_prefill_after_assignment(
                    gpu_id,
                    server_args,
                    compile_utils_loader=compile_utils.load,
                )
                initialized = bool(w13_prefill_dispatch_state()["enabled"])
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

    if swiglu_quant_requested:
        initialize_swiglu_quant_prefill_after_assignment(gpu_id, server_args)

    # Opt-in GLM-5.2 experimental DeepGEMM overlay (does not replace stock import).
    try:
        from sglang.srt.layers.glm52_opt.config import deepgemm_variant, is_enabled

        if (
            not decode_requested
            and not prefill_requested
            and not swiglu_quant_requested
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
        logger.warning("GLM-5.2 DeepGEMM overlay load skipped: %s", exc)


@contextmanager
def configure_deep_gemm_num_sms(num_sms):
    if num_sms is None or not ENABLE_JIT_DEEPGEMM:
        yield
    else:
        original_num_sms = deep_gemm.get_num_sms()
        deep_gemm.set_num_sms(num_sms)
        try:
            yield
        finally:
            deep_gemm.set_num_sms(original_num_sms)


def _sanity_check_input(x_fp8: Tuple[torch.Tensor, torch.Tensor]):
    if not _SANITY_CHECK:
        return

    x, x_scale = x_fp8

    if x_scale.dtype == torch.int:
        return

    from sglang.srt.layers.quantization.fp8_utils import ceil_to_ue8m0

    x_scale_ceil = ceil_to_ue8m0(x_scale)
    assert torch.all(x_scale == x_scale_ceil), f"{x_scale=} {x_scale_ceil=}"
