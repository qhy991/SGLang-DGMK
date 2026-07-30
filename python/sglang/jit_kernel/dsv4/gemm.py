import torch

from sglang.srt.environ import envs
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.utils import get_bool_env_var, is_hip

_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _use_aiter:
    from aiter.tuned_gemm import tgemm

_linear_bf16_fp32_algo = envs.SGLANG_OPT_BF16_FP32_GEMM_ALGO.get()


def linear_bf16_fp32(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if _use_aiter:
        return tgemm.mm(x, y, otype=x.dtype).float()
    elif _linear_bf16_fp32_algo == "deep_gemm":
        z = torch.empty(x.size(0), y.size(0), dtype=torch.float32, device=x.device)
        deep_gemm_wrapper.gemm_nt_bf16bf16f32(x, y, z)
        return z
    else:
        return torch.mm(x, y.t(), out_dtype=torch.float32)


def router_linear_bf16_fp32(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
) -> torch.Tensor:
    """GLM-5.2 router interception with the unchanged GEMM as fallback."""
    from sglang.srt.layers.glm52_opt import config as glm52_opt_config

    if glm52_opt_config.is_enabled():
        from sglang.srt.layers.glm52_opt.dispatch import (
            try_dispatch_router_logit_gemm,
        )

        candidate = try_dispatch_router_logit_gemm(hidden_states, router_weight)
        if candidate is not None:
            return candidate
    return linear_bf16_fp32(hidden_states, router_weight)
