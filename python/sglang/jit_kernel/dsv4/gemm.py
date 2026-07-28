import os

import torch

from sglang.srt.environ import envs
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.utils import get_bool_env_var, is_hip

_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _use_aiter:
    from aiter.tuned_gemm import tgemm

_linear_bf16_fp32_algo = envs.SGLANG_OPT_BF16_FP32_GEMM_ALGO.get()
_ROUTER_PROFILE_ID = "task31-b200-cublas-noaux-v1"


def _parse_router_tactics(raw: str) -> dict[int, str]:
    if not raw:
        return {}
    parsed: dict[int, str] = {}
    for item in raw.split(","):
        key, separator, tactic = item.partition("=")
        if (
            not separator
            or key not in ("m16", "m32")
            or tactic not in ("A", "B", "C", "D")
        ):
            raise ValueError(
                "SGLANG_GLM52_ROUTER_LOGIT_GEMM_TACTICS must use "
                "'m16=A,m32=C' syntax with tactics A-D"
            )
        m = int(key[1:])
        if m in parsed:
            raise ValueError(f"duplicate router tactic for M={m}")
        parsed[m] = tactic
    return parsed


_router_profile = os.environ.get("SGLANG_GLM52_ROUTER_LOGIT_GEMM_PROFILE", "")
_router_tactics = _parse_router_tactics(
    os.environ.get("SGLANG_GLM52_ROUTER_LOGIT_GEMM_TACTICS", "")
)
if bool(_router_profile) != bool(_router_tactics):
    raise ValueError(
        "router profile and tactic map must either both be set or both be absent"
    )
if _router_profile and _router_profile != _ROUTER_PROFILE_ID:
    raise ValueError(f"unsupported router experiment profile {_router_profile!r}")


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
    """Router-only, default-off Task 31 dispatch around the frozen cublas path."""

    m = hidden_states.shape[0] if hidden_states.dim() == 2 else -1
    tactic = _router_tactics.get(m)
    if tactic is None:
        return linear_bf16_fp32(hidden_states, router_weight)

    # An exact configured bucket is an explicit experiment.  Fail before
    # candidate import if its denominator or ABI is not the frozen contract.
    if _linear_bf16_fp32_algo != "cublas":
        raise RuntimeError(
            "Task 31 requires SGLANG_OPT_BF16_FP32_GEMM_ALGO=cublas "
            "to be frozen before import"
        )
    if (
        not hidden_states.is_cuda
        or hidden_states.device != router_weight.device
        or hidden_states.dtype != torch.bfloat16
        or router_weight.dtype != torch.bfloat16
        or tuple(hidden_states.shape) != (m, 6144)
        or tuple(router_weight.shape) != (256, 6144)
        or tuple(hidden_states.stride()) != (6144, 1)
        or tuple(router_weight.stride()) != (6144, 1)
        or hidden_states.storage_offset() != 0
        or router_weight.storage_offset() != 0
        or hidden_states.data_ptr() % 32
        or router_weight.data_ptr() % 32
    ):
        raise RuntimeError("configured Task 31 bucket violated its exact router ABI")
    if torch.cuda.get_device_capability(hidden_states.device) != (10, 0):
        raise RuntimeError("configured Task 31 bucket requires an SM100 device")

    from sglang.jit_kernel.cutedsl_glm52_router_logit_gemm import (
        cutedsl_glm52_router_logit_gemm,
    )

    # Once candidate import begins, every failure propagates.  There is no
    # exception handler and no stock execution after this call.
    return cutedsl_glm52_router_logit_gemm(
        hidden_states,
        router_weight,
        tactic=tactic,
    )
