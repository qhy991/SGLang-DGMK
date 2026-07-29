from __future__ import annotations

from typing import TYPE_CHECKING, Final

import torch

from sglang.jit_kernel.glm52_router_sigmoid_topk import (
    _validate_glm52_router_abi,
)
from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args
from sglang.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from tvm_ffi.module import Module


GLM52_ROUTER_CUDA_WARPS: Final[dict[str, int]] = {
    "C1": 4,
    "C2": 8,
}


@cache_once
def _jit_glm52_router_sigmoid_topk_cuda_module(
    warps_per_cta: int,
) -> Module:
    args = make_cpp_args(warps_per_cta, True)
    return load_jit(
        "glm52_router_sigmoid_topk_cuda",
        *args,
        cuda_files=["moe/glm52_router_sigmoid_topk.cuh"],
        cuda_wrappers=[
            (
                "glm52_router_sigmoid_topk",
                f"Glm52RouterSigmoidTopKKernel<{args}>::run",
            )
        ],
    )


@debug_kernel_api
def glm52_router_sigmoid_topk_cuda(
    scores: torch.Tensor,
    bias: torch.Tensor,
    *,
    tactic: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Native exact-shape SM100 candidates for the GLM-5.2 decode router."""

    selected = _validate_glm52_router_abi(scores, bias, tactic)
    if selected.backend != "cuda":
        raise ValueError(f"tactic {tactic!r} is not a CUDA router tactic")

    m = scores.shape[0]
    weights = torch.empty((m, 8), dtype=torch.float32, device=scores.device)
    indices = torch.empty((m, 8), dtype=torch.int32, device=scores.device)
    module = _jit_glm52_router_sigmoid_topk_cuda_module(selected.num_warps)
    module.glm52_router_sigmoid_topk(scores, bias, weights, indices)
    return weights, indices
