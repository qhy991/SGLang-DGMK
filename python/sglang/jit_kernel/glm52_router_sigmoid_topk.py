from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
import triton

from sglang.jit_kernel.moe_fused_gate import _router_triton_kernel
from sglang.jit_kernel.utils import is_arch_support_pdl
from sglang.kernel_api_logging import debug_kernel_api


@dataclass(frozen=True)
class Glm52RouterTactic:
    backend: str
    block_m: int
    num_warps: int
    description: str


GLM52_ROUTER_TACTICS: Final[dict[str, Glm52RouterTactic]] = {
    "T1": Glm52RouterTactic(
        backend="triton",
        block_m=2,
        num_warps=1,
        description="two rows per one-warp CTA",
    ),
    "T2": Glm52RouterTactic(
        backend="triton",
        block_m=4,
        num_warps=1,
        description="four rows per one-warp CTA",
    ),
    "T3": Glm52RouterTactic(
        backend="triton",
        block_m=4,
        num_warps=4,
        description="four rows per four-warp CTA; generated layout must prove row ownership",
    ),
    "C1": Glm52RouterTactic(
        backend="cuda",
        block_m=4,
        num_warps=4,
        description="native CUDA four rows per CTA; one row owned by each warp",
    ),
    "C2": Glm52RouterTactic(
        backend="cuda",
        block_m=8,
        num_warps=8,
        description="native CUDA eight rows per CTA; one row owned by each warp",
    ),
}


def _validate_glm52_router_abi(
    scores: torch.Tensor,
    bias: torch.Tensor,
    tactic: str,
) -> Glm52RouterTactic:
    if tactic not in GLM52_ROUTER_TACTICS:
        raise ValueError(
            f"unsupported GLM-5.2 router tactic {tactic!r}; "
            f"expected one of {tuple(GLM52_ROUTER_TACTICS)}"
        )
    if not scores.is_cuda or not bias.is_cuda:
        raise RuntimeError("GLM-5.2 router candidate requires CUDA tensors")
    if scores.device != bias.device:
        raise RuntimeError("scores and correction bias must use the same CUDA device")
    if torch.cuda.get_device_capability(scores.device) != (10, 0):
        raise RuntimeError("GLM-5.2 router candidate is restricted to NVIDIA SM100")
    if scores.dtype != torch.float32:
        raise TypeError("GLM-5.2 router scores must be FP32")
    if bias.dtype != torch.float32:
        raise TypeError("GLM-5.2 router correction bias must be FP32")
    if scores.ndim != 2 or tuple(scores.shape) not in ((16, 256), (32, 256)):
        raise ValueError("GLM-5.2 router scores must have exact shape [16|32, 256]")
    if tuple(scores.stride()) != (256, 1) or scores.storage_offset() != 0:
        raise ValueError(
            "GLM-5.2 router scores must use the production contiguous row-major layout"
        )
    if bias.ndim != 1 or tuple(bias.shape) != (256,):
        raise ValueError("GLM-5.2 router correction bias must have exact shape [256]")
    if tuple(bias.stride()) != (1,) or bias.storage_offset() != 0:
        raise ValueError(
            "GLM-5.2 router correction bias must use the production contiguous layout"
        )
    return GLM52_ROUTER_TACTICS[tactic]


@debug_kernel_api
def glm52_router_sigmoid_topk(
    scores: torch.Tensor,
    bias: torch.Tensor,
    *,
    tactic: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact-shape, default-off GLM-5.2 sigmoid no-aux TopK candidate.

    The observable ABI matches the active SGLang ``moe_fused_gate`` wrapper:
    FP32 ``[M,256]`` scores and FP32 ``[256]`` correction bias produce newly
    allocated FP32 weights and int32 ordered expert IDs with shape ``[M,8]``.
    The fixed boundary does not apply routed scale 2.5 inside TopK.

    Unsupported metadata raises before any candidate launch.  There is no
    candidate-to-stock fallback.
    """

    selected = _validate_glm52_router_abi(scores, bias, tactic)
    if selected.backend != "triton":
        raise ValueError(f"tactic {tactic!r} is not a Triton router tactic")
    m = scores.shape[0]
    weights = torch.empty((m, 8), dtype=torch.float32, device=scores.device)
    indices = torch.empty((m, 8), dtype=torch.int32, device=scores.device)

    use_pdl = is_arch_support_pdl()
    launch_options = {"launch_pdl": True} if use_pdl else {}
    grid = (triton.cdiv(m, selected.block_m),)
    _router_triton_kernel[grid](
        scores,
        bias,
        weights,
        indices,
        m,
        1.0,
        0.0,
        N=256,
        K=8,
        K_ROUTED=8,
        BLOCK_M=selected.block_m,
        BLOCK_N=256,
        BLOCK_K=8,
        N_GROUP=1,
        TOPK_GROUP=1,
        EXPERTS_PER_GROUP=256,
        BLOCK_G=1,
        SCORING_FUNC=0,
        HAS_SOFTCAP=False,
        RENORMALIZE=True,
        APPLY_SCALE=False,
        USE_PDL=use_pdl,
        stride_sm=scores.stride(0),
        stride_sn=scores.stride(1),
        stride_wm=weights.stride(0),
        stride_wk=weights.stride(1),
        stride_im=indices.stride(0),
        stride_ik=indices.stride(1),
        num_warps=selected.num_warps,
        **launch_options,
    )
    return weights, indices
