"""GLM-5.2 phase-aware optimized kernel dispatch for SGLang."""

from sglang.srt.layers.glm52_opt.config import is_enabled, profile_name
from sglang.srt.layers.glm52_opt.context import (
    get_forward_mode,
    get_op_name,
    op_context,
    set_forward_mode,
)
from sglang.srt.layers.glm52_opt.dispatch import try_dispatch_fp8_gemm, try_dispatch_moe_masked

__all__ = [
    "is_enabled",
    "profile_name",
    "get_forward_mode",
    "get_op_name",
    "op_context",
    "set_forward_mode",
    "try_dispatch_fp8_gemm",
    "try_dispatch_moe_masked",
]
