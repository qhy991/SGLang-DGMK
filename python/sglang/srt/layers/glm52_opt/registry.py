"""Registry of GLM-5.2 optimized kernels keyed by (op, phase).

Aligned with Kernel-Harness ``llm_flops_style/_common.py`` DECODE/PREFILL_SWAPS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

from sglang.srt.layers.glm52_opt.config import profile_name

KernelKind = Literal[
    "fp8_gemm",
    "moe_masked",
    "bmm",
    "dsa",
    "score_mqa",
    "bf16_gemm",
    "indexer",
]

RunFn = Callable[[dict], object]


@dataclass(frozen=True)
class KernelSpec:
    op: str
    phase: str
    archive_ref: str
    kind: KernelKind
    enabled: bool = True


_DECODE: dict[str, KernelSpec] = {
    "fused_qkv_a_proj": KernelSpec(
        "fused_qkv_a_proj",
        "decode",
        "best-hechenxi-0720/fused_qkv_a_decode",
        "fp8_gemm",
    ),
    "q_b_proj": KernelSpec("q_b_proj", "decode", "best/q_b_decode", "fp8_gemm"),
    "o_proj": KernelSpec("o_proj", "decode", "best/o_proj_decode_hbm35", "fp8_gemm"),
    "index_k_proj": KernelSpec(
        "index_k_proj", "decode", "best/index_k_proj_decode", "fp8_gemm"
    ),
    "index_q_upproj": KernelSpec(
        "index_q_upproj",
        "decode",
        "best-hechenxi-0720/index_q_upproj_decode",
        "fp8_gemm",
    ),
    "moe_gate_proj": KernelSpec(
        "moe_gate_proj", "decode", "best/moe_gate_proj_decode_hbm40", "moe_masked"
    ),
    "moe_up_proj": KernelSpec(
        "moe_up_proj", "decode", "best/moe_up_proj_decode_hbm40", "moe_masked"
    ),
    "moe_down_proj": KernelSpec(
        "moe_down_proj", "decode", "best/moe_down_proj_decode_hbm40", "moe_masked"
    ),
    "dsa_decode_attn": KernelSpec(
        "dsa_decode_attn",
        "decode",
        "best-hechenxi-0720/dsa_decode_attn",
        "dsa",
    ),
    # Fusion path (wk_weights_proj) already matches hechenxi intent; kept for docs.
    "index_weights_proj": KernelSpec(
        "index_weights_proj",
        "decode",
        "best-hechenxi-0720/index_weights_proj",
        "bf16_gemm",
    ),
}

# Prefill winners from PREFILL_SWAPS (including decode kernels that also win on prefill).
_PREFILL_FULL: dict[str, KernelSpec] = {
    "fused_qkv_a_proj": KernelSpec(
        "fused_qkv_a_proj", "prefill", "fused_qkv_a_prefill.py", "fp8_gemm"
    ),
    "q_b_proj": KernelSpec("q_b_proj", "prefill", "q_b_prefill.py", "fp8_gemm"),
    "o_proj": KernelSpec(
        "o_proj", "prefill", "best/o_proj_decode_hbm35", "fp8_gemm"
    ),
    "index_q_upproj": KernelSpec(
        "index_q_upproj", "prefill", "index_q_upproj_prefill.py", "fp8_gemm"
    ),
    "index_k_proj": KernelSpec(
        "index_k_proj", "prefill", "best/index_k_proj_decode", "fp8_gemm"
    ),
    "index_weights_proj": KernelSpec(
        "index_weights_proj", "prefill", "index_weights_proj.py", "bf16_gemm"
    ),
    # moe_gate prefill: CUPTI win but CUDA Graph M=4096 regresses — keep stock.
    "moe_up_proj": KernelSpec(
        "moe_up_proj", "prefill", "best/moe_up_proj_decode_hbm40", "moe_masked"
    ),
    "moe_down_proj": KernelSpec(
        "moe_down_proj", "prefill", "best/moe_down_proj_decode_hbm40", "moe_masked"
    ),
}


def _active_prefill() -> dict[str, KernelSpec]:
    return dict(_PREFILL_FULL)


def lookup(op_name: Optional[str], phase: str) -> Optional[KernelSpec]:
    if not op_name:
        return None
    if phase == "decode":
        spec = _DECODE.get(op_name)
    elif profile_name() == "full":
        spec = _active_prefill().get(op_name)
    else:
        return None
    if spec is None or not spec.enabled:
        return None
    return spec


def list_enabled(phase: str) -> list[KernelSpec]:
    if phase == "decode":
        return [s for s in _DECODE.values() if s.enabled]
    if profile_name() == "full":
        return [s for s in _active_prefill().values() if s.enabled]
    return []
