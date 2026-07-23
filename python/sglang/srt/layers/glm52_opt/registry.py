"""Registry of GLM-5.2 optimized kernels keyed by (op, phase).

Aligned with Kernel-Harness ``llm_flops_style/_common.py`` DECODE/PREFILL_SWAPS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

from sglang.srt.layers.glm52_opt.config import (
    e2e_candidate_ops,
    opt_m_buckets,
    opt_ops_allowlist,
    profile_name,
)

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
# Intentionally omit:
#   - moe_gate: CUDA Graph regresses at large M
#   - moe_up/down decode_hbm40 drop-ins: llm_flops_style B300 M=4096 showed ~0.87–1.0×
#   - dsa / index_score / absorbed_W: no safe flashmla_kv hook or ceiling-bound
_PREFILL_FULL: dict[str, KernelSpec] = {
    "fused_qkv_a_proj": KernelSpec(
        "fused_qkv_a_proj", "prefill", "fused_qkv_a_prefill.py", "fp8_gemm"
    ),
    "q_b_proj": KernelSpec("q_b_proj", "prefill", "q_b_prefill.py", "fp8_gemm"),
    # Native packed UE8M0 path in fp8_gemm.py (archive_ref unused for o_proj).
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
}


def _decode_table() -> dict[str, KernelSpec]:
    """Profile / allowlist gated decode registry.

    - serving_safe (default): no implicit swap; OPT_OPS selects explicit trials
    - e2e_candidates: archived leaf winners for explicit e2e (default o_proj)
    - decode_max / full: all legacy decode swaps (optionally filtered by OPT_OPS)
    - q_b_only: only q_b_proj
    - SGLANG_GLM52_OPT_OPS=a,b: intersect with active table (ablation)
    """
    name = profile_name()
    allow = opt_ops_allowlist()
    if name == "q_b_only":
        table = {}
        spec = _DECODE.get("q_b_proj")
        if spec is not None:
            table["q_b_proj"] = spec
    elif name == "serving_safe":
        table = (
            {op: _DECODE[op] for op in sorted(allow) if op in _DECODE}
            if allow is not None
            else {}
        )
    elif name == "e2e_candidates":
        # Prefill MoE PSUM is wired in the contig runner, not via this table.
        table = {
            op: _DECODE[op]
            for op in sorted(e2e_candidate_ops())
            if op in _DECODE
        }
    elif name in ("decode_max", "full"):
        table = dict(_DECODE)
    else:
        table = {}

    if allow is not None and name not in ("serving_safe", "e2e_candidates"):
        table = {k: v for k, v in table.items() if k in allow}
    return table


def _active_prefill() -> dict[str, KernelSpec]:
    return dict(_PREFILL_FULL)


def lookup(
    op_name: Optional[str], phase: str, m: Optional[int] = None
) -> Optional[KernelSpec]:
    """Look up a replacement, optionally gated by the current local M.

    M gating is deliberately evaluated after the profile/op allowlist.  It is
    a selective fallback policy, not another way to enable an op.
    """
    if not op_name:
        return None
    name = profile_name()
    if phase == "decode":
        spec = _decode_table().get(op_name)
    elif name == "full":
        spec = _active_prefill().get(op_name)
    else:
        # e2e_candidates MoE PSUM does not use KernelSpec lookup.
        return None
    if spec is None or not spec.enabled:
        return None
    allowed_m = opt_m_buckets().get(op_name)
    if allowed_m is not None and (m is None or int(m) not in allowed_m):
        return None
    return spec


def list_enabled(phase: str) -> list[KernelSpec]:
    if phase == "decode":
        return [s for s in _decode_table().values() if s.enabled]
    if profile_name() == "full":
        return [s for s in _active_prefill().values() if s.enabled]
    return []
