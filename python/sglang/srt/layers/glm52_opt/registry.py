"""Registry of GLM-5.2 optimized kernels keyed by (op, phase).

Aligned with Kernel-Harness ``llm_flops_style/_common.py`` DECODE/PREFILL_SWAPS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

from sglang.srt.layers.glm52_opt.config import (
    e2e_candidate_ops,
    hotspot_candidate_ops,
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
KernelImplementation = Literal["auto", "fixed_nk", "hotspot_plugin"]

RunFn = Callable[[dict], object]


@dataclass(frozen=True)
class KernelSpec:
    op: str
    phase: str
    archive_ref: str
    kind: KernelKind
    enabled: bool = True
    implementation: KernelImplementation = "auto"
    profiler_name: Optional[str] = None
    m_values: tuple[int, ...] | None = None
    n: int | None = None
    k: int | None = None
    num_groups: int | None = None
    slab_m: int | None = None
    expected_m_values: tuple[int, ...] | None = None
    topk: int | None = None
    q_heads: int | None = None
    qk_dim: int | None = None
    v_dim: int | None = None
    page_size: int | None = None
    kv_dim: int | None = None
    # When True, try_dispatch may select the candidate only while the current
    # CUDA stream is capturing a graph. Eager calls fall back to stock before
    # any provider launch. Used by MoE W2 decode: production is graph-bound
    # and the Python API-v1 provider tax makes eager containing unreachable.
    graph_only: bool = False


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

_E2E_DECODE: dict[str, KernelSpec] = {
    "o_proj": KernelSpec(
        op="o_proj",
        phase="decode",
        archive_ref="",
        kind="fp8_gemm",
        implementation="fixed_nk",
        profiler_name="infini_kernel_glm52_attn_o_decode_nk",
        m_values=(16, 32),
        n=6144,
        k=16384,
        # Decode o_proj is production graph-bound and the eager glm52_opt
        # dispatch tax (Python lookup/alloc/hit-accounting) vetoes the fixed-N/K
        # device win in every eager paired session (goal-10). Restrict selection
        # to CUDA-graph capture so eager decode stays on stock with no provider
        # launch; SGLANG_GLM52_O_PROJ_GRAPH_ONLY=0 forces eager for a diagnostic
        # leaf. Mirrors moe_down_proj / FlashMLA dsa_decode_attn.
        graph_only=True,
    ),
    "index_q_upproj": KernelSpec(
        op="index_q_upproj",
        phase="decode",
        archive_ref="",
        kind="fp8_gemm",
        implementation="fixed_nk",
        profiler_name="infini_kernel_glm52_index_q_upproj_decode_nk",
        m_values=(16, 32),
        n=4096,
        k=2048,
    ),
    "fused_qkv_a_proj": KernelSpec(
        op="fused_qkv_a_proj",
        phase="decode",
        archive_ref="",
        kind="fp8_gemm",
        implementation="fixed_nk",
        profiler_name="infini_kernel_glm52_fused_qkv_a_decode_nk",
        m_values=(16, 32),
        n=2624,
        k=6144,
        # Decode fused_qkv_a_proj (prepare_qkv_latent down-projection) is
        # production graph-bound; the eager glm52_opt dispatch tax vetoes the
        # fixed-N/K device win in eager paired sessions exactly like o_proj.
        # Restrict selection to CUDA-graph capture so eager decode stays on stock
        # with no provider launch; SGLANG_GLM52_FUSED_QKV_A_GRAPH_ONLY=0 forces
        # eager for a diagnostic leaf. Mirrors o_proj / moe_down_proj.
        graph_only=True,
    ),
}

_E2E_PREFILL: dict[str, KernelSpec] = {
    "fused_qkv_a_proj": KernelSpec(
        op="fused_qkv_a_proj",
        phase="prefill",
        archive_ref="",
        kind="fp8_gemm",
        implementation="fixed_nk",
        profiler_name="infini_kernel_glm52_fused_qkv_a_prefill_nk",
        m_values=(4096,),
        n=2624,
        k=6144,
    ),
}

# Three default-off production-interface hooks selected from the GLM-5.2
# decode Nsight profile.  The provider is supplied out of tree so a PTX/SASS,
# CUDA/CuTe, CUTLASS, or Triton implementation can be A/B tested without
# changing SGLang call sites.  Every unsupported bucket falls back to stock.
_HOTSPOT_DECODE: dict[str, KernelSpec] = {
    "dsa_decode_attn": KernelSpec(
        op="dsa_decode_attn",
        phase="decode",
        archive_ref="",
        kind="dsa",
        implementation="hotspot_plugin",
        profiler_name="infini_kernel_glm52_flashmla_sparse_decode_fp8_topk2048",
        m_values=(16, 32),
        topk=2048,
        q_heads=64,
        qk_dim=576,
        v_dim=512,
        page_size=64,
        kv_dim=656,
    ),
    # SGLang executes gate+up as one fused W13 grouped GEMM.
    "moe_gate_proj": KernelSpec(
        op="moe_gate_proj",
        phase="decode",
        archive_ref="",
        kind="moe_masked",
        implementation="hotspot_plugin",
        profiler_name="infini_kernel_glm52_moe_w13_decode",
        m_values=(16, 32),
        n=4096,
        k=6144,
        num_groups=32,
        slab_m=1024,
        expected_m_values=(4, 5, 8, 9),
    ),
    "moe_down_proj": KernelSpec(
        op="moe_down_proj",
        phase="decode",
        archive_ref="",
        kind="moe_masked",
        implementation="hotspot_plugin",
        profiler_name="infini_kernel_glm52_moe_w2_decode",
        m_values=(16, 32),
        n=6144,
        k=2048,
        num_groups=32,
        slab_m=1024,
        expected_m_values=(4, 5, 8, 9),
        graph_only=True,
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
        # Using a separate table also prevents those names from accidentally
        # enabling the archived decode MoE kernels.
        table = {
            op: _E2E_DECODE[op]
            for op in sorted(e2e_candidate_ops())
            if op in _E2E_DECODE
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


def _e2e_prefill_table() -> dict[str, KernelSpec]:
    return {
        op: _E2E_PREFILL[op]
        for op in sorted(e2e_candidate_ops())
        if op in _E2E_PREFILL
    }


def _hotspot_decode_table() -> dict[str, KernelSpec]:
    return {
        op: _HOTSPOT_DECODE[op]
        for op in sorted(hotspot_candidate_ops())
        if op in _HOTSPOT_DECODE
    }


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
        spec = (
            _hotspot_decode_table().get(op_name)
            if name == "hotspot_candidates"
            else _decode_table().get(op_name)
        )
    elif name == "full":
        spec = _active_prefill().get(op_name)
    elif name == "e2e_candidates":
        spec = _e2e_prefill_table().get(op_name)
    else:
        # e2e_candidates MoE PSUM does not use KernelSpec lookup.
        return None
    if spec is None or not spec.enabled:
        return None
    if spec.m_values is not None and (m is None or int(m) not in spec.m_values):
        return None
    allowed_m = opt_m_buckets().get(op_name)
    if allowed_m is not None and (m is None or int(m) not in allowed_m):
        return None
    return spec


def list_enabled(phase: str) -> list[KernelSpec]:
    if phase == "decode":
        if profile_name() == "hotspot_candidates":
            return [s for s in _hotspot_decode_table().values() if s.enabled]
        return [s for s in _decode_table().values() if s.enabled]
    if profile_name() == "full":
        return [s for s in _active_prefill().values() if s.enabled]
    if profile_name() == "e2e_candidates":
        return [s for s in _e2e_prefill_table().values() if s.enabled]
    return []
