"""Environment-driven GLM-5.2 optimization switches."""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[5]
_DEFAULT_MANIFEST = _REPO_ROOT / "glm52_opt" / "manifest.json"
_DEFAULT_ENV_FILE = Path("/home/ubuntu/wwxq/cache/sglang/glm52_opt.env")

_GLM52_ENV_KEYS = frozenset(
    {
        "SGLANG_GLM52_OPT",
        "SGLANG_GLM52_OPT_PROFILE",
        "SGLANG_GLM52_OPT_OPS",
        "SGLANG_GLM52_OPT_M_BUCKETS",
        "SGLANG_GLM52_ALLOW_ABI_ADAPTER",
        "SGLANG_GLM52_MANIFEST",
        "SGLANG_GLM52_DEEPGEMM_VARIANT",
        "SGLANG_GLM52_ARCHIVE",
        "SGLANG_GLM52_DEEPGEMM_OVERLAY",
        "SGLANG_GLM52_DEEPGEMM_MANIFEST",
        "SGLANG_GLM52_ENV_FILE",
        "SGLANG_GLM52_NSYS_GATE",
        "SGLANG_GLM52_NSYS_TRIGGER",
        "SGLANG_GLM52_NSYS_SECONDS",
        "SGLANG_GLM52_INFINI_KERNEL_NVTX",
        "SGLANG_GLM52_HOTSPOT_MODULE",
        "SGLANG_OPT_MOE_SWIGLU_QUANT_VARIANT",
        "SGLANG_INFINI_V_APPLY_QUANT",
        # 1 (default): pick the MoE masked-grouped M tile from expected_m.
        # 0: always use DeepGEMM stock 128. See infini_moe_align.py.
        "SGLANG_GLM52_INFINI_MOE_ALIGN",
        # 1 (default for graph_only specs): only select under graph capture.
        # 0: allow eager selection for diagnostic leaf timing.
        "SGLANG_GLM52_W13_GRAPH_ONLY",
        "SGLANG_GLM52_W2_GRAPH_ONLY",
        "SGLANG_GLM52_O_PROJ_GRAPH_ONLY",
        "SGLANG_GLM52_FUSED_QKV_A_GRAPH_ONLY",
        "SGLANG_GLM52_INDEX_Q_UPPROJ_GRAPH_ONLY",
        "SGLANG_GLM52_FLASHMLA_GRAPH_ONLY",
        "SGLANG_GLM52_DSA_PREFILL_GRAPH_ONLY",
    }
)

_env_applied = False
logger = logging.getLogger(__name__)


def _truthy(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def _resolve_repo_path(value: str | Path) -> Path:
    """Resolve manifest paths: absolute stay as-is; relative resolve against repo root."""
    path = Path(value)
    if path.is_absolute():
        return path
    return (_REPO_ROOT / path).resolve()


def ensure_glm52_env() -> None:
    """Load GLM52 opt env from a side-channel file into os.environ.

    Multiprocessing spawn / DP scheduler workers sometimes drop unregistered
    ``SGLANG_GLM52_*`` exports from the parent. The launch script writes
    ``SGLANG_GLM52_ENV_FILE`` (or the default path) so workers can re-apply.
    """
    global _env_applied
    if _env_applied:
        return
    _env_applied = True

    path_str = os.environ.get("SGLANG_GLM52_ENV_FILE", "").strip()
    candidates: list[Path] = []
    if path_str:
        candidates.append(Path(path_str))
    candidates.extend(
        [
            _DEFAULT_ENV_FILE,
            _REPO_ROOT / "glm52_opt" / "runtime.env",
        ]
    )

    env_path: Path | None = None
    for cand in candidates:
        if cand.is_file():
            env_path = cand
            break
    if env_path is None:
        try:
            from sglang.srt.layers.glm52_opt.nsys_gate import maybe_start_nsys_gate

            maybe_start_nsys_gate()
        except Exception:
            pass
        return

    applied: list[str] = []
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key not in _GLM52_ENV_KEYS:
            continue
        os.environ[key] = value
        applied.append(key)

    if applied:
        # Manifest may have been cached before the file was applied.
        load_manifest.cache_clear()
        logger.info(
            "glm52_opt: applied %s from %s (OPT=%s profile=%s variant=%s)",
            applied,
            env_path,
            os.environ.get("SGLANG_GLM52_OPT"),
            os.environ.get("SGLANG_GLM52_OPT_PROFILE"),
            os.environ.get("SGLANG_GLM52_DEEPGEMM_VARIANT"),
        )

    # Arm nsys capture-range watcher even when OPT=0 (baseline traces).
    try:
        from sglang.srt.layers.glm52_opt.nsys_gate import maybe_start_nsys_gate

        maybe_start_nsys_gate()
    except Exception:
        pass


def is_enabled() -> bool:
    ensure_glm52_env()
    return _truthy("SGLANG_GLM52_OPT")


def profile_name() -> str:
    ensure_glm52_env()
    # Fail closed: historical harness winners are not serving-safe merely
    # because GLM52_OPT is enabled.  They must be selected explicitly until a
    # production-ABI + end-to-end validation promotes them.
    return os.environ.get("SGLANG_GLM52_OPT_PROFILE", "serving_safe").strip().lower()


# Leaf/component winners archived for explicit e2e trials.  Never implied by
# serving_safe.  See glm52_opt/history/e2e_candidates_20260723/INDEX.md.
_E2E_DEFAULT_OPS = frozenset({"o_proj", "moe_gate_proj", "moe_down_proj"})
_E2E_CONTIG_PSUM_OPS = frozenset({"moe_gate_proj", "moe_down_proj"})
_E2E_EXPLICIT_OPS = frozenset({"fused_qkv_a_proj", "index_q_upproj"})
_E2E_ALLOWED_OPS = _E2E_DEFAULT_OPS | _E2E_EXPLICIT_OPS

# Exact production-interface hooks for out-of-tree PTX/SASS, CUDA/CuTe, or
# Triton experiments.  These are intentionally isolated from e2e_candidates:
# selecting the hotspot profile must never also turn on an older archive swap.
_HOTSPOT_DEFAULT_OPS = frozenset({"dsa_decode_attn", "moe_gate_proj", "moe_down_proj"})
# dsa_prefill_attn is selectable but deliberately NOT in the default set: adding
# a prefill op to the defaults would silently change every existing decode
# hotspot campaign that selects the profile without SGLANG_GLM52_OPT_OPS.  It
# must be named explicitly.
_HOTSPOT_ALLOWED_OPS = _HOTSPOT_DEFAULT_OPS | frozenset({"dsa_prefill_attn"})
_HOTSPOT_OP_ALIASES = {
    "flashmla_sparse_decode": "dsa_decode_attn",
    "flashmla_kv": "dsa_decode_attn",
    "flashmla_kv_prefill": "dsa_prefill_attn",
    "flashmla_sparse_prefill": "dsa_prefill_attn",
    "moe_w13": "moe_gate_proj",
    "moe_gate_up": "moe_gate_proj",
    "moe_w2": "moe_down_proj",
}

# Joint serving swap: FlashMLA r2a+c2 + prefill b3_b5 + fixed-N/K GEMMs + MoE align.
# MoE uses stock ``moe_masked`` (not hotspot_plugin) so ``infini_mk_alignment`` applies.
# MoE BM16 hotspot providers stay off (region historically < 1.03).
_COMBINED_DEFAULT_OPS = frozenset(
    {
        "dsa_decode_attn",
        "dsa_prefill_attn",
        "o_proj",
        "fused_qkv_a_proj",
        "index_q_upproj",
        "moe_gate_proj",
        "moe_up_proj",
        "moe_down_proj",
    }
)
_COMBINED_OP_ALIASES = {
    **_HOTSPOT_OP_ALIASES,
}
_COMBINED_HOTSPOT_OPS = frozenset({"dsa_decode_attn", "dsa_prefill_attn"})
_COMBINED_ALLOWED_OPS = _COMBINED_DEFAULT_OPS | frozenset({"moe_swiglu_quant"})


def opt_ops_allowlist() -> frozenset[str] | None:
    """Optional comma-separated decode op allowlist (SGLANG_GLM52_OPT_OPS).

    When set, only listed ops are eligible for glm52_opt dispatch.
    Example: ``q_b_proj,fused_qkv_a_proj``.
    """
    ensure_glm52_env()
    raw = os.environ.get("SGLANG_GLM52_OPT_OPS", "").strip()
    if not raw:
        return None
    return frozenset(x.strip() for x in raw.split(",") if x.strip())


def e2e_candidate_ops() -> frozenset[str]:
    """Ops active under ``SGLANG_GLM52_OPT_PROFILE=e2e_candidates``.

    Empty ``OPT_OPS`` selects the archived default set.  A non-empty allowlist
    intersects the audited E2E set so ablation cannot accidentally enable
    unrelated historical archive swaps. Fixed-N/K QKV-A and indexer candidates
    are explicit-only and therefore never join the legacy default set.
    """
    allow = opt_ops_allowlist()
    if allow is None:
        return _E2E_DEFAULT_OPS
    return frozenset(op for op in allow if op in _E2E_ALLOWED_OPS)


def hotspot_candidate_ops() -> frozenset[str]:
    """Ops selected by ``SGLANG_GLM52_OPT_PROFILE=hotspot_candidates``.

    User-facing aliases describe the fused production operation (``moe_w13``)
    while the returned names remain compatible with the existing SGLang op
    contexts (``moe_gate_proj``).
    """
    allow = opt_ops_allowlist()
    if allow is None:
        return _HOTSPOT_DEFAULT_OPS
    normalized = {_HOTSPOT_OP_ALIASES.get(op, op) for op in allow}
    unknown = normalized - _HOTSPOT_ALLOWED_OPS
    if unknown:
        raise ValueError(
            "Unsupported SGLANG_GLM52_OPT_OPS for hotspot_candidates: "
            + ", ".join(sorted(unknown))
        )
    return frozenset(normalized)


def hotspot_module_ref() -> str:
    """Python module name or absolute provider ``.py`` path for hotspot ops."""
    ensure_glm52_env()
    return os.environ.get("SGLANG_GLM52_HOTSPOT_MODULE", "").strip()


def combined_winner_ops() -> frozenset[str]:
    """Ops active under ``SGLANG_GLM52_OPT_PROFILE=combined_winners``."""
    allow = opt_ops_allowlist()
    if allow is None:
        return _COMBINED_DEFAULT_OPS
    normalized = {_COMBINED_OP_ALIASES.get(op, op) for op in allow}
    unknown = normalized - _COMBINED_ALLOWED_OPS
    if unknown:
        raise ValueError(
            "Unsupported SGLANG_GLM52_OPT_OPS for combined_winners: "
            + ", ".join(sorted(unknown))
        )
    return frozenset(normalized)


def needs_hotspot_provider() -> bool:
    """Whether this profile must load ``SGLANG_GLM52_HOTSPOT_MODULE``."""
    if not is_enabled():
        return False
    name = profile_name()
    if name == "hotspot_candidates":
        return True
    if name == "combined_winners":
        return bool(combined_winner_ops() & _COMBINED_HOTSPOT_OPS)
    return False


def hotspot_provider_ops() -> frozenset[str]:
    """Ops that the hotspot provider must register for the active profile.

    ``combined_winners`` only wires FlashMLA decode/prefill; MoE stays on stock
    ``moe_masked`` so ``infini_mk_alignment`` can apply.
    """
    name = profile_name()
    if name == "hotspot_candidates":
        return hotspot_candidate_ops()
    if name == "combined_winners":
        return combined_winner_ops() & _COMBINED_HOTSPOT_OPS
    return frozenset()


@lru_cache(maxsize=1)
def emit_infini_kernel_nvtx() -> bool:
    """Whether selected kernels get profiler-only ``infini_kernel`` ranges."""
    ensure_glm52_env()
    return _truthy("SGLANG_GLM52_INFINI_KERNEL_NVTX")


# Per-op override for ``KernelSpec.graph_only``.  Each entry names the env var
# that can disable graph-only selection for diagnostic eager timing.
_GRAPH_ONLY_ENV_BY_OP = {
    "moe_gate_proj": "SGLANG_GLM52_W13_GRAPH_ONLY",
    "moe_up_proj": "SGLANG_GLM52_W13_GRAPH_ONLY",
    "moe_down_proj": "SGLANG_GLM52_W2_GRAPH_ONLY",
    "o_proj": "SGLANG_GLM52_O_PROJ_GRAPH_ONLY",
    "fused_qkv_a_proj": "SGLANG_GLM52_FUSED_QKV_A_GRAPH_ONLY",
    "index_q_upproj": "SGLANG_GLM52_INDEX_Q_UPPROJ_GRAPH_ONLY",
    "dsa_decode_attn": "SGLANG_GLM52_FLASHMLA_GRAPH_ONLY",
    "dsa_prefill_attn": "SGLANG_GLM52_DSA_PREFILL_GRAPH_ONLY",
}


def graph_only_enabled(op_name: str) -> bool:
    """Whether a ``graph_only`` spec keeps its capture-only restriction.

    MoE W13/W2 decode and several fixed-N/K replacements are
    production-graph-bound. A ``graph_only`` spec defaults to selecting under
    CUDA-graph capture only. Set the op's env to ``0`` to force eager selection
    for a diagnostic leaf measurement.
    """
    ensure_glm52_env()
    key = _GRAPH_ONLY_ENV_BY_OP.get(op_name)
    if key is None:
        return True
    raw = os.environ.get(key, "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def contig_psum_kwargs(op_name: str) -> dict[str, object]:
    """Kwargs for DeepGEMM contiguous grouped GEMM PSUM layout (goals 08/09).

    Returns ``{}`` unless OPT is on, profile is ``e2e_candidates``, and ``op_name``
    is an enabled MoE contig PSUM candidate.  Callers must also supply the
    ``expert_start_loc`` endpoint tensor from ``ep_scatter`` as the layout.
    """
    if not is_enabled() or profile_name() != "e2e_candidates":
        return {}
    if op_name not in _E2E_CONTIG_PSUM_OPS or op_name not in e2e_candidate_ops():
        return {}
    return {
        "compiled_dims": "nk",
        "use_psum_layout": True,
        "ensure_zero_padding": False,
        "expected_m_for_psum_layout": 1024,
    }


def opt_m_buckets() -> dict[str, frozenset[int]]:
    """Optional per-op M allowlist for shape-selective dispatch.

    ``SGLANG_GLM52_OPT_M_BUCKETS=q_b_proj:16|32,moe_down_proj:32``
    restricts only the named ops.  An op absent from this map keeps its normal
    profile/OPT_OPS behavior.  If an op is present, dispatch fails closed when
    the current forward M is unknown or not listed, so the caller naturally
    falls back to the production SGLang implementation.
    """
    ensure_glm52_env()
    raw = os.environ.get("SGLANG_GLM52_OPT_M_BUCKETS", "").strip()
    if not raw:
        return {}

    parsed: dict[str, set[int]] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        op, sep, values = entry.partition(":")
        op = op.strip()
        if not sep or not op or not values.strip():
            raise ValueError(
                "Invalid SGLANG_GLM52_OPT_M_BUCKETS entry "
                f"{entry!r}; expected op:16|32"
            )
        try:
            buckets = {int(value.strip()) for value in values.split("|")}
        except ValueError as exc:
            raise ValueError(
                "Invalid SGLANG_GLM52_OPT_M_BUCKETS entry "
                f"{entry!r}; M values must be integers"
            ) from exc
        if not buckets or any(m <= 0 for m in buckets):
            raise ValueError(
                "Invalid SGLANG_GLM52_OPT_M_BUCKETS entry "
                f"{entry!r}; M values must be positive"
            )
        parsed.setdefault(op, set()).update(buckets)
    return {op: frozenset(values) for op, values in parsed.items()}


def allow_abi_adapter() -> bool:
    """Allow packed-UE8M0 -> f32 adapters for legacy harness candidates.

    This is intentionally opt-in: the conversion launches and temporary
    tensors are part of serving latency and caused several microbench winners
    to regress after integration.
    """
    ensure_glm52_env()
    return _truthy("SGLANG_GLM52_ALLOW_ABI_ADAPTER")


@lru_cache(maxsize=1)
def swiglu_quant_variant() -> str | None:
    """Return the explicitly armed B300 masked-MoE activation variant.

    The candidate is absent from every default op set. It is reachable only
    when an allowed campaign profile, the exact op allowlist, and the bounded
    implementation are all selected before worker startup.
    """

    from sglang.srt.environ import envs

    ensure_glm52_env()
    if not is_enabled() or profile_name() not in {
        "b300_moe_swiglu_quant",
        "combined_winners",
    }:
        return None
    allow = opt_ops_allowlist()
    if allow is None or "moe_swiglu_quant" not in allow:
        return None
    variant = envs.SGLANG_OPT_MOE_SWIGLU_QUANT_VARIANT.get()
    if variant is None:
        return None
    variant = variant.strip()
    if not variant:
        return None
    if variant != "cuda_grid_stride":
        raise ValueError(
            "SGLANG_OPT_MOE_SWIGLU_QUANT_VARIANT must be "
            f"'cuda_grid_stride', got {variant!r}"
        )
    return variant


def deepgemm_variant() -> str | None:
    ensure_glm52_env()
    value = os.environ.get("SGLANG_GLM52_DEEPGEMM_VARIANT", "").strip()
    if value:
        return value
    manifest = load_manifest()
    commit = str(manifest.get("deepgemm_commit") or "").strip()
    if commit:
        return commit[:7]
    variant = str(manifest.get("deepgemm_variant") or "").strip()
    return variant or None


@lru_cache(maxsize=1)
def load_manifest() -> dict[str, Any]:
    ensure_glm52_env()
    path = Path(os.environ.get("SGLANG_GLM52_MANIFEST", str(_DEFAULT_MANIFEST)))
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def archive_root() -> Path:
    manifest = load_manifest()
    env = os.environ.get("SGLANG_GLM52_ARCHIVE")
    if env:
        return _resolve_repo_path(env)
    return _resolve_repo_path(
        manifest.get(
            "kernel_harness_archive",
            "third_party/kernel-archive/0720-Best-GLM-52",
        )
    )


def deepgemm_overlay_path() -> Path:
    manifest = load_manifest()
    env = os.environ.get("SGLANG_GLM52_DEEPGEMM_OVERLAY")
    if env:
        return _resolve_repo_path(env)
    return _resolve_repo_path(
        manifest.get(
            "deepgemm_overlay_path",
            "third_party/deepgemm_glm52",
        )
    )
