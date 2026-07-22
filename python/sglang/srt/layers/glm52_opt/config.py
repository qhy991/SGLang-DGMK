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
        "SGLANG_GLM52_MANIFEST",
        "SGLANG_GLM52_DEEPGEMM_VARIANT",
        "SGLANG_GLM52_ARCHIVE",
        "SGLANG_GLM52_DEEPGEMM_OVERLAY",
        "SGLANG_GLM52_DEEPGEMM_MANIFEST",
        "SGLANG_GLM52_ENV_FILE",
        "SGLANG_GLM52_NSYS_GATE",
        "SGLANG_GLM52_NSYS_TRIGGER",
        "SGLANG_GLM52_NSYS_SECONDS",
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
    return os.environ.get("SGLANG_GLM52_OPT_PROFILE", "decode_max").strip().lower()


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
