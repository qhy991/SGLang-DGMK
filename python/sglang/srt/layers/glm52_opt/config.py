"""Environment-driven GLM-5.2 optimization switches."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[5]
_DEFAULT_MANIFEST = _REPO_ROOT / "glm52_opt" / "manifest.json"


def _truthy(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def _resolve_repo_path(value: str | Path) -> Path:
    """Resolve manifest paths: absolute stay as-is; relative resolve against repo root."""
    path = Path(value)
    if path.is_absolute():
        return path
    return (_REPO_ROOT / path).resolve()


def is_enabled() -> bool:
    return _truthy("SGLANG_GLM52_OPT")


def profile_name() -> str:
    return os.environ.get("SGLANG_GLM52_OPT_PROFILE", "decode_max").strip().lower()


def deepgemm_variant() -> str | None:
    value = os.environ.get("SGLANG_GLM52_DEEPGEMM_VARIANT", "").strip()
    return value or None


@lru_cache(maxsize=1)
def load_manifest() -> dict[str, Any]:
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
