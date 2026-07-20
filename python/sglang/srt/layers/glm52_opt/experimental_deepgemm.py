"""Load DeepGEMM-GLM52 experimental overlay alongside stock deep_gemm."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from sglang.srt.layers.glm52_opt.config import deepgemm_overlay_path, deepgemm_variant


def _manifest_path() -> Path:
    overlay = deepgemm_overlay_path()
    env = os.environ.get("SGLANG_GLM52_DEEPGEMM_MANIFEST")
    if env:
        return Path(env)
    return overlay / "manifest.json"


def _read_manifest() -> dict[str, Any]:
    path = _manifest_path()
    if not path.is_file():
        raise FileNotFoundError(f"DeepGEMM overlay manifest missing: {path}")
    return json.loads(path.read_text())


def ensure_stock_deep_gemm():
    if "deep_gemm" in sys.modules:
        mod = sys.modules["deep_gemm"]
    else:
        mod = importlib.import_module("deep_gemm")
    path = Path(getattr(mod, "__file__", "") or "").resolve()
    if "DeepGEMM-GLM52" in str(path) or "deep_gemm_experimental" in str(path):
        raise RuntimeError(f"stock deep_gemm resolved to fork path: {path}")
    return mod


@lru_cache(maxsize=1)
def get_experimental_deep_gemm():
    if not deepgemm_variant():
        return None
    ensure_stock_deep_gemm()
    manifest = _read_manifest()
    pkg_dir = Path(manifest["package_dir"]).resolve()
    jit_cache = Path(manifest["jit_cache_dir"]).resolve()
    init_py = pkg_dir / "__init__.py"
    if not init_py.is_file():
        raise FileNotFoundError(f"overlay package missing: {init_py}")
    jit_cache.mkdir(parents=True, exist_ok=True)
    os.environ["DG_JIT_CACHE_DIR"] = str(jit_cache)

    if (
        "deep_gemm_experimental" in sys.modules
        and getattr(sys.modules["deep_gemm_experimental"], "_C", None) is not None
    ):
        return sys.modules["deep_gemm_experimental"]

    spec = importlib.util.spec_from_file_location(
        "deep_gemm_experimental",
        init_py,
        submodule_search_locations=[str(pkg_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load experimental deep_gemm from {init_py}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["deep_gemm_experimental"] = module
    module.__path__ = [str(pkg_dir)]  # type: ignore[attr-defined]
    module.__package__ = "deep_gemm_experimental"
    spec.loader.exec_module(module)
    return module


def has_fused_fp8_gemm_nt() -> bool:
    mod = get_experimental_deep_gemm()
    return mod is not None and hasattr(mod, "fp8_gemm_nt_fused")
