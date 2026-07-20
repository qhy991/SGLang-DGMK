"""Load stock DeepGEMM and the GLM-5.2 experimental fork side-by-side.

Contract:
  - ``import deep_gemm`` keeps pointing at Kernel-Harness stock
    ``sgl-deep-gemm==0.1.4`` in the harness venv site-packages.
  - The fork is exposed only as ``deep_gemm_experimental``.
  - Fork JIT uses a commit-partitioned ``DG_JIT_CACHE_DIR`` so stock/fork
    caches never collide.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_MANIFEST = _SCRIPT_DIR / "manifest.json"
_STOCK_HINT = Path(
    "/home/qinhaiyan/Kernel-Harness/.venv/lib/python3.12/site-packages/deep_gemm"
)


def _read_manifest(manifest_path: Path | None = None) -> dict:
    path = Path(manifest_path) if manifest_path else _DEFAULT_MANIFEST
    if not path.is_file():
        raise FileNotFoundError(
            f"DeepGEMM-GLM52 manifest missing: {path}. "
            "Run third_party/deepgemm_glm52/build_overlay.sh first."
        )
    return json.loads(path.read_text())


def ensure_stock_deep_gemm():
    """Import stock deep_gemm and assert it is not the experimental overlay."""
    # Prefer the already-imported module (Harness loads glm52_ops before candidates).
    if "deep_gemm" in sys.modules:
        mod = sys.modules["deep_gemm"]
    else:
        mod = importlib.import_module("deep_gemm")
    path = Path(getattr(mod, "__file__", "") or "").resolve()
    if "DeepGEMM-GLM52" in str(path) or "deep_gemm_experimental" in str(path):
        raise RuntimeError(
            f"stock deep_gemm resolved to fork path unexpectedly: {path}"
        )
    if _STOCK_HINT.exists() and "site-packages" not in str(path):
        # Soft warning path — still allow editable stock installs, but surface it.
        pass
    return mod


def load_deep_gemm_experimental(
    manifest_path: Path | str | None = None,
    *,
    force_reload: bool = False,
):
    """Load/return the fork as module ``deep_gemm_experimental``.

    Sets ``DG_JIT_CACHE_DIR`` to the overlay's commit-local cache **before**
    the fork's ``_C`` extension initializes its Compiler singleton.
    Stock must already be importable; this function never replaces it.
    """
    ensure_stock_deep_gemm()

    if (
        not force_reload
        and "deep_gemm_experimental" in sys.modules
        and getattr(sys.modules["deep_gemm_experimental"], "_C", None) is not None
    ):
        return sys.modules["deep_gemm_experimental"]

    manifest = _read_manifest(Path(manifest_path) if manifest_path else None)
    pkg_dir = Path(manifest["package_dir"]).resolve()
    jit_cache = Path(manifest["jit_cache_dir"]).resolve()
    init_py = pkg_dir / "__init__.py"
    if not init_py.is_file():
        raise FileNotFoundError(f"overlay package missing __init__.py: {pkg_dir}")
    if not (pkg_dir / "_C.so").is_file():
        raise FileNotFoundError(f"overlay package missing _C.so: {pkg_dir}")

    jit_cache.mkdir(parents=True, exist_ok=True)
    # Isolate fork JIT from stock ~/.deep_gemm (and from other fork commits).
    os.environ["DG_JIT_CACHE_DIR"] = str(jit_cache)

    if force_reload and "deep_gemm_experimental" in sys.modules:
        # Drop package and submodules; leave stock deep_gemm alone.
        doomed = [
            k
            for k in sys.modules
            if k == "deep_gemm_experimental" or k.startswith("deep_gemm_experimental.")
        ]
        for k in doomed:
            del sys.modules[k]

    # Build a proper package so relative imports (utils/testing/legacy/mega) work.
    spec = importlib.util.spec_from_file_location(
        "deep_gemm_experimental",
        init_py,
        submodule_search_locations=[str(pkg_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create spec for {init_py}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["deep_gemm_experimental"] = module
    # Pre-register package path for submodule imports during exec_module.
    module.__path__ = [str(pkg_dir)]  # type: ignore[attr-defined]
    module.__package__ = "deep_gemm_experimental"
    spec.loader.exec_module(module)

    module.__dg_glm52_manifest__ = manifest  # type: ignore[attr-defined]
    module.__dg_glm52_jit_cache__ = str(jit_cache)  # type: ignore[attr-defined]
    module.__dg_glm52_package_dir__ = str(pkg_dir)  # type: ignore[attr-defined]
    return module


def load_dual(manifest_path: Path | str | None = None):
    """Return ``(stock_deep_gemm, deep_gemm_experimental)``."""
    stock = ensure_stock_deep_gemm()
    fork = load_deep_gemm_experimental(manifest_path)
    return stock, fork


def describe_dual(manifest_path: Path | str | None = None) -> dict:
    stock, fork = load_dual(manifest_path)
    return {
        "stock_file": getattr(stock, "__file__", None),
        "fork_file": getattr(fork, "__file__", None),
        "stock_version": getattr(stock, "__version__", None),
        "fork_version": getattr(fork, "__version__", None),
        "fork_jit_cache": getattr(fork, "__dg_glm52_jit_cache__", None),
        "fork_package_dir": getattr(fork, "__dg_glm52_package_dir__", None),
        "env_dg_jit_cache_dir": os.environ.get("DG_JIT_CACHE_DIR"),
        "same_module_object": stock is fork,
    }
