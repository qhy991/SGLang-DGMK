"""Prebuilt-first API-v1 provider for FlashMLA ``p1_consumer_scale`` (main only).

Prefer the stack provider ``flashmla_combine_decode_provider.py`` for the
combine_c2 win. This module exposes the P1-only tip binary for A/B against
stock combine.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch


INFINI_KERNEL_API_VERSION = 1

_HERE = Path(__file__).resolve().parent
_PREBUILT_DIR = _HERE / "prebuilt"
_VARIANT = os.environ.get("GLM52_FLASHMLA_VARIANT", "p1_consumer_scale").strip()
_KNOWN = {
    "p1_consumer_scale": (
        "infini_kernel_glm52_flashmla_sparse_decode_p1_consumer_scale_a39236323dc57a97"
    ),
}
if _VARIANT not in _KNOWN:
    raise RuntimeError(
        f"unsupported GLM52_FLASHMLA_VARIANT={_VARIANT!r}; "
        f"vendored prebuilt supports {sorted(_KNOWN)}"
    )
_MODULE_NAME = _KNOWN[_VARIANT]
_DEFAULT_SO = _PREBUILT_DIR / f"{_MODULE_NAME}.so"


def _truthy(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def _want_prebuilt() -> bool:
    if os.environ.get("GLM52_FLASHMLA_PREBUILT_SO", "").strip():
        return True
    if "GLM52_FLASHMLA_USE_PREBUILT" in os.environ:
        return _truthy(os.environ.get("GLM52_FLASHMLA_USE_PREBUILT", ""))
    return _DEFAULT_SO.is_file()


def _resolve_prebuilt_so() -> Path:
    explicit = os.environ.get("GLM52_FLASHMLA_PREBUILT_SO", "").strip()
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if path.is_dir():
            matches = sorted(path.glob(f"*p1_consumer_scale_*.so"))
            if not matches:
                raise RuntimeError(f"no P1 prebuilt under {path}")
            return matches[0]
        if not path.is_file():
            raise RuntimeError(f"GLM52_FLASHMLA_PREBUILT_SO={explicit!r} missing")
        return path
    if not _DEFAULT_SO.is_file():
        raise RuntimeError(f"missing vendored prebuilt {_DEFAULT_SO}")
    manifest = _HERE / "MANIFEST.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text())
        for entry in data.get("binaries", []):
            if entry.get("variant") == _VARIANT:
                digest = hashlib.sha256(_DEFAULT_SO.read_bytes()).hexdigest()
                expected = entry.get("sha256")
                if expected and digest != expected:
                    raise RuntimeError(
                        f"prebuilt sha256 mismatch: got {digest}, expected {expected}"
                    )
                break
    return _DEFAULT_SO


def _load_prebuilt(so_path: Path):
    name = so_path.stem
    spec = importlib.util.spec_from_file_location(name, so_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {so_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, name


if not _want_prebuilt():
    raise RuntimeError(
        "vendored P1 provider is prebuilt-only in SGLang; set "
        "GLM52_FLASHMLA_USE_PREBUILT=1 or point GLM52_FLASHMLA_PREBUILT_SO"
    )

_PREBUILT_SO = _resolve_prebuilt_so()
_EXTENSION, _LOADED_NAME = _load_prebuilt(_PREBUILT_SO)

PROVIDER_INFO = {
    "name": f"glm52_flashmla_sparse_decode_{_VARIANT}",
    "role": "experimental",
    "variant": _VARIANT,
    "module_name": _LOADED_NAME,
    "prebuilt_so": str(_PREBUILT_SO),
    "main_symbol_prefix": "infini_kernel_glm52_flashmla_sparse_decode",
}

_WORKSPACES: dict[int, tuple[torch.Tensor, ...]] = {}


def initialize(*, gpu_id: int | None) -> None:
    device = torch.device("cuda" if gpu_id is None else f"cuda:{gpu_id}")
    for m in (16, 32):
        out = torch.empty((m, 1, 64, 512), dtype=torch.bfloat16, device=device)
        lse_base = torch.empty((m, 1, 64), dtype=torch.float32, device=device)
        lse = lse_base.transpose(1, 2)
        lse_accum = torch.empty((m + 148, 1, 64), dtype=torch.float32, device=device)
        o_accum = torch.empty(
            (m + 148, 1, 64, 512), dtype=torch.float32, device=device
        )
        _WORKSPACES[m] = (out, lse_base, lse, lse_accum, o_accum)
    _EXTENSION.reset_launch_count()


def flashmla_sparse_decode(
    *,
    q,
    k_cache,
    cache_seqlens,
    head_dim_v,
    tile_scheduler_metadata,
    num_splits,
    softmax_scale,
    indices,
    block_table,
    is_fp8_kvcache,
):
    del cache_seqlens, block_table
    if head_dim_v != 512 or softmax_scale != 0.0625 or not is_fp8_kvcache:
        raise RuntimeError("selected provider received a non-promotional ABI")
    workspace = _WORKSPACES.get(q.shape[0])
    if workspace is None:
        raise RuntimeError("selected provider received an unsupported M")
    out, lse_base, lse, lse_accum, o_accum = workspace
    _EXTENSION.launch(
        q,
        k_cache,
        indices,
        tile_scheduler_metadata,
        num_splits,
        out,
        lse_base,
        lse_accum,
        o_accum,
    )
    return out, lse


def candidate_evidence() -> dict[str, object]:
    return {
        **PROVIDER_INFO,
        "extension_file": str(Path(_EXTENSION.__file__).resolve()),
        "launch_count": int(_EXTENSION.launch_count()),
    }
