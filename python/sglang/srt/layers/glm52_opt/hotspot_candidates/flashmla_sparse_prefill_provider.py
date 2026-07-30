"""Prebuilt-first API-v1 provider for FlashMLA ``dsa_prefill_attn`` (b3_b5).

Selected identity: ``b3_b5_native_exact`` (NOT decode ``p1_consumer_scale``).
Measured graph leaf/region vs stock at M1024/2048/4096 ≈ 1.05–1.10×, bit-exact.
Default off; point ``SGLANG_GLM52_HOTSPOT_MODULE`` at this file with
``SGLANG_GLM52_OPT_OPS=dsa_prefill_attn``.
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
_VARIANT = os.environ.get("GLM52_DSA_PREFILL_VARIANT", "b3_b5_native_exact").strip()
_KNOWN = {
    "b3_b5_native_exact": (
        "infini_kernel_glm52_dsa_prefill_attn_b3_b5_native_exact_e7e997d8404f2dd5"
    ),
}
if _VARIANT not in _KNOWN:
    raise RuntimeError(
        f"unsupported GLM52_DSA_PREFILL_VARIANT={_VARIANT!r}; "
        f"vendored prebuilt supports {sorted(_KNOWN)}"
    )
_MODULE_NAME = _KNOWN[_VARIANT]
_DEFAULT_SO = _PREBUILT_DIR / f"{_MODULE_NAME}.so"
_SUPPORTED_M = (1024, 2048, 4096)
_NUM_HEADS = 64
_HEAD_DIM_V = 512
_NUM_SM_PARTS = 148


def _truthy(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def _want_prebuilt() -> bool:
    if os.environ.get("GLM52_DSA_PREFILL_PREBUILT_SO", "").strip():
        return True
    if "GLM52_DSA_PREFILL_USE_PREBUILT" in os.environ:
        return _truthy(os.environ.get("GLM52_DSA_PREFILL_USE_PREBUILT", ""))
    return _DEFAULT_SO.is_file()


def _resolve_prebuilt_so() -> Path:
    explicit = os.environ.get("GLM52_DSA_PREFILL_PREBUILT_SO", "").strip()
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f"GLM52_DSA_PREFILL_PREBUILT_SO={explicit!r} missing")
        return path
    if not _DEFAULT_SO.is_file():
        raise RuntimeError(f"missing vendored prebuilt {_DEFAULT_SO}")
    manifest = _HERE / "MANIFEST.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text())
        for entry in data.get("binaries", []):
            if entry.get("variant") == _VARIANT and entry.get("phase") == "prefill":
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
        "dsa_prefill_attn vendor provider requires prebuilt .so "
        "(set GLM52_DSA_PREFILL_USE_PREBUILT=1 or ship prebuilt/)"
    )
_PREBUILT_SO = _resolve_prebuilt_so()
_EXTENSION, _LOADED_NAME = _load_prebuilt(_PREBUILT_SO)

PROVIDER_INFO = {
    "name": f"glm52_dsa_prefill_{_VARIANT}",
    "role": "experimental",
    "op": "dsa_prefill_attn",
    "variant": _VARIANT,
    "module_name": _LOADED_NAME,
    "prebuilt_so": str(_PREBUILT_SO),
    "supported_m": list(_SUPPORTED_M),
    "graph_only": True,
    "api_version": INFINI_KERNEL_API_VERSION,
}

_BUFFERS: dict[tuple[int, int], dict[str, torch.Tensor]] = {}


def _allocate(m: int, device: torch.device) -> dict[str, torch.Tensor]:
    key = (m, device.index if device.index is not None else 0)
    cached = _BUFFERS.get(key)
    if cached is not None:
        return cached
    total_splits = m + _NUM_SM_PARTS
    cached = {
        "out": torch.empty(
            (m, 1, _NUM_HEADS, _HEAD_DIM_V), device=device, dtype=torch.bfloat16
        ),
        "lse": torch.empty((m, 1, _NUM_HEADS), device=device, dtype=torch.float32),
        "lse_accum": torch.empty(
            (total_splits, 1, _NUM_HEADS), device=device, dtype=torch.float32
        ),
        "o_accum": torch.empty(
            (total_splits, 1, _NUM_HEADS, _HEAD_DIM_V),
            device=device,
            dtype=torch.float32,
        ),
    }
    _BUFFERS[key] = cached
    return cached


def initialize(*, gpu_id: int | None = None) -> None:
    device = torch.device("cuda" if gpu_id is None else f"cuda:{gpu_id}")
    for m in _SUPPORTED_M:
        _allocate(m, device)
    if hasattr(_EXTENSION, "reset_launch_count"):
        _EXTENSION.reset_launch_count()


def flashmla_sparse_prefill(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: float,
    indices: torch.Tensor,
    block_table: torch.Tensor,
    is_fp8_kvcache: bool,
):
    del cache_seqlens, block_table, is_fp8_kvcache, head_dim_v, softmax_scale
    m = int(q.shape[0])
    if m not in _SUPPORTED_M:
        raise RuntimeError(f"selected prefill provider received unsupported M={m}")
    buffers = _allocate(m, q.device)
    _EXTENSION.launch(
        q,
        k_cache,
        indices,
        tile_scheduler_metadata,
        num_splits,
        buffers["out"],
        buffers["lse"],
        buffers["lse_accum"],
        buffers["o_accum"],
    )
    return buffers["out"], buffers["lse"]


def candidate_evidence() -> dict[str, object]:
    out = dict(PROVIDER_INFO)
    out["extension_file"] = str(Path(_EXTENSION.__file__).resolve())
    if hasattr(_EXTENSION, "launch_count"):
        out["launch_count"] = int(_EXTENSION.launch_count())
    return out
