"""Prebuilt-first API-v1 provider: FlashMLA ``r2a_prologue_overlap`` + combine_c2.

Round-2 main-kernel identity on top of the frozen combine_c2 stack.
Measured vs P1+c2: M16 graph leaf/region ~1.09–1.10×; M32 ~neutral (≥1.0, not
a regression). Registered under the policy that a clear single-bucket win is
enough to ship default-off for A/B (bs=16 / M16).

Point ``SGLANG_GLM52_HOTSPOT_MODULE`` at this file with
``SGLANG_GLM52_OPT_OPS=flashmla_sparse_decode`` (alias of ``dsa_decode_attn``).
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
_STACK_KEY = "r2a_prologue_overlap__combine_c2_bucket_stages"
_MODULE_NAME = (
    "infini_kernel_glm52_flashmla_sparse_decode_stack_"
    f"{_STACK_KEY}_4f22d65fcd15e4b8"
)
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
        if not path.is_file():
            raise RuntimeError(f"GLM52_FLASHMLA_PREBUILT_SO={explicit!r} missing")
        return path
    if not _DEFAULT_SO.is_file():
        raise RuntimeError(f"missing vendored prebuilt {_DEFAULT_SO}")
    manifest = _HERE / "MANIFEST.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text())
        for entry in data.get("binaries", []):
            if entry.get("variant") == "r2a_prologue_overlap":
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
        "r2a+c2 stack provider requires prebuilt .so "
        "(GLM52_FLASHMLA_USE_PREBUILT=1 or ship prebuilt/)"
    )
_PREBUILT_SO = _resolve_prebuilt_so()
_EXTENSION, _LOADED_NAME = _load_prebuilt(_PREBUILT_SO)

PROVIDER_INFO = {
    "name": "glm52_flashmla_stack_r2a_c2",
    "role": "experimental",
    "main_variant": "r2a_prologue_overlap",
    "combine_variant": "combine_c2_bucket_stages",
    "module_name": _LOADED_NAME,
    "prebuilt_so": str(_PREBUILT_SO),
    "api_version": INFINI_KERNEL_API_VERSION,
    "note": "M16 ~1.09x vs P1+c2; M32 ~neutral; default-off single-bucket win policy",
}

_WORKSPACES: dict[int, tuple[torch.Tensor, ...]] = {}


def initialize(*, gpu_id: int | None = None) -> None:
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
    if hasattr(_EXTENSION, "reset_launch_count"):
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
    workspace = _WORKSPACES.get(int(q.shape[0]))
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
    out = dict(PROVIDER_INFO)
    out["extension_file"] = str(Path(_EXTENSION.__file__).resolve())
    if hasattr(_EXTENSION, "launch_count"):
        out["launch_count"] = int(_EXTENSION.launch_count())
    return out
