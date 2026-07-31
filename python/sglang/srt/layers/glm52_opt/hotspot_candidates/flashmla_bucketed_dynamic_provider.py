"""Experimental GLM-5.2 FlashMLA decode candidates with dynamic KV capacity.

The explicitly selected candidate buckets use different main kernels:

* M16: ``r2a_prologue_overlap`` + ``combine_c2_bucket_stages``
* M32: ``p1_consumer_scale_m32_rr_inline`` + ``combine_c2_bucket_stages``

Both extensions implement the same dynamic-page ABI.  They are loaded once at
worker initialization, and the hot path selects an extension using only the
host-known ``q.shape[0]``.  Both receive SGLang's canonical request-major
scheduler metadata; M32 maps physical CTAs to canonical rows inside its main
kernel.  There is no framework-side reorder or device-to-host metadata read.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType

import torch
from sglang.srt.layers.glm52_opt.flashmla_scheduler_contract import (
    scheduler_contract,
)

INFINI_KERNEL_API_VERSION = 1

_HERE = Path(__file__).resolve().parent
_MANIFEST = _HERE / "MANIFEST.json"
_BUCKETS = {
    16: {
        "role": "decode_stack_dynamic_pages_m16",
        "main_variant": "r2a_prologue_overlap",
        "combine_variant": "combine_c2_bucket_stages",
        "variant_token": "r2a_c2_dynamic",
        "env": "GLM52_FLASHMLA_M16_SO",
    },
    32: {
        "role": "decode_stack_dynamic_pages_m32_rr_inline",
        "main_variant": "p1_consumer_scale_m32_rr_inline",
        "combine_variant": "combine_c2_bucket_stages",
        "variant_token": "p1_c2_m32_rr_inline_dynamic",
        "env": "GLM52_FLASHMLA_M32_SO",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_entries() -> list[dict[str, object]]:
    if not _MANIFEST.is_file():
        return []
    raw = json.loads(_MANIFEST.read_text())
    entries = raw.get("binaries", [])
    if not isinstance(entries, list):
        raise TypeError("FlashMLA manifest binaries must be a list")
    return [dict(entry) for entry in entries if isinstance(entry, dict)]


def _resolve_bucket(m: int) -> tuple[Path, dict[str, object]]:
    contract = _BUCKETS[m]
    scheduler = scheduler_contract(m)
    explicit = os.environ.get(str(contract["env"]), "").strip()
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f"{contract['env']}={explicit!r} is missing")
        variant_token = str(contract["variant_token"])
        if variant_token not in path.stem:
            raise RuntimeError(
                f"{contract['env']} must name the audited M{m} variant token "
                f"{variant_token!r}: {path.name}"
            )
        return path, {
            **contract,
            **scheduler,
            "bucket_m": m,
            "so_file": str(path),
            "sha256": _sha256(path),
            "promotion_status": "explicit_development_override",
            "source": "explicit_development_override",
        }

    matches = [
        entry
        for entry in _manifest_entries()
        if entry.get("role") == contract["role"] and entry.get("bucket_m") == m
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one manifest entry for M{m} role={contract['role']!r}, "
            f"found {len(matches)}"
        )
    entry = matches[0]
    if entry.get("variant") != contract["variant_token"]:
        raise RuntimeError(
            f"M{m} manifest variant={entry.get('variant')!r}, "
            f"expected {contract['variant_token']!r}"
        )
    for key in ("main_variant", "combine_variant"):
        if entry.get(key) != contract[key]:
            raise RuntimeError(
                f"M{m} manifest {key}={entry.get(key)!r}, expected {contract[key]!r}"
            )
    for key, expected in scheduler.items():
        if entry.get(key) != expected:
            raise RuntimeError(
                f"M{m} manifest {key}={entry.get(key)!r}, expected {expected!r}"
            )
    promotion_status = entry.get("promotion_status")
    if not isinstance(promotion_status, str) or not promotion_status.startswith(
        "explicit_experimental_profile_only"
    ):
        raise RuntimeError(
            f"M{m} manifest is not fail-closed experimental: {promotion_status!r}"
        )
    so_file = entry.get("so_file")
    expected_sha = entry.get("sha256")
    if not isinstance(so_file, str) or not isinstance(expected_sha, str):
        raise TypeError(f"M{m} manifest entry lacks so_file/sha256")
    path = (_HERE / so_file).resolve()
    if not path.is_file():
        raise RuntimeError(f"M{m} prebuilt is missing: {path}")
    if entry.get("module_name") != path.stem:
        raise RuntimeError(
            f"M{m} manifest module_name={entry.get('module_name')!r}, "
            f"but the DSO identity is {path.stem!r}"
        )
    actual_sha = _sha256(path)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"M{m} prebuilt sha256 mismatch: got {actual_sha}, expected {expected_sha}"
        )
    return path, {**entry, "source": "vendored_manifest"}


def _load_extension(path: Path) -> ModuleType:
    module_name = path.stem
    existing = sys.modules.get(module_name)
    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        if existing_file is None or Path(existing_file).resolve() != path:
            raise RuntimeError(
                f"extension module {module_name!r} is already bound to "
                f"{existing_file!r}, not {str(path)!r}"
            )
        return existing
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import FlashMLA extension {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


_BUCKET_PATHS: dict[int, Path] = {}
_BUCKET_MANIFEST: dict[int, dict[str, object]] = {}
for _m in sorted(_BUCKETS):
    _path, _entry = _resolve_bucket(_m)
    _BUCKET_PATHS[_m] = _path
    _BUCKET_MANIFEST[_m] = _entry
if len(set(_BUCKET_PATHS.values())) != len(_BUCKET_PATHS):
    raise RuntimeError("M16 and M32 must use distinct FlashMLA DSOs")
if len({_sha256(path) for path in _BUCKET_PATHS.values()}) != len(_BUCKET_PATHS):
    raise RuntimeError("M16 and M32 FlashMLA DSOs must have distinct content hashes")

PROVIDER_INFO = {
    "name": "glm52_flashmla_bucketed_dynamic_pages",
    "role": "experimental",
    "api_version": INFINI_KERNEL_API_VERSION,
    "routing": "host q.shape[0] only",
    "dynamic_kv_pages": True,
    "canonical_scheduler_metadata": True,
    "framework_side_scheduler_reorder": False,
    "buckets": {
        str(m): {
            "main_variant": _BUCKETS[m]["main_variant"],
            "combine_variant": _BUCKETS[m]["combine_variant"],
            "module_name": _BUCKET_PATHS[m].stem,
            "extension_file": str(_BUCKET_PATHS[m]),
            "sha256": _sha256(_BUCKET_PATHS[m]),
            **scheduler_contract(m),
            "provenance": _BUCKET_MANIFEST[m],
        }
        for m in sorted(_BUCKETS)
    },
}

_EXTENSIONS: dict[int, ModuleType] = {}
_WORKSPACES: dict[int, tuple[torch.Tensor, ...]] = {}


def initialize(*, gpu_id: int | None = None) -> None:
    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)
    device = torch.device("cuda" if gpu_id is None else f"cuda:{gpu_id}")
    for m in sorted(_BUCKETS):
        if m not in _EXTENSIONS:
            _EXTENSIONS[m] = _load_extension(_BUCKET_PATHS[m])
        out = torch.empty((m, 1, 64, 512), dtype=torch.bfloat16, device=device)
        lse_base = torch.empty((m, 1, 64), dtype=torch.float32, device=device)
        lse = lse_base.transpose(1, 2)
        lse_accum = torch.empty((m + 148, 1, 64), dtype=torch.float32, device=device)
        o_accum = torch.empty((m + 148, 1, 64, 512), dtype=torch.float32, device=device)
        _WORKSPACES[m] = (out, lse_base, lse, lse_accum, o_accum)
        reset = getattr(_EXTENSIONS[m], "reset_launch_count", None)
        if callable(reset):
            reset()


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
    m = int(q.shape[0])
    extension = _EXTENSIONS.get(m)
    workspace = _WORKSPACES.get(m)
    if extension is None or workspace is None:
        raise RuntimeError(f"provider is uninitialized or received unsupported M={m}")
    out, lse_base, lse, lse_accum, o_accum = workspace
    extension.launch(
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
    evidence = dict(PROVIDER_INFO)
    evidence["buckets"] = {
        str(m): {
            **dict(PROVIDER_INFO["buckets"][str(m)]),
            "launch_count": (
                int(_EXTENSIONS[m].launch_count()) if m in _EXTENSIONS else None
            ),
            "workspace_pointers": (
                {
                    "out": int(_WORKSPACES[m][0].data_ptr()),
                    "lse_base": int(_WORKSPACES[m][1].data_ptr()),
                    "lse_view": int(_WORKSPACES[m][2].data_ptr()),
                    "lse_stride": list(_WORKSPACES[m][2].stride()),
                    "lse_accum": int(_WORKSPACES[m][3].data_ptr()),
                    "o_accum": int(_WORKSPACES[m][4].data_ptr()),
                }
                if m in _WORKSPACES
                else None
            ),
        }
        for m in sorted(_BUCKETS)
    }
    return evidence
