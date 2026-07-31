"""Composite API-v1 provider for the accel-bundle joint swap.

Exports both:
  - ``flashmla_sparse_decode``  (default serving: P1+combine_c2; optional r2a)
  - ``flashmla_sparse_prefill`` (b3_b5_native_exact)

Point ``SGLANG_GLM52_HOTSPOT_MODULE`` here under
``SGLANG_GLM52_OPT_PROFILE=combined_winners`` so one worker init wires decode
and prefill FlashMLA candidates together with e2e fixed-N/K GEMMs.

``GLM52_FLASHMLA_DECODE_STACK=r2a`` selects the leaf-preferred r2a stack; that
identity currently fails production CUDA-graph capture (``invalid kv shape for
bucket``), so the default is ``p1_c2``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

INFINI_KERNEL_API_VERSION = 1

_HERE = Path(__file__).resolve().parent


def _load_sibling(filename: str):
    path = _HERE / filename
    digest_name = f"sglang_glm52_accel_sibling_{path.stem}"
    spec = importlib.util.spec_from_file_location(digest_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load sibling provider {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[digest_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(digest_name, None)
        raise
    return module


# Serving CUDA-graph capture rejects the registered r2a stack SO with
# ``invalid kv shape for bucket`` (leaf harness shapes ≠ production KV pool).
# Default to the serving-proven P1+combine_c2 stack; set
# GLM52_FLASHMLA_DECODE_STACK=r2a to force the preferred leaf identity.
_DECODE_STACK = os.environ.get("GLM52_FLASHMLA_DECODE_STACK", "p1_c2").strip().lower()
_DECODE_FILE = (
    "flashmla_stack_r2a_c2_provider.py"
    if _DECODE_STACK in ("r2a", "r2a_c2", "stack_r2a")
    else "flashmla_combine_decode_provider.py"
)
_DECODE = _load_sibling(_DECODE_FILE)
_PREFILL = _load_sibling("flashmla_sparse_prefill_provider.py")

if getattr(_DECODE, "INFINI_KERNEL_API_VERSION", None) != INFINI_KERNEL_API_VERSION:
    raise RuntimeError("decode sibling API mismatch")
if getattr(_PREFILL, "INFINI_KERNEL_API_VERSION", None) != INFINI_KERNEL_API_VERSION:
    raise RuntimeError("prefill sibling API mismatch")

PROVIDER_INFO = {
    "name": "glm52_flashmla_accel_bundle",
    "role": "experimental",
    "decode": dict(getattr(_DECODE, "PROVIDER_INFO", {})),
    "prefill": dict(getattr(_PREFILL, "PROVIDER_INFO", {})),
    "api_version": INFINI_KERNEL_API_VERSION,
}


def initialize(*, gpu_id: int | None = None) -> None:
    _DECODE.initialize(gpu_id=gpu_id)
    _PREFILL.initialize(gpu_id=gpu_id)


def flashmla_sparse_decode(**kwargs):
    return _DECODE.flashmla_sparse_decode(**kwargs)


def flashmla_sparse_prefill(**kwargs):
    return _PREFILL.flashmla_sparse_prefill(**kwargs)


def candidate_evidence() -> dict[str, object]:
    out = dict(PROVIDER_INFO)
    if hasattr(_DECODE, "candidate_evidence"):
        out["decode_evidence"] = _DECODE.candidate_evidence()
    if hasattr(_PREFILL, "candidate_evidence"):
        out["prefill_evidence"] = _PREFILL.candidate_evidence()
    return out
