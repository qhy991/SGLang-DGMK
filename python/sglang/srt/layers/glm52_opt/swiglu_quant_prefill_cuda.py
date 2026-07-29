"""Bounded precise-CUDA fallback for Task 29."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import torch

CUDA_VARIANTS = {
    "cuda_s8_v16_b128": 0,
    "cuda_s16_v8_b256": 1,
    "cuda_s8_v16_b128_cgld": 2,
}
_CXX_FLAGS = ("-O3", "-std=c++20")
_CUDA_FLAGS = (
    "-O3",
    "-std=c++20",
    "-lineinfo",
    "--expt-relaxed-constexpr",
)
_SOURCE = (
    Path(__file__).resolve().parent
    / "kernels"
    / "cuda"
    / "swiglu_quant_prefill.cu"
)


@lru_cache(maxsize=1)
def load_extension():
    """Compile/load the content-addressed extension outside the hot path."""

    if not _SOURCE.is_file():
        raise RuntimeError(f"Task-29 CUDA source is missing: {_SOURCE}")
    fingerprint = hashlib.sha256()
    fingerprint.update(_SOURCE.read_bytes())
    for flag in (*_CXX_FLAGS, *_CUDA_FLAGS):
        fingerprint.update(b"\0")
        fingerprint.update(flag.encode())
    source_hash = fingerprint.hexdigest()[:16]
    from torch.utils.cpp_extension import load

    return load(
        name=f"sglang_glm52_swiglu_quant_prefill_{source_hash}",
        sources=[str(_SOURCE)],
        extra_cflags=list(_CXX_FLAGS),
        extra_cuda_cflags=list(_CUDA_FLAGS),
        verbose=False,
    )


def run_into(
    gateup_output: torch.Tensor,
    output: torch.Tensor,
    scale_storage: torch.Tensor,
    m_indices: torch.Tensor,
    endpoint: torch.Tensor,
    *,
    variant: str,
) -> None:
    try:
        variant_id = CUDA_VARIANTS[variant]
    except KeyError as exc:
        raise RuntimeError(
            f"unmaterialized Task-29 CUDA variant {variant!r}"
        ) from exc
    load_extension().run_cuda(
        gateup_output,
        output,
        scale_storage,
        m_indices,
        endpoint,
        variant_id,
    )


def source_sha256() -> str:
    return hashlib.sha256(_SOURCE.read_bytes()).hexdigest()


def build_fingerprint() -> str:
    digest = hashlib.sha256()
    digest.update(_SOURCE.read_bytes())
    for flag in (*_CXX_FLAGS, *_CUDA_FLAGS):
        digest.update(b"\0")
        digest.update(flag.encode())
    return digest.hexdigest()
