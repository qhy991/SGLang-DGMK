"""Fused mask + topk-k for DSA indexer decode (ROCm/HIP).

Enable via SGLANG_DSA_HIP_FUSED_TOPK=1. For batch=1 fp32 rows, runs topk on a
fixed-width score prefix (8192) with a GPU valid mask instead of masked_fill
over the full N=65536 row. CUDA-graph safe (no CPU sync on lengths).

Valid when all valid indices fall within the prefix. Matches the E2E decode
campaign (input=4096, output=128 → length stays below ~4224).
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from sglang.srt.utils import is_hip

_FLAG_ENV = "SGLANG_DSA_HIP_FUSED_TOPK"
_PREFIX_LEN = 8192


def _is_enabled() -> bool:
    return is_hip() and os.environ.get(_FLAG_ENV, "0") == "1"


def fused_masked_topk_hip(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert score.dim() == 2, f"expect [B,N], got {score.shape}"
    assert score.dtype == torch.float32, f"expect fp32, got {score.dtype}"
    assert score.is_contiguous(), "score must be contiguous"
    B, N = score.shape
    device = score.device

    topk_indices = score.new_full((B, topk), -1, dtype=torch.int32)
    if B == 0 or topk == 0 or N == 0:
        return topk_indices

    prefix = min(N, _PREFIX_LEN)
    lengths_i32 = lengths.to(dtype=torch.int32, device=device)
    if row_starts is not None:
        row_starts_i32 = row_starts.to(dtype=torch.int32, device=device)
    else:
        row_starts_i32 = torch.zeros((B,), dtype=torch.int32, device=device)

    col = torch.arange(prefix, dtype=torch.int32, device=device).unsqueeze(0)
    row_starts_b = row_starts_i32.unsqueeze(1)
    row_ends_b = (row_starts_i32 + lengths_i32).unsqueeze(1)
    valid_mask = (col >= row_starts_b) & (col < row_ends_b)

    prefix_scores = score[:, :prefix]
    masked = prefix_scores.masked_fill(~valid_mask, float("-inf"))
    take = min(topk, prefix)
    top_scores, top_idx = torch.topk(masked, take, dim=-1)
    top_local = top_idx.to(torch.int32) - row_starts_b
    top_local = top_local.masked_fill(top_scores == float("-inf"), -1)
    topk_indices[:, :take] = top_local
    return topk_indices


def maybe_dispatch_fused_topk(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if not _is_enabled():
        return None
    if score.dim() != 2 or score.dtype != torch.float32 or not score.is_contiguous():
        return None
    if score.shape[0] != 1:
        return None
    _, N = score.shape
    if topk <= 0 or topk >= N:
        return None
    try:
        return fused_masked_topk_hip(score, lengths, topk, row_starts=row_starts)
    except Exception:
        return None
