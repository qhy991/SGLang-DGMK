"""Fail-closed SM100 DeepGEMM paged-MQA adapter for GLM-5.2 prefill.

This path removes the ordinary prefill indexer's paged-to-contiguous K/scale
gather.  It intentionally supports only uniform query counts per request: that
shape maps to DeepGEMM's compact [B, next_n, H, D] ABI and keeps the block table
at [B, pages] without materializing an [M, pages] repeated table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


GLM_HEADS = 32
HEAD_DIM = 128
PAGE_SIZE = 64
PAGE_BYTES = PAGE_SIZE * (HEAD_DIM + 4)
LOGITS_ALIGNMENT = 256

# B300 measurements at context=100K, including production gather/logits/top-k:
#   * every tested shape through total M=256 was positive;
#   * uniform Q<=128 stayed positive through total M=2048;
#   * Q256 x 16 (M=4096) and Q512 x 1 were negative.
VALIDATED_ANY_Q_MAX_M = 256
VALIDATED_MAX_Q_PER_REQUEST = 128
VALIDATED_MAX_TOTAL_Q = 2048


@dataclass(frozen=True)
class Glm52PagedPrefillMqaPlan:
    context_lens_2d: torch.Tensor
    schedule_metadata: torch.Tensor
    local_row_starts: torch.Tensor
    requests: int
    queries_per_request: int
    total_q: int
    max_context: int


def prepare_glm52_paged_prefill_mqa_plan(
    *,
    seq_lens_expanded: torch.Tensor,
    block_tables: torch.Tensor,
    extend_lens_cpu: Sequence[int],
    max_context: int,
    logits_budget_bytes: int,
    max_total_q: int | None = None,
) -> Glm52PagedPrefillMqaPlan | None:
    """Return a reusable DeepGEMM schedule or None for stock fallback."""

    lengths = tuple(int(length) for length in extend_lens_cpu)
    total_q = int(seq_lens_expanded.numel())
    if total_q == 0 or not lengths:
        return None
    if any(length <= 0 for length in lengths) or len(set(lengths)) != 1:
        return None
    queries_per_request = lengths[0]
    if sum(lengths) != total_q:
        return None

    if max_total_q is None:
        admitted = total_q <= VALIDATED_ANY_Q_MAX_M or (
            queries_per_request <= VALIDATED_MAX_Q_PER_REQUEST
            and total_q <= VALIDATED_MAX_TOTAL_Q
        )
        if not admitted:
            return None
    else:
        if max_total_q <= 0:
            raise ValueError("max_total_q must be positive")
        if total_q > max_total_q:
            return None

    if seq_lens_expanded.dtype != torch.int32 or not seq_lens_expanded.is_cuda:
        raise RuntimeError("paged prefill MQA sequence lengths must be CUDA int32")
    if block_tables.dtype != torch.int32 or not block_tables.is_cuda:
        raise RuntimeError("paged prefill MQA block table must be CUDA int32")
    if block_tables.ndim != 2 or block_tables.shape[0] != len(lengths):
        raise RuntimeError("paged prefill MQA block table must be [requests,pages]")
    if max_context <= 0 or max_context > block_tables.shape[1] * PAGE_SIZE:
        raise RuntimeError("max_context exceeds the page-table capacity")

    padded_context = (
        (max_context + LOGITS_ALIGNMENT - 1) // LOGITS_ALIGNMENT
    ) * LOGITS_ALIGNMENT
    logits_bytes = total_q * padded_context * torch.empty(
        (), dtype=torch.float32
    ).element_size()
    if logits_bytes > logits_budget_bytes:
        return None

    context_lens_2d = seq_lens_expanded.view(
        len(lengths), queries_per_request
    ).contiguous()

    import deep_gemm

    schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
        context_lens_2d, PAGE_SIZE, deep_gemm.get_num_sms()
    )
    return Glm52PagedPrefillMqaPlan(
        context_lens_2d=context_lens_2d,
        schedule_metadata=schedule_metadata,
        local_row_starts=torch.zeros_like(seq_lens_expanded),
        requests=len(lengths),
        queries_per_request=queries_per_request,
        total_q=total_q,
        max_context=max_context,
    )


def run_glm52_paged_prefill_mqa(
    *,
    q: torch.Tensor,
    raw_kv_cache: torch.Tensor,
    weights: torch.Tensor,
    block_tables: torch.Tensor,
    plan: Glm52PagedPrefillMqaPlan,
) -> torch.Tensor:
    """Run stock DeepGEMM directly on GLM's raw page-64 index cache."""

    if q.shape != (plan.total_q, GLM_HEADS, HEAD_DIM):
        raise RuntimeError(
            f"GLM paged prefill Q must be [{plan.total_q},32,128], got {tuple(q.shape)}"
        )
    if q.dtype != torch.float8_e4m3fn or not q.is_contiguous():
        raise RuntimeError("GLM paged prefill Q must be contiguous e4m3fn")
    if weights.shape != (plan.total_q, GLM_HEADS):
        raise RuntimeError("GLM paged prefill weights must be [M,32]")
    if weights.dtype != torch.float32 or not weights.is_contiguous():
        raise RuntimeError("GLM paged prefill weights must be contiguous float32")
    if raw_kv_cache.dtype != torch.uint8 or raw_kv_cache.ndim != 2:
        raise RuntimeError("GLM paged prefill raw KV must be 2D uint8")
    if raw_kv_cache.shape[1] != PAGE_BYTES or not raw_kv_cache.is_contiguous():
        raise RuntimeError(
            f"GLM paged prefill KV pages must be contiguous [{PAGE_BYTES}] rows"
        )

    import deep_gemm

    fused_kv = raw_kv_cache.view(-1, PAGE_SIZE, 1, HEAD_DIM + 4)
    return deep_gemm.fp8_paged_mqa_logits(
        q.view(
            plan.requests,
            plan.queries_per_request,
            GLM_HEADS,
            HEAD_DIM,
        ),
        fused_kv,
        weights,
        plan.context_lens_2d,
        block_tables,
        plan.schedule_metadata,
        plan.max_context,
        clean_logits=False,
    )


__all__ = [
    "Glm52PagedPrefillMqaPlan",
    "prepare_glm52_paged_prefill_mqa_plan",
    "run_glm52_paged_prefill_mqa",
    "VALIDATED_ANY_Q_MAX_M",
    "VALIDATED_MAX_Q_PER_REQUEST",
    "VALIDATED_MAX_TOTAL_Q",
]
