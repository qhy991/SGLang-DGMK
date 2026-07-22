"""HIP/ROCm sparse MLA via AITER unified_attention_sparse_mla (TS=64/ns=1)."""

from __future__ import annotations

from functools import lru_cache

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import get_bool_env_var, is_hip

_is_hip = is_hip()


@lru_cache(maxsize=1)
def should_use_aiter_sparse_mla() -> bool:
    """Whether the optimized AITER sparse MLA kernel is available on this runtime."""
    if not _is_hip:
        return False
    if not get_bool_env_var("SGLANG_USE_AITER"):
        return False
    if not envs.SGLANG_DSA_USE_AITER_SPARSE_MLA.get():
        return False
    try:
        from aiter.ops.triton.attention.unified_attention_sparse_mla import (  # noqa: F401
            unified_attention_sparse_mla,
        )

        return True
    except ImportError:
        return False


def aiter_sparse_mla_min_kv_len() -> int:
    return envs.SGLANG_DSA_AITER_SPARSE_MLA_MIN_KV_LEN.get()


def should_route_aiter_sparse_mla(max_seq_len_k: int) -> bool:
    """Use unified_attention_sparse_mla only when KV is long enough to amortize gather."""
    return should_use_aiter_sparse_mla() and max_seq_len_k >= aiter_sparse_mla_min_kv_len()


_BLOCK_TABLE_CACHE: dict[tuple[torch.device, int], torch.Tensor] = {}


def _get_block_table(device: torch.device, num_blocks: int) -> torch.Tensor:
    key = (device, num_blocks)
    tbl = _BLOCK_TABLE_CACHE.get(key)
    if tbl is None:
        tbl = torch.arange(num_blocks, dtype=torch.int32, device=device)
        _BLOCK_TABLE_CACHE[key] = tbl
    elif tbl.shape[0] != num_blocks:
        # Defensive fail-fast: stale entry with mismatched size would silently
        # corrupt attention. ReqToTokenPool has no runtime resize path today
        # (memory_pool.py:146-172 allocates once at init), so this branch is
        # unreachable in current sglang — but it protects against a future
        # pool-resize refactor. Also guards against cache aliasing bugs.
        raise RuntimeError(
            f"aiter sparse MLA block_table cache: cached tensor size "
            f"{tbl.shape[0]} != requested num_blocks {num_blocks} for key {key}"
        )
    return tbl


def clear_aiter_block_table_cache() -> None:
    """Evict all cached block_table tensors.

    Call this if the KV pool is ever resized at runtime. Not wired into any
    caller today (see _get_block_table docstring on why it's unreachable);
    provided for explicit invalidation by future pool-resize paths.
    """
    _BLOCK_TABLE_CACHE.clear()


def aiter_sparse_mla_fwd(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_indices: torch.Tensor,
    sm_scale: float,
    v_head_dim: int,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    seq_lens: torch.Tensor,
    max_seqlen_k: int,
    block_size: int = 1,
) -> torch.Tensor:
    """Run sparse MLA with AITER unified_attention_sparse_mla."""
    from aiter.ops.triton.attention.unified_attention_sparse_mla import (
        unified_attention_sparse_mla,
    )

    num_tokens, num_heads, head_dim = q.shape
    if topk_indices.dim() == 3:
        topk_indices = topk_indices[:, 0, :]
    if topk_indices.dtype != torch.int32:
        topk_indices = topk_indices.to(torch.int32)

    num_slots = kv_cache.shape[0]
    if block_size <= 0:
        block_size = 1
    if block_size == 1:
        kv_blocked = kv_cache.view(num_slots, 1, 1, head_dim)
        num_blocks = num_slots
    else:
        padded_slots = ((num_slots + block_size - 1) // block_size) * block_size
        if padded_slots > num_slots:
            kv_pad = kv_cache.new_zeros(padded_slots - num_slots, 1, head_dim)
            kv_flat = torch.cat([kv_cache, kv_pad], dim=0)
        else:
            kv_flat = kv_cache
        num_blocks = padded_slots // block_size
        kv_blocked = kv_flat.view(num_blocks, block_size, 1, head_dim)

    num_seqs = seq_lens.shape[0]
    block_table = _get_block_table(q.device, num_blocks).unsqueeze(0).expand(
        num_seqs, -1
    )

    if seq_lens.dtype != torch.int32:
        seq_lens = seq_lens.to(torch.int32)

    out = torch.empty(
        num_tokens, num_heads, v_head_dim, dtype=q.dtype, device=q.device
    )
    unified_attention_sparse_mla(
        q,
        kv_blocked,
        out,
        cu_seqlens_q,
        max_seqlen_q,
        seq_lens,
        max_seqlen_k,
        float(sm_scale),
        topk_indices,
        block_table,
        v_head_dim,
    )
    return out
