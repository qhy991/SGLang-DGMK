"""Sparse MLA decode kernel optimized for MI300X (gfx942, bf16 KV).

All query heads share the same KV (H_KV=1); one program per query position
loads KV once and computes a block of heads, eliminating redundant KV reads.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _sparse_mla_fwd_kernel(
    Q_ptr,
    KV_ptr,
    IDX_ptr,
    OUT_ptr,
    M_ptr,
    L_ptr,
    sm_scale,
    stride_q0: tl.int64,
    stride_q1: tl.int64,
    stride_kv0: tl.int64,
    stride_idx0: tl.int64,
    stride_out0: tl.int64,
    stride_out1: tl.int64,
    H_Q: tl.constexpr,
    KV_LORA_RANK: tl.constexpr,
    ROPE_RANK: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_H: tl.constexpr,
    TILE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_h_blocks: tl.constexpr = H_Q // BLOCK_H
    q_pos = pid // num_h_blocks
    h_block = pid % num_h_blocks

    offs_h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_lora = tl.arange(0, KV_LORA_RANK)
    offs_rope = tl.arange(0, ROPE_RANK)

    q_base = Q_ptr + q_pos * stride_q0
    q_lora = tl.load(q_base + offs_h[:, None] * stride_q1 + offs_lora[None, :])
    q_rope = tl.load(
        q_base + offs_h[:, None] * stride_q1 + (KV_LORA_RANK + offs_rope[None, :])
    )

    m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_H], 0.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, KV_LORA_RANK], dtype=tl.float32)

    idx_base = IDX_ptr + q_pos * stride_idx0

    for t in range(tl.cdiv(topk, TILE_K)):
        tile_start = t * TILE_K
        offs_t = tl.arange(0, TILE_K)
        valid = (tile_start + offs_t) < topk

        kv_idx = tl.load(idx_base + tile_start + offs_t, mask=valid, other=0)

        kv_base = KV_ptr + kv_idx[:, None] * stride_kv0
        k_lora = tl.load(kv_base + offs_lora[None, :], mask=valid[:, None], other=0.0)
        k_rope = tl.load(
            kv_base + (KV_LORA_RANK + offs_rope[None, :]), mask=valid[:, None], other=0.0
        )

        s = tl.dot(q_lora, tl.trans(k_lora))
        s += tl.dot(q_rope, tl.trans(k_rope))
        s *= sm_scale
        s = tl.where(valid[None, :], s, float("-inf"))

        m_j = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_j)
        m_new = tl.where(m_new > float("-inf"), m_new, 0.0)

        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_j = tl.sum(p, axis=1)

        l_i = l_i * alpha + l_j
        acc = acc * alpha[:, None]
        m_i = m_new

        acc = tl.dot(p.to(k_lora.dtype), k_lora, acc=acc)

    acc = acc / l_i[:, None]

    out_base = OUT_ptr + q_pos * stride_out0
    tl.store(
        out_base + offs_h[:, None] * stride_out1 + offs_lora[None, :],
        acc.to(tl.bfloat16),
    )


def sparse_mla_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int,
) -> torch.Tensor:
    if kv.dim() == 3:
        kv = kv.squeeze(1)
    if indices.dim() == 3:
        indices = indices[:, 0, :]

    s_q, H_Q, D_QK = q.shape
    topk = indices.shape[1]
    rope_rank = D_QK - d_v

    out = torch.empty(s_q, H_Q, d_v, dtype=q.dtype, device=q.device)
    m_buf = torch.empty(s_q, H_Q, dtype=torch.float32, device=q.device)
    l_buf = torch.empty(s_q, H_Q, dtype=torch.float32, device=q.device)

    BLOCK_H = 8
    TILE_K = 64

    grid = (s_q * (H_Q // BLOCK_H),)

    _sparse_mla_fwd_kernel[grid](
        q,
        kv,
        indices.to(torch.int64),
        out,
        m_buf,
        l_buf,
        sm_scale,
        q.stride(0),
        q.stride(1),
        kv.stride(0),
        indices.stride(0),
        out.stride(0),
        out.stride(1),
        H_Q,
        d_v,
        rope_rank,
        topk,
        BLOCK_H=BLOCK_H,
        TILE_K=TILE_K,
        num_stages=1,
        num_warps=4,
    )
    return out
