"""Split-KV wrapper for AITER MLA prefill (gfx942 bf16).

Raises ``num_kv_splits`` above the AITER default (=1) to improve MI300X
occupancy on GLM-5 absorbed MLA prefill shapes, then fuses the partial
LSE combine in Triton.
"""
from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

import aiter
from aiter import dtypes
from aiter.mla import mla_prefill_fwd as _mla_prefill_fwd_stock


@triton.jit
def _lse_combine_kernel(
    split_data_ptr,
    split_lse_ptr,
    out_ptr,
    M: tl.int64,
    N: tl.int64,
    H: tl.int64,
    Dv: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Dv

    e_max = tl.full((1,), -float("inf"), dtype=tl.float32)
    e_sum = tl.zeros((1,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_DV,), dtype=tl.float32)

    base_data = (pid_m * N * H * Dv) + (pid_h * Dv)
    stride_split_data = H * Dv
    base_lse = (pid_m * N * H) + pid_h
    stride_split_lse = H

    for k in range(0, N):
        lse_k = tl.load(split_lse_ptr + base_lse + k * stride_split_lse).to(tl.float32)
        n_e_max = tl.maximum(lse_k, e_max)
        old_scale = tl.exp(e_max - n_e_max)
        acc = acc * old_scale
        exp_logic = tl.exp(lse_k - n_e_max)
        tv = tl.load(
            split_data_ptr + base_data + k * stride_split_data + offs_d,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        acc = acc + exp_logic * tv
        e_sum = e_sum * old_scale + exp_logic
        e_max = n_e_max

    out = acc / e_sum
    tl.store(out_ptr + (pid_m * H * Dv) + (pid_h * Dv) + offs_d, out, mask=mask_d)


@triton.jit
def _lse_combine_kernel_constn(
    split_data_ptr,
    split_lse_ptr,
    out_ptr,
    M: tl.int64,
    H: tl.int64,
    Dv: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    N_SPLITS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Dv

    e_max = tl.full((1,), -float("inf"), dtype=tl.float32)
    e_sum = tl.zeros((1,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_DV,), dtype=tl.float32)

    base_data = (pid_m * N_SPLITS * H * Dv) + (pid_h * Dv)
    stride_split_data = H * Dv
    base_lse = (pid_m * N_SPLITS * H) + pid_h
    stride_split_lse = H

    for k in tl.static_range(N_SPLITS):
        lse_k = tl.load(split_lse_ptr + base_lse + k * stride_split_lse).to(tl.float32)
        n_e_max = tl.maximum(lse_k, e_max)
        old_scale = tl.exp(e_max - n_e_max)
        acc = acc * old_scale
        exp_logic = tl.exp(lse_k - n_e_max)
        tv = tl.load(
            split_data_ptr + base_data + k * stride_split_data + offs_d,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        acc = acc + exp_logic * tv
        e_sum = e_sum * old_scale + exp_logic
        e_max = n_e_max

    out = acc / e_sum
    tl.store(out_ptr + (pid_m * H * Dv) + (pid_h * Dv) + offs_d, out, mask=mask_d)


def _fused_lse_combine(
    split_data: torch.Tensor, split_lse: torch.Tensor, out: torch.Tensor
) -> None:
    M, N, H, Dv = split_data.shape
    BLOCK_DV = triton.next_power_of_2(Dv)
    grid = (M, H)
    if N == 10:
        _lse_combine_kernel_constn[grid](
            split_data,
            split_lse,
            out,
            M,
            H,
            Dv,
            BLOCK_DV=BLOCK_DV,
            N_SPLITS=10,
            num_warps=8,
            num_stages=2,
        )
    elif N == 8:
        _lse_combine_kernel_constn[grid](
            split_data,
            split_lse,
            out,
            M,
            H,
            Dv,
            BLOCK_DV=BLOCK_DV,
            N_SPLITS=8,
            num_warps=8,
            num_stages=2,
        )
    elif N == 7:
        _lse_combine_kernel_constn[grid](
            split_data,
            split_lse,
            out,
            M,
            H,
            Dv,
            BLOCK_DV=BLOCK_DV,
            N_SPLITS=7,
            num_warps=8,
            num_stages=2,
        )
    else:
        _lse_combine_kernel[grid](
            split_data,
            split_lse,
            out,
            M,
            N,
            H,
            Dv,
            BLOCK_DV=BLOCK_DV,
            num_warps=4,
            num_stages=2,
        )


def choose_num_kv_splits(m: int) -> int:
    if m <= 3000:
        return 8
    return 10


def mla_prefill_split_fwd(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    o: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_lens: torch.Tensor,
    max_seqlen_q: int,
    sm_scale: Optional[float] = None,
    logit_cap: float = 0.0,
    num_kv_splits: Optional[int] = None,
) -> torch.Tensor:
    """Absorbed MLA prefill with configurable KV-split parallelism."""
    if num_kv_splits is None:
        num_kv_splits = choose_num_kv_splits(max_seqlen_q)
    if num_kv_splits <= 1:
        _mla_prefill_fwd_stock(
            q,
            kv_buffer,
            o,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            max_seqlen_q=max_seqlen_q,
            sm_scale=sm_scale,
            logit_cap=logit_cap,
        )
        return o

    if num_kv_splits < 2:
        raise ValueError(
            f"num_kv_splits={num_kv_splits} is invalid for the split path; "
            "the AITER ASM kernel requires >=2 splits with fp32 splitData "
            "to keep the log-sum-exp combine correct."
        )

    device = q.device
    _num_page, _page_size, _nhead_kv, qk_head_dim = kv_buffer.shape
    if sm_scale is None:
        sm_scale = 1.0 / (qk_head_dim**0.5)

    bs, nhead, v_head_dim = o.shape

    split_data = torch.empty(
        (bs, num_kv_splits, nhead, v_head_dim),
        dtype=dtypes.fp32,
        device=device,
    )
    split_lse = torch.empty(
        (bs, num_kv_splits, nhead, 1),
        dtype=dtypes.fp32,
        device=device,
    )

    aiter.mla_prefill_asm_fwd(
        q,
        kv_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        max_seqlen_q,
        sm_scale,
        split_data,
        split_lse,
    )

    _fused_lse_combine(split_data, split_lse, o)
    return o
