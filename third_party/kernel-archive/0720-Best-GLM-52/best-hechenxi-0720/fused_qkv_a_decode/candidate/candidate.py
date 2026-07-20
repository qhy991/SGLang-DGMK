"""GLM-5.2 Indexer Q Up-Projection (decode) plain-load split-free FP8 GEMM.

PROFILE (M=32, split-K TMA kernel): device time is a ~6.1us N-INDEPENDENT fixed cost
(N=128 slice 6.27us; full N=4096 8.35us) on top of a ~2.7us single-kernel launch/timing
floor, plus only ~2us for the actual 8MB weight stream. The grid runs 0.25-0.58 waves
(256-1024 tiny CTAs / 148 SMs) so DRAM never exceeds ~10% -- pure latency bound, not BW.
Two changes cut the fixed cost: (1) drop TMA -- for a [BN,128] fp8 tile only 4KB, the
TensorDescriptor build + tensormap prologue costs more than it saves, so plain vectorised
tl.load of the K-contiguous weight rows is faster; (2) drop split-K -- once TMA is gone a
single deep K-loop (16 groups) with many software-pipeline stages fills the pipe without a
DRAM partial round-trip. BLOCK_N=16 gives 256 N-tiles to spread over the SMs.
"""
from __future__ import annotations
import os
import torch
import triton
import triton.language as tl


def _env_int(name, default):
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else default


_GROUP = 128


@triton.jit
def _notma_gemm(
    w_ptr, x_ptr, sx_ptr, sw_ptr, partial_ptr, lock_ptr, out_ptr,
    M, N, K,
    stride_wn, stride_wk,
    stride_xm, stride_xk,
    stride_sxm, stride_sxj,
    stride_swn, stride_swj,
    stride_ps, stride_pn, stride_pm,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUPS_PER_SPLIT: tl.constexpr, SPLIT_K: tl.constexpr, NUM_STAGES: tl.constexpr,
):
    n_tile = tl.program_id(0)
    sid = tl.program_id(1)
    n0 = n_tile * BLOCK_N
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = n0 + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_blk = n0 // 128
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    g0 = sid * GROUPS_PER_SPLIT
    for gg in tl.range(GROUPS_PER_SPLIT, num_stages=NUM_STAGES):
        kb = g0 + gg
        k0 = kb * BLOCK_K
        w_t = tl.load(w_ptr + offs_n[:, None] * stride_wn + (k0 + offs_k)[None, :] * stride_wk)
        x_t = tl.load(x_ptr + offs_m[:, None] * stride_xm + (k0 + offs_k)[None, :] * stride_xk,
                      mask=m_mask[:, None], other=0.0)
        dot = tl.dot(w_t, tl.trans(x_t), out_dtype=tl.float32)
        sx = tl.load(sx_ptr + offs_m * stride_sxm + kb * stride_sxj, mask=m_mask, other=0.0)
        sw = tl.load(sw_ptr + n_blk * stride_swn + kb * stride_swj)
        acc += dot * (sw * sx[None, :])

    if SPLIT_K == 1:
        tl.store(out_ptr + offs_n[:, None] * stride_on + offs_m[None, :] * stride_om,
                 acc.to(tl.bfloat16), mask=m_mask[None, :])
        return
    tl.store(partial_ptr + sid * stride_ps + offs_n[:, None] * stride_pn
             + offs_m[None, :] * stride_pm, acc, mask=m_mask[None, :])
    arrived = tl.atomic_add(lock_ptr + n_tile, 1, sem="acq_rel", scope="gpu")
    if arrived == SPLIT_K - 1:
        red = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
        for s in tl.static_range(SPLIT_K):
            red += tl.load(partial_ptr + s * stride_ps + offs_n[:, None] * stride_pn
                           + offs_m[None, :] * stride_pm, mask=m_mask[None, :], other=0.0)
        tl.store(out_ptr + offs_n[:, None] * stride_on + offs_m[None, :] * stride_om,
                 red.to(tl.bfloat16), mask=m_mask[None, :])
        tl.store(lock_ptr + n_tile, 0)


_SEM = {}
def _sem(device, n):
    key = (device.index or 0, torch.cuda.current_stream(device).stream_id, n)
    l = _SEM.get(key)
    if l is None:
        l = torch.zeros(n, dtype=torch.int32, device=device); _SEM[key] = l
    return l


def _cfg(M):
    bn = _env_int("IQU_BLOCK_N", 32)
    sk = _env_int("IQU_SPLIT_K", 6)
    if M <= 16:
        warps = _env_int("IQU_NUM_WARPS", 2)
        stages = _env_int("IQU_STAGES", 4)
    else:
        warps = _env_int("IQU_NUM_WARPS", 2)
        stages = _env_int("IQU_STAGES", 4)
    return bn, sk, warps, stages


@torch.no_grad()
def run(inputs):
    x = inputs["x_fp8"]; sx = inputs["x_scale"]; w = inputs["w_fp8"]; sw = inputs["w_scale"]
    M, K = x.shape; N = w.shape[0]
    groups = K // _GROUP
    block_n, split_k, warps, stages = _cfg(M)
    n_tiles = N // block_n
    block_m = max(16, triton.next_power_of_2(M))
    out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    if split_k == 1:
        partial = out; lock = out; ps0 = ps1 = ps2 = 0
    else:
        partial = torch.empty((split_k, N, M), dtype=torch.float32, device=x.device)
        lock = _sem(x.device, n_tiles)
        ps0, ps1, ps2 = partial.stride(0), partial.stride(1), partial.stride(2)
    _notma_gemm[(n_tiles, split_k)](
        w, x, sx, sw, partial, lock, out,
        M, N, K,
        w.stride(0), w.stride(1),
        x.stride(0), x.stride(1),
        sx.stride(0), sx.stride(1),
        sw.stride(0), sw.stride(1),
        ps0, ps1, ps2,
        out.stride(0), out.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=_GROUP,
        GROUPS_PER_SPLIT=groups // split_k, SPLIT_K=split_k, NUM_STAGES=stages,
        num_warps=warps, num_stages=stages,
    )
    return out
