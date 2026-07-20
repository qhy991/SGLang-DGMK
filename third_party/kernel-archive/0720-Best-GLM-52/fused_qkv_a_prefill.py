"""Candidate 6 (WINNER): parallelize the scale-pack across the group dimension
(2D grid) + deep_gemm compiled_dims='nk' + Programmatic Dependent Launch (PDL).

candidate-4 packed with a 1-D grid where each program serially walked all 12
int32 groups (48 f32 loads) — latency-bound at a ~8.7us floor, dominated by the
x-scale pack. This kernel uses a 2-D grid (row-block, group): every program does
just 4 loads + 1 store, so all 12 groups run concurrently. x-pack drops from
~8.7us to ~2.6us (near the ~2us kernel floor); the fused x+w pack is ~2.6us total.
Output is byte-identical to candidate-4 (torch.equal on xs/ws, all shapes).

Separately, deep_gemm's fp8_gemm_nt is fastest here with compiled_dims='nk'
(measured 45.7us vs 46.8us default at M=4096).

Net peak MFU (M=4096) 61.5% vs candidate-4's 52.8% (baseline deep_gemm f32 46.3%).
Per shape: M1024 39.4%, M2048 51.5%, M4096 61.5%. calc_diff == 0 on every shape.
Gemm-only ceiling here is ~64% (deep_gemm tcgen05 floor), so the 70% target is
above what this GEMM can deliver without rewriting the MMA (prohibited/loses).
"""
from __future__ import annotations
import torch
import triton
import triton.language as tl
import deep_gemm

# Programmatic Dependent Launch: lets the GEMM kernel run its prologue while the
# preceding pack kernel's tail is still finishing, hiding part of the ~2.6us pack.
# The GEMM still grid-dep-waits before consuming scales, so it is correctness-safe
# (the Triton pack emits no early trigger -> the wait resolves on full completion).
deep_gemm.set_pdl(True)


@triton.jit
def _pack_fused_2d(xin_ptr, xout_ptr, x_stride_kb, x_stride_g, M, GX,
                   win_ptr, wout_ptr, w_stride_g, N,
                   BLOCK: tl.constexpr):
    b = tl.program_id(0)
    g = tl.program_id(1)
    if b < GX:
        ms = b * BLOCK + tl.arange(0, BLOCK)
        mask = ms < M
        packed = tl.zeros((BLOCK,), dtype=tl.int32)
        for i in tl.static_range(4):
            kb = g * 4 + i
            f = tl.load(xin_ptr + ms + kb * x_stride_kb, mask=mask, other=0.0)
            e = (f.to(tl.int32, bitcast=True) >> 23) & 0xFF
            packed = packed | (e << (8 * i))
        tl.store(xout_ptr + ms + g * x_stride_g, packed, mask=mask)
    else:
        wb = b - GX
        ns = wb * BLOCK + tl.arange(0, BLOCK)
        mask = ns < N
        blk = ns // 128
        packed = tl.zeros((BLOCK,), dtype=tl.int32)
        for i in tl.static_range(4):
            kb = g * 4 + i
            f = tl.load(win_ptr + blk * 48 + kb, mask=mask, other=0.0)
            e = (f.to(tl.int32, bitcast=True) >> 23) & 0xFF
            packed = packed | (e << (8 * i))
        tl.store(wout_ptr + ns + g * w_stride_g, packed, mask=mask)


def run(inputs: dict):
    out = inputs["out"]
    M = inputs["rows"]; N = inputs["N"]
    x_scale = inputs["x_scale"]; w_scale = inputs["w_scale"]
    dev = x_scale.device
    xs = torch.empty((12, M), dtype=torch.int32, device=dev).t()   # (M,12) stride(1,M)
    ws = torch.empty((12, N), dtype=torch.int32, device=dev).t()   # (N,12) stride(1,N)
    BLOCK = 128
    GX = triton.cdiv(M, BLOCK)
    GW = triton.cdiv(N, BLOCK)
    _pack_fused_2d[(GX + GW, 12)](
        x_scale, xs, x_scale.stride(1), xs.stride(1), M, GX,
        w_scale, ws, ws.stride(1), N, BLOCK=BLOCK, num_warps=4)
    deep_gemm.fp8_gemm_nt((inputs["x_fp8"], xs), (inputs["w_fp8"], ws), out,
                          compiled_dims='nk', disable_ue8m0_cast=True)
    return out
