"""Custom fused-scale blockwise FP8 GEMM (Triton) — task8 spike (in-kernel ue8m0).

run() passes the RAW f32 block scales inputs["x_scale"], inputs["w_scale"]
directly into the kernel; the reference-equivalent ue8m0 rounding is done
INSIDE the kernel via the exact bit manipulation deep_gemm.ceil_to_ue8m0 uses
(bitcast f32->int, exp = ((bits>>23)&0xFF) + (mantissa!=0), clamp[1,254],
(exp<<23) back to float). No deep_gemm.ceil_to_ue8m0 / prepacking / transform
helper is called before launch. Stateless: inputs are never mutated or cached.
"""
from __future__ import annotations
import torch, triton, triton.language as tl

_CFGS = [
    triton.Config({'BM': bm, 'BN': bn, 'BK': 128}, num_warps=nw, num_stages=ns)
    for bm in (64, 128) for bn in (128, 256) for nw in (4, 8) for ns in (3, 4)
]


@triton.jit
def _ue8m0(s):  # exact deep_gemm.ceil_to_ue8m0 on f32 tensor, in-kernel
    b = s.to(tl.int32, bitcast=True)
    exp = ((b >> 23) & 0xFF) + ((b & 0x7FFFFF) != 0).to(tl.int32)
    exp = tl.minimum(tl.maximum(exp, 1), 254)
    return (exp << 23).to(tl.float32, bitcast=True)


@triton.autotune(configs=_CFGS, key=['M', 'N', 'K'])
@triton.jit
def _fp8_bw_gemm(
    x_ptr, w_ptr, xs_ptr, ws_ptr, out_ptr, M, N, K,
    sx_m, sx_k, sw_n, sw_k, so_m, so_n, sxs_m, sxs_k, sws_n, sws_k,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kb in range(0, K // BK):
        koff = kb * BK
        x = tl.load(x_ptr + rm[:, None] * sx_m + (koff + rk)[None, :] * sx_k,
                    mask=rm[:, None] < M, other=0.0)
        w = tl.load(w_ptr + rn[:, None] * sw_n + (koff + rk)[None, :] * sw_k,
                    mask=rn[:, None] < N, other=0.0)
        p = tl.dot(x, tl.trans(w), out_dtype=tl.float32)
        # raw f32 scales loaded here; ue8m0 rounding done in-kernel:
        xs = _ue8m0(tl.load(xs_ptr + rm * sxs_m + kb * sxs_k, mask=rm < M, other=0.0))
        ws = _ue8m0(tl.load(ws_ptr + (rn // 128) * sws_n + kb * sws_k, mask=rn < N, other=0.0))
        acc += p * xs[:, None] * ws[None, :]
    tl.store(out_ptr + rm[:, None] * so_m + rn[None, :] * so_n, acc.to(tl.bfloat16),
             mask=(rm[:, None] < M) & (rn[None, :] < N))


def run(inputs: dict):
    x = inputs["x_fp8"]; w = inputs["w_fp8"]
    xs = inputs["x_scale"]; ws = inputs["w_scale"]        # RAW f32 scales, unmodified
    out = inputs["out"]
    M, K = x.shape; N = w.shape[0]
    grid = lambda meta: (triton.cdiv(M, meta['BM']), triton.cdiv(N, meta['BN']))
    _fp8_bw_gemm[grid](
        x, w, xs, ws, out, M, N, K,
        x.stride(0), x.stride(1), w.stride(0), w.stride(1), out.stride(0), out.stride(1),
        xs.stride(0), xs.stride(1), ws.stride(0), ws.stride(1),
    )
    return out
