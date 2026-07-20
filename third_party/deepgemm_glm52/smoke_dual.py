#!/usr/bin/env python3
"""Dual-version smoke: stock deep_gemm vs deep_gemm_experimental isolation."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from loader import describe_dual, load_dual  # noqa: E402


def _calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x32, y32 = x.float(), y.float()
    denom = (x32 * x32 + y32 * y32).sum().clamp_min(1e-12)
    return float(((x32 - y32).square().sum() / denom).sqrt().item())


def _run_fp8_nt(mod, m: int, n: int, k: int, device: str):
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    w = torch.randn(n, k, device=device, dtype=torch.bfloat16)
    # Match DeepGEMM block-scale layout used by glm52 (K/128).
    x_fp8 = x.to(torch.float8_e4m3fn)
    w_fp8 = w.to(torch.float8_e4m3fn)
    x_scale = torch.ones(m, k // 128, device=device, dtype=torch.float32)
    w_scale = torch.ones(n // 128, k // 128, device=device, dtype=torch.float32)
    out = torch.empty(m, n, device=device, dtype=torch.bfloat16)
    mod.fp8_gemm_nt((x_fp8, x_scale), (w_fp8, w_scale), out)
    return out, (x_fp8, x_scale, w_fp8, w_scale)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--n", type=int, default=16384)
    ap.add_argument("--k", type=int, default=2048)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("FAIL: CUDA unavailable")
        return 2

    device = args.device
    torch.cuda.set_device(device)

    info = describe_dual()
    print("describe:", info)
    if info["same_module_object"]:
        print("FAIL: stock and fork are the same module object")
        return 1
    if "site-packages" not in str(info["stock_file"]):
        print("FAIL: stock deep_gemm is not from site-packages:", info["stock_file"])
        return 1
    if "deep_gemm_experimental" not in str(info["fork_file"]):
        print("FAIL: fork path unexpected:", info["fork_file"])
        return 1
    if not info["fork_jit_cache"] or "overlays" not in str(info["fork_jit_cache"]):
        print("FAIL: fork JIT cache not overlay-partitioned:", info["fork_jit_cache"])
        return 1

    stock, fork = load_dual()

    # Setter isolation: change fork knobs, stock getters must stay put.
    stock_sms0 = stock.get_num_sms()
    fork_sms0 = fork.get_num_sms()
    print(f"initial sms stock={stock_sms0} fork={fork_sms0}")
    fork.set_num_sms(max(1, fork_sms0 // 2))
    stock_sms1 = stock.get_num_sms()
    fork_sms1 = fork.get_num_sms()
    print(f"after fork.set_num_sms stock={stock_sms1} fork={fork_sms1}")
    if stock_sms1 != stock_sms0:
        print("FAIL: fork setter leaked into stock get_num_sms")
        return 1
    if fork_sms1 == fork_sms0:
        print("FAIL: fork set_num_sms had no effect")
        return 1
    # Restore fork
    fork.set_num_sms(fork_sms0)

    # Also verify stock setter does not move fork.
    stock.set_num_sms(max(1, stock_sms0 // 2))
    if fork.get_num_sms() != fork_sms0:
        print("FAIL: stock setter leaked into fork")
        return 1
    stock.set_num_sms(stock_sms0)

    # Numeric parity on identical packed-ready float32-scale inputs.
    # Use the same tensors for both calls.
    m, n, k = args.m, args.n, args.k
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    w = torch.randn(n, k, device=device, dtype=torch.bfloat16)
    x_fp8 = x.to(torch.float8_e4m3fn)
    w_fp8 = w.to(torch.float8_e4m3fn)
    x_scale = torch.ones(m, k // 128, device=device, dtype=torch.float32)
    w_scale = torch.ones((n + 127) // 128, k // 128, device=device, dtype=torch.float32)
    # w_scale for fp8_gemm_nt expects (ceil_n/128?); glm52 uses (128,16) for N=16384 -> N/128.
    w_scale = torch.ones(n // 128, k // 128, device=device, dtype=torch.float32)

    out_s = torch.empty(m, n, device=device, dtype=torch.bfloat16)
    out_f = torch.empty(m, n, device=device, dtype=torch.bfloat16)
    stock.fp8_gemm_nt((x_fp8, x_scale), (w_fp8, w_scale), out_s)
    fork.fp8_gemm_nt((x_fp8.clone(), x_scale.clone()), (w_fp8.clone(), w_scale.clone()), out_f)
    torch.cuda.synchronize()

    diff = _calc_diff(out_s, out_f)
    max_abs = float((out_s.float() - out_f.float()).abs().max().item())
    print(f"parity M={m}: calc_diff={diff:.3e} max_abs={max_abs:.3e}")
    if diff > 5e-6 and max_abs > 0:
        # Unmodified fork at same commit should be bit-exact or calc_diff~0.
        print("FAIL: stock/fork numeric mismatch beyond tolerance")
        return 1

    # Elementwise equality preferred for unmodified fork.
    if not torch.equal(out_s, out_f):
        print("WARN: outputs not bitwise equal; calc_diff still within tol")

    print("PASS: dual-version smoke")
    print("DG_JIT_CACHE_DIR=", os.environ.get("DG_JIT_CACHE_DIR"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
