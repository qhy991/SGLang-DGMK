#!/usr/bin/env python3
"""Probe a two-node GLM-5.2 o_proj -> RMSNorm region on one B300.

The production path is::

    DeepGEMM(A, B) -> hidden_bf16
    fused_add_rmsnorm(hidden_bf16, residual_bf16, weight)

SM100 DeepGEMM already supports an in-place BF16 C/D accumulation through
TMA_REDUCE_ADD.  This probe tests the narrower candidate::

    DeepGEMM(A, B, D=residual, C=residual)
    rmsnorm(residual, weight) -> hidden_bf16

It first requires byte-exact residual and normalized outputs, then compares
captured two-node CUDA graphs in an interleaved A/B/B/A schedule.  This is a
mechanism probe only; promotion still requires a production 8-rank nsys trace
and the fixed-32K-KV global-BS-128 serving gate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
from pathlib import Path

import deep_gemm
import sgl_kernel
import torch
import tvm_ffi


M = 16
N = 6144
K = 16384
EPS = 1.0e-6


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _describe(values: list[float]) -> dict[str, float | int]:
    return {
        "samples": len(values),
        "mean": statistics.fmean(values),
        "stdev": statistics.pstdev(values),
        "p10": _quantile(values, 0.10),
        "p50": _quantile(values, 0.50),
        "p90": _quantile(values, 0.90),
        "min": min(values),
        "max": max(values),
    }


def _packed_ue8m0_ones(rows: int, k_groups: int) -> torch.Tensor:
    scales = torch.ones((rows, k_groups), dtype=torch.float32, device="cuda")
    return deep_gemm.utils.layout.get_mn_major_tma_aligned_packed_ue8m0_tensor(
        scales
    )


def _load_fork_extension(fork_root: Path, runtime_root: Path):
    """Load the GLM-5.2 fork's _C without replacing stock ``deep_gemm``.

    The source worktree intentionally contains only the compiled extension in
    ``sgl_deep_gemm``; loading its package ``__init__`` would require the wheel
    staging tree.  Loading the extension under a private package name keeps the
    installed production module untouched while still exercising the fork JIT.
    ``runtime_root/include`` is prepared by the caller with the fork's DeepGEMM
    and CUTLASS headers.
    """

    so_path = fork_root / "sgl_deep_gemm" / "_C.so"
    if not so_path.is_file():
        raise FileNotFoundError(f"fork extension missing: {so_path}")
    required = (
        runtime_root / "include" / "deep_gemm",
        runtime_root / "include" / "cute",
        runtime_root / "include" / "cutlass",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "fork runtime header links are missing; prepare them before the probe: "
            + ", ".join(missing)
        )

    module = tvm_ffi.load_module(str(so_path))
    module.init(str(runtime_root), os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    set_block_multiple = getattr(module, "set_block_size_multiple_of", None)
    if callable(set_block_multiple):
        set_block_multiple(16)
    set_pdl = getattr(module, "set_pdl", None)
    if callable(set_pdl):
        set_pdl(True)
    return module


def _fork_gemm(fork_c, lhs, rhs, out, *, c=None) -> None:
    fork_c.fp8_fp4_gemm_nt(
        lhs[0],
        lhs[1],
        rhs[0],
        rhs[1],
        out,
        c,
        None,
        None,
        None,
        "nk",
        False,
    )


def _time_replay(graph: torch.cuda.CUDAGraph) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) * 1_000.0)


def _git_rev() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def run(
    *,
    warmup: int,
    iters: int,
    seed: int,
    fork_root: Path,
    runtime_root: Path,
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if major != 10:
        raise RuntimeError(f"this probe requires SM100, got sm{major}{minor}")
    fork_c = _load_fork_extension(fork_root.resolve(), runtime_root.resolve())

    torch.manual_seed(seed)
    # Small representable FP8 values keep repeated in-place graph replays away
    # from BF16 overflow without changing the memory or scheduling mechanism.
    a = (torch.randn((M, K), device="cuda") * 0.0625).to(
        torch.float8_e4m3fn
    )
    b = (torch.randn((N, K), device="cuda") * 0.0625).to(
        torch.float8_e4m3fn
    )
    a_scale = _packed_ue8m0_ones(M, K // 128)
    b_scale = _packed_ue8m0_ones(N, K // 128)
    lhs = (a, a_scale)
    rhs = (b, b_scale)
    weight = (torch.randn((N,), device="cuda", dtype=torch.float32) * 0.02 + 1.0).to(
        torch.bfloat16
    )
    residual_seed = (torch.randn((M, N), device="cuda") * 0.25).to(
        torch.bfloat16
    )

    # Compile/warm both kWithAccumulation specializations before correctness.
    compile_plain = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
    _fork_gemm(fork_c, lhs, rhs, compile_plain)
    compile_accum = residual_seed.clone()
    _fork_gemm(fork_c, lhs, rhs, compile_accum, c=compile_accum)
    torch.cuda.synchronize()

    stock_hidden = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
    stock_residual = residual_seed.clone()
    _fork_gemm(fork_c, lhs, rhs, stock_hidden)
    sgl_kernel.fused_add_rmsnorm(
        stock_hidden, stock_residual, weight, EPS, enable_pdl=True
    )

    candidate_residual = residual_seed.clone()
    candidate_hidden = torch.empty_like(candidate_residual)
    _fork_gemm(fork_c, lhs, rhs, candidate_residual, c=candidate_residual)
    sgl_kernel.rmsnorm(
        candidate_residual,
        weight,
        EPS,
        out=candidate_hidden,
        enable_pdl=True,
    )
    torch.cuda.synchronize()

    residual_equal = bool(torch.equal(stock_residual, candidate_residual))
    hidden_equal = bool(torch.equal(stock_hidden, candidate_hidden))
    residual_mismatch = int(
        torch.count_nonzero(stock_residual.view(torch.int16) != candidate_residual.view(torch.int16)).item()
    )
    hidden_mismatch = int(
        torch.count_nonzero(stock_hidden.view(torch.int16) != candidate_hidden.view(torch.int16)).item()
    )
    correctness = {
        "residual_byte_exact": residual_equal,
        "normalized_hidden_byte_exact": hidden_equal,
        "residual_mismatched_bf16_elements": residual_mismatch,
        "normalized_hidden_mismatched_bf16_elements": hidden_mismatch,
        "residual_max_abs": float(
            (stock_residual.float() - candidate_residual.float()).abs().max().item()
        ),
        "normalized_hidden_max_abs": float(
            (stock_hidden.float() - candidate_hidden.float()).abs().max().item()
        ),
    }

    # Fail before timing: a numerically different residual is externally
    # observable by the next layer and is not a valid fusion candidate.
    if not (residual_equal and hidden_equal):
        return {
            "schema_version": 1,
            "workload": {"m": M, "n": N, "k": K, "eps": EPS},
            "correctness": correctness,
            "decision": "REJECT_NUMERICS",
        }

    graph_stock_hidden = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
    graph_stock_residual = residual_seed.clone()
    graph_candidate_residual = residual_seed.clone()
    graph_candidate_hidden = torch.empty_like(graph_candidate_residual)

    # Warm calls on a side stream make every lazy module and kernel ready before
    # graph capture.  The captured graphs contain only the two production nodes.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _fork_gemm(fork_c, lhs, rhs, graph_stock_hidden)
        sgl_kernel.fused_add_rmsnorm(
            graph_stock_hidden,
            graph_stock_residual,
            weight,
            EPS,
            enable_pdl=True,
        )
        _fork_gemm(
            fork_c,
            lhs,
            rhs,
            graph_candidate_residual,
            c=graph_candidate_residual,
        )
        sgl_kernel.rmsnorm(
            graph_candidate_residual,
            weight,
            EPS,
            out=graph_candidate_hidden,
            enable_pdl=True,
        )
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    stock_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(stock_graph):
        _fork_gemm(fork_c, lhs, rhs, graph_stock_hidden)
        sgl_kernel.fused_add_rmsnorm(
            graph_stock_hidden,
            graph_stock_residual,
            weight,
            EPS,
            enable_pdl=True,
        )

    candidate_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(candidate_graph):
        _fork_gemm(
            fork_c,
            lhs,
            rhs,
            graph_candidate_residual,
            c=graph_candidate_residual,
        )
        sgl_kernel.rmsnorm(
            graph_candidate_residual,
            weight,
            EPS,
            out=graph_candidate_hidden,
            enable_pdl=True,
        )

    for _ in range(warmup):
        stock_graph.replay()
        candidate_graph.replay()
    torch.cuda.synchronize()

    stock_us: list[float] = []
    candidate_us: list[float] = []
    # A/B/B/A cancels monotonic clock and thermal drift without relying on a
    # single long unpaired run.
    for _ in range(iters):
        stock_us.append(_time_replay(stock_graph))
        candidate_us.append(_time_replay(candidate_graph))
        candidate_us.append(_time_replay(candidate_graph))
        stock_us.append(_time_replay(stock_graph))

    stock_desc = _describe(stock_us)
    candidate_desc = _describe(candidate_us)
    speedup = float(stock_desc["p50"]) / float(candidate_desc["p50"])
    saved = float(stock_desc["p50"]) - float(candidate_desc["p50"])
    return {
        "schema_version": 1,
        "git_rev": _git_rev(),
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "deep_gemm_stock": getattr(deep_gemm, "__version__", None),
        "deep_gemm_fork_root": str(fork_root.resolve()),
        "deep_gemm_fork_runtime_root": str(runtime_root.resolve()),
        "workload": {"m": M, "n": N, "k": K, "eps": EPS},
        "mechanism": {
            "stock": "DeepGEMM identity store -> fused_add_rmsnorm",
            "candidate": "DeepGEMM in-place BF16 TMA_REDUCE_ADD -> rmsnorm",
            "cuda_graph_nodes_per_arm": 2,
            "enable_pdl": True,
        },
        "correctness": correctness,
        "timing_us": {
            "stock": stock_desc,
            "candidate": candidate_desc,
            "candidate_saved_p50": saved,
            "candidate_speedup_p50": speedup,
        },
        "decision": "MECHANISM_WIN" if speedup > 1.01 else "NO_MECHANISM_WIN",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--fork-root",
        type=Path,
        default=Path(
            os.environ.get(
                "DEEPGEMM_GLM52_ROOT",
                "third_party/DeepGEMM-GLM52",
            )
        ),
    )
    parser.add_argument(
        "--fork-runtime-root",
        type=Path,
        default=Path(
            os.environ.get(
                "DG_GLM52_RUNTIME_ROOT",
                "/mnt/b300-shared/home/qinhaiyan/wwxq/cache/deep_gemm_glm52_runtime",
            )
        ),
    )
    args = parser.parse_args()
    if args.warmup < 1 or args.iters < 1:
        parser.error("--warmup and --iters must be positive")
    result = run(
        warmup=args.warmup,
        iters=args.iters,
        seed=args.seed,
        fork_root=args.fork_root,
        runtime_root=args.fork_runtime_root,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    return 0 if result["decision"] != "REJECT_NUMERICS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
