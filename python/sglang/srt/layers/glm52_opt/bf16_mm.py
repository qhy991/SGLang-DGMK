"""BF16 GEMM helpers for GLM-5.2 indexer weights_proj.

Mirrors ``archive/.../index_weights_proj.py``: CUDA-graph capture of
``torch.mm(x, w.t(), out_dtype=float32)`` so launch overhead is paid once.
In SGLang, decode/prefill CUDA graphs may already cover this; the cache still
helps eager / non-graph paths and matches the harness measurement.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from sglang.srt.layers.glm52_opt import config
from sglang.srt.layers.glm52_opt.context import get_forward_mode
from sglang.srt.layers.glm52_opt.phase import infer_glm52_phase
from sglang.srt.layers.glm52_opt.registry import lookup

_CACHE: Dict[Tuple[int, int, int], Tuple[torch.cuda.CUDAGraph, torch.Tensor]] = {}
_LAST_X: Optional[torch.Tensor] = None
_LAST_ENTRY: Optional[Tuple[torch.cuda.CUDAGraph, torch.Tensor]] = None


def _capture(x: torch.Tensor, w: torch.Tensor) -> Tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    wt = w.t()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            torch.mm(x, wt, out_dtype=torch.float32)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = torch.mm(x, wt, out_dtype=torch.float32)
    return graph, static_out


def bf16_mm_f32_out(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x [M,K] bf16, w [N,K] bf16 -> [M,N] f32."""
    global _LAST_X, _LAST_ENTRY
    if x is _LAST_X and _LAST_ENTRY is not None:
        graph, static_out = _LAST_ENTRY
        graph.replay()
        return static_out
    key = (x.data_ptr(), w.data_ptr(), int(x.shape[0]))
    entry = _CACHE.get(key)
    if entry is None:
        entry = _capture(x, w)
        _CACHE[key] = entry
    _LAST_X, _LAST_ENTRY = x, entry
    graph, static_out = entry
    graph.replay()
    return static_out


def try_index_weights_proj(x: torch.Tensor, weight: torch.Tensor) -> Optional[torch.Tensor]:
    """Return optimized f32 output if glm52_opt enables index_weights_proj."""
    if not config.is_enabled():
        return None
    m = int(x.shape[0]) if x.ndim == 2 else -1
    phase = infer_glm52_phase(get_forward_mode(), m)
    spec = lookup("index_weights_proj", phase, m=m)
    if spec is None or spec.kind != "bf16_gemm":
        return None
    if spec.implementation == "graph_replay":
        from sglang.srt.layers.glm52_opt.dispatch import (
            _nvtx_range,
            _profiler_range_name,
            _record_hit,
            _record_miss,
        )

        if (
            spec.n is None
            or spec.k is None
            or not x.is_cuda
            or not weight.is_cuda
            or x.device != weight.device
            or x.dtype != torch.bfloat16
            or weight.dtype != torch.bfloat16
            or tuple(x.shape) != (m, spec.k)
            or tuple(weight.shape) != (spec.n, spec.k)
            or tuple(x.stride()) != (spec.k, 1)
            or tuple(weight.stride()) != (spec.k, 1)
            or x.storage_offset() != 0
            or weight.storage_offset() != 0
        ):
            _record_miss("index_weights_graph_replay_abi", spec.op, phase, m=m)
            return None
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "index_weights_proj graph_replay cannot be nested inside the "
                "SGLang CUDA graph; run this diagnostic with CUDA graph disabled"
            )
        # An explicitly selected diagnostic candidate either replays once or
        # propagates its failure.  Do not turn a failed candidate into a hidden
        # stock torch.mm while still labeling the arm as graph_replay.
        with _nvtx_range(_profiler_range_name(spec, m)):
            out = bf16_mm_f32_out(x, weight)
        _record_hit("bf16_gemm/graph_replay", spec.op, phase, m=m)
        return out
    try:
        from sglang.srt.layers.glm52_opt.dispatch import _record_hit

        out = bf16_mm_f32_out(x, weight)
        _record_hit("bf16_gemm", "index_weights_proj", phase, m=m)
        return out
    except Exception:
        # Fall back to eager mm if graph capture fails (e.g. already capturing).
        from sglang.srt.layers.glm52_opt.dispatch import _record_hit

        out = torch.mm(x, weight.t(), out_dtype=torch.float32)
        _record_hit("bf16_gemm", "index_weights_proj", phase, m=m)
        return out
