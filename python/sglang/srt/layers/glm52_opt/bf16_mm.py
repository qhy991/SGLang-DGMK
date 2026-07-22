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
    phase = infer_glm52_phase(get_forward_mode(), int(x.shape[0]))
    spec = lookup("index_weights_proj", phase)
    if spec is None or spec.kind != "bf16_gemm":
        return None
    try:
        from sglang.srt.layers.glm52_opt.dispatch import _record_hit

        out = bf16_mm_f32_out(x, weight)
        _record_hit("bf16_gemm", "index_weights_proj", phase)
        return out
    except Exception:
        # Fall back to eager mm if graph capture fails (e.g. already capturing).
        from sglang.srt.layers.glm52_opt.dispatch import _record_hit

        out = torch.mm(x, weight.t(), out_dtype=torch.float32)
        _record_hit("bf16_gemm", "index_weights_proj", phase)
        return out
