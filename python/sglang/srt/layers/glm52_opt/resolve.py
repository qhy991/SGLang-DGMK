"""Resolve which backend a (op, phase) would use — for smoke / CI."""

from __future__ import annotations

from typing import Optional

from sglang.srt.layers.glm52_opt.fp8_gemm import _NATIVE_FORK_OPS, _NATIVE_PACKED_OPS
from sglang.srt.layers.glm52_opt.registry import lookup


def resolve_backend(op_name: str, phase: str) -> Optional[str]:
    spec = lookup(op_name, phase)
    if spec is None:
        return None
    if spec.kind == "fp8_gemm":
        if op_name in _NATIVE_FORK_OPS and phase == "decode":
            return "deepgemm_fork_fused"
        if op_name in _NATIVE_PACKED_OPS:
            return "native_packed_ue8m0"
        return f"archive:{spec.archive_ref}"
    if spec.kind == "moe_masked":
        return "native_moe_pack_pdl"
    if spec.kind == "dsa":
        return f"archive:{spec.archive_ref}"
    if spec.kind == "bf16_gemm":
        return "native_bf16_cuda_graph_mm"
    return spec.kind
