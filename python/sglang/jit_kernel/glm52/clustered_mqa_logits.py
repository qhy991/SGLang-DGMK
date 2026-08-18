"""SM103 clustered FP8 paged-MQA adapter for the GLM-5.2 DSA indexer.

The migrated DSV4 kernel shares each KV tile across a two-CTA cluster covering
16 adjacent causal queries. This GLM specialization changes the template from
H64/D128 to the native H32/D128 contract; the index-cache page ABI is already
identical: page=64 and 8448 bytes (64 * (128 FP8 + one FP32 scale)).
"""

from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from sglang.jit_kernel.utils import cache_once


GLM_HEADS = 32
HEAD_DIM = 128
PAGE_SIZE = 64
PAGE_BYTES = PAGE_SIZE * (HEAD_DIM + 4)
QUERIES_PER_CLUSTER = 16
SUPPORTED_QUERIES_PER_CLUSTER = (16, 32)
LOGITS_ALIGNMENT = 256
VALIDATED_MAX_M = 384
VALIDATED_MAX_M_Q16 = 256
VALIDATED_MAX_M_Q32 = 384
VALIDATED_M512_REQUESTS = 16
VALIDATED_M512_MIN_CONTEXT = 64_000

_HERE = Path(__file__).resolve().parent
_SOURCE_DIR = _HERE.parent / "csrc" / "glm52" / "clustered_mqa_logits"
_SOURCES = (_SOURCE_DIR / "v2_binding.cpp", _SOURCE_DIR / "v2_kernel.cu")
_DEPENDENCIES = (
    *_SOURCES,
    _SOURCE_DIR / "v2_mqa_logits_layout.cuh",
    _SOURCE_DIR / "v2_sm100_mqa_logits.cuh",
    _SOURCE_DIR / "v2_sm100_paged_mqa_logits.cuh",
)


@dataclass(frozen=True)
class ClusteredMqaChunk:
    start: int
    end: int
    seq_lens: torch.Tensor
    page_table: torch.Tensor
    schedule: torch.Tensor


@dataclass(frozen=True)
class Glm52ClusteredMqaPlan:
    chunks: tuple[ClusteredMqaChunk, ...]
    logits_workspace: torch.Tensor
    max_context: int
    padded_context: int
    total_q: int
    queries_per_cluster: int
    extension: Any


def _source_digest() -> str:
    digest = hashlib.sha256()
    for path in _DEPENDENCIES:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:16]


@contextmanager
def _torch_arch(arch: str):
    key = "TORCH_CUDA_ARCH_LIST"
    previous = os.environ.get(key)
    os.environ[key] = arch
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


@cache_once
def load_glm52_clustered_mqa_extension():
    capability = torch.cuda.get_device_capability()
    if capability != (10, 3):
        raise RuntimeError(
            "GLM-5.2 clustered MQA is B300-only; expected SM103, got "
            f"sm{capability[0]}{capability[1]}"
        )
    import deep_gemm
    from torch.utils.cpp_extension import load

    deep_gemm_include = Path(deep_gemm.__file__).resolve().parent / "include"
    module_name = f"sglang_glm52_clustered_mqa_h32native_{_source_digest()}"
    with _torch_arch("10.3a"):
        return load(
            name=module_name,
            sources=[str(path) for path in _SOURCES],
            extra_include_paths=[str(_SOURCE_DIR), str(deep_gemm_include)],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=[
                "-O3",
                "-std=c++17",
                "-lineinfo",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
            ],
            extra_ldflags=["-lcuda"],
            with_cuda=True,
            verbose=os.environ.get("SGLANG_GLM52_CLUSTERED_MQA_VERBOSE", "0")
            == "1",
        )


def prepare_glm52_clustered_mqa_plan(
    *,
    seq_lens_expanded: torch.Tensor,
    token_to_batch_idx: torch.Tensor,
    block_tables: torch.Tensor,
    extend_lens_cpu: Sequence[int],
    max_context: int,
    logits_budget_bytes: int,
    max_total_q: int | None = None,
    queries_per_cluster: int | None = None,
) -> Glm52ClusteredMqaPlan | None:
    """Build a reusable Q16/Q32 plan, or return None for a safe fallback."""

    lengths = tuple(int(length) for length in extend_lens_cpu)
    total_q = int(seq_lens_expanded.numel())
    if max_total_q is not None and max_total_q <= 0:
        raise ValueError("max_total_q must be positive")
    if queries_per_cluster is None:
        queries_per_cluster = (
            32 if lengths and all(length % 32 == 0 for length in lengths) else 16
        )
    if queries_per_cluster not in SUPPORTED_QUERIES_PER_CLUSTER:
        raise ValueError("queries_per_cluster must be 16 or 32")
    if total_q == 0 or not lengths:
        return None
    if max_total_q is None:
        ordinary_limit = (
            VALIDATED_MAX_M_Q32
            if queries_per_cluster == 32
            else VALIDATED_MAX_M_Q16
        )
        admitted_m512 = (
            queries_per_cluster == 32
            and total_q == 512
            and len(lengths) == VALIDATED_M512_REQUESTS
            and all(length == 32 for length in lengths)
            and max_context >= VALIDATED_M512_MIN_CONTEXT
        )
        if total_q > ordinary_limit and not admitted_m512:
            return None
    elif total_q > max_total_q:
        return None
    if sum(lengths) != total_q or any(
        length <= 0 or length % queries_per_cluster for length in lengths
    ):
        return None
    if total_q % queries_per_cluster:
        return None
    if (
        seq_lens_expanded.dtype != torch.int32
        or token_to_batch_idx.dtype != torch.int32
        or block_tables.dtype != torch.int32
    ):
        raise RuntimeError("clustered MQA metadata tensors must be int32")
    if not (
        seq_lens_expanded.is_cuda
        and token_to_batch_idx.is_cuda
        and block_tables.is_cuda
    ):
        raise RuntimeError("clustered MQA metadata tensors must be CUDA")
    if token_to_batch_idx.numel() != total_q:
        raise RuntimeError("token_to_batch_idx must have one entry per query")
    if block_tables.ndim != 2:
        raise RuntimeError("block_tables must be [batch,max_pages]")
    if block_tables.shape[0] != len(lengths):
        raise RuntimeError("block_tables must have one row per request")
    if max_context <= 0 or max_context > block_tables.shape[1] * PAGE_SIZE:
        raise RuntimeError("max_context exceeds the page-table capacity")

    padded_context = (
        (max_context + LOGITS_ALIGNMENT - 1) // LOGITS_ALIGNMENT
    ) * LOGITS_ALIGNMENT
    bytes_per_row = padded_context * torch.empty((), dtype=torch.float32).element_size()
    max_rows = min(total_q, logits_budget_bytes // bytes_per_row)
    max_rows = (max_rows // queries_per_cluster) * queries_per_cluster
    if max_rows < queries_per_cluster:
        return None

    import deep_gemm

    groups_per_request = torch.tensor(
        [length // queries_per_cluster for length in lengths],
        dtype=torch.int64,
        device=seq_lens_expanded.device,
    )
    group_batch_all = torch.repeat_interleave(
        torch.arange(
            len(lengths), dtype=torch.int64, device=seq_lens_expanded.device
        ),
        groups_per_request,
    )
    chunks: list[ClusteredMqaChunk] = []
    for start in range(0, total_q, max_rows):
        end = min(start + max_rows, total_q)
        rows = end - start
        if rows % queries_per_cluster:
            raise RuntimeError("internal clustered-MQA chunk lost query-group alignment")
        grouped_lens = seq_lens_expanded[start:end].view(
            -1, queries_per_cluster
        )
        group_start = start // queries_per_cluster
        group_end = end // queries_per_cluster
        group_batch = group_batch_all[group_start:group_end]
        grouped_page_table = block_tables.index_select(0, group_batch)
        schedule = deep_gemm.get_paged_mqa_logits_metadata(
            grouped_lens,
            PAGE_SIZE,
            deep_gemm.get_num_sms() // 2,
        )
        chunks.append(
            ClusteredMqaChunk(
                start=start,
                end=end,
                seq_lens=grouped_lens,
                page_table=grouped_page_table,
                schedule=schedule,
            )
        )

    device = seq_lens_expanded.device
    logits_workspace = torch.empty(
        (max_rows, padded_context), dtype=torch.float32, device=device
    )
    return Glm52ClusteredMqaPlan(
        chunks=tuple(chunks),
        logits_workspace=logits_workspace,
        max_context=max_context,
        padded_context=padded_context,
        total_q=total_q,
        queries_per_cluster=queries_per_cluster,
        extension=load_glm52_clustered_mqa_extension(),
    )


def run_glm52_clustered_mqa_chunk(
    *,
    q: torch.Tensor,
    raw_kv_cache: torch.Tensor,
    weights: torch.Tensor,
    plan: Glm52ClusteredMqaPlan,
    chunk: ClusteredMqaChunk,
) -> torch.Tensor:
    rows = chunk.end - chunk.start
    if q.shape != (rows, GLM_HEADS, HEAD_DIM):
        raise RuntimeError(
            f"GLM clustered MQA Q must be [{rows},32,128], got {tuple(q.shape)}"
        )
    if q.dtype != torch.float8_e4m3fn:
        raise RuntimeError(f"GLM clustered MQA Q must be e4m3fn, got {q.dtype}")
    if weights.shape != (rows, GLM_HEADS) or weights.dtype != torch.float32:
        raise RuntimeError("GLM clustered MQA weights must be [rows,32] float32")
    if raw_kv_cache.dtype != torch.uint8 or raw_kv_cache.ndim != 2:
        raise RuntimeError("GLM clustered MQA raw KV must be 2D uint8 pages")
    if raw_kv_cache.shape[1] != PAGE_BYTES or not raw_kv_cache.is_contiguous():
        raise RuntimeError(
            f"GLM clustered MQA KV pages must be contiguous [{PAGE_BYTES}] rows"
        )

    if not q.is_contiguous() or not weights.is_contiguous():
        raise RuntimeError("GLM clustered MQA Q and weights must be contiguous")
    fused_kv = raw_kv_cache.view(-1, PAGE_SIZE, 1, HEAD_DIM + 4)
    logits_full = plan.logits_workspace[:rows]
    return plan.extension.forward_out(
        q,
        fused_kv,
        weights,
        chunk.seq_lens,
        chunk.page_table,
        chunk.schedule,
        logits_full,
        plan.max_context,
        plan.queries_per_cluster,
    )


__all__ = [
    "ClusteredMqaChunk",
    "Glm52ClusteredMqaPlan",
    "load_glm52_clustered_mqa_extension",
    "prepare_glm52_clustered_mqa_plan",
    "run_glm52_clustered_mqa_chunk",
    "VALIDATED_MAX_M",
    "VALIDATED_MAX_M_Q16",
    "VALIDATED_MAX_M_Q32",
    "VALIDATED_M512_REQUESTS",
    "VALIDATED_M512_MIN_CONTEXT",
    "SUPPORTED_QUERIES_PER_CLUSTER",
]
