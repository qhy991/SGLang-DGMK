# SPDX-License-Identifier: Apache-2.0
"""B300 CP8 direct-layout multicast for GLM-5.2 packed MLA-KV rows.

Each context-parallel rank owns two chunks of a single zigzag-split sequence.
This kernel multicasts the rank-local final cache rows directly into their
global zigzag positions, eliminating NCCL rank-major AllGather and the
subsequent split/cat rerange.  The output remains the existing uint8[M,656]
CUDA DSA FP8-cache ABI and is consumed by the existing page-store kernel.

The public entry point is deliberately strict: it accepts one sequence,
exactly two local zigzag blocks, CP8, SM103, eager execution, and a final row
count within the preallocated symmetric buffer.  Unsupported inputs raise;
the selected path never silently falls back to NCCL.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import accumulate
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

from sglang.srt.distributed.device_communicators.triton_symm_mem_ag import (
    _blockwise_barrier,
    _get_flat_tid,
    _local_ld_128,
    _multimem_st_128,
    _sync_threads,
)


_BLOCKS = 32
_BLOCK_THREADS = 1024
_CHUNK_BYTES = 16
_ROW_BYTES = 656
_ROW_CHUNKS = _ROW_BYTES // _CHUNK_BYTES


@dataclass
class _DirectPackedKVState:
    group: dist.ProcessGroup
    rank: int
    world: int
    device: torch.device
    max_global_rows: int
    buffer: torch.Tensor
    handle: Any


_state: _DirectPackedKVState | None = None


@triton.jit
def _direct_two_block_scatter_kernel(
    input_ptr,
    multicast_ptr,
    signal_pad_ptr,
    local_rows,
    first_rows,
    first_global_start,
    second_global_start,
    ROW_CHUNKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
):
    _blockwise_barrier(signal_pad_ptr, RANK, WORLD, sem="relaxed")
    _sync_threads()

    total_chunks = local_rows * ROW_CHUNKS
    chunk = tl.program_id(0) * BLOCK_SIZE + _get_flat_tid()
    stride = tl.num_programs(0) * BLOCK_SIZE
    while chunk < total_chunks:
        mask = chunk < total_chunks
        local_row = chunk // ROW_CHUNKS
        row_chunk = chunk % ROW_CHUNKS
        in_first = local_row < first_rows
        global_row = tl.where(
            in_first,
            first_global_start + local_row,
            second_global_start + local_row - first_rows,
        )

        source = input_ptr.to(tl.pointer_type(tl.uint64)) + chunk * 2
        destination_chunk = global_row * ROW_CHUNKS + row_chunk
        destination = (
            multicast_ptr.to(tl.int64).to(tl.pointer_type(tl.uint64))
            + destination_chunk * 2
        )
        x, y, z, w = _local_ld_128(source, mask)
        _multimem_st_128(destination, x, y, z, w, mask)
        chunk += stride

    _sync_threads()
    _blockwise_barrier(signal_pad_ptr, RANK, WORLD, sem="acq_rel")


def _get_or_create_state(
    group: dist.ProcessGroup,
    *,
    rank: int,
    world: int,
    device: torch.device,
    max_global_rows: int,
) -> _DirectPackedKVState:
    global _state

    if _state is not None:
        if (
            _state.group is not group
            or _state.rank != rank
            or _state.world != world
            or _state.device != device
            or _state.max_global_rows != max_global_rows
        ):
            raise RuntimeError("direct packed MLA-KV symmetric state contract changed")
        return _state

    signal_bytes = _BLOCKS * world * 4
    symm_mem.set_signal_pad_size(max(symm_mem.get_signal_pad_size(), signal_bytes))
    with torch.inference_mode(False), torch.no_grad():
        buffer = symm_mem.empty(
            (max_global_rows, _ROW_BYTES), dtype=torch.uint8, device=device
        )
    handle = symm_mem.rendezvous(buffer, group=group)
    if handle.rank != rank or handle.world_size != world:
        raise RuntimeError(
            f"symmetric handle mismatch: rank={handle.rank}/{rank} "
            f"world={handle.world_size}/{world}"
        )
    if handle.multicast_ptr == 0:
        raise RuntimeError("direct packed MLA-KV multicast pointer is zero")
    _state = _DirectPackedKVState(
        group=group,
        rank=rank,
        world=world,
        device=device,
        max_global_rows=max_global_rows,
        buffer=buffer,
        handle=handle,
    )
    return _state


def direct_zigzag_packed_mla_kv_all_gather(
    packed_local: torch.Tensor,
    metadata: Any,
    *,
    group: dist.ProcessGroup,
    rank: int,
    world: int,
    max_global_rows: int,
) -> torch.Tensor:
    """Multicast two local zigzag blocks into final global row order."""

    if torch.cuda.get_device_capability(packed_local.device) != (10, 3):
        raise RuntimeError("direct packed MLA-KV requires NVIDIA SM103")
    if world != 8:
        raise RuntimeError(f"direct packed MLA-KV requires CP8, got {world}")
    if (
        packed_local.dtype is not torch.uint8
        or packed_local.ndim != 2
        or packed_local.shape[1] != _ROW_BYTES
        or not packed_local.is_contiguous()
        or packed_local.data_ptr() % _CHUNK_BYTES != 0
    ):
        raise RuntimeError(
            "direct packed MLA-KV requires aligned contiguous uint8[M,656]; "
            f"got dtype={packed_local.dtype} shape={tuple(packed_local.shape)} "
            f"contiguous={packed_local.is_contiguous()} "
            f"alignment={packed_local.data_ptr() % _CHUNK_BYTES}"
        )
    metadata_bs = getattr(metadata, "bs", None)
    if metadata_bs != 1:
        raise RuntimeError(
            f"direct packed MLA-KV requires metadata.bs=1: got {metadata_bs}"
        )
    split_list = getattr(metadata, "split_list", None)
    zigzag_index = getattr(metadata, "zigzag_index", None)
    if split_list is None or len(split_list) != 2 * world:
        raise RuntimeError(
            f"direct packed MLA-KV requires {2 * world} zigzag split lengths"
        )
    if zigzag_index is None or len(zigzag_index) != 2:
        raise RuntimeError("direct packed MLA-KV requires exactly two local blocks")
    expected_indices = [rank, 2 * world - 1 - rank]
    if list(zigzag_index) != expected_indices:
        raise RuntimeError(
            f"unexpected zigzag ownership: got={zigzag_index} expected={expected_indices}"
        )

    global_rows = sum(int(value) for value in split_list)
    first_index, second_index = expected_indices
    first_rows = int(split_list[first_index])
    second_rows = int(split_list[second_index])
    if first_rows + second_rows != packed_local.shape[0]:
        raise RuntimeError(
            "local packed rows do not match owned zigzag blocks: "
            f"local={packed_local.shape[0]} blocks={first_rows}+{second_rows}"
        )
    if global_rows <= 0 or global_rows > max_global_rows:
        raise RuntimeError(
            f"global rows {global_rows} exceed symmetric capacity {max_global_rows}"
        )
    starts = [0] + list(accumulate(int(value) for value in split_list))

    state = _get_or_create_state(
        group,
        rank=rank,
        world=world,
        device=packed_local.device,
        max_global_rows=max_global_rows,
    )
    _direct_two_block_scatter_kernel[(_BLOCKS, 1, 1)](
        packed_local,
        state.handle.multicast_ptr,
        state.handle.signal_pad_ptrs_dev,
        packed_local.shape[0],
        first_rows,
        starts[first_index],
        starts[second_index],
        ROW_CHUNKS=_ROW_CHUNKS,
        BLOCK_SIZE=_BLOCK_THREADS,
        RANK=rank,
        WORLD=world,
        num_warps=_BLOCK_THREADS // 32,
    )
    return state.buffer[:global_rows]
