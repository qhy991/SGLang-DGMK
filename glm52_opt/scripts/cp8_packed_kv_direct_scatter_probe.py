#!/usr/bin/env python3
"""CP8 operator screen for packed MLA-KV direct-layout multicast.

The target is the real GLM-5.2 MLA cache producer shape observed under the
90K-cache + 10K-input CP8 serving workload: local BF16 ``[10048, 576]`` and a
final packed cache row of 656 bytes.  For one zigzag-split sequence, rank ``r``
owns block ``r`` followed by block ``15-r``.  The candidate writes those two
local blocks directly into their final global rows through ``multimem.st``.

Compared boundaries:

* production control: BF16 NCCL AllGather -> zigzag rerange -> global pack;
* packed NCCL: local pack -> NCCL AllGather -> zigzag rerange;
* direct layout: local pack -> multicast stores into final zigzag rows.

This is an eager operator boundary test.  It is not serving promotion evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

import sglang.srt.distributed.parallel_state as ps
from sglang.srt.distributed.device_communicators.triton_symm_mem_ag import (
    _blockwise_barrier,
    _get_flat_tid,
    _local_ld_128,
    _multimem_st_128,
    _sync_threads,
)
from sglang.srt.layers.attention.dsa.quant_k_cache import (
    quantize_k_cache_separate,
)


BLOCKS = 32
BLOCK_THREADS = 1024
CHUNK_BYTES = 16
ROW_BYTES = 656


@dataclass
class DirectState:
    comm_buffer: torch.Tensor
    handle: object
    max_global_m: int
    row_bytes: int
    rank: int
    world: int


@triton.jit
def _direct_zigzag_row_scatter_kernel(
    input_ptr,
    row_map_ptr,
    multicast_ptr,
    signal_pad_ptr,
    local_m,
    ROW_CHUNKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
):
    _blockwise_barrier(signal_pad_ptr, RANK, WORLD, sem="relaxed")
    _sync_threads()

    total_chunks = local_m * ROW_CHUNKS
    pid = tl.program_id(0)
    tid = _get_flat_tid()
    chunk = pid * BLOCK_SIZE + tid
    stride = tl.num_programs(0) * BLOCK_SIZE
    while chunk < total_chunks:
        mask = chunk < total_chunks
        local_row = chunk // ROW_CHUNKS
        row_chunk = chunk % ROW_CHUNKS
        global_row = tl.load(row_map_ptr + local_row, mask=mask, other=0)

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


def _create_direct_state(
    group: dist.ProcessGroup,
    rank: int,
    world: int,
    global_m: int,
) -> DirectState:
    if ROW_BYTES % CHUNK_BYTES != 0:
        raise AssertionError("packed row must be 16-byte aligned")
    signal_bytes = BLOCKS * world * 4
    symm_mem.set_signal_pad_size(max(symm_mem.get_signal_pad_size(), signal_bytes))
    with torch.inference_mode(False), torch.no_grad():
        comm_buffer = symm_mem.empty(
            (global_m, ROW_BYTES), dtype=torch.uint8, device="cuda"
        )
    handle = symm_mem.rendezvous(comm_buffer, group=group)
    if handle.rank != rank or handle.world_size != world:
        raise RuntimeError(
            f"symmetric handle mismatch: rank={handle.rank}/{rank} "
            f"world={handle.world_size}/{world}"
        )
    if handle.multicast_ptr == 0:
        raise RuntimeError("multimem multicast pointer is zero")
    return DirectState(comm_buffer, handle, global_m, ROW_BYTES, rank, world)


def _pack_mla(kv: torch.Tensor) -> torch.Tensor:
    nope, rope = quantize_k_cache_separate(kv[:, :512], kv[:, 512:])
    packed = torch.cat(
        (nope.reshape(kv.shape[0], -1), rope.reshape(kv.shape[0], -1)), dim=-1
    )
    if packed.dtype is not torch.uint8 or packed.shape[1] != ROW_BYTES:
        raise RuntimeError(
            f"unexpected MLA packed ABI: dtype={packed.dtype} shape={packed.shape}"
        )
    return packed


def _zigzag_maps(local_m: int, rank: int, world: int, device) -> tuple[torch.Tensor, list[int]]:
    if local_m % 2 != 0:
        raise ValueError("single-sequence zigzag local M must contain two equal blocks")
    half = local_m // 2
    local_to_global = list(range(rank * half, (rank + 1) * half)) + list(
        range((2 * world - 1 - rank) * half, (2 * world - rank) * half)
    )
    global_m = local_m * world
    inverse = [-1] * global_m
    for owner in range(world):
        owner_map = list(range(owner * half, (owner + 1) * half)) + list(
            range(
                (2 * world - 1 - owner) * half,
                (2 * world - owner) * half,
            )
        )
        for local_row, global_row in enumerate(owner_map):
            inverse[global_row] = owner * local_m + local_row
    if sorted(local_to_global) != local_to_global or sorted(inverse) != list(
        range(global_m)
    ):
        # local_to_global has two monotonic segments, not one global monotonic list.
        if sorted(inverse) != list(range(global_m)):
            raise AssertionError("zigzag inverse is not a permutation")
    return torch.tensor(local_to_global, device=device, dtype=torch.int64), inverse


def _direct_scatter(
    state: DirectState, packed_local: torch.Tensor, row_map: torch.Tensor
) -> torch.Tensor:
    if (
        packed_local.dtype is not torch.uint8
        or not packed_local.is_contiguous()
        or packed_local.ndim != 2
        or packed_local.shape[1] != state.row_bytes
        or row_map.shape != (packed_local.shape[0],)
    ):
        raise ValueError(
            f"direct scatter requires contiguous uint8[M,{state.row_bytes}] and map[M]; "
            f"got packed={packed_local.dtype}/{tuple(packed_local.shape)} "
            f"map={tuple(row_map.shape)}"
        )
    if packed_local.data_ptr() % CHUNK_BYTES != 0:
        raise ValueError("packed input is not 16-byte aligned")
    _direct_zigzag_row_scatter_kernel[(BLOCKS, 1, 1)](
        packed_local,
        row_map,
        state.handle.multicast_ptr,
        state.handle.signal_pad_ptrs_dev,
        packed_local.shape[0],
        ROW_CHUNKS=state.row_bytes // CHUNK_BYTES,
        BLOCK_SIZE=BLOCK_THREADS,
        RANK=state.rank,
        WORLD=state.world,
        num_warps=BLOCK_THREADS // 32,
    )
    return state.comm_buffer[: state.max_global_m]


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - position) + ordered[hi] * (position - lo)


def _time_one(fn: Callable[[], torch.Tensor], group: dist.ProcessGroup) -> float:
    dist.barrier(group)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    end.synchronize()
    if out.numel() == 0:
        raise RuntimeError("unexpected empty output")
    return float(start.elapsed_time(end))


def _rank_max(local: list[float], world: int, group: dist.ProcessGroup) -> list[float]:
    value = torch.tensor(local, dtype=torch.float64, device="cuda")
    gathered = torch.empty((world, len(local)), dtype=torch.float64, device="cuda")
    dist.all_gather_into_tensor(gathered, value, group=group)
    return gathered.max(dim=0).values.cpu().tolist()


def _paired_case(
    name: str,
    baseline: Callable[[], torch.Tensor],
    candidate: Callable[[], torch.Tensor],
    group: dist.ProcessGroup,
    world: int,
    warmup: int,
    repeat: int,
) -> dict:
    reference = baseline().clone()
    result = candidate().clone()
    torch.cuda.synchronize()
    equal = torch.tensor(
        [int(torch.equal(reference, result))], device="cuda", dtype=torch.int32
    )
    dist.all_reduce(equal, op=dist.ReduceOp.MIN, group=group)
    if not equal.item():
        raise RuntimeError(f"{name}: candidate is not byte-exact")

    for iteration in range(warmup):
        if iteration % 2 == 0:
            baseline()
            candidate()
        else:
            candidate()
            baseline()
    torch.cuda.synchronize()

    baseline_ms: list[float] = []
    candidate_ms: list[float] = []
    order: list[str] = []
    for iteration in range(repeat + 1):
        if iteration % 2 == 0:
            order.append("BC")
            baseline_ms.append(_time_one(baseline, group))
            candidate_ms.append(_time_one(candidate, group))
        else:
            order.append("CB")
            candidate_ms.append(_time_one(candidate, group))
            baseline_ms.append(_time_one(baseline, group))

    baseline_wall = _rank_max(baseline_ms[1:], world, group)
    candidate_wall = _rank_max(candidate_ms[1:], world, group)
    speedups = [b / c for b, c in zip(baseline_wall, candidate_wall)]
    return {
        "name": name,
        "byte_exact_all_ranks": True,
        "pair_order_after_discard": order[1:],
        "baseline_rank_max_ms": baseline_wall,
        "candidate_rank_max_ms": candidate_wall,
        "baseline_median_ms": statistics.median(baseline_wall),
        "candidate_median_ms": statistics.median(candidate_wall),
        "median_speedup": statistics.median(speedups),
        "speedup_p10": _percentile(speedups, 0.10),
        "speedup_p90": _percentile(speedups, 0.90),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-m", type=int, default=10048)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 8:
        raise RuntimeError(f"requires CP8, got world={world}")
    ps._WORLD = ps.init_world_group(
        ranks=list(range(world)), local_rank=local_rank, backend="nccl"
    )
    group = ps._WORLD.device_group
    if group is None:
        raise RuntimeError("SGLang NCCL device group was not created")

    device = torch.device(f"cuda:{local_rank}")
    global_m = args.local_m * world
    row_map, inverse = _zigzag_maps(args.local_m, rank, world, device)
    inverse_tensor = torch.tensor(inverse, device=device, dtype=torch.int64)
    direct_state = _create_direct_state(group, rank, world, global_m)

    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260818 + rank)
    kv_local = torch.randn(
        (args.local_m, 576),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    bf16_rank_major = torch.empty(
        (global_m, 576), dtype=torch.bfloat16, device=device
    )
    packed_rank_major = torch.empty(
        (global_m, ROW_BYTES), dtype=torch.uint8, device=device
    )

    def production_control() -> torch.Tensor:
        dist.all_gather_into_tensor(bf16_rank_major, kv_local, group=group)
        reranged = bf16_rank_major.index_select(0, inverse_tensor)
        return _pack_mla(reranged)

    def packed_nccl() -> torch.Tensor:
        packed = _pack_mla(kv_local)
        dist.all_gather_into_tensor(packed_rank_major, packed, group=group)
        return packed_rank_major.index_select(0, inverse_tensor)

    def direct_layout() -> torch.Tensor:
        return _direct_scatter(direct_state, _pack_mla(kv_local), row_map)

    cases = [
        _paired_case(
            "production_bf16_nccl_rerange_pack_vs_direct_packed_multicast",
            production_control,
            direct_layout,
            group,
            world,
            args.warmup,
            args.repeat,
        ),
        _paired_case(
            "packed_nccl_rerange_vs_direct_packed_multicast",
            packed_nccl,
            direct_layout,
            group,
            world,
            args.warmup,
            args.repeat,
        ),
    ]
    result = {
        "schema": "glm52-cp8-packed-kv-direct-scatter-v1",
        "status": "PASS",
        "classification": "source/runtime graph with inline PTX transport",
        "scope": "operator boundary only; serving integration still required",
        "gpu": torch.cuda.get_device_name(local_rank),
        "world_size": world,
        "local_m": args.local_m,
        "global_m": global_m,
        "row_bytes": ROW_BYTES,
        "payload_bytes_per_rank": args.local_m * ROW_BYTES,
        "zigzag_blocks_per_rank": 2,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "timing": "CUDA event, rank max, balanced BC/CB, first pair discarded",
        "cases": cases,
        "promotion_boundary": "requires independent replicate, fail-closed integration, exact x1, and paired x11 serving",
    }
    if rank == 0:
        rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
        print(rendered, end="", flush=True)
        if args.output is not None:
            args.output.write_text(rendered)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
