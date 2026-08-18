#!/usr/bin/env python3
"""Fail-closed 8-rank precondition for SGLang's multimem all-gather.

The registered SGLang test imports pytest, which is intentionally absent from
the frozen serving venv.  This dependency-free runner exercises the same
production ``create_state`` and ``all_gather_inner`` functions.  It verifies
that B300 exposes a non-zero multicast pointer and that repeated, mutated BF16
inputs remain byte-exact against NCCL for every safe/entry-sync combination.

This proves only the symmetric-memory transport precondition.  It does not
promote a CP row-scatter or serving candidate.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.srt.distributed.device_communicators.triton_symm_mem_ag import (
    all_gather_inner,
    create_state,
)


def _rank_max_ms(
    local_ms: list[float], world: int, group: dist.ProcessGroup
) -> list[float]:
    local = torch.tensor(local_ms, device="cuda", dtype=torch.float64)
    gathered = torch.empty((world, len(local_ms)), device="cuda", dtype=torch.float64)
    dist.all_gather_into_tensor(gathered, local, group=group)
    return gathered.max(dim=0).values.cpu().tolist()


def _time_one(fn, group: dist.ProcessGroup) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    dist.barrier(group)
    start.record()
    out = fn()
    end.record()
    end.synchronize()
    if out.numel() == 0:
        raise RuntimeError("unexpected empty output")
    return float(start.elapsed_time(end))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--loops", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    # Match the production registered test: use Gloo as the launcher/default
    # group and let SGLang create/register the NCCL device group used by the
    # symmetric-memory allocator and rendezvous protocol.
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world != 8:
        raise RuntimeError(f"requires exactly eight ranks, got {world}")
    if args.hidden % world != 0 or (args.hidden // world) % 8 != 0:
        raise RuntimeError("hidden width must yield an 8-BF16-aligned local shard")

    ps._WORLD = ps.init_world_group(
        ranks=list(range(world)),
        local_rank=local_rank,
        backend="nccl",
    )
    group = ps._WORLD.device_group
    if group is None:
        raise RuntimeError("SGLang NCCL device group was not created")
    state = create_state(
        group=group,
        rank_in_group=rank,
        max_tokens=args.tokens,
        hidden_size=args.hidden,
    )
    multicast_available = int(state.symm_mem_hdl.multicast_ptr != 0)
    multicast_all = torch.tensor([multicast_available], device="cuda", dtype=torch.int32)
    dist.all_reduce(multicast_all, op=dist.ReduceOp.MIN, group=group)
    if multicast_all.item() != 1:
        raise RuntimeError("multimem multicast pointer is zero on at least one rank")

    local_hidden = args.hidden // world
    nccl_buffer = torch.empty(
        (world * args.tokens, local_hidden),
        dtype=torch.bfloat16,
        device="cuda",
    )
    cases = []
    for safe in (False, True):
        for skip_entry_sync in (False, True):
            nccl_ms: list[float] = []
            multimem_ms: list[float] = []
            all_equal = True
            for iteration in range(args.loops + 4):
                generator = torch.Generator(device="cuda")
                generator.manual_seed(20260818 + rank * 1009 + iteration)
                x = torch.randn(
                    (args.tokens, local_hidden),
                    dtype=torch.bfloat16,
                    device="cuda",
                    generator=generator,
                )

                def nccl():
                    dist.all_gather_into_tensor(nccl_buffer, x, group=group)
                    return (
                        nccl_buffer.view(world, args.tokens, local_hidden)
                        .movedim(0, 1)
                        .reshape(args.tokens, args.hidden)
                    )

                def multimem():
                    return all_gather_inner(
                        state,
                        x,
                        tp_hidden_dim=args.hidden,
                        skip_entry_sync=skip_entry_sync,
                        safe=safe,
                    )

                # Correctness is checked before either output buffer can be reused.
                dist.barrier(group)
                reference = nccl().clone()
                candidate = multimem().clone()
                torch.cuda.synchronize()
                equal = torch.tensor(
                    [int(torch.equal(reference, candidate))],
                    device="cuda",
                    dtype=torch.int32,
                )
                dist.all_reduce(equal, op=dist.ReduceOp.MIN, group=group)
                all_equal = all_equal and bool(equal.item())

                if iteration >= 4:
                    if iteration % 2 == 0:
                        nccl_ms.append(_time_one(nccl, group))
                        multimem_ms.append(_time_one(multimem, group))
                    else:
                        multimem_ms.append(_time_one(multimem, group))
                        nccl_ms.append(_time_one(nccl, group))

            if not all_equal:
                raise RuntimeError(
                    f"byte mismatch for safe={safe} skip_entry_sync={skip_entry_sync}"
                )
            nccl_wall = _rank_max_ms(nccl_ms, world, group)
            multimem_wall = _rank_max_ms(multimem_ms, world, group)
            cases.append(
                {
                    "safe": safe,
                    "skip_entry_sync": skip_entry_sync,
                    "byte_exact_all_ranks_all_iterations": all_equal,
                    "nccl_rank_max_median_ms": statistics.median(nccl_wall),
                    "multimem_rank_max_median_ms": statistics.median(multimem_wall),
                    "median_speedup": statistics.median(
                        b / c for b, c in zip(nccl_wall, multimem_wall)
                    ),
                    "nccl_rank_max_ms": nccl_wall,
                    "multimem_rank_max_ms": multimem_wall,
                }
            )

    result = {
        "schema": "sglang-multimem-precondition-v1",
        "status": "PASS",
        "scope": "transport precondition only; not CP row-scatter or serving evidence",
        "gpu": torch.cuda.get_device_name(local_rank),
        "world_size": world,
        "tokens": args.tokens,
        "hidden": args.hidden,
        "loops": args.loops,
        "multicast_nonzero_all_ranks": True,
        "cases": cases,
    }
    if rank == 0:
        rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
        print(rendered, end="", flush=True)
        if args.output is not None:
            args.output.write_text(rendered)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
