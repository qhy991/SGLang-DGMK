#!/usr/bin/env python3
"""Delayed-rank and single-slot reuse stress for direct packed MLA-KV."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.jit_kernel.cp8_packed_kv_direct_scatter import (
    direct_zigzag_packed_mla_kv_all_gather,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-m", type=int, default=10_048)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--delay-rank", type=int, default=7)
    parser.add_argument("--delay-ms", type=float, default=50.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 8 or args.local_m % 2:
        raise RuntimeError("stress requires CP8 and an even local M")
    ps._WORLD = ps.init_world_group(
        ranks=list(range(world)), local_rank=local_rank, backend="nccl"
    )
    group = ps._WORLD.device_group
    if group is None:
        raise RuntimeError("SGLang NCCL group is missing")

    half = args.local_m // 2
    global_m = args.local_m * world
    metadata = SimpleNamespace(
        bs=1,
        split_list=[half] * (2 * world),
        zigzag_index=[rank, 2 * world - 1 - rank],
    )
    inverse = [-1] * global_m
    for owner in range(world):
        owner_rows = list(range(owner * half, (owner + 1) * half)) + list(
            range((2 * world - 1 - owner) * half, (2 * world - owner) * half)
        )
        for local_row, global_row in enumerate(owner_rows):
            inverse[global_row] = owner * args.local_m + local_row
    inverse_tensor = torch.tensor(inverse, dtype=torch.int64, device="cuda")
    gathered = torch.empty((global_m, 656), dtype=torch.uint8, device="cuda")

    all_equal = True
    delayed_epochs = []
    output_hashes = []
    for epoch in range(args.epochs):
        generator = torch.Generator(device="cuda")
        generator.manual_seed(20260818 + rank * 1009 + epoch)
        local = torch.randint(
            0,
            256,
            (args.local_m, 656),
            dtype=torch.uint8,
            device="cuda",
            generator=generator,
        )
        dist.all_gather_into_tensor(gathered, local, group=group)
        reference = gathered.index_select(0, inverse_tensor)

        if epoch % 5 == 2 and rank == args.delay_rank:
            delayed_epochs.append(epoch)
            time.sleep(args.delay_ms / 1_000.0)
        direct = direct_zigzag_packed_mla_kv_all_gather(
            local,
            metadata,
            group=group,
            rank=rank,
            world=world,
            max_global_rows=global_m,
        )
        # Clone models the existing page-store consumer completing on the same
        # stream before the next epoch reuses the one symmetric slot.
        consumed = direct.clone()
        torch.cuda.synchronize()
        equal = torch.tensor(
            [int(torch.equal(reference, consumed))],
            dtype=torch.int32,
            device="cuda",
        )
        dist.all_reduce(equal, op=dist.ReduceOp.MIN, group=group)
        all_equal = all_equal and bool(equal.item())
        if rank == 0:
            output_hashes.append(
                hashlib.sha256(consumed.cpu().numpy().tobytes()).hexdigest()
            )

    unique_outputs_ok = rank != 0 or len(set(output_hashes)) == args.epochs
    if not all_equal or not unique_outputs_ok:
        raise RuntimeError(
            f"direct stress failed: rank={rank} equal={all_equal} "
            f"unique={len(set(output_hashes))}"
        )
    result = {
        "schema": "glm52-cp8-direct-packed-delayed-rank-stress-v1",
        "status": "PASS",
        "world_size": world,
        "local_m": args.local_m,
        "global_m": global_m,
        "epochs": args.epochs,
        "delay_rank": args.delay_rank,
        "delay_ms": args.delay_ms,
        "delayed_epochs": delayed_epochs if rank == args.delay_rank else list(range(2, args.epochs, 5)),
        "byte_exact_all_ranks_all_epochs": all_equal,
        "unique_output_hashes": len(set(output_hashes)) if rank == 0 else args.epochs,
        "single_symmetric_slot_reused": True,
    }
    if rank == 0:
        rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
        print(rendered, end="", flush=True)
        if args.output:
            args.output.write_text(rendered)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
