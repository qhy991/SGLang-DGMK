#!/usr/bin/env python3

"""GLM-5.2 production-shape FlashMLA sparse-decode validation and timing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from flashmla_glm52_contract import (
    DISPATCH_STATE_NAMES,
    flat_index_dispatch_hit,
    production_abi_failures,
)
from sgl_kernel import flashmla_ops
from sgl_kernel.flash_mla import flash_mla_with_kvcache, get_mla_metadata


OUTPUT_ATOL = 8.0e-4
OUTPUT_RTOL = 2.01 / 128
LSE_ATOL = 1.0e-6
LSE_RTOL = 8.01 / 65536
HEADS_Q = 64
HEAD_DIM_QK = 576
HEAD_DIM_V = 512
PACKED_ROW_BYTES = 656
LOCAL_HEADS_Q = 8


@dataclass
class Fixture:
    batch_size: int
    heads_q: int
    local_heads_q: int
    valid_lengths: tuple[int, ...]
    topk: int
    page_size: int
    kv_block_padding_rows: int
    row_cache_tokens: int
    q: torch.Tensor
    packed_kv: torch.Tensor
    dequant_kv: torch.Tensor
    indices: torch.Tensor
    block_table: torch.Tensor
    cache_seqlens: torch.Tensor
    metadata: torch.Tensor
    num_splits: torch.Tensor


def _sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_fingerprint(tensor: torch.Tensor) -> dict[str, object]:
    contiguous = tensor.detach().contiguous().cpu()
    raw = contiguous.view(torch.uint8).numpy().tobytes()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "stride": list(tensor.stride()),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _quantize_v32_kv(input_kv: torch.Tensor) -> torch.Tensor:
    """Pack BF16 [nope512, rope64] into FlashMLA's 656-byte V3.2 row."""
    num_blocks, page_size, h_k, d_qk = input_kv.shape
    assert h_k == 1 and d_qk == HEAD_DIM_QK
    source = input_kv.squeeze(2)
    packed = torch.empty(
        (num_blocks, page_size, PACKED_ROW_BYTES),
        dtype=torch.float8_e4m3fn,
        device=source.device,
    )
    packed_nope = packed[..., :HEAD_DIM_V]
    packed_scales = packed[..., HEAD_DIM_V : HEAD_DIM_V + 16].view(torch.float32)
    packed_rope = packed[..., HEAD_DIM_V + 16 :].view(torch.bfloat16)
    packed_rope.copy_(source[..., HEAD_DIM_V:])
    for tile in range(4):
        values = source[..., tile * 128 : (tile + 1) * 128]
        inverse_scale = (values.abs().amax(dim=-1) / 448.0).clamp_min_(1.0e-8)
        packed_scales[..., tile].copy_(inverse_scale)
        packed_nope[..., tile * 128 : (tile + 1) * 128].copy_(
            (values.float() / inverse_scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
        )
    return packed.view(num_blocks, page_size, 1, PACKED_ROW_BYTES)


def _dequantize_v32_kv(packed: torch.Tensor) -> torch.Tensor:
    num_blocks, page_size, h_k, row_bytes = packed.shape
    assert h_k == 1 and row_bytes == PACKED_ROW_BYTES
    flat = packed.view(num_blocks, page_size, row_bytes)
    output = torch.empty(
        (num_blocks, page_size, 1, HEAD_DIM_QK),
        dtype=torch.bfloat16,
        device=packed.device,
    )
    output_flat = output.squeeze(2)
    scales = flat[..., HEAD_DIM_V : HEAD_DIM_V + 16].view(torch.float32)
    output_flat[..., HEAD_DIM_V:].copy_(
        flat[..., HEAD_DIM_V + 16 :].view(torch.bfloat16)
    )
    for tile in range(4):
        output_flat[..., tile * 128 : (tile + 1) * 128].copy_(
            flat[..., tile * 128 : (tile + 1) * 128].float()
            * scales[..., tile].unsqueeze(-1)
        )
    return output


def production_lengths(batch_size: int) -> tuple[int, ...]:
    """Deterministically span the captured 512..575 decode-length window."""
    if batch_size == 1:
        return (575,)
    return tuple(
        512 + round(index * 63 / (batch_size - 1)) for index in range(batch_size)
    )


def resolve_lengths(
    batch_size: int,
    active_tokens: int,
    length_pattern: str,
) -> tuple[int, ...]:
    if length_pattern == "production":
        return production_lengths(batch_size)
    if length_pattern == "uniform":
        return (active_tokens,) * batch_size
    if length_pattern == "mixed":
        boundary = (0, 1, 63, 64, 127, 511, 512, 575)
        return tuple(boundary[index % len(boundary)] for index in range(batch_size))
    raise ValueError(f"unknown length pattern: {length_pattern}")


def make_fixture(
    batch_size: int,
    heads_q: int,
    local_heads_q: int,
    valid_lengths: tuple[int, ...],
    topk: int,
    page_size: int,
    kv_block_padding_rows: int,
    seed: int,
) -> Fixture:
    if len(valid_lengths) != batch_size:
        raise ValueError("valid_lengths must have one entry per batch row")
    if topk <= 0 or topk % 64:
        raise ValueError("topk must be a positive multiple of 64")
    if kv_block_padding_rows < 0:
        raise ValueError("kv_block_padding_rows must be nonnegative")
    if any(length < 0 or length > topk for length in valid_lengths):
        raise ValueError("valid lengths must be within [0, topk]")
    if heads_q not in (64, 128) or not 0 < local_heads_q <= heads_q:
        raise ValueError("heads_q/local_heads_q must describe a supported head count")
    torch.manual_seed(seed)
    max_valid = max(max(valid_lengths), 1)
    row_cache_tokens = max(page_size, math.ceil(max_valid / page_size) * page_size)
    blocks_per_batch = row_cache_tokens // page_size
    num_blocks = batch_size * blocks_per_batch

    q = torch.zeros(
        batch_size,
        1,
        heads_q,
        HEAD_DIM_QK,
        dtype=torch.bfloat16,
        device="cuda",
    )
    q[:, :, :local_heads_q] = (
        torch.randn(
            batch_size,
            1,
            local_heads_q,
            HEAD_DIM_QK,
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 10
    ).clamp_(-1, 1)
    unpacked_kv = (
        torch.randn(
            num_blocks,
            page_size,
            1,
            HEAD_DIM_QK,
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 10
    ).clamp_(-1, 1)
    packed_kv = _quantize_v32_kv(unpacked_kv)
    if kv_block_padding_rows:
        padded_kv = torch.empty(
            (
                num_blocks,
                page_size + kv_block_padding_rows,
                1,
                PACKED_ROW_BYTES,
            ),
            dtype=packed_kv.dtype,
            device=packed_kv.device,
        )
        padded_kv[:, :page_size].copy_(packed_kv)
        packed_kv = padded_kv[:, :page_size]
    dequant_kv = _dequantize_v32_kv(packed_kv)

    indices = torch.full(
        (batch_size, 1, topk),
        -1,
        dtype=torch.int32,
        device="cuda",
    )
    for batch_idx, valid_length in enumerate(valid_lengths):
        if valid_length == 0:
            continue
        row_offset = batch_idx * row_cache_tokens
        physical = (
            torch.randperm(row_cache_tokens, device="cuda", dtype=torch.int64)[
                :valid_length
            ]
            + row_offset
        )
        positions = torch.randperm(topk, device="cuda")[:valid_length]
        indices[batch_idx, 0, positions] = physical.to(torch.int32)

    cache_seqlens = torch.tensor(
        [max(length, 1) for length in valid_lengths],
        dtype=torch.int32,
        device="cuda",
    )
    metadata, num_splits = get_mla_metadata(
        cache_seqlens=cache_seqlens,
        num_q_tokens_per_head_k=heads_q,
        num_heads_k=1,
        num_heads_q=heads_q,
        is_fp8_kvcache=True,
        topk=topk,
    )
    # This empty tensor is the exact production ABI. Sparse physical indices
    # already address the cache, so the operator does not consume a page table.
    block_table = torch.empty(
        (batch_size, 0), dtype=torch.int32, device="cuda"
    )
    torch.cuda.synchronize()
    return Fixture(
        batch_size=batch_size,
        heads_q=heads_q,
        local_heads_q=local_heads_q,
        valid_lengths=valid_lengths,
        topk=topk,
        page_size=page_size,
        kv_block_padding_rows=kv_block_padding_rows,
        row_cache_tokens=row_cache_tokens,
        q=q,
        packed_kv=packed_kv,
        dequant_kv=dequant_kv,
        indices=indices,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        metadata=metadata,
        num_splits=num_splits,
    )


def run_flashmla(
    fixture: Fixture,
    metadata: torch.Tensor | None = None,
    num_splits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if metadata is None:
        metadata = fixture.metadata
    if num_splits is None:
        num_splits = fixture.num_splits
    return flash_mla_with_kvcache(
        q=fixture.q,
        k_cache=fixture.packed_kv,
        block_table=fixture.block_table,
        cache_seqlens=fixture.cache_seqlens,
        head_dim_v=HEAD_DIM_V,
        tile_scheduler_metadata=metadata,
        num_splits=num_splits,
        softmax_scale=1.0 / math.sqrt(HEAD_DIM_QK),
        causal=False,
        is_fp8_kvcache=True,
        indices=fixture.indices,
    )


def reference(fixture: Fixture) -> tuple[torch.Tensor, torch.Tensor]:
    kv_flat = fixture.dequant_kv.view(-1, HEAD_DIM_QK).float()
    output = torch.empty(
        (fixture.batch_size, 1, fixture.heads_q, HEAD_DIM_V),
        dtype=torch.bfloat16,
        device="cuda",
    )
    lse = torch.empty(
        (fixture.batch_size, fixture.heads_q, 1),
        dtype=torch.float32,
        device="cuda",
    )
    for batch_idx in range(fixture.batch_size):
        physical_indices = fixture.indices[batch_idx, 0]
        physical_indices = physical_indices[physical_indices >= 0].long()
        if physical_indices.numel() == 0:
            output[batch_idx].zero_()
            lse[batch_idx].fill_(float("inf"))
            continue
        selected = kv_flat[physical_indices]
        scores = (
            fixture.q[batch_idx, 0].float() @ selected.T
            / math.sqrt(HEAD_DIM_QK)
        )
        row_lse = torch.logsumexp(scores, dim=-1)
        probabilities = torch.softmax(scores, dim=-1)
        row_output = probabilities @ selected[..., :HEAD_DIM_V]
        output[batch_idx, 0].copy_(row_output.to(torch.bfloat16))
        lse[batch_idx, :, 0].copy_(row_lse)
    return output, lse


def _finite_json_number(value: float) -> float | str:
    if math.isfinite(value):
        return value
    if math.isnan(value):
        return "nan"
    return "inf" if value > 0 else "-inf"


def compare(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    actual_f = actual.float()
    expected_f = expected.float()
    close = torch.isclose(actual_f, expected_f, atol=atol, rtol=rtol)
    same_infinity = (
        torch.isinf(actual_f)
        & torch.isinf(expected_f)
        & (actual_f == expected_f)
    )
    close |= same_infinity
    finite = torch.isfinite(actual_f) & torch.isfinite(expected_f)
    abs_error = torch.full_like(actual_f, float("inf"))
    abs_error[same_infinity] = 0
    abs_error[finite] = (actual_f[finite] - expected_f[finite]).abs()
    rel_error = torch.full_like(actual_f, float("inf"))
    rel_error[same_infinity] = 0
    rel_error[finite] = abs_error[finite] / expected_f[finite].abs().clamp_min(
        1.0e-12
    )
    result: dict[str, object] = {
        "pass": bool(close.all()),
        "shape": list(actual.shape),
        "atol": atol,
        "rtol": rtol,
        "max_abs": _finite_json_number(float(abs_error.max().item())),
        "max_rel": _finite_json_number(float(rel_error.max().item())),
        "first_failing_index": None,
    }
    if result["pass"]:
        return result
    flat_index = int((~close).flatten().nonzero()[0].item())
    coordinates: list[int] = []
    residual = flat_index
    for dimension in reversed(actual.shape):
        coordinates.append(residual % dimension)
        residual //= dimension
    coordinates.reverse()
    index = tuple(coordinates)
    result["first_failing_index"] = coordinates
    result["first_actual"] = _finite_json_number(float(actual_f[index].item()))
    result["first_expected"] = _finite_json_number(float(expected_f[index].item()))
    return result


def _capture_graph(
    function: Callable[[], tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.cuda.CUDAGraph, tuple[torch.Tensor, torch.Tensor]]:
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        for _ in range(3):
            function()
    torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = function()
    for output in outputs:
        output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    return graph, outputs


def validate(fixture: Fixture) -> dict[str, object]:
    expected_output, expected_lse = reference(fixture)
    eager_output, eager_lse = run_flashmla(fixture)
    torch.cuda.synchronize()
    eager_checks: dict[str, dict[str, object]] = {
        "output_local_heads_after_production_trim": compare(
            eager_output[:, :, : fixture.local_heads_q],
            expected_output[:, :, : fixture.local_heads_q],
            atol=OUTPUT_ATOL,
            rtol=OUTPUT_RTOL,
        ),
        "lse": compare(eager_lse, expected_lse, atol=LSE_ATOL, rtol=LSE_RTOL),
    }
    if fixture.local_heads_q < fixture.heads_q:
        eager_checks["output_zero_padded_heads"] = compare(
            eager_output[:, :, fixture.local_heads_q :],
            expected_output[:, :, fixture.local_heads_q :],
            atol=OUTPUT_ATOL,
            rtol=OUTPUT_RTOL,
        )

    graph, graph_outputs = _capture_graph(lambda: run_flashmla(fixture))
    replay_one = tuple(output.clone() for output in graph_outputs)
    for output in graph_outputs:
        output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    replay_two = tuple(output.clone() for output in graph_outputs)
    graph_checks: dict[str, dict[str, object]] = {
        "replay_one_output_local_heads_after_production_trim": compare(
            replay_one[0][:, :, : fixture.local_heads_q],
            expected_output[:, :, : fixture.local_heads_q],
            atol=OUTPUT_ATOL,
            rtol=OUTPUT_RTOL,
        ),
        "replay_one_lse": compare(
            replay_one[1], expected_lse, atol=LSE_ATOL, rtol=LSE_RTOL
        ),
        "replay_two_output_local_heads_after_production_trim": compare(
            replay_two[0][:, :, : fixture.local_heads_q],
            expected_output[:, :, : fixture.local_heads_q],
            atol=OUTPUT_ATOL,
            rtol=OUTPUT_RTOL,
        ),
        "replay_two_lse": compare(
            replay_two[1], expected_lse, atol=LSE_ATOL, rtol=LSE_RTOL
        ),
    }
    if fixture.local_heads_q < fixture.heads_q:
        graph_checks["replay_one_output_zero_padded_heads"] = compare(
            replay_one[0][:, :, fixture.local_heads_q :],
            expected_output[:, :, fixture.local_heads_q :],
            atol=OUTPUT_ATOL,
            rtol=OUTPUT_RTOL,
        )
        graph_checks["replay_two_output_zero_padded_heads"] = compare(
            replay_two[0][:, :, fixture.local_heads_q :],
            expected_output[:, :, fixture.local_heads_q :],
            atol=OUTPUT_ATOL,
            rtol=OUTPUT_RTOL,
        )
    checks = list(eager_checks.values()) + list(graph_checks.values())
    return {
        "verdict": "PASS" if all(check["pass"] for check in checks) else "FAIL",
        "tolerances": {
            "output": {"atol": OUTPUT_ATOL, "rtol": OUTPUT_RTOL},
            "lse": {"atol": LSE_ATOL, "rtol": LSE_RTOL},
        },
        "eager": eager_checks,
        "cuda_graph": graph_checks,
    }


def _percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(samples_us: list[float]) -> dict[str, object]:
    return {
        "unit": "us",
        "count": len(samples_us),
        "p10": _percentile(samples_us, 0.10),
        "p50": _percentile(samples_us, 0.50),
        "p90": _percentile(samples_us, 0.90),
        "mean": statistics.fmean(samples_us),
        "stddev": statistics.pstdev(samples_us),
        "raw": samples_us,
    }


def time_cuda(
    function: Callable[[], object],
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for index in range(iterations):
        starts[index].record()
        function()
        ends[index].record()
    torch.cuda.synchronize()
    samples_us = [
        start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)
    ]
    return summarize(samples_us)


def metadata_plus_operator(fixture: Fixture) -> tuple[torch.Tensor, torch.Tensor]:
    metadata, num_splits = get_mla_metadata(
        cache_seqlens=fixture.cache_seqlens,
        num_q_tokens_per_head_k=fixture.heads_q,
        num_heads_k=1,
        num_heads_q=fixture.heads_q,
        is_fp8_kvcache=True,
        topk=fixture.topk,
    )
    return run_flashmla(fixture, metadata, num_splits)


def benchmark(
    fixture: Fixture,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    eager = time_cuda(lambda: run_flashmla(fixture), warmup, iterations)
    containing_region = time_cuda(
        lambda: metadata_plus_operator(fixture), warmup, iterations
    )
    graph, graph_outputs = _capture_graph(lambda: run_flashmla(fixture))
    # Keep both graph output tensors live for the complete replay benchmark.
    assert graph_outputs[0].numel() and graph_outputs[1].numel()
    graph_replay = time_cuda(graph.replay, warmup, iterations)
    return {
        "eager_main_plus_combine": eager,
        "eager_metadata_plus_main_plus_combine": containing_region,
        "cuda_graph_main_plus_combine": graph_replay,
    }


def profile(
    fixture: Fixture,
    iterations: int,
    region: str,
) -> dict[str, object]:
    graph_outputs: tuple[torch.Tensor, torch.Tensor] | None = None
    if region == "operator":
        function: Callable[[], object] = lambda: run_flashmla(fixture)
    elif region == "containing":
        function = lambda: metadata_plus_operator(fixture)
    elif region == "graph":
        graph, graph_outputs = _capture_graph(lambda: run_flashmla(fixture))
        function = graph.replay
    else:
        raise ValueError(f"unknown profile region: {region}")
    for _ in range(10):
        function()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push(f"flashmla_glm52_{region}")
    try:
        for _ in range(iterations):
            function()
        torch.cuda.synchronize()
    finally:
        torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()
    return {
        "region": region,
        "iterations": iterations,
        "nvtx": f"flashmla_glm52_{region}",
        "graph_outputs_retained": graph_outputs is not None,
    }


def fixture_record(fixture: Fixture, length_pattern: str) -> dict[str, object]:
    valid = fixture.indices >= 0
    return {
        "synthetic_fixture": True,
        "local_heads_q": fixture.local_heads_q,
        "padded_heads_q": fixture.heads_q,
        "topk_length": None,
        "attn_sink": None,
        "extra_topk_length": None,
        "length_pattern": length_pattern,
        "kv_block_padding_rows": fixture.kv_block_padding_rows,
        "valid_lengths": list(fixture.valid_lengths),
        "row_cache_tokens": fixture.row_cache_tokens,
        "valid_index_count": int(valid.sum().item()),
        "invalid_minus_one_count": int((fixture.indices == -1).sum().item()),
        "tensors": {
            "q": _tensor_fingerprint(fixture.q),
            "packed_kv": _tensor_fingerprint(fixture.packed_kv),
            "indices": _tensor_fingerprint(fixture.indices),
            "cache_seqlens": _tensor_fingerprint(fixture.cache_seqlens),
            "block_table": _tensor_fingerprint(fixture.block_table),
            "metadata": _tensor_fingerprint(fixture.metadata),
            "num_splits": _tensor_fingerprint(fixture.num_splits),
        },
    }


def dispatch_record(fixture: Fixture) -> dict[str, object]:
    inputs = {
        "original_h_q": fixture.heads_q,
        "kernel_h_q": 64 if fixture.heads_q == 128 else fixture.heads_q,
        "s_q": int(fixture.q.shape[1]),
        "d_qk": int(fixture.q.shape[3]),
        "topk": int(fixture.indices.shape[2]),
        "page_block_size": int(fixture.packed_kv.shape[1]),
        "stride_kv_row": int(fixture.packed_kv.stride(1)),
        "stride_kv_block": int(fixture.packed_kv.stride(0)),
        "extra_topk": 0,
        "has_topk_length": False,
        "has_extra_topk_length": False,
        "has_attn_sink": False,
    }
    contract_hit = flat_index_dispatch_hit(**inputs)
    try:
        operator = torch.ops.sgl_kernel.flashmla_glm52_flat_index_dispatch_state
        state_code = int(
            operator.default(
                inputs["original_h_q"],
                inputs["kernel_h_q"],
                inputs["s_q"],
                inputs["d_qk"],
                inputs["topk"],
                inputs["page_block_size"],
                inputs["stride_kv_row"],
                inputs["stride_kv_block"],
                inputs["extra_topk"],
                inputs["has_topk_length"],
                inputs["has_extra_topk_length"],
                inputs["has_attn_sink"],
            )
        )
    except AttributeError:
        return {
            "available": False,
            "state": "unavailable",
            "state_code": None,
            "contract_hit": contract_hit,
            "inputs": inputs,
        }
    if state_code not in DISPATCH_STATE_NAMES:
        raise AssertionError(f"unknown dispatch state code: {state_code}")
    state = DISPATCH_STATE_NAMES[state_code]
    if state != "disabled" and (state == "hit") != contract_hit:
        raise AssertionError(
            f"extension dispatch state {state!r} disagrees with contract "
            f"hit={contract_hit}"
        )
    return {
        "available": True,
        "state": state,
        "state_code": state_code,
        "contract_hit": contract_hit,
        "inputs": inputs,
    }


def assert_expected_dispatch(
    record: dict[str, object],
    expected: str,
) -> None:
    if expected == "ignore":
        return
    actual = record["state"]
    if actual != expected:
        raise AssertionError(
            f"flat-index dispatch state {actual!r}, expected {expected!r}"
        )


def production_abi_record(fixture: Fixture) -> dict[str, object]:
    zero_padded_heads = fixture.q[:, :, fixture.local_heads_q :]
    valid_indices = fixture.indices >= 0
    physical_indices_within_batch_rows = True
    for batch_index in range(fixture.batch_size):
        row = fixture.indices[batch_index][valid_indices[batch_index]]
        lower = batch_index * fixture.row_cache_tokens
        upper = lower + fixture.row_cache_tokens
        if row.numel() and not ((row >= lower) & (row < upper)).all().item():
            physical_indices_within_batch_rows = False
            break
    return {
        "batch_size": fixture.batch_size,
        "local_heads_q": fixture.local_heads_q,
        "padded_heads_q": fixture.heads_q,
        "page_size": fixture.page_size,
        "topk": fixture.topk,
        "head_dim_qk": HEAD_DIM_QK,
        "head_dim_v": HEAD_DIM_V,
        "packed_row_bytes": PACKED_ROW_BYTES,
        "zero_padded_heads_all_zero": bool(
            zero_padded_heads.numel() > 0
            and (zero_padded_heads == 0).all().item()
        ),
        "topk_length_present": False,
        "extra_topk_length_present": False,
        "attn_sink_present": False,
        "extra_kv_present": False,
        "invalid_indices_all_minus_one": bool(
            (fixture.indices[~valid_indices] == -1).all().item()
        ),
        "physical_indices_within_batch_rows": physical_indices_within_batch_rows,
        "valid_lengths": list(fixture.valid_lengths),
        "cache_seqlens_values": fixture.cache_seqlens.cpu().tolist(),
        "valid_index_count": int(valid_indices.sum().item()),
        "q": _tensor_fingerprint(fixture.q),
        "packed_kv": _tensor_fingerprint(fixture.packed_kv),
        "indices": _tensor_fingerprint(fixture.indices),
        "block_table": _tensor_fingerprint(fixture.block_table),
        "cache_seqlens": _tensor_fingerprint(fixture.cache_seqlens),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--heads-q", type=int, choices=(64, 128), default=HEADS_Q)
    parser.add_argument("--local-heads-q", type=int, default=LOCAL_HEADS_Q)
    parser.add_argument("--active-tokens", type=int, default=576)
    parser.add_argument(
        "--length-pattern",
        choices=("uniform", "production", "mixed"),
        default="production",
    )
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--kv-block-padding-rows", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--label", default="unknown")
    parser.add_argument(
        "--expected-dispatch",
        choices=("ignore", "unavailable", "disabled", "miss", "hit"),
        default="ignore",
    )
    parser.add_argument("--assert-production-abi", action="store_true")
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--profile-iterations", type=int, default=5)
    parser.add_argument(
        "--profile-region",
        choices=("operator", "containing", "graph"),
        default="containing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.time()
    result: dict[str, object] = {
        "schema_version": 2,
        "label": args.label,
        "shape": {
            "batch_size": args.batch_size,
            "s_q": 1,
            "h_q": args.heads_q,
            "local_h_q_before_padding": args.local_heads_q,
            "d_qk": HEAD_DIM_QK,
            "d_v": HEAD_DIM_V,
            "active_tokens": args.active_tokens,
            "topk": args.topk,
            "page_size": args.page_size,
            "kv_block_padding_rows": args.kv_block_padding_rows,
            "packed_row_bytes": PACKED_ROW_BYTES,
            "length_pattern": args.length_pattern,
            "seed": args.seed,
        },
    }
    exit_code = 0
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        torch.cuda.set_device(0)
        torch.set_grad_enabled(False)
        valid_lengths = resolve_lengths(
            args.batch_size, args.active_tokens, args.length_pattern
        )
        fixture = make_fixture(
            batch_size=args.batch_size,
            heads_q=args.heads_q,
            local_heads_q=args.local_heads_q,
            valid_lengths=valid_lengths,
            topk=args.topk,
            page_size=args.page_size,
            kv_block_padding_rows=args.kv_block_padding_rows,
            seed=args.seed,
        )
        result["fixture"] = fixture_record(fixture, args.length_pattern)
        result["dispatch"] = dispatch_record(fixture)
        assert_expected_dispatch(result["dispatch"], args.expected_dispatch)
        if args.assert_production_abi:
            abi = production_abi_record(fixture)
            failures = production_abi_failures(abi)
            result["production_abi"] = {
                "verdict": "PASS" if not failures else "FAIL",
                "failures": failures,
                "record": abi,
            }
            if failures:
                raise AssertionError(
                    "production ABI mismatch: " + "; ".join(failures)
                )
        correctness = validate(fixture)
        result["schedule"] = {
            "metadata_shape": list(fixture.metadata.shape),
            "num_splits": fixture.num_splits.cpu().tolist(),
        }
        result["correctness"] = correctness
        if correctness["verdict"] != "PASS":
            exit_code = 1
        if args.profile_only and exit_code == 0:
            result["profile"] = profile(
                fixture, args.profile_iterations, args.profile_region
            )
            result["timings"] = None
        elif args.correctness_only or exit_code != 0:
            result["timings"] = None
        else:
            result["timings"] = benchmark(fixture, args.warmup, args.iterations)
        extension_path = Path(flashmla_ops.__file__).resolve()
        result["environment"] = {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "extension": str(extension_path),
            "extension_sha256": _sha256(extension_path),
            "commit": os.environ.get("B300_COMMIT"),
            "run_id": os.environ.get("B300_RUN_ID"),
        }
    except Exception as error:
        exit_code = 1
        result["correctness"] = {
            "verdict": "ERROR",
            "exception_type": type(error).__name__,
            "exception": str(error),
            "traceback": traceback.format_exc(),
        }
        result["timings"] = None
    result["elapsed_seconds"] = time.time() - start_time
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(args.output)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    if exit_code:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
