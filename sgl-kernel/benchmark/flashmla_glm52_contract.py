"""Pure-Python contracts for the GLM-5.2 FlashMLA dispatch experiment."""

from __future__ import annotations

from typing import Mapping

DISPATCH_BUILD_DISABLED = 0
DISPATCH_MISS = 1
DISPATCH_HIT = 2

DISPATCH_STATE_NAMES = {
    DISPATCH_BUILD_DISABLED: "disabled",
    DISPATCH_MISS: "miss",
    DISPATCH_HIT: "hit",
}


def generic_token_byte_offset(
    token_index: int,
    *,
    page_block_size: int = 64,
    stride_kv_row: int = 656,
    stride_kv_block: int = 64 * 656,
) -> int:
    block_index, index_in_block = divmod(token_index, page_block_size)
    return block_index * stride_kv_block + index_in_block * stride_kv_row


def flat_token_byte_offset(token_index: int, *, bytes_per_token: int = 656) -> int:
    return token_index * bytes_per_token


def flat_index_dispatch_hit(
    *,
    original_h_q: int,
    kernel_h_q: int,
    s_q: int,
    d_qk: int,
    topk: int,
    page_block_size: int,
    stride_kv_row: int,
    stride_kv_block: int,
    extra_topk: int,
    has_topk_length: bool,
    has_extra_topk_length: bool,
    has_attn_sink: bool,
) -> bool:
    """Mirror the exact host-side dispatch predicate for independent checking."""
    bytes_per_token = 656
    return (
        original_h_q == 64
        and kernel_h_q == 64
        and s_q == 1
        and d_qk == 576
        and topk == 2048
        and page_block_size == 64
        and stride_kv_row == bytes_per_token
        and stride_kv_block == 64 * bytes_per_token
        and extra_topk == 0
        and not has_topk_length
        and not has_extra_topk_length
        and not has_attn_sink
    )


def expected_dispatch_state(*, build_enabled: bool, **inputs: object) -> int:
    if not build_enabled:
        return DISPATCH_BUILD_DISABLED
    return DISPATCH_HIT if flat_index_dispatch_hit(**inputs) else DISPATCH_MISS


def production_abi_failures(record: Mapping[str, object]) -> list[str]:
    """Return exact GLM TP8 production-ABI mismatches without importing torch."""
    batch_size = record.get("batch_size")
    q = record.get("q")
    packed_kv = record.get("packed_kv")
    indices = record.get("indices")
    block_table = record.get("block_table")
    cache_seqlens = record.get("cache_seqlens")
    failures: list[str] = []

    if not isinstance(batch_size, int) or batch_size <= 0:
        failures.append("batch_size must be a positive integer")
        return failures

    expected = {
        "q": {
            "shape": [batch_size, 1, 64, 576],
            "dtype": "torch.bfloat16",
            "stride": [64 * 576, 64 * 576, 576, 1],
        },
        "packed_kv": {
            "shape_tail": [64, 1, 656],
            "dtype": "torch.float8_e4m3fn",
            "stride": [64 * 656, 656, 656, 1],
        },
        "indices": {
            "shape": [batch_size, 1, 2048],
            "dtype": "torch.int32",
            "stride": [2048, 2048, 1],
        },
        "block_table": {
            "shape": [batch_size, 0],
            "dtype": "torch.int32",
        },
        "cache_seqlens": {
            "shape": [batch_size],
            "dtype": "torch.int32",
            "stride": [1],
        },
    }
    tensors = {
        "q": q,
        "packed_kv": packed_kv,
        "indices": indices,
        "block_table": block_table,
        "cache_seqlens": cache_seqlens,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, Mapping):
            failures.append(f"{name} descriptor is missing")
            continue
        for field, wanted in expected[name].items():
            if field == "shape_tail":
                shape = tensor.get("shape")
                actual = shape[-3:] if isinstance(shape, list) else None
            else:
                actual = tensor.get(field)
            if actual != wanted:
                failures.append(f"{name}.{field}: {actual!r} != {wanted!r}")

    scalar_expectations = {
        "local_heads_q": 8,
        "padded_heads_q": 64,
        "page_size": 64,
        "topk": 2048,
        "head_dim_qk": 576,
        "head_dim_v": 512,
        "packed_row_bytes": 656,
        "zero_padded_heads_all_zero": True,
        "topk_length_present": False,
        "extra_topk_length_present": False,
        "attn_sink_present": False,
        "extra_kv_present": False,
        "invalid_indices_all_minus_one": True,
        "physical_indices_within_batch_rows": True,
    }
    for field, wanted in scalar_expectations.items():
        actual = record.get(field)
        if actual != wanted:
            failures.append(f"{field}: {actual!r} != {wanted!r}")
    expected_lengths = (
        [575]
        if batch_size == 1
        else [
            512 + round(index * 63 / (batch_size - 1))
            for index in range(batch_size)
        ]
    )
    if record.get("valid_lengths") != expected_lengths:
        failures.append(
            f"valid_lengths: {record.get('valid_lengths')!r} != "
            f"{expected_lengths!r}"
        )
    if record.get("cache_seqlens_values") != expected_lengths:
        failures.append(
            f"cache_seqlens_values: {record.get('cache_seqlens_values')!r} != "
            f"{expected_lengths!r}"
        )
    expected_valid_indices = sum(expected_lengths)
    if record.get("valid_index_count") != expected_valid_indices:
        failures.append(
            f"valid_index_count: {record.get('valid_index_count')!r} != "
            f"{expected_valid_indices!r}"
        )
    return failures
