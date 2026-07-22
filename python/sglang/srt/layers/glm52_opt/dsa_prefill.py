"""Pure predicates for fail-closed GLM-5.2 DSA prefill trials."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Optional


def select_trtllm_prefill_enable_pdl(
    *,
    allowed_m: Collection[int],
    is_plain_extend: bool,
    is_prefill: bool,
    q_shape: Sequence[int],
    kv_shape: Sequence[int],
    block_tables_shape: Sequence[int],
    seq_lens_shape: Sequence[int],
    max_seq_len: int,
    q_is_fp8_e4m3: bool,
    kv_is_fp8_e4m3: bool,
) -> Optional[bool]:
    """Select PDL-off only for the measured M4096/context32768 leaf ABI.

    ``None`` preserves FlashInfer's stock device-dependent selection.  The
    caller additionally gates the static model, backend, graph, topology, and
    head/page dimensions once during backend initialization.
    """
    q_shape = tuple(q_shape)
    kv_shape = tuple(kv_shape)
    block_tables_shape = tuple(block_tables_shape)
    seq_lens_shape = tuple(seq_lens_shape)
    m = q_shape[0] if q_shape else 0
    if (
        m not in allowed_m
        or not is_plain_extend
        or not is_prefill
        or q_shape != (4096, 1, 64, 576)
        or len(kv_shape) != 4
        or kv_shape[1:] != (1, 64, 576)
        or block_tables_shape != (4096, 1, 2048)
        or seq_lens_shape != (4096,)
        or max_seq_len != 32768
        or not q_is_fp8_e4m3
        or not kv_is_fp8_e4m3
    ):
        return None
    return False
