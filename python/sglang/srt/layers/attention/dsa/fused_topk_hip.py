"""Fused mask + topk-k kernel for DSA indexer decode fast path.

Round-3 status: SCAFFOLDING, NOT WIRED. The bucketed algorithm below is
correct for the fully-valid case (length == N) but silently loses candidates
when the valid range is much smaller than N (length << N with topk ≥
length/num_blocks). That is exactly the DSA decode profile shape
(N=65536, length ~4108, topk=2048), so this kernel MUST NOT be wired into
dsa_topk_backend._topk_unfused as-is. Round-4 needs a different algorithm
for the length << N regime — either:
  (a) a length-adaptive single-block sort when length ≤ CHUNK, or
  (b) a two-level tournament that adjusts k_per_block by the number of
      chunks that overlap the valid range at graph-capture time.

Existing correctness evidence (torch.manual_seed(0), N=65536, K=2048):
  - length == N          : 2048/2048 overlap (PASS)
  - length == 4108 (prof): 264/2048  (FAIL)
  - length == 8192       : 512/2048  (FAIL)
  - length == 2049       : 256/2048  (FAIL — 1 valid chunk, K=2048 > 256)

Speed on gfx942 for the length=4108, K=2048 case: 432 us fused vs 1590 us
for `masked_fill + torch.topk` — 3.68× speedup — but the fused result is
wrong. Rank-4 potential is real; algorithm choice needs revisiting.

The `maybe_dispatch_fused_topk` guard below returns None for any call whose
shape/dtype would exercise the buggy path, so importing this file is safe:
callers stay on torch.topk. Set `SGLANG_DSA_HIP_FUSED_TOPK=1` AND meet the
guard's fully-valid check (`lengths.item() == score.shape[-1]`) to opt-in.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import triton
import triton.language as tl


_FLAG_ENV = "SGLANG_DSA_HIP_FUSED_TOPK"


def _is_enabled() -> bool:
    return os.environ.get(_FLAG_ENV, "0") == "1"


@triton.jit
def _bucket_topk_kernel(
    score_ptr,
    lengths_ptr,
    row_starts_ptr,
    out_scores_ptr,
    out_indices_ptr,
    N: tl.constexpr,
    K_PER_BLOCK: tl.constexpr,
    CHUNK: tl.constexpr,
    HAS_ROW_STARTS: tl.constexpr,
):
    """One Triton program per (batch_row, chunk). Extract top-K_PER_BLOCK
    scores + their absolute indices from `score[row, chunk_lo:chunk_hi]`,
    honouring the valid range `[row_start, row_start+length)`."""
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    num_chunks = tl.num_programs(1)

    length = tl.load(lengths_ptr + row).to(tl.int32)
    if HAS_ROW_STARTS:
        row_start = tl.load(row_starts_ptr + row).to(tl.int32)
    else:
        row_start = 0
    row_end = row_start + length

    chunk_lo = chunk * CHUNK
    tile_off = tl.arange(0, CHUNK)
    offs = chunk_lo + tile_off
    mask_bounds = offs < N
    mask_valid = (offs >= row_start) & (offs < row_end)
    mask = mask_bounds & mask_valid

    scores = tl.load(
        score_ptr + row * N + offs,
        mask=mask,
        other=float("-inf"),
    )

    # Repeated argmax to extract top-K_PER_BLOCK. Uses tl.argmax which returns
    # the tile-local index of the maximum. After picking, set that slot to -inf
    # so the next argmax picks the next-highest.
    for k in tl.static_range(K_PER_BLOCK):
        max_val = tl.max(scores, axis=0)
        rel_idx = tl.argmax(scores, axis=0).to(tl.int32)
        abs_idx = chunk_lo + rel_idx

        # Where does this write go? (row, chunk*K_PER_BLOCK + k)
        out_slot = row * (num_chunks * K_PER_BLOCK) + chunk * K_PER_BLOCK + k
        tl.store(out_scores_ptr + out_slot, max_val)
        stored_idx = tl.where(max_val == float("-inf"), -1, abs_idx)
        tl.store(out_indices_ptr + out_slot, stored_idx)

        # Suppress the just-selected element so next argmax picks the next
        suppress = tile_off == rel_idx
        scores = tl.where(suppress, float("-inf"), scores)


def fused_masked_topk_hip(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
    num_blocks: Optional[int] = None,
    k_per_block: Optional[int] = None,
) -> torch.Tensor:
    """Drop-in replacement for the mask+torch.topk path in
    dsa_topk_backend._topk_unfused, for the batch=1 decode shape.

    Args:
      score:      [B, N] fp32
      lengths:    [B] int32/int64, valid range per row is [row_start, row_start+length)
      topk:       int, K
      row_starts: [B] optional int, default 0
      num_blocks: number of chunk-level Triton programs; auto-picked to give
                  ~2*topk candidates when the data is concentrated in a
                  smaller range. For N=65536, K=2048 we default to
                  num_blocks=16, k_per_block=256 (candidate pool 4096).
      k_per_block: top-K per chunk to extract; must satisfy
                   num_blocks * k_per_block >= topk (else silently loses
                   candidates).

    Returns:
      topk_indices: [B, topk] int32, positions RELATIVE to row_start (matches
      the _topk_unfused contract), padded with -1 for invalid slots.
    """
    assert score.dim() == 2, f"expect [B,N], got {score.shape}"
    assert score.dtype == torch.float32, f"expect fp32, got {score.dtype}"
    assert score.is_contiguous(), "score must be contiguous"
    B, N = score.shape
    device = score.device
    dtype_out = torch.int32

    topk_indices = score.new_full((B, topk), -1, dtype=dtype_out)
    if B == 0 or topk == 0 or N == 0:
        return topk_indices

    # Auto-sizing:
    # - k_per_block must be >= topk / num_blocks (else we lose candidates when
    #   valid data concentrates in few chunks — the DSA decode workload has
    #   length ~4108 out of N=65536, i.e. only ~7% of chunks are valid, so we
    #   need enough per-chunk capacity to gather all top-K from those chunks).
    # - Total per-block work is O(CHUNK * k_per_block) so both should stay
    #   moderate. Rule of thumb: k_per_block = 2*topk / num_blocks (2x margin).
    if k_per_block is None or num_blocks is None:
        if num_blocks is None:
            # Small default: 16 chunks lets CHUNK be pow2 for N up to 65536.
            num_blocks = 16
        if k_per_block is None:
            # 2x margin so the top-K survives even if all valid data lands in
            # 1/2 of the chunks.
            k_per_block = max(topk // (num_blocks // 2), 32)
            # Round up to a power of 2 (Triton static_range prefers pow2)
            k_per_block = 1 << (k_per_block - 1).bit_length()

    # Round CHUNK up so num_blocks * CHUNK >= N
    chunk = (N + num_blocks - 1) // num_blocks
    chunk = triton.next_power_of_2(chunk)
    num_blocks_actual = (N + chunk - 1) // chunk

    cand_len = num_blocks_actual * k_per_block
    cand_scores = score.new_full(
        (B, cand_len), float("-inf"), dtype=torch.float32
    )
    cand_indices = score.new_full(
        (B, cand_len), -1, dtype=dtype_out
    )

    lengths_i32 = lengths.to(dtype=torch.int32, device=device)
    if row_starts is not None:
        row_starts_i32 = row_starts.to(dtype=torch.int32, device=device)
    else:
        row_starts_i32 = torch.zeros((B,), dtype=torch.int32, device=device)

    grid = (B, num_blocks_actual)
    _bucket_topk_kernel[grid](
        score,
        lengths_i32,
        row_starts_i32,
        cand_scores,
        cand_indices,
        N=N,
        K_PER_BLOCK=k_per_block,
        CHUNK=chunk,
        HAS_ROW_STARTS=(row_starts is not None),
    )

    # Final top-K from the candidate pool via torch.topk on the smaller tensor.
    # cand_len is much smaller than N when num_blocks * k_per_block << N.
    take = min(topk, cand_len)
    top_scores, top_local = torch.topk(cand_scores, take, dim=-1)
    top_abs = torch.gather(cand_indices, dim=-1, index=top_local.to(torch.int64))

    row_starts_b = row_starts_i32.unsqueeze(1).to(dtype_out)
    top_rel = top_abs - row_starts_b
    top_rel = torch.where(
        top_scores == float("-inf"), torch.full_like(top_rel, -1), top_rel
    )

    topk_indices[:, :take] = top_rel
    return topk_indices


def maybe_dispatch_fused_topk(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Return fused topk result iff the environment flag is set AND the shape
    matches the ONLY currently-verified correct case: fully-valid row (all
    lengths equal N, no row_starts offset). Otherwise return None so the
    caller falls back to the original torch.topk path. Zero-cost when disabled.

    See module docstring: the bucketed algorithm silently loses candidates
    when length << N with topk ≥ length/num_blocks. Round 4 must replace the
    algorithm before this guard can be widened."""
    if not _is_enabled():
        return None
    if score.dim() != 2 or score.dtype != torch.float32 or not score.is_contiguous():
        return None
    B, N = score.shape
    if B != 1:
        return None
    if topk >= N:
        return None
    if row_starts is not None:
        return None
    # Fully-valid row check. Reading lengths.item() forces a CPU sync — this
    # guard is intentionally restrictive and only intended for benchmarking /
    # correctness verification runs, NOT for the cudagraph decode path.
    try:
        if int(lengths.max().item()) != N:
            return None
        return fused_masked_topk_hip(score, lengths, topk, row_starts=None)
    except Exception:
        return None
