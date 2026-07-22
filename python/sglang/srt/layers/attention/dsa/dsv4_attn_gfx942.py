"""
DSV4 FP8 Attention — PyTorch Reference for gfx942 (MI300X)

Pure PyTorch implementation of the DSV4 FP8 sparse decode attention that works
on AMD MI300X. This replaces the tilelang kernel which generates NVIDIA WGMMA
instructions incompatible with gfx942.

KV Cache Format (MODEL1_FP8Sparse) — struct-of-arrays within each page:
  Per token = 584 bytes:
    - 448 bytes: FP8 e4m3fnuz nope values (7 tiles x 64d)
    - 128 bytes: BF16 rope values (64 elems x 2 bytes)
    - 8 bytes: ue8m0 per-tile scales (7 tiles + 1 pad)
"""

import math
from typing import Any, Optional, Tuple

import torch

from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz

DIM_NOPE = 448
DIM_ROPE = 64
DIM_TOTAL = DIM_NOPE + DIM_ROPE
TILE_SIZE = 64
NUM_TILES = DIM_NOPE // TILE_SIZE
PACKED_BYTES = DIM_NOPE + DIM_ROPE * 2
SCALE_BYTES = 8
PACKED_W4 = PACKED_BYTES // 4
SCALE_W4 = SCALE_BYTES // 4


def _dequant_fp8_nope(nope_u8: torch.Tensor, scale_u8: torch.Tensor) -> torch.Tensor:
    """Dequant FP8 nope with per-tile ue8m0 scales → BF16."""
    device = nope_u8.device
    bias_offset = 8 if is_fp8_fnuz() else 7

    b = nope_u8.to(torch.int32)
    sign_bf = (b & 0x80) << 8
    exp_e4 = (b & 0x78) >> 3
    mant_bf = (b & 0x07) << 4

    tile_idx = torch.arange(DIM_NOPE, device=device) // TILE_SIZE
    scale = scale_u8[..., tile_idx].to(torch.int32)

    exp_combined = (exp_e4 + scale - bias_offset).clamp(0, 255)
    bf16_bits = sign_bf | (exp_combined << 7) | mant_bf
    return bf16_bits.to(torch.int16).view(torch.bfloat16)


def _unpack_tokens(
    k_combined: torch.Tensor,
    indices: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Unpack KV tokens from packed int32 cache → [N, 512] BF16."""
    device = k_combined.device
    N = indices.shape[0]
    NOPE_ROPE_U32_PER_BLOCK = block_size * PACKED_W4

    valid_mask = indices >= 0
    safe_indices = indices.clamp(min=0)
    block_ids = safe_indices // block_size
    t_in_blocks = safe_indices % block_size

    col_offsets_packed = torch.arange(PACKED_W4, device=device)
    gather_cols = (t_in_blocks.unsqueeze(1) * PACKED_W4 + col_offsets_packed).long()
    rows = k_combined[block_ids.long()]
    packed_data = rows.gather(1, gather_cols)

    col_offsets_scale = torch.arange(SCALE_W4, device=device)
    gather_cols_s = (
        NOPE_ROPE_U32_PER_BLOCK
        + t_in_blocks.unsqueeze(1) * SCALE_W4
        + col_offsets_scale
    ).long()
    scale_data = rows.gather(1, gather_cols_s)

    packed_u8 = packed_data.contiguous().view(torch.uint8).reshape(N, PACKED_BYTES)
    scale_u8 = scale_data.contiguous().view(torch.uint8).reshape(N, SCALE_BYTES)

    nope_bf16 = _dequant_fp8_nope(packed_u8[:, :DIM_NOPE], scale_u8[:, :NUM_TILES])
    rope_bf16 = (
        packed_u8[:, DIM_NOPE : DIM_NOPE + DIM_ROPE * 2]
        .contiguous()
        .view(torch.bfloat16)
    )

    K_bf16 = torch.cat([nope_bf16, rope_bf16], dim=-1)
    K_bf16[~valid_mask] = 0.0
    return K_bf16


def _build_combined_view(k_cache: torch.Tensor):
    """Reinterpret KV cache as flat int32 view per block."""
    k_u8 = k_cache.view(torch.uint8) if k_cache.dtype != torch.uint8 else k_cache
    num_blocks = k_u8.shape[0]
    block_size = k_u8.shape[1]
    block_pad = k_u8.stride(0) // 4
    storage = k_u8.untyped_storage()
    flat = torch.empty(0, dtype=torch.int32, device=k_u8.device).set_(
        storage, 0, (storage.nbytes() // 4,), (1,)
    )
    combined = torch.as_strided(
        flat,
        size=(num_blocks, block_pad),
        stride=(block_pad, 1),
        storage_offset=k_u8.storage_offset() // 4,
    )
    return combined, num_blocks, block_size


def _build_validity_mask(indices, topk_length, topk, device):
    valid = indices >= 0
    if topk_length is not None:
        pos = torch.arange(topk, device=device)
        tk_mask = pos < topk_length.unsqueeze(-1)
        if tk_mask.dim() == 2:
            tk_mask = tk_mask.unsqueeze(1)
        valid = valid & tk_mask
    return valid


def dsv4_fp8_attention_fwd_torch(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: Optional[torch.Tensor],
    cache_seqlens: Optional[torch.Tensor],
    head_dim_v: int,
    tile_scheduler_metadata: Any = None,
    num_splits: None = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    is_fp8_kvcache: bool = False,
    indices: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference DSV4 FP8 sparse decode attention for gfx942.

    Returns:
        output: [batch, seq_len, num_heads, head_dim_v] bfloat16
        lse: [batch, seq_len, num_heads] float32 (natural log)
    """
    batch, seq_len, num_heads, head_dim = q.shape

    if softmax_scale is None:
        softmax_scale = head_dim**-0.5

    k1, _, bs1 = _build_combined_view(k_cache)
    topk_1 = indices.shape[-1]
    K1 = _unpack_tokens(k1, indices.reshape(-1), bs1).reshape(
        batch, seq_len, topk_1, DIM_TOTAL
    )
    valid_1 = _build_validity_mask(indices, topk_length, topk_1, q.device)

    Q_f = q.float()
    K1_f = K1.float()
    scores_1 = torch.einsum("bshd,bstd->bsht", Q_f, K1_f) * softmax_scale
    scores_1.masked_fill_(~valid_1.unsqueeze(2), float("-inf"))

    if extra_k_cache is not None:
        k2, _, bs2 = _build_combined_view(extra_k_cache)
        topk_2 = extra_indices_in_kvcache.shape[-1]
        K2 = _unpack_tokens(k2, extra_indices_in_kvcache.reshape(-1), bs2).reshape(
            batch, seq_len, topk_2, DIM_TOTAL
        )
        valid_2 = _build_validity_mask(
            extra_indices_in_kvcache, extra_topk_length, topk_2, q.device
        )
        K2_f = K2.float()
        scores_2 = torch.einsum("bshd,bstd->bsht", Q_f, K2_f) * softmax_scale
        scores_2.masked_fill_(~valid_2.unsqueeze(2), float("-inf"))

        all_scores = torch.cat([scores_1, scores_2], dim=-1)
        all_K = torch.cat([K1_f, K2_f], dim=2)
    else:
        all_scores = scores_1
        all_K = K1_f

    attn_weights = torch.softmax(all_scores, dim=-1)
    all_V = all_K[..., :head_dim_v]
    output = torch.einsum("bsht,bstd->bshd", attn_weights, all_V)
    lse = torch.logsumexp(all_scores, dim=-1)

    if attn_sink is not None:
        sink = attn_sink.view(1, 1, num_heads)
        o_scale = torch.sigmoid(lse - sink)
        output = output * o_scale.unsqueeze(-1)

    output = output.to(torch.bfloat16)
    return output, lse
