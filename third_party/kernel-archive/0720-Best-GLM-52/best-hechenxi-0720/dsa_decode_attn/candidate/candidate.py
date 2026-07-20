"""GLM-5.2 DSA Sparse Attention (decode) — flashinfer trtllm-gen backend.

Replaces the stock sgl_kernel.flash_mla.flash_mla_sparse_fwd (grid==M, occupancy-
starved) with flashinfer's trtllm-gen sparse-MLA decode kernel, which parallelizes
the topk-KV read across many CTAs. Numerically equivalent (calc_diff ~4e-6 < 5e-6).

Mapping (frozen inputs, no re-quant / re-seed):
  q[M,64,576]  -> query[M, 1, 64, 576]           (576 = kv_lora 512 + rope 64)
  kv[65536,1,576] -> kv_cache[1024, 1, 64, 576]   (page_size 64, flat token pool)
  indices[M,1,2048] -> block_tables[M,1,2048]     (flat token indices, == sparse topk)
  sm_scale -> bmm1_scale (already 1/sqrt(576))
"""
from __future__ import annotations
import torch
import flashinfer.decode as _fd

_PAGE = 64
_KV_LORA = 512
_QK_ROPE = 64
_QK_NOPE = 192
_WS = None  # reused workspace


def run(inputs: dict):
    q = inputs["q"]; kv = inputs["kv"]; idx = inputs["indices"]
    sm = inputs["sm_scale"]
    M = q.shape[0]
    S = kv.shape[0]; HD = kv.shape[-1]
    num_pages = S // _PAGE

    query = q.view(M, 1, q.shape[1], HD)
    kv_cache = kv.view(num_pages, 1, _PAGE, HD)
    block_tables = idx.view(M, 1, idx.shape[-1])
    seq_lens = torch.full((M,), S, dtype=torch.int32, device=q.device)

    global _WS
    if _WS is None:
        _WS = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)

    out = _fd.trtllm_batch_decode_with_kv_cache_mla(
        query=query, kv_cache=kv_cache, workspace_buffer=_WS,
        qk_nope_head_dim=_QK_NOPE, kv_lora_rank=_KV_LORA, qk_rope_head_dim=_QK_ROPE,
        block_tables=block_tables, seq_lens=seq_lens, max_seq_len=S,
        sparse_mla_top_k=idx.shape[-1], bmm1_scale=float(sm), backend="trtllm-gen")
    if out.ndim == 4 and out.shape[1] == 1:
        out = out.squeeze(1)
    return out
