"""GLM-5.2 DSA sparse attention (decode) — split-KV FlashMLA, B300.

Replaces the stock ``sgl_kernel.flash_mla.flash_mla_sparse_fwd`` on the
``flashmla_sparse`` DSA decode path. Loaded through the glm52_opt registry as a
``kind="dsa"`` archive candidate, so the per-call cost is a dict build plus an
``lru_cache`` hit in ``archive_loader.load_run_fn`` -- NOT the API-v1
hotspot_plugin provider path, whose measured +17.4 us of host-side Python is
larger than this kernel's entire margin.

Diagnosis
---------
Stock ``sm100::fwd::head64::sparse_attn_fwd_kernel`` launches
``<<<params.s_q, 384>>>`` -- one CTA per query token (FlashMLA
``csrc/sm100/prefill/sparse/fwd/head64/phase1.cuh:669``). At decode shapes that
is 16 or 32 CTAs on a 148-SM B300, and measured latency is bit-identical at
M=16 and M=32 (42.18 us warm): doubling the work costs nothing.

The inner loop is left BYTE-IDENTICAL -- tcgen05 UMMA with the accumulator in
TMEM, Q staged to TMEM by UTCCP, TMA ``tile::gather4`` for the 512 NoPE dims,
``cp.async`` for the 64 RoPE dims, 3-deep pipeline, dual-GEMM N=128 re-view.
Only the grid changes, to ``(num_splits, s_q)``, plus an LSE combine.

Measured on the kernel-harness gate (B300, idle, three runs):

    reference sgl_kernel   M16 55.02 us   M32 56.54 us
    this candidate         M16 26.73 us   M32 32.52 us   geomean 1.89x
    calc_diff 3.84e-6 against a 5e-6 tolerance; COMPLETE_WIN

For context, the previously registered ``best-hechenxi-0720/dsa_decode_attn``
(flashinfer trtllm-gen sparse MLA) measures **0.91x on this host** -- correct
(calc_diff 3.8e-6) but a regression at both shapes.

Safety
------
Every shape this kernel cannot serve falls back to the stock call, reproducing
the backend's own head-padding, so registering this can only change which
kernel runs, never whether the path works. Guarded on: bf16 contiguous q/kv,
h_q == 64, d_qk == 576, int32 indices, and topk divisible into whole 64-key
blocks by the chosen split count.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_MANIFEST = _HERE / "MANIFEST.json"

D_QK = 576
D_V = 512
ROPE = D_QK - D_V
H_Q = 64
B_TOPK = 64        # kernel's key-block size; a slice must be a whole multiple

# Splits per M, measured on an idle B300 under the harness gate's own protocol.
# Both land 128 CTAs of 148. Deeper splits lose: 256 CTAs is two waves, and the
# per-split partial buffer grows linearly (at M=16, splits=16 writes 16.8 MB of
# partials against a 36 MB KV gather). Measured M=16: splits 4/8/16/32 ->
# 1.99x / 2.19x / 1.58x / 1.05x.
_SPLITS = {16: 8, 32: 4}
_ENV_SPLITS = os.environ.get("SGLANG_GLM52_MLA_SPLITS", "").strip()
if _ENV_SPLITS:
    _SPLITS = {int(k): int(v)
               for k, v in (p.split(":") for p in _ENV_SPLITS.split(";") if p.strip())}


def _load_extension():
    """Prebuilt .so first; JIT only if explicitly asked for.

    A served deployment runs one worker per GPU, and letting eight of them race
    a torch.utils.cpp_extension JIT into one cache directory is a known way to
    lose. The vendored .so is sha256-pinned in MANIFEST.json, matching the
    convention in layers/glm52_opt/hotspot_candidates/.
    """
    if os.environ.get("SGLANG_GLM52_MLA_JIT", "").strip().lower() in ("1", "true", "yes", "on"):
        sys.path.insert(0, str(_HERE / "ext"))
        import build_split  # noqa: E402
        return build_split.build(verbose=False)

    entry = json.loads(_MANIFEST.read_text())["binaries"][0]
    so = _HERE / entry["so_file"]
    if not so.is_file():
        raise RuntimeError(f"missing vendored prebuilt {so}")
    digest = hashlib.sha256(so.read_bytes()).hexdigest()
    if digest != entry["sha256"]:
        raise RuntimeError(f"prebuilt sha256 mismatch for {so}: "
                           f"got {digest}, expected {entry['sha256']}")
    name = so.stem
    spec = importlib.util.spec_from_file_location(name, so)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {so}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_EXT = _load_extension()
_WS: dict = {}


def _workspace(M: int, hq: int, splits: int, device):
    key = (M, hq, splits, device)
    ws = _WS.get(key)
    if ws is None:
        ws = (torch.empty((splits, M, hq, D_V), dtype=torch.bfloat16, device=device),
              torch.empty((splits, M, hq), dtype=torch.float32, device=device))
        _WS[key] = ws
    return ws


def _stock(q, kv, indices, sm_scale, d_v):
    """The stock call, for every shape this kernel declines.

    Pads only to a head count FlashMLA actually dispatches on. Its sm100 path
    takes h_q == 64 (head64) and h_q == 128 (head128) -- FlashMLA
    `csrc/api/sparse_fwd.h:216-219`. The backend
    (dsa_backend.py:2318-2336) pads unconditionally to 128 on Blackwell, which
    sends a 64-head GLM-5.2 decode through the head128 kernel: twice the MMA
    work, and head128 additionally requires topk % 128 == 0, so a topk of e.g.
    1984 (= 31*64) trips an assert that head64 would have accepted.
    """
    from sgl_kernel.flash_mla import flash_mla_sparse_fwd

    num_tokens, num_heads, head_dim = q.shape
    if num_heads in (64, 128):
        target = num_heads
    elif 64 % num_heads == 0:
        target = 64                       # cheaper than padding all the way to 128
    elif 128 % num_heads == 0:
        target = 128
    else:
        raise ValueError(f"h_q={num_heads} cannot be padded to 64 or 128")

    if target != num_heads:
        q_in = q.new_zeros((num_tokens, target, head_dim))
        q_in[:, :num_heads, :] = q
    else:
        q_in = q
    o, _, _ = flash_mla_sparse_fwd(q=q_in, kv=kv, indices=indices,
                                   sm_scale=sm_scale, d_v=d_v)
    return o[:, :num_heads, :] if target != num_heads else o


def _serviceable(q, kv, indices, splits) -> bool:
    if splits is None or splits <= 1:
        return False
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        return False                      # fp8 / paged KV is a different path
    if indices.dtype != torch.int32:
        return False
    if q.dim() != 3 or kv.dim() != 3 or indices.dim() != 3:
        return False
    if q.shape[1] != H_Q or q.shape[2] != D_QK or kv.shape[1] != 1 or kv.shape[2] != D_QK:
        return False
    if q.stride(2) != 1 or kv.stride(2) != 1 or indices.stride(2) != 1:
        return False
    topk = indices.shape[2]
    # Every slice must be a whole number of B_TOPK key blocks.
    return topk % (splits * B_TOPK) == 0


def run(inputs: dict):
    q = inputs["q"]
    kv = inputs["kv"]
    indices = inputs["indices"]
    sm_scale = inputs["sm_scale"]
    # The registry hook (dsa_backend.py:2305-2313) passes no d_v; on the MLA
    # absorbed path it is always the latent rank, i.e. d_qk minus the rope dims.
    d_v = int(inputs.get("d_v") or (q.shape[2] - ROPE))

    splits = _SPLITS.get(int(q.shape[0]))
    if d_v != D_V or not _serviceable(q, kv, indices, splits):
        return _stock(q, kv, indices, sm_scale, d_v)

    o_acc, l_acc = _workspace(int(q.shape[0]), int(q.shape[1]), splits, q.device)
    return _EXT.fwd_split(q, kv, indices, sm_scale, d_v, o_acc, l_acc, splits)
