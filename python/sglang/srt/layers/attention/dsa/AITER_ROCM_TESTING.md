# Testing: aiter paths for GLM-5.2 on ROCm (gfx942 / MI300X)

This branch (`decode-fusion-r1`) routes several GLM-5.2 attention/quantization
operators through **aiter** on ROCm. This file tells an agent how to exercise
and A/B-test each path. (For the separate HIP fused-topk decode optimization,
see `FUSED_TOPK_HIP_TESTING.md` in this same directory.)

## What was added

| Area | Module(s) | What it does |
|---|---|---|
| DSA sparse MLA (decode) | `dsa/dsa_aiter_sparse_mla.py`, routed in `dsa_backend.py` | Sparse MLA decode via aiter `unified_attention_sparse_mla` (TS=64/ns=1) |
| DSA Triton sparse MLA (decode) | `triton_sparse_mla_fwd.py` | Triton sparse MLA decode kernel for MI300X (bf16 KV), shared-KV |
| MLA prefill (split-KV) | `aiter_mla_prefill_split.py`, routed in `aiter_backend.py` | Raises `num_kv_splits` above aiter default (=1) for better MI300X occupancy on GLM-5 absorbed MLA prefill; fuses LSE combine in Triton |
| DSV4 FP8 attention | `dsa/dsv4_attn_gfx942.py` | PyTorch reference for DSV4 FP8 sparse decode, replacing the tilelang kernel (which emits NVIDIA WGMMA, incompatible with gfx942) |
| FP8 linear (quant) | `quantization/fp8.py`, `quantization/fp8_utils.py` | Route block-FP8 / per-1x128 FP8 linear through `aiter.ops.gemm_op_a8w8` on gfx942 |
| infra | `environ.py`, `mem_cache/memory_pool.py`, `hip_flash_mla.py` | New env flags, KV pool tweaks, flashmla backend override |

## Prerequisites

- ROCm / gfx942 (MI300X). Every path is `is_hip()`-gated.
- `aiter` importable (the sglang-ROCm kernel library). The aiter MLA / GEMM ops must JIT successfully on this node.
- `SGLANG_USE_AITER=1` (global aiter enable; required for the FP8 + MLA prefill paths).

## Flag matrix

| Flag | Default | Controls |
|---|---|---|
| `SGLANG_USE_AITER` | unset | Global aiter enable. Required for FP8 aiter GEMM + MLA prefill split. |
| `SGLANG_DSA_USE_AITER_SPARSE_MLA` | `0` (off) | **Opt-in** aiter sparse MLA decode. Off → original path. |
| `SGLANG_DSA_AITER_SPARSE_MLA_MIN_KV_LEN` | `2048` | Only route to aiter sparse MLA when KV length ≥ this. |
| `SGLANG_DSA_DECODE_INSTR` | `0` (off) | Per-step decode instrumentation (`[DECODE_INSTR]` to stderr). Zero overhead off. Use to confirm dispatch. |
| `SGLANG_DISABLE_GFX942_BPRESHUFFLE` | unset | Set to disable the aiter bpreshuffle FP8 linear path (for A/B). |
| `SGLANG_HACK_FLASHMLA_BACKEND` | unset | Override the flashmla backend selection (debug escape-hatch). |

⚠️ `should_use_aiter_sparse_mla()` is `@lru_cache`d — the flag is read **once at
first call**. Set `SGLANG_DSA_USE_AITER_SPARSE_MLA` (and `SGLANG_USE_AITER`)
**before launching the server / importing sglang**, not at runtime.

## Test A — dispatch smoke (no model, ~2 s)

Confirms the code is wired and gating works on this node. Verified on MI300X:

```python
# SGLANG_USE_AITER=1 must be set in the env before running
import os, torch
from sglang.srt.layers.attention.dsa.dsa_aiter_sparse_mla import (
    should_use_aiter_sparse_mla, should_route_aiter_sparse_mla)
from sglang.srt.layers.attention.aiter_mla_prefill_split import mla_prefill_split_fwd
from sglang.srt.layers.attention.triton_sparse_mla_fwd import _sparse_mla_fwd_kernel
from sglang.srt.layers.quantization.fp8_utils import _is_gfx942, _use_aiter_bpreshuffle_gfx942
print("is_hip:", torch.version.hip is not None)
print("should_use_aiter_sparse_mla:", should_use_aiter_sparse_mla())         # False unless flag set pre-import
print("should_route_aiter_sparse_mla(3000):", should_route_aiter_sparse_mla(3000))
print("_is_gfx942:", _is_gfx942(), "| fp8 aiter bpreshuffle on:", _use_aiter_bpreshuffle_gfx942)
```

Expected on MI300X with `SGLANG_USE_AITER=1`: `is_hip True`, `_is_gfx942 True`,
`_use_aiter_bpreshuffle_gfx942 True`, all 4 modules import. The sparse-MLA flags
read `False` unless `SGLANG_DSA_USE_AITER_SPARSE_MLA=1` was set before import.

## Test B — E2E GLM-5.2 DSA decode A/B (primary)

This is the test that measures real impact. Reuse the existing GLM-5 DSA E2E
tests as the harness:

- `test/registered/models_e2e/test_dsa_glm5_tp_mtp.py`
- `test/registered/models_e2e/test_dsa_glm5_dp_mtp.py`

Run the decode workload with the aiter paths **off** (baseline) then **on**,
keeping everything else identical, and compare per-token decode latency:

```bash
# Baseline: aiter sparse MLA off, FP8 aiter GEMM off
SGLANG_USE_AITER=1 SGLANG_DISABLE_GFX942_BPRESHUFFLE=1 \
SGLANG_DSA_USE_AITER_SPARSE_MLA=0 \
python -m sglang.launch_server --model-path <GLM-5.2> --dsa-decode-backend aiter ...   # drive decode, record latency

# Optimized: aiter sparse MLA on + FP8 aiter GEMM on
SGLANG_USE_AITER=1 \
SGLANG_DSA_USE_AITER_SPARSE_MLA=1 \
python -m sglang.launch_server --model-path <GLM-5.2> --dsa-decode-backend aiter ...   # same workload
```

Requirements for a meaningful comparison:
- `--dsa-decode-backend aiter` on **both** runs (the sparse-MLA path lives in
  `dsa_backend._forward_aiter`; it is not reached from other decode backends).
- Decode KV length ≥ `SGLANG_DSA_AITER_SPARSE_MLA_MIN_KV_LEN` (default 2048) —
  i.e. a prompt long enough that the sparse-MLA route actually engages. Shorter
  sequences fall through to the default path by design.
- Compare per-token decode latency (median / p10), not just throughput.

To A/B the **FP8 aiter GEMM** path independently of sparse MLA, toggle
`SGLANG_DISABLE_GFX942_BPRESHUFFLE` (it's auto-on for gfx942 when
`SGLANG_USE_AITER=1`).

## Confirming each path actually ran

Set `SGLANG_DSA_DECODE_INSTR=1` — it emits `[DECODE_INSTR] ...` lines to stderr
from `dsa_backend._run_aiter_mla_decode_fwd` / `_apply_cuda_graph_metadata`,
including which path was taken. If you see no `[DECODE_INSTR]` lines on a decode
step, the aiter decode path wasn't reached (check `--dsa-decode-backend`, the
sparse-MLA flag, and the KV-length threshold).

For the FP8 path, confirm via `SGLANG_DISABLE_GFX942_BPRESHUFFLE` toggling a
measurable latency change on a dense FP8 GEMM shape, or by profiling
(`rocprof`) for `gemm_a8w8_blockscale` / aiter GEMM kernels.

## Caveats

- ROCm/gfx942 only. On CUDA none of these paths engage.
- Sparse MLA is **opt-in** (`SGLANG_DSA_USE_AITER_SPARSE_MLA=1`); the FP8 aiter
  GEMM is **auto-on** for gfx942 when `SGLANG_USE_AITER=1` (disable with
  `SGLANG_DISABLE_GFX942_BPRESHUFFLE`).
- lru_cache on the gating flag: set env vars **before** sglang import.
- `dsv4_attn_gfx942.py` is a **PyTorch reference** (the correctness baseline for
  DSV4 FP8 sparse decode on gfx942), not an aiter call itself; it exists because
  the tilelang kernel emits WGMMA and won't compile on gfx942.
- These paths are deep in the attention/quant stacks; standalone microbenchmarks
  need model-shaped KV/index inputs — prefer the E2E A/B above.
