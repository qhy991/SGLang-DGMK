"""Verify the split-KV DSA decode kernel as SGLang will actually call it.

The kernel-harness gate proves the kernel. This proves the *registration*: the
call SGLang makes is not the call the harness makes.

  harness : run({"q", "kv", "indices", "sm_scale", "d_v"})
  SGLang  : run({"q", "kv", "indices", "sm_scale"})     <- no d_v

(dsa_backend.py:2305-2313, inside _forward_flashmla_sparse, reached before the
head-padding block.) So d_v has to be inferred, and this checks that inference
against the stock kernel on the same inputs.

Also checks the two properties that make the change safe to land:
  * default-OFF   -- lookup() returns None unless the glm52_opt env gates are set
  * fallback      -- every shape the kernel declines still returns the stock answer
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, "/mnt/b300-shared/home/qinhaiyan/wwxq/SGLang-DGMK/python")

from sglang.srt.layers.glm52_opt.archive_loader import load_run_fn  # noqa: E402
from sglang.srt.layers.glm52_opt.registry import lookup  # noqa: E402
from sgl_kernel.flash_mla import flash_mla_sparse_fwd  # noqa: E402

REF = "best-splitkv-0730/dsa_decode_attn"
S_KV, D_QK, D_V, TOPK, H_Q = 65536, 576, 512, 2048, 64
rc = 0


def report(name, ok, detail=""):
    global rc
    if not ok:
        rc = 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}", flush=True)


print("\n=== 1. default-OFF ===", flush=True)
for var in ("SGLANG_GLM52_OPT", "SGLANG_GLM52_OPT_PROFILE", "SGLANG_GLM52_OPT_OPS"):
    os.environ.pop(var, None)
import sglang.srt.layers.glm52_opt.config as cfg  # noqa: E402
cfg.reset_cache() if hasattr(cfg, "reset_cache") else None
report("lookup() is None with no glm52_opt env", lookup("dsa_decode_attn", "decode", m=16) is None)

print("\n=== 2. numerics through SGLang's call signature (no d_v) ===", flush=True)
run = load_run_fn(REF)
torch.manual_seed(0)
kv = torch.randn(S_KV, 1, D_QK, dtype=torch.bfloat16, device="cuda")

for M in (16, 32):
    q = torch.randn(M, H_Q, D_QK, dtype=torch.bfloat16, device="cuda")
    idx = torch.stack([torch.randperm(S_KV, device="cuda")[:TOPK]
                       for _ in range(M)]).view(M, 1, TOPK).to(torch.int32)
    sm_scale = D_QK ** -0.5

    ref, _, _ = flash_mla_sparse_fwd(q=q, kv=kv, indices=idx, sm_scale=sm_scale, d_v=D_V)
    got = run({"q": q, "kv": kv, "indices": idx, "sm_scale": sm_scale})   # exactly SGLang's dict

    shape_ok = tuple(got.shape) == tuple(ref.shape) and got.dtype == ref.dtype
    x, y = ref.double().reshape(-1), got.double().reshape(-1)
    calc_diff = (1 - 2 * (x * y).sum() / (x * x + y * y).sum()).item()
    report(f"M={M} shape/dtype {tuple(got.shape)} {got.dtype}", shape_ok)
    report(f"M={M} calc_diff {calc_diff:.3e} <= 5e-6", abs(calc_diff) <= 5e-6)

print("\n=== 3. fallback on shapes the kernel declines ===", flush=True)
q = torch.randn(16, H_Q, D_QK, dtype=torch.bfloat16, device="cuda")
idx = torch.stack([torch.randperm(S_KV, device="cuda")[:TOPK]
                   for _ in range(16)]).view(16, 1, TOPK).to(torch.int32)
sm_scale = D_QK ** -0.5
ref, _, _ = flash_mla_sparse_fwd(q=q, kv=kv, indices=idx, sm_scale=sm_scale, d_v=D_V)

cases = {
    # M not in the tuned split table -> splits is None -> stock
    "untuned M=8": dict(q=q[:8], kv=kv, indices=idx[:8]),
    # topk not a whole multiple of splits*64 -> stock
    "topk=1984 (not 8*64-divisible)": dict(q=q, kv=kv, indices=idx[..., :1984].contiguous()),
}
for name, kw in cases.items():
    try:
        out = run({**kw, "sm_scale": sm_scale})
        r, _, _ = flash_mla_sparse_fwd(q=kw["q"], kv=kw["kv"], indices=kw["indices"],
                                       sm_scale=sm_scale, d_v=D_V)
        report(f"{name} -> stock, bit-exact", torch.equal(out, r),
               f"shape {tuple(out.shape)}")
    except Exception as exc:  # a fallback that raises is worse than a slow one
        report(f"{name} -> stock", False, f"{type(exc).__name__}: {exc}")

# h_q != 64 exercises the padding path inside the fallback.
try:
    q8 = torch.randn(16, 8, D_QK, dtype=torch.bfloat16, device="cuda")
    out = run({"q": q8, "kv": kv, "indices": idx, "sm_scale": sm_scale})
    report("h_q=8 -> stock with head padding", tuple(out.shape) == (16, 8, D_V),
           f"shape {tuple(out.shape)}")
except Exception as exc:
    report("h_q=8 -> stock with head padding", False, f"{type(exc).__name__}: {exc}")

print("\nRESULT:", "ALL PASS" if rc == 0 else "FAILURES", flush=True)
sys.exit(rc)
