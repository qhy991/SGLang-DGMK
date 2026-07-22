# Testing: HIP fused topk for DSA indexer decode (`fused_topk_hip.py`)

This branch (`decode-fusion-r1`) adds an optimized DSA indexer decode topk for
**ROCm/HIP only**. This file tells an agent how to verify and benchmark it.

## What the optimization does

In the DSA indexer decode path, the topk over the score row is computed by
`masked_fill`-ing the full width (`N=65536`) and then `torch.topk`. The fused
path instead runs `torch.topk` on a **fixed-width prefix (8192)** with a GPU
valid mask, skipping the masked-fill over the tail. CUDA-graph safe (no CPU sync
on `lengths`).

**Valid only when every valid index falls inside the prefix**, i.e. current
sequence length `< 8192`. The E2E decode campaign this was built for stays well
under that: input≈4096 + output≈128 ⇒ length ≈4224.

## When it activates (read this before testing)

All of the following must hold, otherwise the fused path is a no-op and the
unfused path runs unchanged:

- `SGLANG_DSA_HIP_FUSED_TOPK=1` (env var; off by default)
- ROCm/HIP (`is_hip()` true)
- `score` is `[B=1, N]`, `float32`, contiguous
- `0 < topk < N`
- all valid columns fall within the first 8192

**Critical routing note:** the dispatch lives inside `_topk_unfused()` in
`dsa_topk_backend.py`. That function is only reached from the **`torch`** DSA
topk backend. The **default** `--dsa-topk-backend sgl-kernel` calls
`sgl_kernel.fast_topk_v2` and **never reaches this optimization**. So in E2E
serving you must force `--dsa-topk-backend torch` (see Test B).

## Test A — standalone correctness + microbench (no model needed)

Fastest test. Runs the real dispatch function on synthetic DSA decode shapes.
Run on a ROCm node (the fused path is `is_hip()`-gated).

```python
# bench_fused_topk_hip.py  — run: SGLANG_DIR=<sglang checkout> PYTHONPATH=$SGLANG_DIR/python python bench_fused_topk_hip.py
import os, torch
from sglang.srt.layers.attention.dsa.dsa_topk_backend import _topk_unfused

dev, N, TOPK, B = "cuda", 65536, 2048, 1
SEQ = 4224  # input 4096 + output 128; must be < prefix 8192
lengths = torch.full((B,), SEQ, dtype=torch.int32, device=dev)
row_starts = torch.zeros(B, dtype=torch.int32, device=dev)
torch.manual_seed(0)
score = torch.randn(B, N, dtype=torch.float32, device=dev)

# correctness: fused vs unfused must agree while length < prefix
os.environ["SGLANG_DSA_HIP_FUSED_TOPK"] = "0"
ref = _topk_unfused(score, lengths, TOPK, row_starts, torch.topk, {"dim": -1})
os.environ["SGLANG_DSA_HIP_FUSED_TOPK"] = "1"
fus = _topk_unfused(score, lengths, TOPK, row_starts, torch.topk, {"dim": -1})
print("correctness match:", bool(torch.equal(ref.sort().values, fus.sort().values)))

def bench(flag, iters=300):
    os.environ["SGLANG_DSA_HIP_FUSED_TOPK"] = flag
    fn = lambda: _topk_unfused(score, lengths, TOPK, row_starts, torch.topk, {"dim": -1})
    for _ in range(30): fn()
    torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000  # us

print(f"unfused full-65536: {bench('0'):.2f} us")
print(f"fused  prefix-8192: {bench('1'):.2f} us")
```

Expected (measured on MI300X, SEQ=4224, TOPK=2048):
- `correctness match: True`
- unfused ≈ 168 µs, fused ≈ 140 µs → **~1.2x kernel speedup**

Also sweep `SEQ` to confirm: identical output while `SEQ < 8192`; once `SEQ ≥ 8192`
the fused path returns `-1` for the out-of-prefix tail by design (it no longer
matches the unfused reference — that is the documented limitation, not a bug).

## Test B — E2E GLM-5.2 decode serving A/B

This is the test that actually exercises the optimization inside a running
server. Reuse the existing GLM-5 DSA E2E test as the harness:

- `test/registered/models_e2e/test_dsa_glm5_tp_mtp.py`
- `test/registered/models_e2e/test_dsa_glm5_dp_mtp.py`

Run the decode workload twice — once with the optimization off (baseline), once
on — keeping everything else identical:

```bash
# Baseline (default sgl-kernel topk OR torch topk without the fused path)
SGLANG_DSA_HIP_FUSED_TOPK=0  python -m sglang.launch_server \
    --model-path <GLM-5.2> --dsa-topk-backend torch ...   # then drive decode, record per-token latency

# Optimized
SGLANG_DSA_HIP_FUSED_TOPK=1  python -m sglang.launch_server \
    --model-path <GLM-5.2> --dsa-topk-backend torch ...   # same workload, record per-token latency
```

Requirements for the comparison to be meaningful:
- `--dsa-topk-backend torch` on **both** runs (without it the fused path is
  never reached — see the routing note above).
- Decode batch size 1, prompt ≈4096 tokens, generate ≈128 so the running length
  stays `< 8192` and the prefix assumption holds.
- Compare per-token decode latency (median/p10), not just throughput.

## Confirming the fused path actually ran

The dispatch is silent. To verify in a one-off run, add a temporary print inside
`maybe_dispatch_fused_topk` (return path) or check that toggling
`SGLANG_DSA_HIP_FUSED_TOPK` between `0`/`1` changes the measured topk latency in
Test A — if it does, dispatch is wired; if not, a precondition
(backend/`is_hip`/batch/dtype/length) is failing.

## Caveats

- ROCm-only (`is_hip()` gate). On CUDA the fused path never engages.
- Helps only the `torch` (unfused) topk backend. If your serving uses
  `sgl-kernel`'s `fast_topk_v2`, this optimization is out of that path.
- Prefix assumption: valid indices must be within the first 8192 columns. For
  sequences longer than that, raise `_PREFIX_LEN` or fall back to the unfused
  path (the dispatch already returns `None` when it can't apply — it does not
  silently drop indices beyond the prefix, but the result is only valid
  in-prefix).
