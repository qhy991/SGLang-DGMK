# MoE W2 hotspot: graph-only dispatch

## What this is

`KernelSpec.graph_only` restricts a hotspot candidate to CUDA-graph capture.
The `moe_down_proj` hotspot spec is the only spec that sets it. Outside
capture, `try_dispatch_moe_masked` returns `False` before the ABI check and
before the hit/miss lock, so eager decode stays on the unmodified stock path
with no provider launch.

`SGLANG_GLM52_W2_GRAPH_ONLY` defaults on. Set it to `0` to force eager
selection for a diagnostic leaf measurement.

## Why

The MoE W2 decode candidate's device kernel is genuinely faster than stock, but
the API-v1 Python provider path costs roughly 23 us per call. That turned a
1.14x device-kernel win into **0.80x** on the selected eager leaf and made the
eager containing-region gate arithmetically unreachable. Production decode is
CUDA-graph-bound, and graph replay executes no Python at all, so restricting
selection to capture keeps the device win and leaves eager on stock.

This mirrors what the FlashMLA hotspot does for `dsa_decode_attn` on its own
branch.

## Verified behaviour

CPU contracts live in `test/registered/kernels/test_glm52_hotspot_registry.py`.
A device contract in the Kernel-Harness worktree
(`serving_native/moe_w2_graph_only_gpu_contract.py`) proved on one B200:

| Property | Result |
| --- | --- |
| eager, graph-only on | declines; 0 provider attempts; caller output untouched |
| eager, `GRAPH_ONLY=0` | selects; exactly 1 provider attempt |
| capture, either setting | 1 graph node carrying the candidate symbol, no forbidden nodes |
| capture on vs off | identical node count, node types, kernel identities |
| eager containing region | 0 provider attempts; estimators 0.9987-1.0068 |

The fourth row matters for measurement: because the captured graph is identical
either way, a graph lane timed with `GRAPH_ONLY=0` replays exactly what
production graph-only replays.

## Status of the W2 candidate itself

The integration is kept, but it did **not** promote a candidate. Under
graph-only the BM16 candidate clears the graph leaf gate for all four
expected-M hints (pooled geomean 1.1008) and fails the graph containing-region
gate for every hint (worst per-series estimators 1.0256-1.0281 against a
required 1.03). NCU shows the kernel is DRAM-bandwidth-bound at 99.26% of its
achievable floor, so no bounded kernel change can close the gap.

The `hotspot_candidates` profile stays default-off and stock remains active.
Full evidence:
`glm52-hotspot-goal-runs/tasks/moe_w2_ptx_graph_only/evidence/FINAL_REPORT.md`.

## Round 2: the region cannot be rescued (`moe_down_proj_decode_r2`)

Round-2 re-measured all eight lanes on a **different** B200 and reproduced the
outcome: graph leaf passes for all four hints (12/12 series, worst 1.07089),
graph containing region fails for all four (8/12 series, worst 1.02746-1.02998).

It also replaced round-1's circular ceiling argument. Round-1 derived the
candidate's "achievable floor" from the candidate's own bandwidth, which can
only ever conclude "bandwidth-bound". Measured independently, the candidate's
**6.134 TB/s is the highest rate seen anywhere on this GPU** — above the stock
W13 GEMM beside it in the same graph (6.022 TB/s) and above the best generic
streaming read (5.776 TB/s). Remaining headroom in the whole W2 kernel is
**0.631 us**.

The one structural rescue route — begin W2's 403 MB static-weight stream during
the preceding SwiGLU/quant kernel, whose 7.04 us leaves DRAM ~97% idle — is
blocked by occupancy, not by correctness. PDL is on and working, but the
producer launches **8192 CTAs x 256 threads** and saturates every SM, while a W2
CTA needs **230,188 B of shared memory**, i.e. essentially a whole SM. W2 CTAs
therefore cannot become resident until the producer drains, and the realized
overlap is only **0.192 us**. A prefetch issued before the dependency wait would
have nothing to run in. Changing that would require changing the stock
activation kernel's launch geometry, which the region gate forbids.

Disposition `no-replacement`; zero identities built. Evidence:
`glm52-hotspot-goal-runs/tasks/moe_down_proj_decode_r2/evidence/FINAL_REPORT.md`.
