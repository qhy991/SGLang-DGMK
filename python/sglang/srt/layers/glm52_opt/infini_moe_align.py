"""infini: M-tile alignment policy for the GLM-5.2 MoE masked grouped GEMM.

WHY
DeepGEMM's masked-grouped heuristic hard-codes the M tile to
``get_mk_alignment_for_contiguous_layout()`` (default 128): the MGroupedMasked branch
short-circuits before config enumeration (csrc/heuristics/sm100.hpp:37), so this path
emits exactly ONE candidate and never compares. At decode only a handful of rows per
expert are live, so a 128-row M tile pads the A read, the UMMA and the epilogue store
by up to ~25x. DeepGEMM's own
``get_theoretical_mk_alignment_for_contiguous_layout()`` returns 32 for
``expected_m`` in {4, 8, 16} -- the masked path simply never calls it.

Choosing the alignment from ``expected_m`` is **bit-exact** (calc_diff == 0.0 on every
shape measured) and worth up to 1.14x on the MoE grouped GEMM.

MEASURED (B300 sm_103, E=32 EP8, K/N = 6144/2048 and 2048/6144, device time with host
launch overhead excluded; cross-checked against nsys cuda_gpu_kern_sum). Production
``expected_m`` follows deepep.py:675,
``(tokens_per_rank * ep_size * topk + num_experts) // num_experts``:

    bs   expected_m  max_rows | align=128   64      32      16     -> best
    16       5           8    |   74.1     --      66.7    65.4      16  (1.13-1.14x)
    32       9          13    |   75.5     --      67.7    66.7      16  (1.13-1.14x)
    64      17          27    |   75.4    70.4     67.9    69.9      32  (1.11x)
    128     33          46    |   75.4    70.2     68.9    82.0      32  (1.08-1.09x)
    256     65          84    |   75.8    76.9     84.1   101.7     128  (stock; all lose)

So the win is real but **bounded**: alignment 16 REGRESSES to 0.92x at expected_m=33
and 0.75x at expected_m=65. The bucket table below encodes exactly the measured
crossovers and falls back to stock (128) above them. Do not widen a bucket without
re-measuring -- the failure mode is a silent throughput loss at high concurrency.

EXACT B300 PHYSICAL-SLAB GRAPH GATE (2026-08-02)
The serving-native follow-up preserves the traced ``[E=32,T=8192]`` expert slab
instead of compacting it to the live rows.  With alignment 16, every production
capture bucket M={1,2,4,8,12,16} passed graph correctness on an unseen seed and
the adjacent paired p10 >= 1.03 gate:

    W13 graph: geomean 1.0808x, minimum p10 1.0549x
    W2  graph: geomean 1.0947x, minimum p10 1.0316x

Execution mode matters.  At M16 the same scoped alignment measured W13 eager
p10=1.0263x (below gate) and W2 eager p10=0.9474x (regression), while graph p10
was 1.0619x and 1.0753x respectively.  Therefore ``combined_winners`` marks both
MoE specs graph-only: capture bakes the selected kernel into every audited graph;
eager/tail calls decline before this context manager and execute stock alignment
128.  Do not remove that execution-mode boundary based on graph results alone.

PREFILL MUST NEVER SEE A SMALL ALIGNMENT. The knob is process-global and also governs
contiguous-layout m-grouped GEMMs:

    moe_up   prefill M=1024   111.8 -> 307.8 us  0.36x  (align=16)
    moe_down prefill M=1024   115.7 -> 354.3 us  0.33x  (align=16)
    moe_down prefill M=1024   115.7 -> 478.7 us  0.24x  (align=32)

Hence the strict get/set/restore in ``infini_mk_alignment`` -- the global is never
left modified after the call returns, so a stock prefill GEMM that falls through
``try_dispatch_moe_masked`` always runs at 128.

COST AND CUDA GRAPHS
``set_mk_alignment_for_contiguous_layout`` is a host-side config setter and launches
no GPU work, so it is graph-capture safe: during capture it selects which kernel gets
baked into the graph, and during replay the host code does not run at all. The
get+set+restore triple costs ~2 us of host time (a set+restore pair measured at
1.7 us), which for graph-captured decode is paid only at capture. The production
``combined_winners`` profile declines eager W13/W2 by default; the per-op
``SGLANG_GLM52_W13_GRAPH_ONLY=0`` and ``SGLANG_GLM52_W2_GRAPH_ONLY=0`` switches
exist only for diagnostic eager replay. ``SGLANG_GLM52_INFINI_MOE_ALIGN=0``
disables the policy entirely.

Exact raw results: ``bench_results/b300_moe_alignment16_graph_bucket_sweep_20260802w``
Earlier independent B200 result (same 16 < 32 < 64 < 128 ordering, shelved because the
selector is process-global): glm52_opt/history/e2e_candidates_20260723/07_moe_w2_decode_bm16
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Optional

import deep_gemm

logger = logging.getLogger(__name__)

STOCK_MK_ALIGNMENT = 128

# (max expected_m inclusive, M-tile alignment). Ordered, first match wins.
# Boundaries sit between measured points: 9 -> 16 wins, 17 -> 32 wins (so 12);
# 33 -> 32 wins, 65 -> nothing wins (so 40).
INFINI_MOE_ALIGN_BUCKETS: tuple[tuple[int, int], ...] = (
    (12, 16),
    (40, 32),
)

_HAVE_KNOB = hasattr(deep_gemm, "set_mk_alignment_for_contiguous_layout") and hasattr(
    deep_gemm, "get_mk_alignment_for_contiguous_layout"
)
if not _HAVE_KNOB:
    logger.warning(
        "glm52_opt infini_moe_align: deep_gemm lacks "
        "get/set_mk_alignment_for_contiguous_layout; MoE M-tile policy disabled"
    )


def infini_moe_align_enabled() -> bool:
    """On by default whenever glm52_opt is on; ``...INFINI_MOE_ALIGN=0`` disables."""
    if not _HAVE_KNOB:
        return False
    return os.environ.get("SGLANG_GLM52_INFINI_MOE_ALIGN", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def infini_select_mk_alignment(expected_m: int) -> Optional[int]:
    """Measured-best M tile for this ``expected_m``, or None to leave stock alone."""
    if not infini_moe_align_enabled():
        return None
    try:
        em = int(expected_m)
    except (TypeError, ValueError):
        return None
    if em <= 0:
        return None
    for limit, alignment in INFINI_MOE_ALIGN_BUCKETS:
        if em <= limit:
            return alignment
    return None


@contextmanager
def infini_mk_alignment(expected_m: int):
    """Scope the process-global M-tile alignment to one masked-grouped GEMM call.

    Restores unconditionally in ``finally``: leaving a small alignment installed
    would cut prefill grouped-GEMM throughput to ~0.33x (see module docstring).
    """
    want = infini_select_mk_alignment(expected_m)
    if want is None:
        yield None
        return
    try:
        prev = deep_gemm.get_mk_alignment_for_contiguous_layout()
    except Exception as exc:  # never let the policy break the GEMM
        logger.warning("glm52_opt infini_moe_align: get alignment failed: %s", exc)
        yield None
        return
    if prev == want:
        yield want
        return
    try:
        deep_gemm.set_mk_alignment_for_contiguous_layout(want)
    except Exception as exc:
        logger.warning(
            "glm52_opt infini_moe_align: set alignment %s failed: %s", want, exc
        )
        yield None
        return
    try:
        yield want
    finally:
        try:
            deep_gemm.set_mk_alignment_for_contiguous_layout(prev)
        except Exception:  # pragma: no cover - would leave global state dirty
            logger.exception(
                "glm52_opt infini_moe_align: FAILED to restore alignment to %s; "
                "prefill grouped GEMMs may run slow until the process restarts",
                prev,
            )
