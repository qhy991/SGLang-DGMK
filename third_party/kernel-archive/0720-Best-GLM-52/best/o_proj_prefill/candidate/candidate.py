"""GLM-5.2 Attention O Projection (prefill) — final PARTIAL WIN candidate.

Vendored from campaign final: `compiled_dims="mnk"` + `deep_gemm.set_pdl(True)`.
Authoritative gate: ~1.033 / 1.019 / 1.013× vs harness reference (exit 0).

NOTE: The earlier archive mistakenly pointed `candidate/` at the task8 Triton
spike (~0.21–0.25×). That exploratory path lives under
`candidates/task8_triton_spike/`; this file is the real final.
"""
from __future__ import annotations

import deep_gemm

# Programmatic dependent launch: overlaps the grid-launch latency of the fp8 GEMM
# with the tail of the preceding scale-layout transform kernels. Measured on B200
# as the only device-time lever that helps this already-tensor-core-bound path.
# Global, idempotent, and value-independent — set once at import, outside run().
try:
    deep_gemm.set_pdl(True)
except Exception:  # pragma: no cover - older builds without the toggle
    pass


def run(inputs: dict):
    # Same math and same frozen input dict as the reference, computed statelessly:
    # no re-quantization, no rebuilt/aliased/cached operand copies. The only change
    # is compile-time specialization of the M/N/K extents, which lets DeepGEMM's JIT
    # emit tighter code for these frozen shapes. Specialization compiles during
    # warmup (outside the timed window); any non-frozen M simply recompiles and
    # still computes correctly, so no explicit fallback branch is needed.
    out = inputs["out"]
    deep_gemm.fp8_gemm_nt(
        (inputs["x_fp8"], inputs["x_scale"]),
        (inputs["w_fp8"], inputs["w_scale"]),
        out,
        compiled_dims="mnk",
    )
    return out
