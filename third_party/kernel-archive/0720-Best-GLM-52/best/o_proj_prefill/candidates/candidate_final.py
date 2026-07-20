"""GLM-5.2 Attention O Projection (prefill) — the one file to edit for this task.

This file is the DEFAULT candidate, not the only one: `./run.sh --candidate PATH`
tests any .py defining run(inputs), from anywhere on disk, without touching the task.
Editing this file is just the convenient path.

Run `./run.sh --describe` for the full contract. The short version:

`inputs` is the frozen dict from glm52_ops.build_inputs. The very same dict feeds
the reference, so do NOT re-quantize, re-seed, or rebuild any tensor inside
run() — that would measure a different problem than the one the gate checked.

Tensors at M=1024:

    x_fp8            (1024, 16384)            torch.float8_e4m3fn
    x_scale          (1024, 128)              torch.float32
    w_fp8            (6144, 16384)            torch.float8_e4m3fn
    w_scale          (48, 128)                torch.float32
    out              (1024, 6144)             torch.bfloat16

Return the output. Correctness against glm52_ops.reference on these inputs is
FlashMLA's three-layer check: matching inf/nan positions, then every element
abs_err < abs_tol OR rel_err < 0.0157, then DeepGEMM's calc_diff
<= 5e-06. `./run.sh --describe` prints all of it.

`inputs["out"]` is pre-allocated and may be written in place, but the harness
NaN-poisons it before calling run(): returning it unwritten FAILS.

Baseline to beat: the call below, timed CUPTI cold-L2 on these same inputs.

    ./run.sh
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
