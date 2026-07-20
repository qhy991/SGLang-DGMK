"""Candidate 4: compiled_dims='mnk' + PDL (programmatic dependent launch).

Fresh levers from the sibling wave. Internal probe (cold-L2, B200 sm100):
  compiled_dims='mnk' consistently >= 'nk' default; PDL hides the prologue.
  M=1024: mnk+pdl -> ~1.065x   M=2048: mnk -> ~1.02x   M=4096: mnk+pdl -> ~1.015x
num_sms reduction (146/144/132/128) REGRESSED at all shapes -> keep default 148.
Correctness-safe: same masked kernel, only config knobs change.
"""
from __future__ import annotations
import deep_gemm

# Set once at import scope (load-time), not per timed call.
deep_gemm.set_pdl(True)


def run(inputs: dict):
    out = inputs["out"]
    deep_gemm.fp8_m_grouped_gemm_nt_masked(
        (inputs["x_fp8"], inputs["x_scale"]),
        (inputs["w_fp8"], inputs["w_scale"]),
        out, inputs["masked_m"], inputs["expected_m"],
        compiled_dims="mnk",
    )
    return out
