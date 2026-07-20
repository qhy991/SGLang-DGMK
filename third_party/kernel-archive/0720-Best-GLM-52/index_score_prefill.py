from __future__ import annotations
import deep_gemm

# Fresh lever #3: programmatic dependent launch (PDL). Hides the prologue.
# Set once at import scope (global _C state), correctness-safe (output identical).
try:
    deep_gemm.set_pdl(True)
except Exception:
    pass


def run(inputs: dict):
    return deep_gemm.fp8_mqa_logits(
        inputs["q_fp8"], (inputs["k_fp8"], inputs["k_scale"]), inputs["weights"],
        inputs["ks"], inputs["ke"], clean_logits=False,
    )
