"""moe_up: pure PDL lever on default nk dims. num_sms=148."""
import deep_gemm
deep_gemm.set_pdl(True)
def run(inputs: dict):
    out = inputs["out"]
    deep_gemm.set_pdl(True)
    deep_gemm.fp8_m_grouped_gemm_nt_masked(
        (inputs["x_fp8"], inputs["x_scale"]),
        (inputs["w_fp8"], inputs["w_scale"]),
        out, inputs["masked_m"], inputs["expected_m"])
    return out
