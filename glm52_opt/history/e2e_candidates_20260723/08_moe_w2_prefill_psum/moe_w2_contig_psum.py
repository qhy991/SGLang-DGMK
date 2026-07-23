"""Use DeepGEMM's PSUM grouped layout for normal-DeepEP prefill W2.

The PSUM endpoints are already produced by normal ``ep_scatter`` as its final
``expert_start_loc`` values. This candidate tests the W2 kernel contract without
adding a timed pack, copy, allocation, or device-to-host synchronization.
"""

from serving_native.candidates._moe_w2_contig_psum import prepare_psum, run_psum


METADATA = {
    "control": "use_psum_layout",
    "ensure_zero_padding": False,
    "expected_m_for_psum_layout": 1024,
    "compiled_dims": "nk",
}



def prepare(inputs, runtime):
    return prepare_psum(inputs, runtime, METADATA)


def run(inputs, runtime):
    return run_psum(inputs, runtime, METADATA)
