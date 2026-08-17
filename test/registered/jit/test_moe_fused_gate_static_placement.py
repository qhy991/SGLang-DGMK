import pytest
import torch

from sglang.jit_kernel.moe_fused_gate import moe_fused_gate


@pytest.mark.parametrize("m,valid", [(1, 1), (17, 13), (10048, 10000)])
def test_glm52_static_placement_matches_stock_postprocess(m: int, valid: int):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    torch.manual_seed(20260812 + m)
    scores = torch.randn((m, 256), device="cuda", dtype=torch.float32)
    bias = torch.randn((256,), device="cuda", dtype=torch.float32) * 0.01
    logical_to_physical = torch.randperm(
        256, device="cuda", dtype=torch.int64
    ).contiguous()
    non_padded = torch.tensor(valid, device="cuda", dtype=torch.int32)

    ref_weights, ref_ids = moe_fused_gate(
        scores,
        bias,
        topk=8,
        scoring_func="sigmoid",
        num_fused_shared_experts=0,
        renormalize=True,
        routed_scaling_factor=2.5,
    )
    ref_ids = ref_ids.to(torch.int64)
    ref_ids = logical_to_physical[ref_ids]
    ref_ids[valid:, :] = -1

    got_weights, got_ids = moe_fused_gate(
        scores,
        bias,
        topk=8,
        scoring_func="sigmoid",
        num_fused_shared_experts=0,
        renormalize=True,
        routed_scaling_factor=2.5,
        num_token_non_padded=non_padded,
        output_ids_int64=True,
        logical_to_physical_map=logical_to_physical,
    )

    torch.testing.assert_close(got_weights, ref_weights, rtol=0, atol=0)
    torch.testing.assert_close(got_ids, ref_ids, rtol=0, atol=0)
