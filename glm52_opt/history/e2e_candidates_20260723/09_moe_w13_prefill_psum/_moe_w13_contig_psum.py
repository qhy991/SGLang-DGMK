"""Shared fail-closed runner for normal-DeepEP fused-W13 PSUM experiments."""

from sglang.srt.layers.deep_gemm_wrapper.entrypoint import (
    grouped_gemm_nt_f8f8bf16_contig,
)


_PREPARED = set()


def _preflight_key(inputs, runtime, metadata):
    return (
        id(inputs),
        id(runtime),
        metadata["compiled_dims"],
        metadata["ensure_zero_padding"],
        metadata["expected_m_for_psum_layout"],
    )


def _compatible(inputs, runtime) -> bool:
    torch = runtime.torch
    p = runtime.workload.params
    scale_k = p["k"] // 512
    return (
        runtime.workload.name == "moe_w13_grouped_prefill_m4096"
        and runtime.world_size == 1
        and runtime.deep_gemm_config is not None
        and runtime.deep_gemm_config["pdl"] is True
        and runtime.deep_gemm_config["mk_alignment"] == p["expert_alignment"] == 128
        and runtime.deep_gemm_config["supports_psum_layout"]
        and inputs["activation_fp8"].shape == (p["all_tokens"], p["k"])
        and inputs["activation_fp8"].is_contiguous()
        and inputs["activation_fp8"].dtype == torch.float8_e4m3fn
        and inputs["activation_fp8"].is_cuda
        and inputs["activation_scale"].shape == (p["all_tokens"], scale_k)
        and inputs["activation_scale"].stride() == (1, p["all_tokens"])
        and inputs["activation_scale"].dtype == torch.int32
        and inputs["activation_scale"].is_cuda
        and inputs["weight_fp8"].shape
        == (p["experts_per_rank"], p["n"], p["k"])
        and inputs["weight_fp8"].is_contiguous()
        and inputs["weight_fp8"].dtype == torch.float8_e4m3fn
        and inputs["weight_fp8"].is_cuda
        and inputs["weight_scale"].shape
        == (p["experts_per_rank"], p["n"], scale_k)
        and inputs["weight_scale"].stride()
        == (p["n"] * scale_k, 1, p["n"])
        and inputs["weight_scale"].dtype == torch.int32
        and inputs["weight_scale"].is_cuda
        and inputs["out"].shape == (p["all_tokens"], p["n"])
        and inputs["out"].is_contiguous()
        and inputs["out"].dtype == torch.bfloat16
        and inputs["out"].is_cuda
        and inputs["m_indices"].shape == (p["all_tokens"],)
        and inputs["m_indices"].dtype == torch.int32
        and inputs["m_indices"].is_cuda
        and inputs["m_indices"].is_contiguous()
        and inputs["psum_layout"].shape == (p["experts_per_rank"],)
        and inputs["psum_layout"].dtype == torch.int32
        and inputs["psum_layout"].is_cuda
        and inputs["psum_layout"].is_contiguous()
        and inputs["recipe_a"] is None
        and inputs["recipe_b"] is None
    )


def prepare_psum(inputs, runtime, metadata):
    """Validate the complete ABI outside the measured CUDA-event interval."""
    if not _compatible(inputs, runtime):
        raise RuntimeError(
            "PSUM candidate is inactive: workload, packed ABI, PDL, or "
            "128-row alignment does not match the production probe"
        )
    expected_m = metadata["expected_m_for_psum_layout"]
    if expected_m not in (None, runtime.workload.params["expected_m_per_expert"]):
        raise RuntimeError(
            "PSUM candidate metadata must use the workload's expected M or None"
        )
    _PREPARED.add(_preflight_key(inputs, runtime, metadata))
    return {
        "active": True,
        "fallback": False,
        "mk_alignment": runtime.deep_gemm_config["mk_alignment"],
        "layout_elements": inputs["psum_layout"].numel(),
        "expected_m_for_psum_layout": expected_m,
    }


def run_psum(inputs, runtime, metadata):
    if _preflight_key(inputs, runtime, metadata) not in _PREPARED:
        prepare_psum(inputs, runtime, metadata)
    grouped_gemm_nt_f8f8bf16_contig(
        (inputs["activation_fp8"], inputs["activation_scale"]),
        (inputs["weight_fp8"], inputs["weight_scale"]),
        inputs["out"],
        inputs["psum_layout"],
        recipe_a=inputs["recipe_a"],
        recipe_b=inputs["recipe_b"],
        compiled_dims=metadata["compiled_dims"],
        use_psum_layout=True,
        ensure_zero_padding=metadata["ensure_zero_padding"],
        expected_m_for_psum_layout=metadata["expected_m_for_psum_layout"],
    )
    return runtime.moe_contig_observed(inputs["out"])
