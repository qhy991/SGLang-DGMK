"""Bounded CUDA fallback for GLM-5.2 Task-25 decode."""

from __future__ import annotations

from pathlib import Path

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)


VARIANT = "cuda_valid_cta"


@cache_once
def _jit_module():
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "task25_silu_mul_quant_valid_cta",
        *args,
        cuda_files=["deepseek_v4/task25_silu_and_mul_masked_post_quant.cuh"],
        cuda_wrappers=[
            (
                "run",
                f"Task25SiluAndMulMaskedPostQuantKernel<{args}>::run",
            )
        ],
        extra_cuda_cflags=["-use_fast_math"],
    )


def launch_into(
    gateup_output: torch.Tensor,
    output: torch.Tensor,
    scale_storage: torch.Tensor,
    masked_m: torch.Tensor,
    *,
    num_real_tokens: int,
) -> None:
    """Launch the exact stock kernel body on only live routed assignments."""

    _jit_module().run(
        gateup_output,
        output,
        scale_storage,
        masked_m,
        8,
        num_real_tokens,
    )


def artifact_paths() -> tuple[str, ...]:
    jit_root = Path(__file__).resolve().parents[3] / "jit_kernel"
    return (
        str(Path(__file__).resolve()),
        str(
            jit_root
            / "csrc"
            / "deepseek_v4"
            / "task25_silu_and_mul_masked_post_quant.cuh"
        ),
        str(
            jit_root
            / "csrc"
            / "deepseek_v4"
            / "silu_and_mul_masked_post_quant.cuh"
        ),
    )
