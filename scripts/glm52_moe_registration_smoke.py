#!/usr/bin/env python3
"""Exercise the registered GLM-5.2 W13 or W2 route through CUDA Graph replay.

Run this script through the matching same-source-stock launcher documented in
``glm52_opt/infini_kernel_hotspot_e2e.md``.  It intentionally uses zero-valued
FP8 tensors: this is a registration/graph-safety smoke, not a performance or
numerical-accuracy benchmark.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from functools import partial
from types import SimpleNamespace
from typing import Any

import torch


class _DecodeMode:
    @staticmethod
    def is_decode() -> bool:
        return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", choices=("w13", "w2"), required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    return parser.parse_args()


def _zeros_strided(
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    value = torch.empty_strided(shape, stride, dtype=dtype, device=device)
    value.zero_()
    return value


def _capture_and_replay(
    run: Callable[[], None],
    *,
    device: torch.device,
) -> None:
    side_stream = torch.cuda.Stream(device=device)
    side_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side_stream):
        run()
    torch.cuda.current_stream(device).wait_stream(side_stream)
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side_stream):
        run()
    graph.replay()
    torch.cuda.synchronize(device)


def _active_rows(out: torch.Tensor, expected_m: int) -> torch.Tensor:
    return out[:, :expected_m].clone()


def _run_w13(device: torch.device) -> dict[str, Any]:
    import deep_gemm
    from sglang.srt.layers.deep_gemm_wrapper import entrypoint
    from sglang.srt.layers.glm52_opt import w13_decode
    from sglang.srt.layers.glm52_opt.hotspot_provider import (
        initialize_hotspot_provider,
        provider_state,
    )
    from sglang.srt.layers.glm52_opt.w13_context import w13_decode_forward_scope

    entrypoint.update_deep_gemm_config(
        device.index,
        SimpleNamespace(chunked_prefill_size=8192, base_gpu_id=device.index),
    )
    initialize_hotspot_provider(device.index)
    state = w13_decode.dispatch_state()
    if not state["enabled"]:
        raise RuntimeError(f"W13 registration is not ready: {state}")

    tensors = w13_decode._allocate_warm_inputs(device)
    expected_m = 4
    tensors["masked_m"].fill_(expected_m)
    stock_out = torch.empty_like(tensors["out"])
    candidate_out = torch.empty_like(tensors["out"])
    graph_out = torch.empty_like(tensors["out"])

    candidate = w13_decode._STATE.candidate_module
    if candidate is None:
        raise RuntimeError("W13 candidate module is absent")
    original_candidate = candidate.fp8_m_grouped_gemm_nt_masked
    candidate_calls = 0

    def traced_candidate(*args, **kwargs):
        nonlocal candidate_calls
        candidate_calls += 1
        return original_candidate(*args, **kwargs)

    candidate.fp8_m_grouped_gemm_nt_masked = traced_candidate
    forward_batch = SimpleNamespace(forward_mode=_DecodeMode())
    try:
        stock_result = entrypoint.grouped_gemm_nt_f8f8bf16_masked(
            (tensors["a"], tensors["a_scale"]),
            (tensors["b"], tensors["b_scale"]),
            stock_out,
            tensors["masked_m"],
            expected_m,
        )
        if stock_result is not None or candidate_calls:
            raise RuntimeError("marker-free W13 reference did not use stock")

        with w13_decode_forward_scope(
            forward_batch,
            16,
            graph_capture=False,
        ):
            candidate_result = entrypoint.grouped_gemm_nt_f8f8bf16_masked(
                (tensors["a"], tensors["a_scale"]),
                (tensors["b"], tensors["b_scale"]),
                candidate_out,
                tensors["masked_m"],
                expected_m,
            )
        if candidate_result is not None or candidate_calls != 1:
            raise RuntimeError("eager W13 registration did not launch once")

        def graph_run() -> None:
            with w13_decode_forward_scope(
                forward_batch,
                16,
                graph_capture=True,
            ):
                result = entrypoint.grouped_gemm_nt_f8f8bf16_masked(
                    (tensors["a"], tensors["a_scale"]),
                    (tensors["b"], tensors["b_scale"]),
                    graph_out,
                    tensors["masked_m"],
                    expected_m,
                )
            if result is not None:
                raise RuntimeError("graph W13 return contract changed")

        _capture_and_replay(graph_run, device=device)
        calls_after_capture_and_replay = candidate_calls
    finally:
        candidate.fp8_m_grouped_gemm_nt_masked = original_candidate

    # One eager launch, one side-stream warmup, and one capture execute Python.
    # Replay must execute the captured kernel nodes without re-entering Python.
    if calls_after_capture_and_replay != 3:
        raise RuntimeError(
            "W13 CUDA Graph replay unexpectedly re-entered Python: "
            f"calls={calls_after_capture_and_replay}"
        )
    torch.testing.assert_close(
        _active_rows(stock_out, expected_m),
        _active_rows(candidate_out, expected_m),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        _active_rows(stock_out, expected_m),
        _active_rows(graph_out, expected_m),
        rtol=0,
        atol=0,
    )
    return {
        "op": "w13",
        "profiler_name": "infini_kernel_glm52_moe_w13_decode",
        "authoritative_stock": str(deep_gemm.__file__),
        "candidate": str(candidate.__file__),
        "dispatch_state": state,
        "provider_state": provider_state(),
        "candidate_python_calls": calls_after_capture_and_replay,
        "cuda_graph_capture_and_replay": True,
        "graph_replay_python_reentry": False,
        "active_rows_match_stock": True,
    }


def _run_w2(device: torch.device) -> dict[str, Any]:
    import deep_gemm
    from sglang.srt.layers.deep_gemm_wrapper import entrypoint
    from sglang.srt.layers.glm52_opt import experimental_deepgemm
    from sglang.srt.layers.glm52_opt.context import op_context
    from sglang.srt.layers.glm52_opt.hotspot_provider import (
        initialize_hotspot_provider,
        provider_state,
    )
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    forward_context = entrypoint.update_deep_gemm_config(
        device.index,
        SimpleNamespace(chunked_prefill_size=8192, base_gpu_id=device.index),
    )
    if forward_context is None:
        raise RuntimeError("W2 forward context was not armed")
    initialize_hotspot_provider(device.index)
    runtime = experimental_deepgemm.get_w2_bm16_prepared_contract()
    if runtime is None:
        raise RuntimeError(
            "W2 runtime was not prepared: "
            f"{experimental_deepgemm.get_w2_bm16_prepare_error()}"
        )

    lhs = (
        torch.zeros((32, 1024, 2048), dtype=torch.float8_e4m3fn, device=device),
        _zeros_strided(
            (32, 1024, 4),
            (4096, 1, 1024),
            torch.int32,
            device,
        ),
    )
    rhs = (
        torch.zeros((32, 6144, 2048), dtype=torch.float8_e4m3fn, device=device),
        _zeros_strided(
            (32, 6144, 4),
            (24576, 1, 6144),
            torch.int32,
            device,
        ),
    )
    rhs[1].format_ue8m0 = True
    masked_m = torch.full((32,), 4, dtype=torch.int32, device=device)
    stock_out = torch.empty((32, 1024, 6144), dtype=torch.bfloat16, device=device)
    candidate_out = torch.empty_like(stock_out)
    graph_out = torch.empty_like(stock_out)
    expected_m = 4

    bound_holder: dict[str, Callable[..., Any]] = {}
    runner_core = SimpleNamespace(
        set_masked_down_gemm=lambda value: bound_holder.setdefault("run", value)
    )
    entrypoint.configure_w2_bm16_masked_down_gemm(
        runner_core,
        w2_weight=rhs[0],
        w2_scale=rhs[1],
        block_shape=[128, 128],
        deep_gemm_backend=True,
        is_fp4_experts=False,
        use_mxfp8=False,
    )
    bound = bound_holder.get("run")
    if bound is None:
        raise RuntimeError("W2 down-GEMM runner was not rebound")
    layer_contract, runtime_contract, callsite_prepare, candidate_dispatch = bound.args
    candidate_calls = 0

    def traced_dispatch(*args, **kwargs):
        nonlocal candidate_calls
        selected = candidate_dispatch(*args, **kwargs)
        candidate_calls += int(selected)
        return selected

    registered = partial(
        bound.func,
        layer_contract,
        runtime_contract,
        callsite_prepare,
        traced_dispatch,
    )

    stock_result = deep_gemm.fp8_m_grouped_gemm_nt_masked(
        lhs,
        rhs,
        stock_out,
        masked_m,
        expected_m,
    )
    if stock_result is not None:
        raise RuntimeError("stock W2 return contract changed")

    def candidate_run(out: torch.Tensor) -> None:
        with (
            forward_context(ForwardMode.DECODE, 16),
            op_context("moe_down_proj"),
        ):
            result = registered(
                lhs,
                rhs,
                out,
                masked_m,
                expected_m,
            )
        if result is not None:
            raise RuntimeError("registered W2 return contract changed")

    candidate_run(candidate_out)
    if candidate_calls != 1:
        raise RuntimeError("eager W2 registration did not launch once")
    _capture_and_replay(lambda: candidate_run(graph_out), device=device)
    if candidate_calls != 3:
        raise RuntimeError(
            "W2 CUDA Graph replay unexpectedly re-entered Python: "
            f"calls={candidate_calls}"
        )
    torch.testing.assert_close(
        _active_rows(stock_out, expected_m),
        _active_rows(candidate_out, expected_m),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        _active_rows(stock_out, expected_m),
        _active_rows(graph_out, expected_m),
        rtol=0,
        atol=0,
    )
    return {
        "op": "w2",
        "profiler_name": "infini_kernel_glm52_moe_w2_decode",
        "authoritative_stock": str(deep_gemm.__file__),
        "candidate": runtime.candidate_module_path,
        "runtime_evidence": runtime.evidence(),
        "provider_state": provider_state(),
        "candidate_python_calls": candidate_calls,
        "cuda_graph_capture_and_replay": True,
        "graph_replay_python_reentry": False,
        "active_rows_match_stock": True,
    }


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA device is required")
    torch.cuda.set_device(args.gpu_id)
    device = torch.device("cuda", args.gpu_id)
    result = _run_w13(device) if args.op == "w13" else _run_w2(device)
    properties = torch.cuda.get_device_properties(device)
    result["device"] = {
        "index": args.gpu_id,
        "name": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "num_sms": properties.multi_processor_count,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
