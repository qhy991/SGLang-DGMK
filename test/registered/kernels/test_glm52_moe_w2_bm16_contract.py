"""CPU-only contracts for the source-scoped GLM-5.2 W2/BM16 path."""

from __future__ import annotations

import hashlib
import json
import inspect
import os
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.layers.deep_gemm_wrapper import entrypoint
from sglang.srt.layers.glm52_opt import config, dispatch
from sglang.srt.layers.glm52_opt import experimental_deepgemm as experimental
from sglang.srt.layers.glm52_opt.context import op_context
from sglang.srt.layers.moe.moe_runner.deep_gemm import DeepGemmRunnerCore
from sglang.srt.model_executor.forward_batch_info import ForwardMode


def _tensor(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    stride: tuple[int, ...],
    *,
    contiguous: bool,
) -> Mock:
    tensor = Mock(spec=torch.Tensor)
    tensor.shape = shape
    tensor.dtype = dtype
    tensor.layout = torch.strided
    tensor.device = torch.device("cuda:0")
    tensor.stride.return_value = stride
    tensor.is_contiguous.return_value = contiguous
    return tensor


def _inputs():
    lhs = (
        _tensor(
            (32, 1024, 2048),
            torch.float8_e4m3fn,
            (2097152, 2048, 1),
            contiguous=True,
        ),
        _tensor(
            (32, 1024, 4),
            torch.int32,
            (4096, 1, 1024),
            contiguous=False,
        ),
    )
    rhs = (
        _tensor(
            (32, 6144, 2048),
            torch.float8_e4m3fn,
            (12582912, 2048, 1),
            contiguous=True,
        ),
        _tensor(
            (32, 6144, 4),
            torch.int32,
            (24576, 1, 6144),
            contiguous=False,
        ),
    )
    rhs[1].format_ue8m0 = True
    out = _tensor(
        (32, 1024, 6144),
        torch.bfloat16,
        (6291456, 6144, 1),
        contiguous=True,
    )
    masked_m = _tensor((32,), torch.int32, (1,), contiguous=True)
    return lhs, rhs, out, masked_m


def _dispatch(
    *,
    expected_m: int,
    op: str = "moe_down_proj",
    mode: ForwardMode = ForwardMode.DECODE,
    local_m: int | None = None,
    prepared: bool = True,
    callsite_eligible: bool = True,
    launch=None,
):
    inputs = _inputs()
    if launch is None:
        launch = Mock(return_value=None)
    if local_m is None:
        local_m = 16 if expected_m in (4, 5) else 32
    contract = (
        SimpleNamespace(
            launch=launch,
            current_forward_state=lambda: (mode, local_m),
        )
        if prepared
        else None
    )
    with op_context(op):
        result = dispatch.try_dispatch_moe_w2_bm16(
            contract,
            *inputs,
            expected_m=expected_m,
            callsite_eligible=callsite_eligible,
        )
    return result, launch, inputs


@pytest.mark.parametrize("expected_m", [4, 5, 8, 9])
def test_exact_decode_expected_m_dispatches_explicit_per_call_override(expected_m: int):
    result, launch, inputs = _dispatch(expected_m=expected_m)

    assert result is True
    launch.assert_called_once_with(
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[3],
        expected_m,
        masked_block_m_override=16,
    )
    inputs[3].cpu.assert_not_called()
    inputs[3].item.assert_not_called()
    inputs[3].tolist.assert_not_called()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"expected_m": 16},
        {"expected_m": 4, "op": "moe_gate_proj"},
        {"expected_m": 4, "mode": ForwardMode.EXTEND},
        {"expected_m": 4, "mode": ForwardMode.TARGET_VERIFY},
        {"expected_m": 4, "callsite_eligible": False},
        {"expected_m": 4, "prepared": False},
    ],
)
def test_unsupported_state_declines_before_launch(kwargs):
    result, launch, _ = _dispatch(**kwargs)
    assert result is False
    launch.assert_not_called()


def test_hot_path_does_not_read_env_stats_forward_m_tensor_abi_or_cuda():
    launch = Mock(return_value=None)
    with (
        patch.object(config, "w2_bm16_enabled", side_effect=AssertionError),
        patch.object(dispatch, "_record_hit", side_effect=AssertionError),
        patch.object(dispatch, "_record_miss", side_effect=AssertionError),
        patch.object(torch.cuda, "get_device_capability", side_effect=AssertionError),
        patch.object(torch.cuda, "current_device", side_effect=AssertionError),
    ):
        result, _, _ = _dispatch(expected_m=4, launch=launch)
    assert result is True
    launch.assert_called_once()


def test_generic_dispatch_phase_lookup_retains_forward_mode_dependency():
    with patch.object(
        dispatch, "get_forward_mode", return_value=ForwardMode.DECODE
    ):
        assert dispatch._current_phase(16) == "decode"
    dispatch_source = Path(dispatch.__file__).read_text()
    assert (
        "from sglang.srt.model_executor.forward_batch_info import ForwardMode"
        not in dispatch_source
    )
    assert "forward_state[0].is_decode()" in dispatch_source


@pytest.mark.parametrize(
    "launch",
    [
        Mock(side_effect=RuntimeError("candidate failed")),
        Mock(return_value=object()),
    ],
)
def test_candidate_failure_or_return_violation_propagates_without_stock(launch):
    stock = Mock()
    inputs = _inputs()
    runtime_contract = SimpleNamespace(
        launch=launch,
        current_forward_state=lambda: (ForwardMode.DECODE, 16),
    )
    layer_contract = SimpleNamespace(
        callsite_checked=True,
        callsite_eligible=True,
    )
    armed_callable = partial(
        entrypoint._grouped_gemm_nt_f8f8bf16_masked_w2_bm16,
        layer_contract,
        runtime_contract,
        Mock(side_effect=AssertionError("latched ABI was rescanned")),
        dispatch.try_dispatch_moe_w2_bm16,
    )
    with (
        patch.object(entrypoint, "deep_gemm", stock, create=True),
        patch.object(entrypoint, "_ensure_cuda", side_effect=lambda value: value),
        patch.object(entrypoint, "_sanity_check_input"),
        patch.object(
            entrypoint.compile_utils,
            "deep_gemm_execution_hook",
            return_value=nullcontext(),
        ),
        op_context("moe_down_proj"),
        pytest.raises(RuntimeError),
    ):
        armed_callable(
            *inputs,
            expected_m=4,
        )
    stock.fp8_m_grouped_gemm_nt_masked.assert_not_called()


def test_layer_and_callsite_contract_are_prevalidated_once_without_mask_read():
    runtime = SimpleNamespace(
        device_index=0,
        current_forward_state=experimental.get_w2_bm16_forward_state,
    )
    lhs, rhs, out, masked_m = _inputs()
    with (
        patch.object(experimental, "_W2_BM16_REQUESTED", True),
        patch.object(experimental, "_W2_BM16_PREPARED", runtime),
    ):
        contract = experimental.create_w2_bm16_layer_contract(
            w2_weight=rhs[0],
            w2_scale=rhs[1],
            block_shape=[128, 128],
            deep_gemm_backend=True,
            is_fp4_experts=False,
            use_mxfp8=False,
        )
        assert contract is not None
        assert contract.static_eligible is True
        with experimental.w2_bm16_forward_context(ForwardMode.DECODE, 16):
            assert experimental.prepare_w2_bm16_callsite_contract(
                contract,
                lhs,
                rhs,
                out,
                masked_m,
                expected_m=4,
                recipe_a=None,
                recipe_b=None,
                overlap_args=None,
            )
        masked_m.cpu.assert_not_called()
        masked_m.item.assert_not_called()
        masked_m.tolist.assert_not_called()

        # The layer's buffers and layouts are immutable after setup. A second
        # call consumes the persistent result and performs no metadata scan.
        lhs[0].shape = (1,)
        lhs[0].stride.side_effect = AssertionError("hot-path ABI rescan")
        with experimental.w2_bm16_forward_context(ForwardMode.DECODE, 16):
            assert experimental.prepare_w2_bm16_callsite_contract(
                contract,
                lhs,
                rhs,
                out,
                masked_m,
                expected_m=4,
                recipe_a=None,
                recipe_b=None,
                overlap_args=None,
            )


@pytest.mark.parametrize(
    "mode,local_m,expected_m,recipe_a,overlap_args",
    [
        (ForwardMode.EXTEND, 4096, 4, None, None),
        (ForwardMode.TARGET_VERIFY, 16, 4, None, None),
        (ForwardMode.DECODE, 16, 8, None, None),
        (ForwardMode.DECODE, 16, 4, (1, 128), None),
        (
            ForwardMode.DECODE,
            16,
            4,
            None,
            SimpleNamespace(num_sms=116),
        ),
    ],
)
def test_first_ineligible_call_does_not_latch_before_later_decode_proof(
    mode,
    local_m,
    expected_m,
    recipe_a,
    overlap_args,
):
    runtime = SimpleNamespace(
        device_index=0,
        current_forward_state=experimental.get_w2_bm16_forward_state,
    )
    lhs, rhs, out, masked_m = _inputs()
    with (
        patch.object(experimental, "_W2_BM16_REQUESTED", True),
        patch.object(experimental, "_W2_BM16_PREPARED", runtime),
    ):
        contract = experimental.create_w2_bm16_layer_contract(
            w2_weight=rhs[0],
            w2_scale=rhs[1],
            block_shape=[128, 128],
            deep_gemm_backend=True,
            is_fp4_experts=False,
            use_mxfp8=False,
        )
        with experimental.w2_bm16_forward_context(mode, local_m):
            assert not experimental.prepare_w2_bm16_callsite_contract(
                contract,
                lhs,
                rhs,
                out,
                masked_m,
                expected_m=expected_m,
                recipe_a=recipe_a,
                recipe_b=None,
                overlap_args=overlap_args,
            )
        assert contract.callsite_checked is False

        with experimental.w2_bm16_forward_context(ForwardMode.DECODE, 16):
            assert experimental.prepare_w2_bm16_callsite_contract(
                contract,
                lhs,
                rhs,
                out,
                masked_m,
                expected_m=4,
                recipe_a=None,
                recipe_b=None,
                overlap_args=None,
            )
        assert contract.callsite_checked is True
        assert contract.callsite_eligible is True


def _fake_runtime(
    *,
    pdl: bool = False,
    missing: str | None = None,
    pdl_settable: bool = True,
):
    state = {"num_sms": 148, "tc_util": 87, "pdl": pdl}

    def get_num_sms():
        return state["num_sms"]

    def set_num_sms(value):
        state["num_sms"] = int(value)

    def get_tc_util():
        return state["tc_util"]

    def set_tc_util(value):
        state["tc_util"] = int(value)

    def get_pdl():
        return state["pdl"]

    def set_pdl(value):
        if pdl_settable:
            state["pdl"] = value

    def launch(*args, masked_block_m_override=0):
        return None

    module = SimpleNamespace(
        __file__="/fake/deep_gemm/__init__.py",
        _C=object(),
        get_num_sms=get_num_sms,
        set_num_sms=set_num_sms,
        get_tc_util=get_tc_util,
        set_tc_util=set_tc_util,
        get_pdl=get_pdl,
        set_pdl=set_pdl,
        fp8_m_grouped_gemm_nt_masked=launch,
    )
    if missing is not None:
        delattr(module, missing)
    return module, state


def _prepare_runtime(
    *,
    stock_pdl: bool = False,
    stock_pdl_settable: bool = True,
    candidate_missing: str | None = None,
):
    manifest_path = experimental._expected_w2_bm16_manifest_path()
    manifest = json.loads(manifest_path.read_text())
    stock, stock_state = _fake_runtime(
        pdl=stock_pdl,
        pdl_settable=stock_pdl_settable,
    )
    candidate, candidate_state = _fake_runtime(missing=candidate_missing)
    candidate.__file__ = "/fake/candidate/__init__.py"
    task_cache_root = Path(
        "/home/qinhaiyan/glm52-v2-goal-runs/cache/"
        "26-moe_w2_decode_scoped_bm16"
    )
    with (
        patch.dict(
            os.environ,
            {
                "DG_JIT_CACHE_DIR": str(task_cache_root / "deepgemm"),
                "SGLANG_DG_CACHE_DIR": str(task_cache_root / "deepgemm"),
                "TRITON_CACHE_DIR": str(task_cache_root / "triton"),
                "TORCH_EXTENSIONS_DIR": str(
                    task_cache_root / "torch_extensions"
                ),
            },
            clear=False,
        ),
        patch.object(experimental, "_W2_BM16_REQUESTED", False),
        patch.object(experimental, "_W2_BM16_PREPARED", None),
        patch.object(experimental, "_W2_BM16_PREPARE_ERROR", None),
        patch.object(
            experimental, "_verify_w2_bm16_manifest", return_value=manifest_path
        ),
        patch.object(
            experimental, "ensure_stock_deep_gemm", return_value=stock
        ),
        patch.object(
            experimental,
            "_load_w2_bm16_candidate",
            return_value=candidate,
        ),
        patch.object(experimental, "_verify_cache_contract"),
        patch.object(experimental, "_sha256", return_value="manifest-hash"),
        patch.object(torch.cuda, "current_device", return_value=0),
        patch.object(torch.cuda, "get_device_capability", return_value=(10, 0)),
        patch.object(
            torch.cuda,
            "get_device_properties",
            return_value=SimpleNamespace(multi_processor_count=148),
        ),
    ):
        try:
            contract = experimental.prepare_w2_bm16_deep_gemm(0)
            error = None
        except Exception as exc:
            contract = None
            error = exc
        state = (
            experimental.w2_bm16_requested(),
            experimental.get_w2_bm16_prepared_contract(),
            experimental.get_w2_bm16_prepare_error(),
        )
    return contract, error, state, stock_state, candidate_state, manifest


def test_prepare_freezes_pdl_num_sms_tc_util_and_identity():
    contract, error, state, stock_state, candidate_state, manifest = _prepare_runtime()
    assert error is None
    assert contract is state[1]
    assert state[0] is True
    assert state[2] is None
    assert contract.stock_pdl is contract.candidate_pdl is True
    assert contract.stock_initial_pdl is False
    assert contract.candidate_initial_pdl is False
    assert contract.stock_num_sms == contract.candidate_num_sms == 148
    assert contract.stock_tc_util == contract.candidate_tc_util == 87
    assert contract.runtime_modules_distinct is True
    assert contract.runtime_extension_modules_distinct is True
    assert contract.independence_probe_num_sms == 146
    assert contract.independence_probe_tc_util == 88
    assert contract.base_commit == manifest["base"]["commit"]
    assert contract.base_version == manifest["base"]["version"]
    assert (
        contract.cutlass_commit
        == manifest["base"]["submodules"]["third-party/cutlass"]
    )
    assert (
        contract.fmt_commit
        == manifest["base"]["submodules"]["third-party/fmt"]
    )
    assert stock_state == candidate_state == {
        "num_sms": 148,
        "tc_util": 87,
        "pdl": True,
    }
    assert contract.stock_extension_sha256 == manifest["stock"]["extension_sha256"]
    assert (
        contract.candidate_extension_sha256
        == manifest["candidate"]["extension_sha256"]
    )


def test_stock_readiness_requires_both_consumers_on_one_exact_module():
    stock = object()
    entrypoint_consumer = SimpleNamespace(deep_gemm=stock)
    compile_utils_consumer = SimpleNamespace(deep_gemm=stock)
    consumers = {
        "sglang.srt.layers.deep_gemm_wrapper.entrypoint": entrypoint_consumer,
        "sglang.srt.layers.deep_gemm_wrapper.compile_utils": compile_utils_consumer,
    }
    with (
        patch.object(
            experimental.importlib,
            "import_module",
            return_value=stock,
        ),
        patch.object(experimental, "_verify_module_package"),
        patch.object(experimental, "_verify_cache_contract"),
        patch.dict(sys.modules, consumers),
    ):
        assert (
            experimental.ensure_stock_deep_gemm(
                {},
                verify_consumers=True,
            )
            is stock
        )
        entrypoint_consumer.deep_gemm = object()
        with pytest.raises(
            RuntimeError,
            match="entrypoint is not bound to the manifest stock",
        ):
            experimental.ensure_stock_deep_gemm(
                {},
                verify_consumers=True,
            )


@pytest.mark.parametrize(
    "stock_pdl_settable,candidate_missing",
    [(False, None), (True, "get_tc_util"), (True, "set_pdl")],
)
def test_prepare_failure_keeps_requested_but_has_no_launch_token(
    stock_pdl_settable: bool, candidate_missing: str | None
):
    contract, error, state, *_ = _prepare_runtime(
        stock_pdl_settable=stock_pdl_settable,
        candidate_missing=candidate_missing,
    )
    assert contract is None
    assert error is not None
    assert state[0] is True
    assert state[1] is None
    assert state[2]


def test_task_private_forward_context_restores_outer_metadata():
    with experimental.w2_bm16_forward_context(ForwardMode.EXTEND, 4096):
        assert experimental.get_w2_bm16_forward_state() == (
            ForwardMode.EXTEND,
            4096,
        )
        with experimental.w2_bm16_forward_context(ForwardMode.DECODE, 32):
            assert experimental.get_w2_bm16_forward_state() == (
                ForwardMode.DECODE,
                32,
            )
        assert experimental.get_w2_bm16_forward_state() == (
            ForwardMode.EXTEND,
            4096,
        )
    assert experimental.get_w2_bm16_forward_state() is None


@pytest.mark.parametrize(
    "module_name,helper_name,consumer_name",
    [
        (
            "sglang.srt.model_executor.runner.decode_cuda_graph_runner",
            "_stacked_capture_contexts",
            "DecodeCudaGraphRunner.capture_one_shape",
        ),
    ],
)
def test_second_context_enter_failure_restores_first_context(
    module_name: str,
    helper_name: str,
    consumer_name: str,
):
    module = __import__(module_name, fromlist=["*"])
    helper = getattr(module, helper_name)
    events = []

    @contextmanager
    def primary():
        events.append("primary-enter")
        try:
            yield
        finally:
            events.append("primary-exit")

    class RaisingContext:
        def __enter__(self):
            events.append("task-enter")
            raise RuntimeError("task context failed")

        def __exit__(self, *_args):
            events.append("task-exit")

    with pytest.raises(RuntimeError, match="task context failed"):
        with helper(primary(), RaisingContext()):
            raise AssertionError("unreachable")
    assert events == ["primary-enter", "task-enter", "primary-exit"]

    owner_name, method_name = consumer_name.split(".")
    consumer = getattr(getattr(module, owner_name), method_name)
    assert helper_name in inspect.getsource(consumer)


def test_model_runner_arms_instance_only_and_restores_on_inner_failure():
    from sglang.srt.model_executor import model_runner as model_runner_mod

    events = []

    @contextmanager
    def task_context():
        events.append("task-enter")
        try:
            yield
        finally:
            events.append("task-exit")

    def stock_forward(_forward_batch):
        events.append("stock-forward")
        raise RuntimeError("inner forward failed")

    runner = SimpleNamespace(_forward_raw=stock_forward)
    stock_identity = runner._forward_raw
    assert "_w2_bm16" not in inspect.getsource(
        model_runner_mod.ModelRunner._forward_raw
    )
    assert runner._forward_raw is stock_identity

    factory = Mock(return_value=task_context())
    model_runner_mod._arm_w2_forward_context(runner, factory)
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        input_ids=SimpleNamespace(shape=(16,)),
    )
    with pytest.raises(RuntimeError, match="inner forward failed"):
        runner._forward_raw(forward_batch)
    factory.assert_called_once_with(ForwardMode.DECODE, 16)
    assert events == ["task-enter", "stock-forward", "task-exit"]


def _capture_boundary_events(mode: ForwardMode):
    from sglang.srt.model_executor.runner import decode_cuda_graph_runner as graph_mod

    events = []
    stage = SimpleNamespace(value="outside")
    launch = Mock(
        side_effect=lambda *args, **kwargs: events.append(
            (
                "candidate",
                stage.value,
                experimental.get_w2_bm16_forward_state(),
            )
        )
    )
    token = SimpleNamespace(
        launch=launch,
        current_forward_state=experimental.get_w2_bm16_forward_state,
    )
    inputs = _inputs()

    forward_batch = SimpleNamespace(
        forward_mode=mode,
        lora_ids=None,
        dp_local_start_pos=None,
        dp_local_num_tokens=None,
        global_dp_buffer_len=None,
        dp_padding_mode=SimpleNamespace(is_max_len=lambda: False),
        global_num_tokens_cpu=None,
        input_ids=object(),
        positions=object(),
    )
    attn_backend = SimpleNamespace(
        init_forward_metadata_out_graph=Mock(),
        init_forward_metadata_in_graph=Mock(),
    )

    class Backend:
        def capture_one(self, shape_key, run_once, **kwargs):
            stage.value = "capture"
            run_once()
            stage.value = "outside"

    runner = object.__new__(graph_mod.DecodeCudaGraphRunner)
    runner.num_tokens_per_req = 1
    runner.ragged_verify_mode = False
    runner.model_runner = SimpleNamespace(
        server_args=SimpleNamespace(debug_cuda_graph=False),
        lora_manager=SimpleNamespace(prepare_lora_batch=Mock()),
        spec_algorithm=SimpleNamespace(is_dflash_family=lambda: False),
        is_draft_worker=False,
        model=object(),
        capture_tail_hooks=[],
        canary_manager=None,
        attn_backend=SimpleNamespace(),
        _w2_bm16_forward_context=experimental.w2_bm16_forward_context,
    )
    runner.capture_prepare = Mock(
        return_value=(forward_batch, attn_backend, None)
    )
    runner.tbo_plugin = SimpleNamespace(capture_one_batch_size=Mock())
    runner.deepep_adapter = SimpleNamespace(capture=Mock())
    runner.pp_size = 1
    runner.backend = Backend()
    runner._capture_graph_size = lambda **kwargs: 16
    runner._make_graph_key = lambda *args: ("key", args)

    def forward(*args, **kwargs):
        with op_context("moe_down_proj"):
            result = dispatch.try_dispatch_moe_w2_bm16(
                token,
                *inputs,
                expected_m=4,
                callsite_eligible=True,
            )
        events.append(
            (
                "dispatch",
                stage.value,
                experimental.get_w2_bm16_forward_state(),
                result,
            )
        )
        return object()

    def warmup(_runner, run_once, **kwargs):
        stage.value = "warmup"
        run_once()
        stage.value = "outside"

    with (
        patch.object(graph_mod, "forward_context", return_value=nullcontext()),
        patch.object(graph_mod, "ForwardContext", return_value=object()),
        patch.object(
            graph_mod,
            "maybe_flashinfer_autotune_speculative_draft",
            side_effect=warmup,
        ),
        patch.object(graph_mod, "set_dp_buffer_len"),
        patch.object(graph_mod, "set_is_extend_in_batch"),
        experimental.w2_bm16_forward_context(ForwardMode.EXTEND, 4096),
    ):
        runner.capture_one_shape(16, forward)
        restored = experimental.get_w2_bm16_forward_state()
    return events, launch, restored


def test_actual_capture_warmup_and_run_once_reach_candidate_and_restore_context():
    events, launch, restored = _capture_boundary_events(ForwardMode.DECODE)
    assert launch.call_count == 2
    assert (
        "candidate",
        "warmup",
        (ForwardMode.DECODE, 16),
    ) in events
    assert (
        "candidate",
        "capture",
        (ForwardMode.DECODE, 16),
    ) in events
    assert restored == (ForwardMode.EXTEND, 4096)


def test_actual_target_verify_capture_boundary_never_reaches_candidate():
    events, launch, restored = _capture_boundary_events(ForwardMode.TARGET_VERIFY)
    assert launch.call_count == 0
    dispatch_events = [event for event in events if event[0] == "dispatch"]
    assert len(dispatch_events) == 2
    assert all(event[-1] is False for event in dispatch_events)
    assert restored == (ForwardMode.EXTEND, 4096)


def _run_stock_entrypoint():
    inputs = _inputs()
    stock = Mock()
    stock.fp8_m_grouped_gemm_nt_masked.return_value = None
    with (
        patch.object(entrypoint, "deep_gemm", stock, create=True),
        patch.object(entrypoint, "_ensure_cuda", side_effect=lambda value: value),
        patch.object(entrypoint, "_sanity_check_input"),
        patch.object(
            entrypoint.compile_utils,
            "deep_gemm_execution_hook",
            return_value=nullcontext(),
        ),
        patch.object(dispatch, "try_dispatch_moe_masked", return_value=False),
        op_context("moe_gate_proj"),
    ):
        result = entrypoint.grouped_gemm_nt_f8f8bf16_masked(
            *inputs,
            expected_m=4,
        )
    return result, stock


def test_stock_wrapper_has_original_signature_and_no_w2_hot_path():
    signature = inspect.signature(
        entrypoint.grouped_gemm_nt_f8f8bf16_masked
    )
    assert "w2_bm16_eligible" not in signature.parameters
    source = inspect.getsource(
        entrypoint.grouped_gemm_nt_f8f8bf16_masked
    )
    source_identity = hashlib.sha256(source.encode()).hexdigest()
    assert (
        source_identity
        == "5edff041abefebe45121e093c8252090ddb1374b6b363870b66ca80f1f51b40a"
    )
    assert "_W2_BM16" not in source
    assert "experimental_deepgemm" not in source

    result, stock = _run_stock_entrypoint()
    assert result is None
    stock.fp8_m_grouped_gemm_nt_masked.assert_called_once()


def test_default_runner_binds_exact_stock_callable_and_unrequested_setup_is_noop():
    core = object.__new__(DeepGemmRunnerCore)
    core._masked_down_gemm = (
        entrypoint.grouped_gemm_nt_f8f8bf16_masked
    )
    assert (
        core._masked_down_gemm
        is entrypoint.grouped_gemm_nt_f8f8bf16_masked
    )

    forbidden = Mock(side_effect=AssertionError("default runner was mutated"))
    core.set_masked_down_gemm = forbidden
    with patch.object(entrypoint, "_W2_BM16_PROFILE_REQUESTED", False):
        entrypoint.configure_w2_bm16_masked_down_gemm(
            core,
            w2_weight=object(),
            w2_scale=object(),
            block_shape=[128, 128],
            deep_gemm_backend=True,
            is_fp4_experts=False,
            use_mxfp8=False,
        )
    forbidden.assert_not_called()


def test_armed_setup_binds_one_per_runner_callable():
    runner_core = SimpleNamespace(set_masked_down_gemm=Mock())
    runtime_contract = SimpleNamespace()
    layer_contract = SimpleNamespace()
    callsite_prepare = Mock()
    candidate_dispatch = Mock()
    with (
        patch.object(entrypoint, "_W2_BM16_PROFILE_REQUESTED", True),
        patch.object(
            entrypoint,
            "_W2_BM16_PREPARED_CONTRACT",
            runtime_contract,
        ),
        patch.object(
            entrypoint,
            "_W2_BM16_CALLSITE_PREPARE",
            callsite_prepare,
        ),
        patch.object(
            entrypoint, "_W2_BM16_DISPATCH", candidate_dispatch
        ),
        patch.object(
            experimental,
            "create_w2_bm16_layer_contract",
            return_value=layer_contract,
        ) as create,
    ):
        entrypoint.configure_w2_bm16_masked_down_gemm(
            runner_core,
            w2_weight="weight",
            w2_scale="scale",
            block_shape=[128, 128],
            deep_gemm_backend=True,
            is_fp4_experts=False,
            use_mxfp8=False,
        )
    create.assert_called_once()
    bound = runner_core.set_masked_down_gemm.call_args.args[0]
    assert isinstance(bound, partial)
    assert (
        bound.func
        is entrypoint._grouped_gemm_nt_f8f8bf16_masked_w2_bm16
    )
    assert bound.args == (
        layer_contract,
        runtime_contract,
        callsite_prepare,
        candidate_dispatch,
    )


def test_armed_w13_still_uses_exact_stock_callable():
    # Only the per-runner down-GEMM callable is replaceable. Gate/up (W13)
    # continues to resolve the unchanged module-level stock function.
    core = object.__new__(DeepGemmRunnerCore)
    core._masked_down_gemm = Mock()
    assert (
        entrypoint.grouped_gemm_nt_f8f8bf16_masked
        is not core._masked_down_gemm
    )
    result, stock = _run_stock_entrypoint()
    assert result is None
    core._masked_down_gemm.assert_not_called()
    stock.fp8_m_grouped_gemm_nt_masked.assert_called_once()


def test_default_off_imports_no_experimental_module_in_fp8_or_graph_runner():
    repo = Path(__file__).resolve().parents[3]
    code = """
import sys
import sglang.srt.layers.quantization.fp8
import sglang.srt.model_executor.runner.decode_cuda_graph_runner
name = "sglang.srt.layers.glm52_opt.experimental_deepgemm"
assert name not in sys.modules, sorted(
    key for key in sys.modules if "experimental_deepgemm" in key
)
"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["SGLANG_GLM52_OPT"] = "0"
    env.pop("SGLANG_GLM52_OPT_PROFILE", None)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo / "python"), str(repo), *sys.path]
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_profile_is_explicit_and_down_projection_only(monkeypatch):
    monkeypatch.setenv("SGLANG_GLM52_OPT", "1")
    monkeypatch.setenv("SGLANG_GLM52_OPT_PROFILE", "serving_safe")
    monkeypatch.delenv("SGLANG_GLM52_OPT_OPS", raising=False)
    assert config.w2_bm16_enabled() is False

    monkeypatch.setenv("SGLANG_GLM52_OPT_PROFILE", config.W2_BM16_PROFILE)
    assert config.w2_bm16_enabled() is True
    monkeypatch.setenv("SGLANG_GLM52_OPT_OPS", "moe_gate_proj")
    assert config.w2_bm16_enabled() is False
    monkeypatch.setenv("SGLANG_GLM52_OPT_OPS", "moe_down_proj")
    assert config.w2_bm16_enabled() is True


def test_source_patch_has_v2_jit_identity_and_optional_override_only():
    repo = Path(__file__).resolve().parents[3]
    source_patch = (
        repo / "third_party" / "deepgemm_w2_bm16" / "source.patch"
    ).read_text()
    assert (
        "sm100_m_grouped_fp8_fp4_gemm_masked_1d1d_"
        "glm52_w2_bm16_v2_em" in source_patch
    )
    assert "masked_block_m_override=0" in source_patch
    assert "set_mk_alignment_for_contiguous_layout(" not in source_patch
    assert "SGLANG_GLM52_W2_BM16_JIT_CACHE" not in source_patch


def test_overlay_source_tree_excludes_generated_python_files():
    repo = Path(__file__).resolve().parents[3]
    overlay_source = repo / "third_party" / "deepgemm_w2_bm16"
    generated = [
        path
        for path in overlay_source.rglob("*")
        if path.is_file() and path.suffix in {".pyc", ".pyo"}
    ]
    assert generated == []
