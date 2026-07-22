import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.distributed import all_reduce_trace as trace
from sglang.srt.distributed import parallel_state
from sglang.srt.distributed.all_reduce_trace import (
    REPLAY_LIMITATION,
    AllReduceTraceRecorder,
    begin_all_reduce_trace,
    finish_all_reduce_trace,
    get_all_reduce_trace_failure_counts,
    select_all_reduce_backend,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class _Device:
    type = "mock"

    def __str__(self):
        return "mock:0"


class _Tensor:
    dtype = "bfloat16"
    device = _Device()
    is_cpu = False

    def __init__(self, pointer: int, shape=(16, 6144)):
        self.pointer = pointer
        self.shape = shape

    def stride(self):
        return (self.shape[1], 1)

    def numel(self):
        return self.shape[0] * self.shape[1]

    def element_size(self):
        return 2

    def data_ptr(self):
        return self.pointer


class _CustomCommunicator:
    disabled = False

    def should_custom_ar(self, _tensor):
        return True

    def _determine_algo(self, _tensor):
        return SimpleNamespace(name="ONE_SHOT_PUSH")


class _Group:
    unique_name = "tp:0"
    ranks = [0, 1, 2, 3]
    world_size = 4
    rank = 0
    rank_in_group = 0
    device_group = object()
    hpu_communicator = None
    xpu_communicator = None
    npu_communicator = None
    pynccl_comm = None
    pymscclpp_comm = None
    ca_comm = _CustomCommunicator()
    qr_comm = None
    torch_symm_mem_comm = None

    def is_symmetric_memory_enabled(self):
        return False


def _record_once(
    recorder,
    input_tensor,
    output_tensor,
    selected_backend,
    *,
    piecewise_cuda_graph=False,
):
    token = begin_all_reduce_trace(
        _Group(),
        input_tensor,
        piecewise_cuda_graph=piecewise_cuda_graph,
        recorder=recorder,
    )
    finish_all_reduce_trace(
        token,
        output_tensor,
        selected_backend=selected_backend,
    )


def _record_from_distinct_caller(recorder, input_tensor, output_tensor):
    token = begin_all_reduce_trace(
        _Group(),
        input_tensor,
        piecewise_cuda_graph=False,
        recorder=recorder,
    )
    finish_all_reduce_trace(
        token,
        output_tensor,
        selected_backend="custom_all_reduce_outplace",
    )


def _outer_caller_one(recorder, input_tensor, output_tensor):
    _record_once(recorder, input_tensor, output_tensor, "custom_all_reduce_outplace")


def _outer_caller_two(recorder, input_tensor, output_tensor):
    _record_once(recorder, input_tensor, output_tensor, "custom_all_reduce_outplace")


def _group_coordinator():
    group = object.__new__(parallel_state.GroupCoordinator)
    group.unique_name = "tp:0"
    group.ranks = [0, 1, 2, 3]
    group.world_size = 4
    group.rank = 0
    group.rank_in_group = 0
    group.local_size = 4
    group.device_group = object()
    group.hpu_communicator = None
    group.xpu_communicator = None
    group.npu_communicator = None
    group.pynccl_comm = None
    group.pymscclpp_comm = None
    group.ca_comm = None
    group.qr_comm = None
    group.torch_symm_mem_comm = None
    group.is_symmetric_memory_enabled = lambda: False
    return group


class TestAllReduceBackendSelection(unittest.TestCase):
    def test_priority(self):
        base = {
            "world_size_is_one": False,
            "input_is_cpu": False,
            "cpu_shm_eligible": False,
            "hpu_eligible": False,
            "xpu_eligible": False,
            "npu_eligible": False,
            "pynccl_symmetric_eligible": False,
            "custom_eligible": False,
            "quick_eligible": False,
            "pymscclpp_eligible": False,
            "torch_symm_mem_eligible": False,
            "piecewise_pynccl_eligible": False,
            "pynccl_inplace_eligible": False,
            "torch_symm_mem_inplace_eligible": False,
        }
        cases = [
            ({"world_size_is_one": True, "custom_eligible": True}, "identity"),
            ({"input_is_cpu": True, "cpu_shm_eligible": True}, "cpu_shm"),
            ({"input_is_cpu": True}, "cpu_c10d"),
            ({"hpu_eligible": True, "custom_eligible": True}, "hpu"),
            (
                {"xpu_eligible": True, "npu_eligible": True},
                "xpu_torch_distributed_inplace",
            ),
            ({"npu_eligible": True, "custom_eligible": True}, "npu"),
            (
                {"pynccl_symmetric_eligible": True, "custom_eligible": True},
                "pynccl_symmetric_inplace",
            ),
            (
                {"custom_eligible": True, "quick_eligible": True},
                "custom_all_reduce_outplace",
            ),
            (
                {"quick_eligible": True, "pymscclpp_eligible": True},
                "quick_all_reduce_outplace",
            ),
            ({"pymscclpp_eligible": True}, "pymscclpp_outplace"),
            ({"torch_symm_mem_eligible": True}, "torch_symm_mem_outplace"),
            ({"piecewise_pynccl_eligible": True}, "pynccl_piecewise_outplace"),
            (
                {
                    "pynccl_inplace_eligible": True,
                    "torch_symm_mem_inplace_eligible": True,
                },
                "pynccl_inplace",
            ),
            (
                {"torch_symm_mem_inplace_eligible": True},
                "torch_symm_mem_inplace",
            ),
            ({}, "torch_distributed_inplace"),
        ]
        for delta, expected in cases:
            predicates = dict(base)
            predicates.update(delta)
            with self.subTest(expected=expected):
                self.assertEqual(select_all_reduce_backend(predicates), expected)


class TestAllReduceTraceSerialization(unittest.TestCase):
    def test_global_failure_counts_api_returns_a_copy(self):
        counts = get_all_reduce_trace_failure_counts()
        counts["injected"] = 1
        self.assertNotIn("injected", get_all_reduce_trace_failure_counts())

    def test_first_signature_is_buffered_and_serialized_once(self):
        recorder = AllReduceTraceRecorder("unused", enabled=True)
        input_tensor = _Tensor(0x1000)
        output_tensor = _Tensor(0x2000)

        for _ in range(2):
            _record_once(
                recorder,
                input_tensor,
                output_tensor,
                "custom_all_reduce_outplace",
            )

        self.assertEqual(len(recorder.records), 1)
        record = recorder.records[0]
        self.assertEqual(record["input"]["shape"], [16, 6144])
        self.assertEqual(record["input"]["stride"], [6144, 1])
        self.assertEqual(record["input"]["bytes"], 196608)
        self.assertEqual(record["group"]["ranks"], [0, 1, 2, 3])
        self.assertEqual(record["group"]["world_size"], 4)
        self.assertIsNotNone(record["caller"]["tag"])
        self.assertTrue(record["caller"]["stack"])
        self.assertTrue(record["selection"]["predicates"]["custom_eligible"])
        self.assertIn(
            "_CustomCommunicator",
            record["selection"]["selected_communicator"],
        )
        self.assertEqual(record["selection"]["selected_algorithm"], "ONE_SHOT_PUSH")
        self.assertTrue(record["selection"]["prediction_matches_selection"])
        self.assertEqual(
            record["selection"]["selected_algorithm_source"],
            "custom_private_selector_mirror_pre_dispatch",
        )
        self.assertFalse(record["alias"]["same_data_ptr"])
        self.assertFalse(record["alias"]["python_object_identity"])
        self.assertEqual(record["graph"]["replay_limitation"], REPLAY_LIMITATION)
        self.assertFalse(record["graph"]["cuda_graph_replay_observed"])

        serialized = AllReduceTraceRecorder.serialize_record(record)
        self.assertEqual(
            json.loads(serialized)["signature"]["id"], record["signature"]["id"]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            destination = str(Path(tmpdir) / "trace.{rank}.{pid}.jsonl")
            path = recorder.flush(destination)
            self.assertIsNotNone(path)
            lines = path.read_text().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(
                json.loads(lines[0])["selection"]["selected_backend"],
                "custom_all_reduce_outplace",
            )

    def test_duplicate_uses_cheap_pre_dedup(self):
        recorder = AllReduceTraceRecorder("unused", enabled=True)
        input_tensor = _Tensor(0x1000)
        output_tensor = _Tensor(0x2000)

        with (
            patch.object(
                trace,
                "_selection_snapshot",
                wraps=trace._selection_snapshot,
            ) as selection_snapshot,
            patch.object(
                trace,
                "_capture_python_stack",
                wraps=trace._capture_python_stack,
            ) as capture_stack,
            patch.object(
                trace,
                "_tensor_metadata",
                wraps=trace._tensor_metadata,
            ) as tensor_metadata,
            patch.object(
                trace.hashlib,
                "sha256",
                wraps=trace.hashlib.sha256,
            ) as sha256,
        ):
            for _ in range(2):
                _record_once(
                    recorder,
                    input_tensor,
                    output_tensor,
                    "custom_all_reduce_outplace",
                )

        self.assertEqual(selection_snapshot.call_count, 1)
        self.assertEqual(capture_stack.call_count, 1)
        self.assertEqual(tensor_metadata.call_count, 2)
        self.assertEqual(sha256.call_count, 1)
        self.assertEqual(len(recorder.records), 1)

    def test_pre_dedup_preserves_shape_graph_and_caller_distinctions(self):
        recorder = AllReduceTraceRecorder("unused", enabled=True)
        output_tensor = _Tensor(0x2000)
        _record_once(
            recorder,
            _Tensor(0x1000),
            output_tensor,
            "custom_all_reduce_outplace",
        )
        _record_once(
            recorder,
            _Tensor(0x3000, shape=(32, 6144)),
            output_tensor,
            "custom_all_reduce_outplace",
        )
        _record_once(
            recorder,
            _Tensor(0x4000),
            output_tensor,
            "custom_all_reduce_outplace",
            piecewise_cuda_graph=True,
        )
        _record_from_distinct_caller(
            recorder,
            _Tensor(0x5000),
            output_tensor,
        )

        self.assertEqual(len(recorder.records), 4)
        self.assertEqual(
            {tuple(record["input"]["shape"]) for record in recorder.records},
            {(16, 6144), (32, 6144)},
        )
        self.assertEqual(
            {record["graph"]["tc_piecewise_cuda_graph"] for record in recorder.records},
            {False, True},
        )
        self.assertEqual(
            len({record["caller"]["tag"] for record in recorder.records}),
            2,
        )

    def test_shared_wrapper_preserves_distinct_outer_callers(self):
        recorder = AllReduceTraceRecorder("unused", enabled=True)
        output_tensor = _Tensor(0x2000)

        _outer_caller_one(recorder, _Tensor(0x1000), output_tensor)
        _outer_caller_two(recorder, _Tensor(0x3000), output_tensor)

        self.assertEqual(len(recorder.records), 2)
        chains = [record["caller"]["key_chain"] for record in recorder.records]
        self.assertEqual(chains[0][0], chains[1][0])
        self.assertNotEqual(chains[0][1], chains[1][1])

    def test_pre_dedup_preserves_entry_stream_identity(self):
        recorder = AllReduceTraceRecorder("unused", enabled=True)
        output_tensor = _Tensor(0x2000)

        for stream_handle in (11, 22):
            stream_state = {
                "present": True,
                "class": "mock.Stream",
                "device": "cuda:0",
                "cuda_stream": stream_handle,
                "priority": 0,
            }
            with patch.object(
                trace,
                "_capture_dispatch_state",
                return_value=(False, None, stream_state),
            ):
                _record_once(
                    recorder,
                    _Tensor(0x1000),
                    output_tensor,
                    "custom_all_reduce_outplace",
                )

        self.assertEqual(len(recorder.records), 2)
        self.assertEqual(
            {record["stream"]["cuda_stream"] for record in recorder.records},
            {11, 22},
        )

    def test_stream_probe_failure_is_recorded_and_fails_open(self):
        recorder = AllReduceTraceRecorder("unused", enabled=True)
        with patch.object(
            trace,
            "_capture_dispatch_state",
            return_value=(
                False,
                None,
                {"present": False, "probe_error": "injected stream failure"},
            ),
        ):
            _record_once(
                recorder,
                _Tensor(0x1000),
                _Tensor(0x2000),
                "custom_all_reduce_outplace",
            )

        self.assertEqual(len(recorder.records), 1)
        self.assertEqual(recorder.failures, {"graph_or_stream_probe": 1})
        self.assertEqual(
            recorder.records[0]["stream"]["probe_error"],
            "injected stream failure",
        )

    def test_begin_and_finish_fail_open(self):
        begin_recorder = AllReduceTraceRecorder("unused", enabled=True)
        with patch.object(
            trace,
            "_selection_snapshot",
            side_effect=RuntimeError("injected begin failure"),
        ):
            token = begin_all_reduce_trace(
                _Group(),
                _Tensor(0x1000),
                piecewise_cuda_graph=False,
                recorder=begin_recorder,
            )
        self.assertIsNone(token)
        self.assertEqual(begin_recorder.failures, {"begin": 1})

        finish_recorder = AllReduceTraceRecorder("unused", enabled=True)
        input_tensor = _Tensor(0x1000)
        token = begin_all_reduce_trace(
            _Group(),
            input_tensor,
            piecewise_cuda_graph=False,
            recorder=finish_recorder,
        )
        self.assertIsNotNone(token)
        with patch.object(
            finish_recorder,
            "record_first",
            side_effect=RuntimeError("injected finish failure"),
        ):
            finish_all_reduce_trace(
                token,
                input_tensor,
                selected_backend="custom_all_reduce_outplace",
            )
        self.assertEqual(finish_recorder.failures, {"finish": 1})
        self.assertEqual(finish_recorder.records, [])


class TestGroupCoordinatorAllReduceTrace(unittest.TestCase):
    def test_disabled_path_does_not_enter_trace_helper(self):
        group = _group_coordinator()
        input_tensor = _Tensor(0x1000)
        with (
            patch.object(parallel_state, "ALL_REDUCE_TRACE_ENABLED", False),
            patch.object(parallel_state, "begin_all_reduce_trace") as traced_begin,
            patch.object(parallel_state, "inplace_all_reduce") as inplace,
            patch.object(
                parallel_state,
                "is_in_tc_piecewise_cuda_graph",
                return_value=False,
            ),
        ):
            output = group.all_reduce(input_tensor)

        self.assertIs(output, input_tensor)
        traced_begin.assert_not_called()
        inplace.assert_called_once_with(input_tensor, group_name="tp:0")

    def test_real_all_reduce_method_records_exact_inplace_fallback(self):
        recorder = AllReduceTraceRecorder("unused", enabled=True)
        group = _group_coordinator()
        input_tensor = _Tensor(0x1000)

        def traced_begin(group, tensor, **kwargs):
            return begin_all_reduce_trace(
                group,
                tensor,
                recorder=recorder,
                **kwargs,
            )

        with (
            patch.object(parallel_state, "ALL_REDUCE_TRACE_ENABLED", True),
            patch.object(
                parallel_state,
                "begin_all_reduce_trace",
                side_effect=traced_begin,
            ),
            patch.object(parallel_state, "inplace_all_reduce") as inplace,
            patch.object(
                parallel_state,
                "is_in_tc_piecewise_cuda_graph",
                return_value=False,
            ),
        ):
            output = group.all_reduce(input_tensor)

        self.assertIs(output, input_tensor)
        inplace.assert_called_once_with(input_tensor, group_name="tp:0")
        self.assertEqual(len(recorder.records), 1)
        record = recorder.records[0]
        self.assertEqual(
            record["selection"]["selected_backend"],
            "torch_distributed_inplace",
        )
        self.assertEqual(
            record["selection"]["selected_backend_source"],
            "pre_dispatch_mirror_of_inplace_selector",
        )
        self.assertTrue(record["alias"]["python_object_identity"])

    def test_real_all_reduce_method_contains_trace_entry_failure(self):
        group = _group_coordinator()
        input_tensor = _Tensor(0x1000)
        with (
            patch.object(parallel_state, "ALL_REDUCE_TRACE_ENABLED", True),
            patch.object(
                parallel_state,
                "begin_all_reduce_trace",
                side_effect=RuntimeError("injected integration failure"),
            ),
            patch.object(
                parallel_state,
                "note_all_reduce_trace_failure",
            ) as note_failure,
            patch.object(parallel_state, "inplace_all_reduce") as inplace,
            patch.object(
                parallel_state,
                "is_in_tc_piecewise_cuda_graph",
                return_value=False,
            ),
        ):
            output = group.all_reduce(input_tensor)

        self.assertIs(output, input_tensor)
        inplace.assert_called_once_with(input_tensor, group_name="tp:0")
        note_failure.assert_called_once_with("parallel_state_begin")


if __name__ == "__main__":
    unittest.main()
