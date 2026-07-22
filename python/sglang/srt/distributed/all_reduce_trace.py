"""Opt-in, buffered reachability tracing for ``GroupCoordinator.all_reduce``.

Set ``SGLANG_ALL_REDUCE_TRACE`` before process start to a JSONL destination.
The destination may contain ``{rank}`` and ``{pid}``; otherwise a rank/pid
suffix is added automatically. ``1`` uses a file under ``/tmp``.

The all-reduce hot path never writes files or synchronizes CUDA. A cheap,
bounded caller-chain/tensor/graph/stream key is checked on every enabled call;
predicates, the full stack, metadata, JSON, and SHA are collected only for the
first observation of a key/backend pair. Records flush at interpreter exit, or when
``flush_all_reduce_trace`` is called explicitly at a known-safe point outside
CUDA graph capture/replay.

Explicit flush is a quiescent-point operation: it holds the recorder lock while
serializing and writing, so it must not run concurrently with traced dispatch.
The diagnostic runner flushes only after every measurement has completed.

This recorder is intended for short reachability diagnostics. Its deduplication
keys are process-lifetime state, so it should not be left enabled indefinitely
on a dynamic-shape production server.

This is a Python dispatch trace: it can prove eager execution or the Python
capture pass, but Python is not re-entered by CUDA graph replay. Correlate the
record with stable kernel names in an Nsight Systems trace for replay evidence.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import pathlib
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional

TRACE_ENV = "SGLANG_ALL_REDUCE_TRACE"
SCHEMA_VERSION = 1
REPLAY_LIMITATION = (
    "Python observes eager dispatch or CUDA graph capture only; CUDA graph replay "
    "does not re-enter this hook. Correlate stable kernel names in an Nsight "
    "Systems trace to prove replay execution."
)
_FALSE_VALUES = {"", "0", "false", "no", "off"}
_CALLER_EXCLUDED_MODULES = {
    __name__,
    "sglang.srt.distributed.parallel_state",
    "sglang.srt.distributed.communication_op",
}


def _class_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _safe_bool_call(
    obj: Any,
    method_name: str,
    *args: Any,
) -> tuple[Optional[bool], Optional[str]]:
    if obj is None:
        return None, None
    method = getattr(obj, method_name, None)
    if method is None:
        return None, f"missing method {method_name}"
    try:
        return bool(method(*args)), None
    except Exception as exc:  # The diagnostic must not change backend reachability.
        return None, f"{type(exc).__name__}: {exc}"


def _safe_data_ptr(tensor: Any) -> tuple[Optional[int], Optional[str]]:
    method = getattr(tensor, "data_ptr", None)
    if method is None:
        return None, "missing data_ptr"
    try:
        return int(method()), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _tensor_metadata(tensor: Any) -> dict[str, Any]:
    if tensor is None:
        return {"present": False}

    errors: dict[str, str] = {}
    try:
        shape = [int(value) for value in tensor.shape]
    except Exception as exc:
        shape = None
        errors["shape"] = f"{type(exc).__name__}: {exc}"
    try:
        stride = [int(value) for value in tensor.stride()]
    except Exception as exc:
        stride = None
        errors["stride"] = f"{type(exc).__name__}: {exc}"
    try:
        numel = int(tensor.numel())
    except Exception as exc:
        numel = None
        errors["numel"] = f"{type(exc).__name__}: {exc}"
    try:
        element_size = int(tensor.element_size())
    except Exception as exc:
        element_size = None
        errors["element_size"] = f"{type(exc).__name__}: {exc}"
    data_ptr, pointer_error = _safe_data_ptr(tensor)
    if pointer_error is not None:
        errors["data_ptr"] = pointer_error

    device = getattr(tensor, "device", None)
    metadata: dict[str, Any] = {
        "present": True,
        "shape": shape,
        "stride": stride,
        "dtype": str(getattr(tensor, "dtype", "unknown")),
        "device": str(device) if device is not None else None,
        "numel": numel,
        "element_size": element_size,
        "bytes": (
            numel * element_size
            if numel is not None and element_size is not None
            else None
        ),
        "data_ptr": data_ptr,
    }
    if errors:
        metadata["errors"] = errors
    return metadata


def _cheap_tensor_signature(tensor: Any) -> tuple[Any, ...]:
    """Return only the tensor fields that distinguish trace signatures."""
    shape = tuple(int(value) for value in tensor.shape)
    stride = tuple(int(value) for value in tensor.stride())
    numel = int(tensor.numel())
    element_size = int(tensor.element_size())
    return shape, stride, str(tensor.dtype), numel * element_size


def _caller_context(limit: int = 8) -> tuple[Optional[str], tuple[str, ...]]:
    """Return a bounded external caller chain without materializing a stack."""
    try:
        frame = sys._getframe(2)
    except ValueError:
        return None, ()
    fallback: list[str] = []
    external: list[str] = []
    while frame is not None and len(external) < limit:
        module = str(frame.f_globals.get("__name__", ""))
        code = frame.f_code
        tag = f"{module}:{code.co_qualname}:{frame.f_lineno}"
        if len(fallback) < limit:
            fallback.append(tag)
        if module not in _CALLER_EXCLUDED_MODULES:
            external.append(tag)
        frame = frame.f_back
    chain = tuple(external or fallback[:1])
    return (chain[0] if chain else None), chain


def _capture_python_stack(
    caller_tag: Optional[str], caller_key_chain: tuple[str, ...], limit: int = 16
) -> dict[str, Any]:
    """Capture code-object metadata only; unlike traceback, this reads no files."""
    frames: list[dict[str, Any]] = []
    try:
        frame = sys._getframe(2)
    except ValueError:
        frame = None
    while frame is not None and len(frames) < limit:
        module = str(frame.f_globals.get("__name__", ""))
        code = frame.f_code
        frames.append(
            {
                "module": module,
                "function": code.co_qualname,
                "filename": code.co_filename,
                "line": int(frame.f_lineno),
            }
        )
        frame = frame.f_back
    return {
        "tag": caller_tag,
        "tag_source": "first_external_python_frame_at_dispatch_entry",
        "key_chain": list(caller_key_chain),
        "key_chain_source": "bounded_external_python_frames_at_dispatch_entry",
        "stack": frames,
    }


def _capture_dispatch_state(
    tensor: Any,
) -> tuple[Optional[bool], Optional[str], dict[str, Any]]:
    """Capture graph and entry-stream identity without synchronizing CUDA."""
    device = getattr(tensor, "device", None)
    if getattr(device, "type", None) != "cuda":
        return None, None, {"present": False}
    capture_state = None
    capture_error = None
    stream_state: dict[str, Any] = {"present": False}
    try:
        import torch

        capture_state = bool(torch.cuda.is_current_stream_capturing())
    except Exception as exc:
        capture_error = f"{type(exc).__name__}: {exc}"
    try:
        import torch

        stream = torch.cuda.current_stream(device=device)
        stream_state = {
            "present": True,
            "class": _class_name(stream),
            "device": str(getattr(stream, "device", device)),
            "cuda_stream": int(getattr(stream, "cuda_stream")),
            "priority": int(getattr(stream, "priority", 0)),
        }
    except Exception as exc:
        stream_state["probe_error"] = f"{type(exc).__name__}: {exc}"
    return capture_state, capture_error, stream_state


def _stream_signature(stream_state: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        bool(stream_state.get("present")),
        stream_state.get("device"),
        stream_state.get("cuda_stream"),
        stream_state.get("priority"),
        stream_state.get("probe_error"),
    )


def select_all_reduce_backend(predicates: Mapping[str, Any]) -> str:
    """Mirror ``GroupCoordinator.all_reduce`` backend priority using pure data."""
    if predicates.get("world_size_is_one"):
        return "identity"
    if predicates.get("input_is_cpu"):
        return "cpu_shm" if predicates.get("cpu_shm_eligible") else "cpu_c10d"
    if predicates.get("hpu_eligible"):
        return "hpu"
    if predicates.get("xpu_eligible"):
        return "xpu_torch_distributed_inplace"
    if predicates.get("npu_eligible"):
        return "npu"
    if predicates.get("pynccl_symmetric_eligible"):
        return "pynccl_symmetric_inplace"
    if predicates.get("custom_eligible"):
        return "custom_all_reduce_outplace"
    if predicates.get("quick_eligible"):
        return "quick_all_reduce_outplace"
    if predicates.get("pymscclpp_eligible"):
        return "pymscclpp_outplace"
    if predicates.get("torch_symm_mem_eligible"):
        return "torch_symm_mem_outplace"
    if predicates.get("piecewise_pynccl_eligible"):
        return "pynccl_piecewise_outplace"
    if predicates.get("pynccl_inplace_eligible"):
        return "pynccl_inplace"
    if predicates.get("torch_symm_mem_inplace_eligible"):
        return "torch_symm_mem_inplace"
    return "torch_distributed_inplace"


def _selection_snapshot(
    group: Any,
    tensor: Any,
    *,
    piecewise_cuda_graph: bool,
    cpu_shm_eligible: Optional[bool],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    errors: dict[str, str] = {}

    pymscclpp = getattr(group, "pymscclpp_comm", None)
    pynccl = getattr(group, "pynccl_comm", None)
    custom = getattr(group, "ca_comm", None)
    quick = getattr(group, "qr_comm", None)
    torch_symm = getattr(group, "torch_symm_mem_comm", None)
    hpu = getattr(group, "hpu_communicator", None)
    xpu = getattr(group, "xpu_communicator", None)
    npu = getattr(group, "npu_communicator", None)

    pymscclpp_should, error = _safe_bool_call(
        pymscclpp, "should_mscclpp_allreduce", tensor
    )
    if error is not None:
        errors["pymscclpp_should"] = error
    custom_should, error = _safe_bool_call(custom, "should_custom_ar", tensor)
    if error is not None:
        errors["custom_should"] = error
    quick_should, error = _safe_bool_call(quick, "should_quick_allreduce", tensor)
    if error is not None:
        errors["quick_should"] = error
    torch_symm_should, error = _safe_bool_call(
        torch_symm, "should_torch_symm_mem_allreduce", tensor
    )
    if error is not None:
        errors["torch_symm_mem_should"] = error
    symmetric_enabled, error = _safe_bool_call(group, "is_symmetric_memory_enabled")
    if error is not None:
        errors["symmetric_memory_enabled"] = error

    custom_disabled = bool(getattr(custom, "disabled", True))
    quick_disabled = bool(getattr(quick, "disabled", True))
    pymscclpp_disabled = bool(getattr(pymscclpp, "disabled", True))
    torch_symm_disabled = bool(getattr(torch_symm, "disabled", True))
    pynccl_disabled = bool(getattr(pynccl, "disabled", True))
    world_size = int(getattr(group, "world_size", 0))
    input_is_cpu = bool(getattr(tensor, "is_cpu", False))
    pymscclpp_eligible = bool(pymscclpp is not None and pymscclpp_should)
    torch_symm_eligible = bool(
        torch_symm is not None and not torch_symm_disabled and torch_symm_should
    )
    predicates: dict[str, Any] = {
        "world_size_is_one": world_size == 1,
        "input_is_cpu": input_is_cpu,
        "cpu_shm_eligible": bool(cpu_shm_eligible),
        "hpu_present": hpu is not None,
        "hpu_disabled": bool(getattr(hpu, "disabled", True)),
        "hpu_eligible": bool(hpu is not None and not getattr(hpu, "disabled", True)),
        "xpu_present": xpu is not None,
        "xpu_disabled": bool(getattr(xpu, "disabled", True)),
        "xpu_eligible": bool(xpu is not None and not getattr(xpu, "disabled", True)),
        "npu_present": npu is not None,
        "npu_disabled": bool(getattr(npu, "disabled", True)),
        "npu_eligible": bool(npu is not None and not getattr(npu, "disabled", True)),
        "pymscclpp_present": pymscclpp is not None,
        "pymscclpp_should": pymscclpp_should,
        "pymscclpp_disabled": pymscclpp_disabled,
        "pymscclpp_eligible": pymscclpp_eligible,
        "symmetric_memory_enabled": symmetric_enabled,
        "pynccl_present": pynccl is not None,
        "pynccl_disabled": pynccl_disabled,
        "pynccl_symmetric_eligible": bool(
            pynccl is not None and symmetric_enabled and not pymscclpp_eligible
        ),
        "custom_present": custom is not None,
        "custom_should": custom_should,
        "custom_disabled": custom_disabled,
        "custom_eligible": bool(
            custom is not None
            and not custom_disabled
            and not pymscclpp_eligible
            and custom_should
        ),
        "quick_present": quick is not None,
        "quick_should": quick_should,
        "quick_disabled": quick_disabled,
        "quick_eligible": bool(
            quick is not None and not quick_disabled and quick_should
        ),
        "torch_symm_mem_present": torch_symm is not None,
        "torch_symm_mem_should": torch_symm_should,
        "torch_symm_mem_disabled": torch_symm_disabled,
        "torch_symm_mem_eligible": torch_symm_eligible,
        "piecewise_cuda_graph": bool(piecewise_cuda_graph),
        "piecewise_pynccl_eligible": bool(piecewise_cuda_graph and pynccl is not None),
        "pynccl_inplace_eligible": bool(pynccl is not None and not pynccl_disabled),
        "torch_symm_mem_inplace_eligible": torch_symm_eligible,
    }

    communicators = {
        "device_group": _class_name(getattr(group, "device_group", None)),
        "hpu": _class_name(hpu),
        "xpu": _class_name(xpu),
        "npu": _class_name(npu),
        "pynccl": _class_name(pynccl),
        "custom": _class_name(custom),
        "quick": _class_name(quick),
        "pymscclpp": _class_name(pymscclpp),
        "torch_symm_mem": _class_name(torch_symm),
    }

    custom_algorithm = None
    determine_algo = getattr(custom, "_determine_algo", None)
    if determine_algo is not None:
        try:
            algo = determine_algo(tensor)
            custom_algorithm = str(getattr(algo, "name", algo))
        except Exception as exc:
            errors["custom_algorithm"] = f"{type(exc).__name__}: {exc}"
    details = {
        "communicators": communicators,
        "custom_algorithm": custom_algorithm,
        "custom_algorithm_source": (
            "custom_private_selector_mirror_pre_dispatch"
            if custom_algorithm is not None
            else None
        ),
    }
    return predicates, details, errors


def _graph_and_stream_state(
    piecewise_cuda_graph: bool,
    capture_state: Optional[bool],
    capture_error: Optional[str],
    entry_stream_state: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    graph_state: dict[str, Any] = {
        "cuda_stream_capturing": capture_state,
        "tc_piecewise_cuda_graph": bool(piecewise_cuda_graph),
        "python_hook_observation": "eager_or_capture_dispatch_only",
        "cuda_graph_replay_observed": False,
        "replay_limitation": REPLAY_LIMITATION,
    }
    if capture_error is not None:
        graph_state["probe_error"] = capture_error
    return graph_state, dict(entry_stream_state)


class AllReduceTraceRecorder:
    def __init__(self, destination: Optional[str], *, enabled: bool = True):
        self.destination = destination
        self.enabled = enabled
        self._metadata_keys: set[tuple[Any, ...]] = set()
        self._selected_keys: set[tuple[Any, ...]] = set()
        self._seen: set[str] = set()
        self._records: list[dict[str, Any]] = []
        self._failures: dict[str, int] = {}
        self._lock = threading.Lock()

    @property
    def records(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._records)

    @property
    def failures(self) -> dict[str, int]:
        with self._lock:
            return dict(self._failures)

    def reserve_metadata(self, key: tuple[Any, ...]) -> bool:
        with self._lock:
            if key in self._metadata_keys:
                return False
            self._metadata_keys.add(key)
            return True

    def release_metadata(self, key: tuple[Any, ...]) -> None:
        with self._lock:
            self._metadata_keys.discard(key)

    def claim_selected(self, key: tuple[Any, ...]) -> bool:
        with self._lock:
            if key in self._selected_keys:
                return False
            self._selected_keys.add(key)
            return True

    def release_selected(self, key: tuple[Any, ...]) -> None:
        with self._lock:
            self._selected_keys.discard(key)

    def note_failure(self, phase: str) -> None:
        with self._lock:
            self._failures[phase] = self._failures.get(phase, 0) + 1

    @staticmethod
    def _signature_payload(record: Mapping[str, Any]) -> dict[str, Any]:
        tensor = record["input"]
        selection = record["selection"]
        return {
            "group": record["group"],
            "caller_tag": record["caller"]["tag"],
            "caller_key_chain": record["caller"]["key_chain"],
            "shape": tensor["shape"],
            "stride": tensor["stride"],
            "dtype": tensor["dtype"],
            "bytes": tensor["bytes"],
            "graph": {
                "cuda_stream_capturing": record["graph"]["cuda_stream_capturing"],
                "tc_piecewise_cuda_graph": record["graph"]["tc_piecewise_cuda_graph"],
            },
            "stream": _stream_signature(record["stream"]),
            "selected_backend": selection["selected_backend"],
            "communicator": selection["selected_communicator"],
            "algorithm": selection["selected_algorithm"],
        }

    def record_first(self, record: dict[str, Any]) -> bool:
        payload = self._signature_payload(record)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        signature = hashlib.sha256(encoded).hexdigest()[:20]
        record["signature"] = {
            "id": signature,
            "dedup_scope": "process",
            "policy": "first hit per group/caller/tensor/graph/backend signature",
        }
        with self._lock:
            if signature in self._seen:
                return False
            self._seen.add(signature)
            self._records.append(record)
        return True

    @staticmethod
    def serialize_record(record: Mapping[str, Any]) -> str:
        return json.dumps(record, sort_keys=True, separators=(",", ":"))

    def _resolve_destination(self, records: list[dict[str, Any]]) -> pathlib.Path:
        raw = self.destination or "1"
        rank = records[0].get("group", {}).get("rank", "unknown")
        pid = os.getpid()
        if raw.strip().lower() in {"1", "true", "yes", "on"}:
            raw = "/tmp/sglang_all_reduce_trace.{rank}.{pid}.jsonl"
        has_placeholder = "{rank}" in raw or "{pid}" in raw
        raw = raw.replace("{rank}", str(rank)).replace("{pid}", str(pid))
        path = pathlib.Path(raw).expanduser()
        if raw.endswith(os.sep) or (path.exists() and path.is_dir()):
            path = path / f"all_reduce_trace.rank{rank}.pid{pid}.jsonl"
        elif not has_placeholder:
            suffix = path.suffix or ".jsonl"
            stem = path.name[: -len(path.suffix)] if path.suffix else path.name
            path = path.with_name(f"{stem}.rank{rank}.pid{pid}{suffix}")
        return path

    def flush(self, destination: Optional[str] = None) -> Optional[pathlib.Path]:
        """Write records at an explicit, quiescent non-capture/replay point."""
        with self._lock:
            if not self._records:
                return None
            records = list(self._records)
            old_destination = self.destination
            if destination is not None:
                self.destination = destination
            try:
                path = self._resolve_destination(records)
            finally:
                self.destination = old_destination
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = "".join(self.serialize_record(item) + "\n" for item in records)
            with path.open("a", encoding="utf-8") as output:
                output.write(payload)
            self._records.clear()
        return path


@dataclass
class AllReduceTraceToken:
    recorder: AllReduceTraceRecorder
    group: Any
    input_tensor: Any
    pre_signature: tuple[Any, ...]
    caller_tag: Optional[str]
    caller_key_chain: tuple[str, ...]
    piecewise_cuda_graph: bool
    cpu_shm_eligible: Optional[bool]
    capture_state: Optional[bool]
    capture_error: Optional[str]
    entry_stream_state: dict[str, Any]
    record: Optional[dict[str, Any]] = None


def _note_failure(recorder: AllReduceTraceRecorder, phase: str) -> None:
    try:
        recorder.note_failure(phase)
    except Exception:
        pass


def _build_record(
    token: AllReduceTraceToken, *, predicate_observation_phase: str
) -> dict[str, Any]:
    group = token.group
    tensor = token.input_tensor
    predicates, details, predicate_errors = _selection_snapshot(
        group,
        tensor,
        piecewise_cuda_graph=token.piecewise_cuda_graph,
        cpu_shm_eligible=token.cpu_shm_eligible,
    )
    predicted = select_all_reduce_backend(predicates)
    graph_state, stream_state = _graph_and_stream_state(
        token.piecewise_cuda_graph,
        token.capture_state,
        token.capture_error,
        token.entry_stream_state,
    )
    input_metadata = _tensor_metadata(tensor)
    if predicate_errors:
        _note_failure(token.recorder, "predicate_probe")
    if "probe_error" in graph_state or "probe_error" in stream_state:
        _note_failure(token.recorder, "graph_or_stream_probe")
    if input_metadata.get("errors"):
        _note_failure(token.recorder, "input_metadata")
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "sglang_all_reduce_reachability",
        "observed_at_unix_ns": time.time_ns(),
        "process": {"pid": os.getpid()},
        "group": {
            "name": str(getattr(group, "unique_name", "unknown")),
            "ranks": [int(value) for value in getattr(group, "ranks", [])],
            "world_size": int(getattr(group, "world_size", 0)),
            "rank": int(getattr(group, "rank", -1)),
            "rank_in_group": int(getattr(group, "rank_in_group", -1)),
        },
        "caller": _capture_python_stack(token.caller_tag, token.caller_key_chain),
        "input": input_metadata,
        "graph": graph_state,
        "stream": stream_state,
        "selection": {
            "predicate_observation_phase": predicate_observation_phase,
            "predicates": predicates,
            "predicate_errors": predicate_errors,
            "communicators": details["communicators"],
            "custom_algorithm": details["custom_algorithm"],
            "custom_algorithm_source": details["custom_algorithm_source"],
            "predicted_backend": predicted,
            "selected_backend": None,
            "selected_backend_source": None,
            "selected_communicator": None,
            "selected_algorithm": None,
        },
    }


def _recorder_from_environment() -> AllReduceTraceRecorder:
    value = os.environ.get(TRACE_ENV, "")
    enabled = value.strip().lower() not in _FALSE_VALUES
    return AllReduceTraceRecorder(value if enabled else None, enabled=enabled)


_RECORDER = _recorder_from_environment()
ALL_REDUCE_TRACE_ENABLED = _RECORDER.enabled


def begin_all_reduce_trace(
    group: Any,
    tensor: Any,
    *,
    piecewise_cuda_graph: bool,
    cpu_shm_eligible: Optional[bool] = None,
    recorder: Optional[AllReduceTraceRecorder] = None,
) -> Optional[AllReduceTraceToken]:
    """Create a fail-open trace token without performing CUDA synchronization."""
    active_recorder = recorder if recorder is not None else _RECORDER
    pre_signature = None
    metadata_reserved = False
    try:
        if not active_recorder.enabled:
            return None
        caller_tag, caller_key_chain = _caller_context()
        capture_state, capture_error, entry_stream_state = _capture_dispatch_state(
            tensor
        )
        group_signature = (
            str(getattr(group, "unique_name", "unknown")),
            tuple(int(value) for value in getattr(group, "ranks", [])),
            int(getattr(group, "world_size", 0)),
            int(getattr(group, "rank", -1)),
            int(getattr(group, "rank_in_group", -1)),
        )
        pre_signature = (
            group_signature,
            caller_key_chain,
            _cheap_tensor_signature(tensor),
            bool(piecewise_cuda_graph),
            capture_state,
            capture_error,
            _stream_signature(entry_stream_state),
        )
        metadata_reserved = active_recorder.reserve_metadata(pre_signature)
        token = AllReduceTraceToken(
            recorder=active_recorder,
            group=group,
            input_tensor=tensor,
            pre_signature=pre_signature,
            caller_tag=caller_tag,
            caller_key_chain=caller_key_chain,
            piecewise_cuda_graph=bool(piecewise_cuda_graph),
            cpu_shm_eligible=cpu_shm_eligible,
            capture_state=capture_state,
            capture_error=capture_error,
            entry_stream_state=entry_stream_state,
        )
        if metadata_reserved:
            token.record = _build_record(
                token, predicate_observation_phase="begin_pre_dispatch"
            )
        return token
    except Exception:
        if metadata_reserved and pre_signature is not None:
            try:
                active_recorder.release_metadata(pre_signature)
            except Exception:
                pass
        _note_failure(active_recorder, "begin")
        return None


def _selected_details(
    record: Mapping[str, Any], selected_backend: Optional[str]
) -> tuple[Any, Any, Optional[str]]:
    if selected_backend is None:
        return None, None, None
    selection = record["selection"]
    comms = selection["communicators"]
    if selected_backend.startswith("custom_all_reduce"):
        return (
            comms["custom"],
            selection.get("custom_algorithm"),
            selection.get("custom_algorithm_source"),
        )
    if selected_backend.startswith("quick_all_reduce"):
        return comms["quick"], "QUICK_REDUCE_SUM", "backend_contract_constant"
    if selected_backend.startswith("pymscclpp"):
        return comms["pymscclpp"], "MSCCLPP_SUM", "backend_contract_constant"
    if selected_backend.startswith("torch_symm_mem"):
        return (
            comms["torch_symm_mem"],
            "TORCH_SYMM_MEM_SUM",
            "backend_contract_constant",
        )
    if selected_backend.startswith("pynccl"):
        return comms["pynccl"], "NCCL_SUM", "backend_contract_constant"
    if (
        selected_backend.startswith("torch_distributed")
        or selected_backend == "cpu_c10d"
    ):
        return comms["device_group"], "SUM", "backend_contract_constant"
    if selected_backend == "hpu":
        return comms["hpu"], "SUM", "backend_contract_constant"
    if selected_backend == "xpu_torch_distributed_inplace":
        return comms["device_group"], "SUM", "backend_contract_constant"
    if selected_backend == "npu":
        return comms["npu"], "SUM", "backend_contract_constant"
    if selected_backend == "cpu_shm":
        return (
            "torch.ops.sgl_kernel.shm_allreduce",
            "SUM",
            "backend_contract_constant",
        )
    if selected_backend == "identity":
        return None, "IDENTITY", "backend_contract_constant"
    return None, None, None


def finish_all_reduce_trace(
    token: Optional[AllReduceTraceToken],
    output: Any,
    *,
    selected_backend: Optional[str],
    selected_backend_source: str = "observed_dispatch_branch",
) -> None:
    """Complete a trace record; every diagnostic failure is contained."""
    if token is None:
        return

    selected_key = None
    selected_claimed = False
    try:
        selected_key = (token.pre_signature, selected_backend)
        selected_claimed = token.recorder.claim_selected(selected_key)
        if not selected_claimed:
            return
        record = token.record
        if record is None:
            record = _build_record(
                token,
                predicate_observation_phase="finish_post_dispatch_new_backend",
            )
        output_metadata = _tensor_metadata(output)
        if output_metadata.get("errors"):
            _note_failure(token.recorder, "output_metadata")
        input_metadata = record["input"]
        input_ptr = input_metadata.get("data_ptr")
        output_ptr = output_metadata.get("data_ptr")
        communicator, algorithm, algorithm_source = _selected_details(
            record, selected_backend
        )
        selection = record["selection"]
        selection["selected_backend"] = selected_backend
        selection["selected_backend_source"] = selected_backend_source
        selection["selected_communicator"] = communicator
        selection["selected_algorithm"] = algorithm
        selection["selected_algorithm_source"] = algorithm_source
        selection["prediction_matches_selection"] = (
            selection["predicted_backend"] == selected_backend
            if selected_backend is not None
            else None
        )
        record["output"] = output_metadata
        record["alias"] = {
            "python_object_identity": output is token.input_tensor,
            "same_data_ptr": (
                input_ptr == output_ptr
                if input_ptr is not None and output_ptr is not None
                else None
            ),
            "input_data_ptr": input_ptr,
            "output_data_ptr": output_ptr,
        }
        token.recorder.record_first(record)
    except Exception:
        if selected_claimed and selected_key is not None:
            try:
                token.recorder.release_selected(selected_key)
            except Exception:
                pass
        try:
            token.recorder.release_metadata(token.pre_signature)
        except Exception:
            pass
        _note_failure(token.recorder, "finish")


def flush_all_reduce_trace(destination: Optional[str] = None) -> Optional[pathlib.Path]:
    """Flush at an explicit non-capture/non-replay safe point, without syncing CUDA."""
    return _RECORDER.flush(destination)


def get_all_reduce_trace_failure_counts() -> dict[str, int]:
    """Return a copy of fail-open diagnostic failures for evidence validation."""
    return _RECORDER.failures


def note_all_reduce_trace_failure(phase: str) -> None:
    """Record a contained integration-hook failure without raising or doing I/O."""
    _note_failure(_RECORDER, phase)


def _flush_at_exit() -> None:
    try:
        _RECORDER.flush()
    except Exception as exc:
        sys.stderr.write(f"Failed to flush all-reduce reachability trace: {exc}\n")


if ALL_REDUCE_TRACE_ENABLED:
    atexit.register(_flush_at_exit)
