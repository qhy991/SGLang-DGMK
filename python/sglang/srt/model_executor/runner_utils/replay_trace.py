# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Opt-in, unprofiled timing of decode CUDA-graph submission.

This diagnostic intentionally measures only the host ``CUDAGraph.replay`` call.
It does not synchronize the device and therefore does not turn asynchronous
serving into a benchmark.  ``time.perf_counter_ns`` is a same-host monotonic
clock, so start timestamps from the eight worker processes can be aligned after
the run.

The trace is off unless ``SGLANG_DECODE_GRAPH_REPLAY_TRACE_DIR`` is set.  An
optional ``SGLANG_DECODE_GRAPH_REPLAY_TRACE_TRIGGER`` delays collection until a
file exists, allowing the fixed-KV warmup to finish before any samples are kept.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional


class DecodeGraphReplayTrace:
    def __init__(
        self,
        *,
        output_dir: Path,
        trigger: Optional[Path],
        sample_limit: int,
        tp_rank: int,
        gpu_id: int,
    ) -> None:
        if sample_limit <= 0:
            raise ValueError(
                "SGLANG_DECODE_GRAPH_REPLAY_TRACE_SAMPLES must be positive"
            )
        self.output_dir = output_dir
        self.trigger = trigger
        self.sample_limit = sample_limit
        self.tp_rank = tp_rank
        self.gpu_id = gpu_id
        self.pid = os.getpid()
        self.armed_at_ns: Optional[int] = None
        self.samples: list[dict[str, Any]] = []
        self.finished = False

    @classmethod
    def from_env(cls, model_runner: Any) -> Optional["DecodeGraphReplayTrace"]:
        raw_dir = os.environ.get("SGLANG_DECODE_GRAPH_REPLAY_TRACE_DIR")
        if not raw_dir:
            return None
        raw_trigger = os.environ.get("SGLANG_DECODE_GRAPH_REPLAY_TRACE_TRIGGER")
        try:
            sample_limit = int(
                os.environ.get("SGLANG_DECODE_GRAPH_REPLAY_TRACE_SAMPLES", "128")
            )
        except ValueError as exc:
            raise ValueError(
                "SGLANG_DECODE_GRAPH_REPLAY_TRACE_SAMPLES must be an integer"
            ) from exc
        return cls(
            output_dir=Path(raw_dir),
            trigger=Path(raw_trigger) if raw_trigger else None,
            sample_limit=sample_limit,
            tp_rank=int(model_runner.tp_rank),
            gpu_id=int(model_runner.gpu_id),
        )

    def should_record(self) -> bool:
        if self.finished:
            return False
        if self.armed_at_ns is not None:
            return True
        if self.trigger is not None and not self.trigger.is_file():
            return False
        self.armed_at_ns = time.perf_counter_ns()
        return True

    def record(
        self,
        *,
        call_start_ns: int,
        call_end_ns: int,
        graph_key: Any,
        raw_bs: int,
        padded_bs: int,
        forward_mode: str,
    ) -> None:
        if self.finished:
            return
        self.samples.append(
            {
                "ordinal": len(self.samples),
                "call_start_ns": call_start_ns,
                "call_end_ns": call_end_ns,
                "call_duration_ns": call_end_ns - call_start_ns,
                "graph_key_size": int(graph_key.size),
                "raw_bs": int(raw_bs),
                "padded_bs": int(padded_bs),
                "forward_mode": forward_mode,
            }
        )
        if len(self.samples) >= self.sample_limit:
            self._write()

    def _write(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output = self.output_dir / (
            f"decode-graph-replay-tp-rank-{self.tp_rank:03d}-pid-{self.pid}.json"
        )
        temporary = output.with_suffix(".json.tmp")
        payload = {
            "schema_version": 1,
            "kind": "sglang_decode_cuda_graph_replay_host_trace",
            "clock": (
                "time.perf_counter_ns; same-host monotonic; no device synchronization"
            ),
            "tp_rank": self.tp_rank,
            "gpu_id": self.gpu_id,
            "pid": self.pid,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "trigger": str(self.trigger) if self.trigger is not None else None,
            "armed_at_ns": self.armed_at_ns,
            "sample_limit": self.sample_limit,
            "samples": self.samples,
            "interpretation_guard": (
                "call_duration is asynchronous host submission time, not graph device "
                "latency. Compare cross-rank call starts and duration periodicity; do "
                "not claim kernel speedup from this trace."
            ),
        }
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, output)
        self.finished = True
