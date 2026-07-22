"""Optional Nsight capture-range gate via cudaProfilerApi + NVTX.

Enabled when ``SGLANG_GLM52_NSYS_GATE=1``. Each CUDA worker polls a trigger file;
when present, it starts an nsys capture window of ``SGLANG_GLM52_NSYS_SECONDS``
(default 28) then stops. Use with:

  nsys profile -c cudaProfilerApi --capture-range-end=stop --kill=none ...

Trigger file default: ``/home/ubuntu/wwxq/cache/sglang/nsys_trigger``
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_STARTED = False
_LOCK = threading.Lock()


def _truthy(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def maybe_start_nsys_gate() -> None:
    """Start background watcher once per process (no-op unless gate env is set)."""
    global _STARTED
    if not _truthy("SGLANG_GLM52_NSYS_GATE"):
        return
    with _LOCK:
        if _STARTED:
            return
        _STARTED = True
    t = threading.Thread(target=_watch_loop, name="glm52-nsys-gate", daemon=True)
    t.start()
    print(
        f"[glm52_opt] nsys gate armed pid={os.getpid()} "
        f"trigger={_trigger_path()} secs={_duration_s()}",
        flush=True,
    )


def _trigger_path() -> Path:
    return Path(
        os.environ.get(
            "SGLANG_GLM52_NSYS_TRIGGER",
            "/home/ubuntu/wwxq/cache/sglang/nsys_trigger",
        )
    )


def _duration_s() -> float:
    try:
        return float(os.environ.get("SGLANG_GLM52_NSYS_SECONDS", "28"))
    except ValueError:
        return 28.0


def _watch_loop() -> None:
    path = _trigger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fired = False
    while True:
        try:
            if path.exists() and not fired:
                fired = True
                _run_capture_window()
                # One-shot per process; wait for trigger removal to re-arm.
            elif not path.exists():
                fired = False
        except Exception as exc:
            print(f"[glm52_opt] nsys gate error: {exc}", flush=True)
        time.sleep(0.2)


def _run_capture_window() -> None:
    import torch

    secs = _duration_s()
    dev = torch.cuda.current_device() if torch.cuda.is_available() else -1
    print(
        f"[glm52_opt] nsys capture START pid={os.getpid()} device={dev} secs={secs}",
        flush=True,
    )
    try:
        torch.cuda.nvtx.range_push("glm52_nsys_window")
    except Exception:
        pass
    try:
        torch.cuda.cudart().cudaProfilerStart()
    except Exception as exc:
        print(f"[glm52_opt] cudaProfilerStart failed: {exc}", flush=True)
    try:
        time.sleep(secs)
    finally:
        try:
            torch.cuda.cudart().cudaProfilerStop()
        except Exception as exc:
            print(f"[glm52_opt] cudaProfilerStop failed: {exc}", flush=True)
        try:
            torch.cuda.nvtx.range_pop()
        except Exception:
            pass
        print(
            f"[glm52_opt] nsys capture STOP pid={os.getpid()} device={dev}",
            flush=True,
        )
