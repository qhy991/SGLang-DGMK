#!/usr/bin/env python3
"""Arm an nsys file trigger after one_batch_server has built the KV cache."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import sglang.benchmark.one_batch_server as one_batch_server


def main() -> None:
    trigger_value = os.environ.get("GLM52_NSYS_TRIGGER")
    if not trigger_value:
        raise SystemExit("GLM52_NSYS_TRIGGER must name the profiler trigger file")
    trigger = Path(trigger_value)
    arm_delay = float(os.environ.get("GLM52_NSYS_ARM_DELAY", "2"))
    if arm_delay < 0:
        raise SystemExit("GLM52_NSYS_ARM_DELAY must be non-negative")

    original_warmup_cache = one_batch_server._warmup_cache
    armed = False

    def warm_then_arm(*args, **kwargs):
        nonlocal armed
        result = original_warmup_cache(*args, **kwargs)
        if armed:
            raise RuntimeError("nsys trigger would be armed more than once")
        trigger.parent.mkdir(parents=True, exist_ok=True)
        temporary = trigger.with_suffix(trigger.suffix + f".tmp.{os.getpid()}")
        temporary.write_text("1\n")
        os.replace(temporary, trigger)
        armed = True
        print(
            f"[VALID] KV warmup complete; armed nsys trigger {trigger}; "
            f"waiting {arm_delay:.3f}s before measured request",
            flush=True,
        )
        time.sleep(arm_delay)
        return result

    one_batch_server.DEFAULT_TIMEOUT = 7200
    one_batch_server._warmup_cache = warm_then_arm
    sys.argv[0] = "one_batch_server"
    one_batch_server.cli_main()
    if not armed:
        raise SystemExit("cache warmup hook was not called; refusing unprofiled result")


if __name__ == "__main__":
    main()
