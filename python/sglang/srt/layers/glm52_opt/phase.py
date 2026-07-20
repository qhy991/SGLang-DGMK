"""Map SGLang forward modes to GLM-5.2 harness phases."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardMode


def infer_glm52_phase(
    forward_mode: Optional["ForwardMode"],
    token_num: int = 0,
) -> str:
    if forward_mode is not None:
        if forward_mode.is_decode():
            return "decode"
        if getattr(forward_mode, "is_target_verify", lambda: False)():
            return "decode"
        if forward_mode.is_extend() or forward_mode.is_mixed():
            return "prefill"
        if forward_mode.name in ("SPLIT_PREFILL", "DLLM_EXTEND"):
            return "prefill"
    # Conservative fallback for unknown modes.
    if 0 < token_num <= 64:
        return "decode"
    return "prefill"
