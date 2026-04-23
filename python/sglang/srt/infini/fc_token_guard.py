from __future__ import annotations

import copy
from typing import Any, Dict, Optional

FC_SPECIAL_TOKENS = (
    "<|tool_calls_section_begin|>",
    "<|tool_calls_section_end|>",
    "<|tool_call_begin|>",
    "<|tool_call_end|>",
    "<|tool_call_argument_begin|>",
)


def strip_kimi_fc_special_substrings(s: Optional[str]) -> Optional[str]:
    """Remove all Kimi FC special literal substrings from text (e.g. for OpenAI output)."""
    if not s:
        return s
    out: str = s
    for tok in FC_SPECIAL_TOKENS:
        if tok in out:
            out = out.replace(tok, "")
    return out


def prefer_engine_finish_on_fc_leak(
    engine_finish: Optional[Dict[str, Any]],
    *,
    default_unexpected: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Use ``default_unexpected`` (typically ``unexpected_state``) only in place of a
    normal engine ``stop``. Never replace a non-``stop`` reason (e.g. ``length``,
    ``abort``, ``content_filter``).

    If ``engine_finish`` has no ``type`` key, the payload is left unchanged (no
    ``unexpected`` unless ``engine_finish`` is entirely missing).
    """
    du = default_unexpected or {
        "type": "unexpected_state",
        "matched": None,
    }
    if not engine_finish:
        return {**du}
    t = engine_finish.get("type")
    if t is not None and t != "stop":
        return copy.deepcopy(engine_finish)
    if t == "stop":
        return {**du}
    return copy.deepcopy(engine_finish)
