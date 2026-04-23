from __future__ import annotations

import json
from typing import Any, Optional

FC_SPECIAL_TOKENS = (
    "<|tool_calls_section_begin|>",
    "<|tool_calls_section_end|>",
    "<|tool_call_begin|>",
    "<|tool_call_end|>",
    "<|tool_call_argument_begin|>",
)


def contains_fc_special_tokens(text: Optional[str]) -> bool:
    if not text:
        return False
    return any(token in text for token in FC_SPECIAL_TOKENS)


def choice_has_fc_special_tokens(
    content: Optional[str],
    reasoning_content: Optional[str],
    tool_calls: Optional[list[Any]],
) -> bool:
    if contains_fc_special_tokens(content) or contains_fc_special_tokens(reasoning_content):
        return True
    for tool_call in tool_calls or []:
        arguments = (
            tool_call.function.arguments
            if getattr(tool_call, "function", None) is not None
            else None
        )
        if isinstance(arguments, str) and contains_fc_special_tokens(arguments):
            return True
    return False


def stream_sse_chunk_has_fc_special_tokens(chunk: str) -> bool:
    if not chunk.startswith("data: {"):
        return False
    try:
        payload = json.loads(chunk[len("data: ") :])
        choices = payload.get("choices", [])
        assert len(choices) == 1, "Expected exactly one choice per streamed chunk"
        delta_obj = choices[0]["delta"]
        content_text = delta_obj.get("content")
        reasoning_text = delta_obj.get("reasoning_content")
        tool_calls = delta_obj.get("tool_calls") or []
        if contains_fc_special_tokens(content_text) or contains_fc_special_tokens(
            reasoning_text
        ):
            return True
        for tool_call in tool_calls:
            function = tool_call.get("function", {})
            if contains_fc_special_tokens(function.get("arguments")):
                return True
        return False
    except Exception:
        # Best-effort guard only; ignore malformed chunks.
        return False
