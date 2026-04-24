"""Kimi K2 (``kimi_k2`` tool parser) helpers for the OpenAI chat layer.

FC special substrings in assistant output are removed via
``strip_kimi_fc_special_substrings`` in :mod:`sglang.srt.infini.fc_token_guard`.
"""

from __future__ import annotations

import logging
import uuid
from typing import List, Optional, Tuple

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest, ToolCall
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.infini.fc_token_guard import strip_kimi_fc_special_substrings

logger = logging.getLogger(__name__)

KIMI_K2_OPENAI_TOOL_PARSER = "kimi_k2"


def is_kimi_k2_openai_serving(tool_call_parser: Optional[str]) -> bool:
    return tool_call_parser == KIMI_K2_OPENAI_TOOL_PARSER


def history_tool_calls_count(request: ChatCompletionRequest) -> int:
    """Count prior assistant tool_calls in the request (for Kimi-style tool_call ids)."""
    messages = getattr(request, "messages", [])
    idx = 0
    for msg in messages:
        if msg.role == "assistant":
            tool_calls = getattr(msg, "tool_calls", None)
            idx += len(list(tool_calls)) if tool_calls is not None else 0
    return idx


def format_openai_tool_call_id(
    tool_call_parser: Optional[str],
    call_item: ToolCallItem,
    history_tool_calls_cnt: int,
) -> str:
    """OpenAI ``tool_calls[].id`` for one parsed call (UUID except Kimi-K2)."""
    if tool_call_parser != KIMI_K2_OPENAI_TOOL_PARSER:
        return f"call_{uuid.uuid4().hex[:24]}"
    tool_call_id = (
        f"functions.{call_item.name}:"
        f"{history_tool_calls_cnt + call_item.tool_index}"
    )
    logger.debug(
        "Process tool call idx, parser: %s, tool_call_id: %s, history_cnt: %s",
        tool_call_parser,
        tool_call_id,
        history_tool_calls_cnt,
    )
    return tool_call_id


def maybe_strip_kimi_fc_substrings(
    text: Optional[str], *, is_kimi: bool
) -> Optional[str]:
    if not is_kimi or text is None:
        return text
    return strip_kimi_fc_special_substrings(text)


def maybe_strip_remaining_tool_args(
    remaining_call: str, tool_call_parser: Optional[str]
) -> str:
    if is_kimi_k2_openai_serving(tool_call_parser):
        return strip_kimi_fc_special_substrings(remaining_call) or ""
    return remaining_call


def strip_kimi_openai_choice_fields(
    is_kimi: bool,
    text: str,
    reasoning_text: Optional[str],
    tool_calls: Optional[List[ToolCall]],
) -> Tuple[str, Optional[str], Optional[List[ToolCall]]]:
    """Strip FC literals from non-streaming assistant message fields when serving Kimi."""
    if not is_kimi:
        return text, reasoning_text, tool_calls
    text = strip_kimi_fc_special_substrings(text)
    if reasoning_text is not None:
        reasoning_text = strip_kimi_fc_special_substrings(reasoning_text)
    if tool_calls is not None:
        stripped_tcs = []
        for tc in tool_calls:
            args = tc.function.arguments
            if isinstance(args, str):
                args = strip_kimi_fc_special_substrings(args)
            stripped_tcs.append(
                tc.model_copy(
                    update={
                        "function": tc.function.model_copy(
                            update={"arguments": args}
                        )
                    }
                )
            )
        tool_calls = stripped_tcs
    return text, reasoning_text, tool_calls
