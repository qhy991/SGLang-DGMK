"""Kimi K2 (``kimi_k2`` tool parser) helpers for the OpenAI chat layer.

FC literal sanitization for streamed and completion payloads lives on
:class:`~sglang.srt.entrypoints.openai.protocol.DeltaMessage` and
:class:`~sglang.srt.entrypoints.openai.protocol.ChatMessage` in ``protocol.py``.
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.function_call.core_types import ToolCallItem

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
