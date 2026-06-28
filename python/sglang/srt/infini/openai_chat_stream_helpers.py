"""OpenAI chat-completion SSE line builders and stream-debug helpers.

Used by :mod:`sglang.srt.entrypoints.openai.serving_chat` for tool streaming and
logprob-aware content chunks.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChoiceLogprobs,
    DeltaMessage,
    ToolCall,
)
from sglang.srt.entrypoints.openai.usage_processor import UsageProcessor

# Greppable prefix for diagnosing missing SSE after reasoning in tool streaming.
STREAM_TOOL_DEBUG = "[stream_tool_debug]"


def parser_incremental_buffer_len(parser: object) -> int:
    inner = getattr(parser, "detector", parser)
    buf = getattr(inner, "_buffer", None)
    return len(buf) if isinstance(buf, str) else 0


def short_repr(s: str, max_len: int = 96) -> str:
    if not s:
        return "<empty>"
    if len(s) <= max_len:
        return repr(s)
    return repr(s[:max_len]) + f"... (+{len(s) - max_len} chars)"


def sse_tool_stream_line(
    index: int,
    chatcmpl_id: str,
    model: str,
    meta_info: Dict[str, Any],
    continuous_usage_stats: bool,
    *,
    content: Optional[str] = None,
    tool_calls: Optional[List[ToolCall]] = None,
) -> str:
    if content is not None:
        delta = DeltaMessage(content=content)
    else:
        delta = DeltaMessage(tool_calls=tool_calls or [])
    choice_data = ChatCompletionResponseStreamChoice(
        index=index,
        delta=delta,
        finish_reason=None,
    )
    chunk = ChatCompletionStreamResponse(
        id=chatcmpl_id,
        created=int(time.time()),
        choices=[choice_data],
        model=model,
    )
    if continuous_usage_stats:
        chunk.usage = UsageProcessor.calculate_token_usage(
            prompt_tokens=meta_info.get("prompt_tokens", 0),
            completion_tokens=meta_info.get("completion_tokens", 0),
            reasoning_tokens=meta_info.get("reasoning_tokens", 0),
        )
    return f"data: {chunk.model_dump_json()}\n\n"


def sse_stream_plain_text_line(
    index: int,
    chatcmpl_id: str,
    model: str,
    to_emit: Optional[str],
    choice_logprobs: Optional[ChoiceLogprobs],
    continuous_usage_stats: bool,
    *,
    usage_prompt_tokens: int = 0,
    usage_completion_tokens: int = 0,
    usage_reasoning_tokens: int = 0,
) -> str:
    """Stream one assistant ``content`` delta (normal text and tool-parse fallback)."""
    choice_data = ChatCompletionResponseStreamChoice(
        index=index,
        delta=DeltaMessage(content=to_emit if to_emit else None),
        finish_reason=None,
        matched_stop=None,
        logprobs=choice_logprobs,
    )
    chunk = ChatCompletionStreamResponse(
        id=chatcmpl_id,
        created=int(time.time()),
        choices=[choice_data],
        model=model,
    )
    if continuous_usage_stats:
        chunk.usage = UsageProcessor.calculate_token_usage(
            prompt_tokens=usage_prompt_tokens,
            reasoning_tokens=usage_reasoning_tokens,
            completion_tokens=usage_completion_tokens,
        )
    return f"data: {chunk.model_dump_json()}\n\n"
