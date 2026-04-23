"""Streaming tool-call deltas → OpenAI SSE strings (Kimi FC strip + debug)."""

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, Dict, Optional, Union

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    FunctionResponse,
    ToolCall,
    ToolChoice,
)
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.json_array_parser import JsonArrayParser
from sglang.srt.infini.kimi_fc_openai_serving import (
    format_openai_tool_call_id,
    history_tool_calls_count,
)
from sglang.srt.infini.openai_chat_stream_helpers import (
    STREAM_TOOL_DEBUG,
    parser_incremental_buffer_len,
    short_repr,
    sse_tool_stream_line,
)
from sglang.srt.infini.tool_call_processing import StreamToolCallCollector

logger = logging.getLogger(__name__)


def ensure_stream_tool_parser(
    index: int,
    parser_dict: Dict[int, Union[FunctionCallParser, JsonArrayParser]],
    request: ChatCompletionRequest,
    tool_call_parser: Optional[str],
) -> Union[FunctionCallParser, JsonArrayParser]:
    if index not in parser_dict:
        if request.tool_choice == "required" or isinstance(
            request.tool_choice, ToolChoice
        ):
            parser_dict[index] = JsonArrayParser()
        else:
            parser_dict[index] = FunctionCallParser(
                tools=request.tools or [],
                tool_call_parser=tool_call_parser,
            )
    return parser_dict[index]


async def iter_tool_call_stream_sse_chunks(
    *,
    index: int,
    delta: str,
    parser_dict: Dict[int, Union[FunctionCallParser, JsonArrayParser]],
    content: Dict[str, Any],
    request: ChatCompletionRequest,
    has_tool_calls: Dict[int, bool],
    stream_tool_call_collector: Optional[StreamToolCallCollector],
    continuous_usage_stats: bool,
    is_kimi: bool,
    tool_call_parser: Optional[str],
) -> AsyncGenerator[str, None]:
    """Parse one engine text delta into zero or more ``data: {...}`` SSE lines."""
    parser = ensure_stream_tool_parser(
        index, parser_dict, request, tool_call_parser
    )

    if isinstance(parser, JsonArrayParser):
        result = parser.parse_streaming_increment(delta, request.tools or [])
        normal_text, calls = result.normal_text, result.calls
    else:
        normal_text, calls = parser.parse_stream_chunk(delta)

    if logger.isEnabledFor(logging.DEBUG):
        buf_len = parser_incremental_buffer_len(parser)
        logger.debug(
            "%s parse_tool_stream rid=%s index=%s parser=%s "
            "delta_in_len=%d normal_text_len=%d num_calls=%d buf_len=%d is_kimi=%s",
            STREAM_TOOL_DEBUG,
            content["meta_info"].get("id"),
            index,
            type(parser).__name__,
            len(delta) if delta else 0,
            len(normal_text) if normal_text else 0,
            len(calls) if calls else 0,
            buf_len,
            is_kimi,
        )
        if not (normal_text or calls):
            logger.debug(
                "%s parse_silent_after_increment rid=%s index=%s "
                "delta_in_preview=%s (parser produced no normal_text and no calls)",
                STREAM_TOOL_DEBUG,
                content["meta_info"].get("id"),
                index,
                short_repr(delta),
            )
    if normal_text:
        yield sse_tool_stream_line(
            index,
            content["meta_info"]["id"],
            request.model,
            content["meta_info"],
            continuous_usage_stats,
            content=normal_text,
        )

    history_tool_calls_cnt = history_tool_calls_count(request)
    mid = content["meta_info"]["id"]
    for call_item in calls:
        has_tool_calls[index] = True
        if call_item.name:
            tool_call_id = format_openai_tool_call_id(
                tool_call_parser, call_item, history_tool_calls_cnt
            )
            function_name = call_item.name
        else:
            tool_call_id = None
            function_name = None

        tool_call = ToolCall(
            id=tool_call_id,
            index=call_item.tool_index,
            function=FunctionResponse(
                name=function_name,
                arguments=call_item.parameters,
            ),
        )
        yield sse_tool_stream_line(
            index,
            mid,
            request.model,
            content["meta_info"],
            continuous_usage_stats,
            tool_calls=[tool_call],
        )
        if stream_tool_call_collector:
            stream_tool_call_collector.ingest_call_item(index, call_item)
