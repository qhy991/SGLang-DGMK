"""Runtime validation for model-emitted tool calls (OpenAI chat serving).

:class:`ToolCallValidator` is **active** only when ``tool_call_parser == "kimi_k2"``.
When inactive it is a strict bypass: no schema validation, no
``finish_reason`` rewriting, and streaming uses an empty :class:`ToolCallStreamSession`.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, Iterable, List, Optional

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.infini.fc_token_guard import prefer_engine_finish_on_fc_leak
from sglang.srt.infini.tool_call_processing import (
    StreamToolCallCollector,
    validate_parsed_tool_call_items,
    validate_required_tool_call_payload,
)
from sglang.srt.infini.tool_call_validation import ToolCallValidationError

_DEFAULT_UNEXPECTED: Dict[str, Any] = {"type": "unexpected_state", "matched": None}
_LOG = logging.getLogger(__name__)

# Single switch: generated tool-call validation + finish_reason tweaks only for Kimi K2.
_ACTIVATE_TOOL_PARSER: str = "kimi_k2"

__all__ = (
    "ToolCallStreamSession",
    "ToolCallValidationError",
    "ToolCallValidator",
)


class ToolCallStreamSession:
    """Per-stream tool-call argument buffering and JSON-schema validation.

    Call :meth:`ingest_after_tool_chunk` after each streamed tool delta,
    :meth:`ingest_remaining_args` for tail fragments from the detector, and
    :meth:`finalize_choice` when the generation ends for that choice index.
    """

    __slots__ = ("_collector", "_defer_merge_on_stop")

    def __init__(self, collector: Optional[StreamToolCallCollector]) -> None:
        self._collector = collector
        self._defer_merge_on_stop: Dict[int, bool] = {}

    @property
    def active(self) -> bool:
        return self._collector is not None

    def ingest_after_tool_chunk(
        self,
        choice_index: int,
        call_item: ToolCallItem,
        meta_finish_reason: Optional[Dict[str, Any]],
        finish_reasons: Dict[int, Any],
        has_tool_calls: Dict[int, bool],
    ) -> None:
        if not self._collector:
            return
        try:
            self._collector.ingest_call_item(choice_index, call_item)
        except ToolCallValidationError as exc:
            _LOG.error(
                "Tool-call schema validation failed in stream chunk "
                "(choice_index=%s, tool_index=%s, tool_name=%s): %s",
                choice_index,
                getattr(call_item, "tool_index", None),
                getattr(call_item, "name", None),
                exc,
            )
            has_tool_calls[choice_index] = False
            fr = meta_finish_reason
            if fr and fr.get("type") == "stop":
                finish_reasons[choice_index] = prefer_engine_finish_on_fc_leak(
                    copy.deepcopy(fr),
                    default_unexpected=_DEFAULT_UNEXPECTED,
                )
            else:
                self._defer_merge_on_stop[choice_index] = True

    def merge_engine_finish_for_storage(
        self, choice_index: int, engine_finish: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply deferred validation failure when the engine ends with ``stop``."""
        fr_copy = copy.deepcopy(engine_finish)
        if self._defer_merge_on_stop.pop(choice_index, False) and fr_copy.get(
            "type"
        ) == "stop":
            return prefer_engine_finish_on_fc_leak(
                fr_copy, default_unexpected=_DEFAULT_UNEXPECTED
            )
        return fr_copy

    def ingest_remaining_args(
        self,
        choice_index: int,
        tool_index: int,
        arguments_fragment: str,
    ) -> None:
        if not self._collector:
            return
        try:
            self._collector.ingest_remaining_args(
                choice_index=choice_index,
                tool_index=tool_index,
                arguments_fragment=arguments_fragment,
            )
        except ToolCallValidationError as exc:
            _LOG.error(
                "Tool-call schema validation failed when ingesting remaining "
                "streamed arguments (choice_index=%s, tool_index=%s): %s",
                choice_index,
                tool_index,
                exc,
            )

    def finalize_choice(
        self,
        choice_index: int,
        engine_finish: Optional[Dict[str, Any]],
        finish_reasons: Dict[int, Any],
        has_tool_calls: Dict[int, bool],
    ) -> None:
        if not self._collector:
            return
        try:
            self._collector.finalize_choice(choice_index)
        except ToolCallValidationError as exc:
            _LOG.error(
                "Tool-call schema validation failed during stream finalization "
                "(choice_index=%s): %s",
                choice_index,
                exc,
            )
            base = finish_reasons.get(choice_index) or engine_finish
            finish_reasons[choice_index] = prefer_engine_finish_on_fc_leak(
                copy.deepcopy(base) if base else None,
                default_unexpected=_DEFAULT_UNEXPECTED,
            )
            has_tool_calls[choice_index] = False


class ToolCallValidator:
    """Generated tool-call policy: **active** only for ``kimi_k2``; otherwise full bypass."""

    __slots__ = ("_tool_call_parser",)

    def __init__(self, tool_call_parser: Optional[str]) -> None:
        self._tool_call_parser = tool_call_parser

    @property
    def activate(self) -> bool:
        """When ``True``, run JSON-schema validation and Kimi-specific finish_reason rules."""
        return self._tool_call_parser == _ACTIVATE_TOOL_PARSER

    @property
    def enabled(self) -> bool:
        """Alias of :attr:`activate` (same ``kimi_k2`` gate)."""
        return self.activate

    def open_stream_session(
        self,
        *,
        tool_choice_active: bool,
        tools: Iterable[Tool],
    ) -> ToolCallStreamSession:
        """Create per-response streaming state (collector + finish-reason helpers)."""
        collector: Optional[StreamToolCallCollector] = None
        if self.activate and tool_choice_active:
            collector = StreamToolCallCollector(tools)
        return ToolCallStreamSession(collector)

    def validate_required_payload(
        self,
        tool_call_data: List[Dict[str, Any]],
        tools: Iterable[Tool],
    ) -> None:
        """Validate a decoded required/named JSON tool array; no-op when not :attr:`activate`."""
        if not self.activate:
            return
        validate_required_tool_call_payload(tool_call_data, tools)

    def validate_parsed_items(
        self,
        call_info_list: List[ToolCallItem],
        tools: Iterable[Tool],
    ) -> None:
        """Validate parser-produced :class:`ToolCallItem` rows; no-op when not :attr:`activate`."""
        if not self.activate:
            return
        validate_parsed_tool_call_items(call_info_list, tools)

    def finish_reason_after_non_stream_validation(
        self,
        *,
        validation_ok: bool,
        has_parsed_tools: bool,
        engine_finish: Dict[str, Any],
    ) -> Dict[str, Any]:
        """When not :attr:`activate`, return a plain deep copy (no ``finish_reason`` changes).

        When active: schema failure maps engine ``stop`` → ``unexpected_state``; success with
        parsed tools maps ``stop`` / ``length`` → ``tool_calls``.
        """
        if not self.activate:
            return copy.deepcopy(engine_finish)
        if not validation_ok:
            return prefer_engine_finish_on_fc_leak(
                copy.deepcopy(engine_finish),
                default_unexpected=_DEFAULT_UNEXPECTED,
            )
        if has_parsed_tools and engine_finish.get("type") in ("stop", "length"):
            out = copy.deepcopy(engine_finish)
            out["type"] = "tool_calls"
            out["matched"] = None
            return out
        return copy.deepcopy(engine_finish)
