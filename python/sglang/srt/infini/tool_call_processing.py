from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional

import orjson
from jsonschema import Draft202012Validator

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.function_call.utils import _is_complete_json
from sglang.srt.infini.tool_call_validation import (
    build_tool_schema_index,
    validate_tool_call,
)


def validate_required_tool_call_payload(
    tool_call_data: List[Dict[str, Any]],
    tools: Iterable[Tool],
    schema_index: Optional[Mapping[str, Draft202012Validator]] = None,
) -> None:
    for tool_call in tool_call_data:
        validate_tool_call(
            name=tool_call["name"],
            parameters_obj=tool_call["parameters"],
            tools=tools,
            schema_index=schema_index,
        )


def validate_parsed_tool_call_items(
    call_info_list: List[ToolCallItem],
    tools: Iterable[Tool],
    schema_index: Optional[Mapping[str, Draft202012Validator]] = None,
) -> None:
    for call_info in call_info_list:
        if not call_info.name:
            raise ValueError(
                "Tool call validation failed for '<unknown>': missing function name"
            )
        try:
            parameters_obj = orjson.loads(call_info.parameters)
        except orjson.JSONDecodeError as exc:
            raise ValueError(
                f"Tool call validation failed for '{call_info.name}': arguments are not valid JSON"
            ) from exc
        validate_tool_call(
            name=call_info.name,
            parameters_obj=parameters_obj,
            tools=tools,
            schema_index=schema_index,
        )


@dataclass
class _CollectedCallState:
    name: Optional[str] = None
    arguments: str = ""
    validated: bool = False


@dataclass
class StreamToolCallCollector:
    """Collect and validate streamed tool call deltas per choice/index."""

    tools: Iterable[Tool]
    schema_index: Mapping[str, Draft202012Validator] = field(init=False)
    _choice_state: Dict[int, Dict[int, _CollectedCallState]] = field(
        default_factory=dict, init=False
    )

    def __post_init__(self):
        self.schema_index = build_tool_schema_index(self.tools)

    def ingest_call_item(self, choice_index: int, call_item: ToolCallItem) -> None:
        if call_item.tool_index is None:
            return
        call_state = self._choice_state.setdefault(choice_index, {}).setdefault(
            call_item.tool_index, _CollectedCallState()
        )
        if call_item.name:
            call_state.name = call_item.name
        if call_item.parameters:
            call_state.arguments += call_item.parameters
        self._validate_if_ready(call_state)

    def ingest_remaining_args(
        self, choice_index: int, tool_index: int, arguments_fragment: str
    ) -> None:
        self.ingest_call_item(
            choice_index,
            ToolCallItem(
                tool_index=tool_index,
                name=None,
                parameters=arguments_fragment,
            ),
        )

    def finalize_choice(self, choice_index: int) -> None:
        for call_state in self._choice_state.get(choice_index, {}).values():
            if call_state.validated:
                continue
            if not call_state.name:
                raise ValueError(
                    "Tool call validation failed for '<unknown>': missing function name"
                )
            if not _is_complete_json(call_state.arguments):
                raise ValueError(
                    f"Tool call validation failed for '{call_state.name}': arguments are not valid JSON"
                )
            self._validate_call(call_state)
            call_state.validated = True

    def _validate_if_ready(self, call_state: _CollectedCallState) -> None:
        if call_state.validated or not call_state.name:
            return
        if not _is_complete_json(call_state.arguments):
            return
        self._validate_call(call_state)
        call_state.validated = True

    def _validate_call(self, call_state: _CollectedCallState) -> None:
        try:
            parameters_obj = orjson.loads(call_state.arguments)
        except orjson.JSONDecodeError as exc:
            raise ValueError(
                f"Tool call validation failed for '{call_state.name}': arguments are not valid JSON"
            ) from exc
        validate_tool_call(
            name=call_state.name,
            parameters_obj=parameters_obj,
            tools=self.tools,
            schema_index=self.schema_index,
        )
