"""Kimi structural-tag helpers kept separate from builtin templates."""

import logging
from typing import Any, Dict, List, Union



from xgrammar.structural_tag import (
    AnyTextFormat,
    AnyTokensFormat,
    ConstStringFormat,
    JSONSchemaFormat,
    RegexFormat,
    SequenceFormat,
    StructuralTag,
    TagFormat,
    TokenFormat,
    TokenTriggeredTagsFormat,
    TriggeredTagsFormat,
)

# Kimi tool-call wire literals.
_KIMI_TOOL_CALLS_SECTION_BEGIN = "<|tool_calls_section_begin|>"
_KIMI_TOOL_CALLS_SECTION_END = "<|tool_calls_section_end|>"
_KIMI_TOOL_CALL_BEGIN = "<|tool_call_begin|>"
_KIMI_TOOL_CALL_ARGUMENTS_BEGIN = "<|tool_call_argument_begin|>"
_KIMI_TOOL_CALL_END = "<|tool_call_end|>"
_KIMI_FC_MARKERS_ALL = (
    _KIMI_TOOL_CALLS_SECTION_BEGIN,
    _KIMI_TOOL_CALLS_SECTION_END,
    _KIMI_TOOL_CALL_BEGIN,
    _KIMI_TOOL_CALL_ARGUMENTS_BEGIN,
    _KIMI_TOOL_CALL_END,
)


fc_token_ids = {
    "tool_calls_section_begin": 163595,
    "tool_calls_section_end": 163596,
    "tool_call_begin": 163597,
    "tool_call_argument_begin": 163598,
    "tool_call_end": 163599,
    "think": 163606,
    
}

token_id_tool_calls_section_begin = 163595
token_id_tool_calls_section_end = 163596
token_id_tool_call_begin = 163597
token_id_tool_call_argument_begin = 163598
token_id_tool_call_end = 163599
token_id_think = 163606

logger = logging.getLogger(__name__)


def _unique_strings_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _get_function_parameters(function: Dict[str, Any]) -> Union[Dict[str, Any], bool]:
    if ("strict" in function and function["strict"] is False) or ("parameters" not in function):
        return True
    return function["parameters"]


def _get_fc_marker_input(input_dict: Dict[str, Any], key: str, default_str: str) -> Union[int, str]:
    """Resolve marker token/id from input_dict['fc_token_ids'], fallback to wire string."""
    token_map = fc_token_ids
    if isinstance(token_map, dict) and key in token_map:
        return token_map[key]
    return default_str


def get_kimi_structural_tag_with_tool_marker_excludes(input_dict: Dict[str, Any]) -> StructuralTag:
    """Build a Kimi-like structural tag with tool marker excludes.

    Wire shape:
    - ``<|tool_calls_section_begin|>``
    - ``<|tool_call_begin|>functions.{name}:``
    - ``<digits><|tool_call_argument_begin|><json>``
    - ``<|tool_call_end|>``
    - ``<|tool_calls_section_end|>``

    Exclusion policy:
    - With tools in free-text: allow section begin marker, block other FC markers + think tags.
    - Without tools: block all FC markers + think tags.

    Optional switches:
    - ``use_exclude_tokens`` (bool, default ``False``): when true, emit token-level
      ``exclude_tokens`` / ``token_triggered_tags`` forms instead of string ``excludes``.
      You can optionally pass ``fc_token_ids`` in input_dict:
      ``{"tool_calls_section_begin": <id_or_token_str>, ...}``
      Keys: ``tool_calls_section_begin``, ``tool_calls_section_end``, ``tool_call_begin``,
      ``tool_call_argument_begin``, ``tool_call_end``.
    """

    tools = input_dict.get("tools", [])
    use_exclude_tokens = input_dict.get("use_exclude_tokens", True)
    logger.debug(
        "kimi_structural_tag input: tools=%d use_exclude_tokens=%s fc_token_ids_keys=%s",
        len(tools),
        use_exclude_tokens,
        sorted(input_dict.get("fc_token_ids", {}).keys())
        if isinstance(input_dict.get("fc_token_ids"), dict)
        else None,
    )

    tool_calls_section_begin = _get_fc_marker_input(
        input_dict, "tool_calls_section_begin", _KIMI_TOOL_CALLS_SECTION_BEGIN
    )
    tool_calls_section_end = _get_fc_marker_input(
        input_dict, "tool_calls_section_end", _KIMI_TOOL_CALLS_SECTION_END
    )
    tool_call_begin = _get_fc_marker_input(input_dict, "tool_call_begin", _KIMI_TOOL_CALL_BEGIN)
    tool_call_argument_begin = _get_fc_marker_input(
        input_dict, "tool_call_argument_begin", _KIMI_TOOL_CALL_ARGUMENTS_BEGIN
    )
    tool_call_end = _get_fc_marker_input(input_dict, "tool_call_end", _KIMI_TOOL_CALL_END)

    tags = []
    for tool in tools:
        if "function" not in tool:
            continue

        function = tool["function"]
        parameters = _get_function_parameters(function)
        name = function["name"]
        content_elements = []
        if use_exclude_tokens:
            content_elements.append(ConstStringFormat(value=f"functions.{name}:"))
        content_elements.extend(
            [
                RegexFormat(pattern=r"\d+"),
                (
                    TokenFormat(token=tool_call_argument_begin)
                    if use_exclude_tokens
                    else ConstStringFormat(value=_KIMI_TOOL_CALL_ARGUMENTS_BEGIN)
                ),
                JSONSchemaFormat(json_schema=parameters),
            ]
        )
        tags.append(
            TagFormat(
                begin=(
                    TokenFormat(token=tool_call_begin)
                    if use_exclude_tokens
                    else f"{_KIMI_TOOL_CALL_BEGIN}functions.{name}:"
                ),
                content=SequenceFormat(elements=content_elements),
                end=TokenFormat(token=tool_call_end) if use_exclude_tokens else _KIMI_TOOL_CALL_END,
            )
        )

    if len(tags) > 0:
        if use_exclude_tokens:
            section_dispatch_exclude_tokens: List[Union[int, str]] = [
                tool_calls_section_begin,
                tool_call_argument_begin,
                tool_call_end,
                "<think>",
                "</think>",
            ]
            section_content = TokenTriggeredTagsFormat(
                trigger_tokens=[tool_call_begin],
                tags=tags,
                exclude_tokens=section_dispatch_exclude_tokens,
            )
        else:
            section_dispatch_excludes = _unique_strings_keep_order(
                [
                    _KIMI_TOOL_CALLS_SECTION_BEGIN,
                    _KIMI_TOOL_CALL_ARGUMENTS_BEGIN,
                    _KIMI_TOOL_CALL_END,
                    "<think>",
                    "</think>",
                ]
            )
            section_content = TriggeredTagsFormat(
                triggers=[_KIMI_TOOL_CALL_BEGIN],
                tags=tags,
                excludes=section_dispatch_excludes,
            )

        tool_section_tag = TagFormat(
            begin=TokenFormat(token=tool_calls_section_begin)
            if use_exclude_tokens
            else _KIMI_TOOL_CALLS_SECTION_BEGIN,
            content=section_content,
            end=TokenFormat(token=tool_calls_section_end)
            if use_exclude_tokens
            else _KIMI_TOOL_CALLS_SECTION_END,
        )
        if use_exclude_tokens:
            # Allow free text and trigger into tool-call section on section_begin.
            # Block all other FC markers and think tags outside the section.
            suffix_dispatch_exclude_tokens: List[Union[int, str]] = [
                tool_calls_section_end,
                tool_call_begin,
                tool_call_argument_begin,
                tool_call_end,
                "<think>",
                "</think>",
            ]
            suffix_tag = TokenTriggeredTagsFormat(
                trigger_tokens=[tool_calls_section_begin],
                tags=[tool_section_tag],
                exclude_tokens=suffix_dispatch_exclude_tokens,
            )
        else:
            # Allow free text and trigger into tool-call section on section_begin.
            # Block all other FC markers and think tags outside the section.
            suffix_dispatch_excludes = _unique_strings_keep_order(
                [
                    _KIMI_TOOL_CALLS_SECTION_END,
                    _KIMI_TOOL_CALL_BEGIN,
                    _KIMI_TOOL_CALL_ARGUMENTS_BEGIN,
                    _KIMI_TOOL_CALL_END,
                    "<think>",
                    "</think>",
                ]
            )
            suffix_tag = TriggeredTagsFormat(
                triggers=[_KIMI_TOOL_CALLS_SECTION_BEGIN],
                tags=[tool_section_tag],
                excludes=suffix_dispatch_excludes,
            )
    else:
        if use_exclude_tokens:
            suffix_tag = AnyTokensFormat(
                exclude_tokens=[
                    tool_calls_section_begin,
                    tool_calls_section_end,
                    tool_call_begin,
                    tool_call_argument_begin,
                    tool_call_end,
                    "<think>",
                    "</think>",
                ]
            )
        else:
            suffix_tag = AnyTextFormat(
                excludes=_unique_strings_keep_order(
                    list(_KIMI_FC_MARKERS_ALL)
                )
                # excludes=_unique_strings_keep_order(
                #     ["<think>", "</think>"] + list(_KIMI_FC_MARKERS_ALL)
                # )
            )
    logger.debug(
        "kimi_structural_tag resolved: tags=%d use_exclude_tokens=%s",
        len(tags),
        use_exclude_tokens,
    )
    return StructuralTag(format=suffix_tag)


def get_kimi_structural_tag_schema_string_with_tool_marker_excludes(input_dict: Dict[str, Any]) -> str:
    """Build the Kimi structural-tag JSON string in modern format.

    The returned JSON is always:
    ``{"type":"structural_tag","format":{...}}``
    and never the deprecated legacy top-level ``structures`` / ``triggers`` shape.
    """
    return get_kimi_structural_tag_with_tool_marker_excludes(input_dict).model_dump_json(
        indent=None
    )


__all__ = [
    "get_kimi_structural_tag_with_tool_marker_excludes",
    "get_kimi_structural_tag_schema_string_with_tool_marker_excludes",
]
