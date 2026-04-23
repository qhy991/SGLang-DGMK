from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping

from jsonschema import Draft202012Validator, ValidationError

from sglang.srt.entrypoints.openai.protocol import Tool

_PERMISSIVE_OBJECT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {},
}


def build_tool_schema_index(tools: Iterable[Tool]) -> Dict[str, Draft202012Validator]:
    """Build validators indexed by tool name."""
    validators: Dict[str, Draft202012Validator] = {}
    for tool in tools:
        schema = tool.function.parameters or _PERMISSIVE_OBJECT_SCHEMA
        validators[tool.function.name] = Draft202012Validator(schema)
    return validators


def _format_validation_error(err: ValidationError) -> str:
    if not err.path:
        return err.message
    path = ".".join(str(x) for x in err.path)
    return f"{path}: {err.message}"


def validate_tool_call(
    name: str,
    parameters_obj: Any,
    tools: Iterable[Tool] | None = None,
    schema_index: Mapping[str, Draft202012Validator] | None = None,
) -> None:
    """Validate one generated tool call against the declared schema."""
    validators = schema_index or build_tool_schema_index(tools or [])
    validator = validators.get(name)
    if validator is None:
        raise ValueError(f"Tool call validation failed for '{name}': unknown tool name")
    try:
        validator.validate(parameters_obj)
    except ValidationError as exc:
        raise ValueError(
            f"Tool call validation failed for '{name}': {_format_validation_error(exc)}"
        ) from exc


def validate_tool_call_list(
    tool_calls: Iterable[Dict[str, Any]],
    tools: Iterable[Tool],
) -> None:
    """Validate a list of {'name': ..., 'parameters': ...} tool calls."""
    schema_index = build_tool_schema_index(tools)
    for tool_call in tool_calls:
        validate_tool_call(
            name=tool_call["name"],
            parameters_obj=tool_call["parameters"],
            schema_index=schema_index,
        )
