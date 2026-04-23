"""Debug helpers for tool-call constraint / validation (grep: sglang-tool-call-debug)."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

_LOG = logging.getLogger(__name__)

TAG = "[sglang-tool-call-debug]"

# Mirrors Loom default widen policies on schema objects (compile_params root abort).
_WIDEN_TRIGGER_KEYS = frozenset(
    {
        "anyOf",
        "allOf",
        "oneOf",
        "not",
        "$ref",
        "$defs",
        "definitions",
        "if",
        "then",
        "else",
        "dependentRequired",
        "dependentSchemas",
    }
)


def parameters_schema_summary(schema: Any) -> Dict[str, Any]:
    """Compact summary of a tool ``parameters`` schema for logs."""
    if schema is None:
        return {"present": False}
    if isinstance(schema, str):
        try:
            schema = json.loads(schema)
        except json.JSONDecodeError:
            return {"present": True, "form": "str", "parse_error": True}
    if not isinstance(schema, dict):
        return {"present": True, "form": type(schema).__name__}

    props = schema.get("properties")
    prop_keys: List[str] = []
    if isinstance(props, dict):
        prop_keys = sorted(props.keys())

    required = schema.get("required")
    required_list: List[str] = []
    if isinstance(required, list):
        required_list = [x for x in required if isinstance(x, str)]

    missing_in_props = [k for k in required_list if k not in prop_keys]
    widen_at_root = sorted(k for k in schema.keys() if k in _WIDEN_TRIGGER_KEYS)

    return {
        "present": True,
        "type": schema.get("type"),
        "property_keys": prop_keys,
        "required": required_list,
        "required_missing_from_properties": missing_in_props,
        "widen_trigger_keys_at_root": widen_at_root,
        "additionalProperties": schema.get("additionalProperties"),
    }


def log_debug(msg: str, *args: Any, **kwargs: Any) -> None:
    _LOG.debug("%s " + msg, TAG, *args, **kwargs)


def log_loom_required_enforcement_for_tool(
    tool_name: str, parameters: Any, *, context: str
) -> None:
    """Log native Loom compile prediction for one tool schema (needs ``loom`` package)."""
    try:
        from loom.required_debug import enabled, estimate_parameters_enforcement, log_required
    except ImportError:
        log_debug(
            "loom required_debug unavailable (import loom) tool=%s context=%s",
            tool_name,
            context,
        )
        return
    if not enabled():
        return
    est = estimate_parameters_enforcement(parameters)
    log_required(
        "sglang %s tool=%s mode=%s required=%s missing_in_properties=%s "
        "json_any_slots=%s enforced=%s",
        context,
        tool_name,
        est.get("enforcement_mode"),
        est.get("required"),
        est.get("required_missing_from_properties"),
        est.get("estimated_json_any_slots"),
        est.get("required_enforced_at_close_brace"),
    )
