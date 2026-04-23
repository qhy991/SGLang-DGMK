"""Kimi K2 (``kimi_k2`` tool parser) helpers for the OpenAI chat layer.

FC special substrings in assistant output are removed via
``strip_kimi_fc_special_substrings`` in :mod:`sglang.srt.infini.fc_token_guard`.
"""

from __future__ import annotations

from typing import Optional

KIMI_K2_OPENAI_TOOL_PARSER = "kimi_k2"


def is_kimi_k2_openai_serving(tool_call_parser: Optional[str]) -> bool:
    return tool_call_parser == KIMI_K2_OPENAI_TOOL_PARSER
