# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Native Loom grammar backend (OpenAI structural_tag → Kimi wire)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

try:
    from loom.grammar_backend import LoomGrammarBackend
    from loom.grammar_backend import init_loom_backend as _init_loom_backend
except ImportError:
    LoomGrammarBackend = None  # type: ignore[misc,assignment]
    _init_loom_backend = None  # type: ignore[misc,assignment]


def create_loom_backend(
    server_args: "ServerArgs",
    tokenizer: Any,
    vocab_size: int,
    eos_token_ids: Optional[set] = None,
) -> Any:
    """Factory used by :func:`~sglang.srt.constrained.base_grammar_backend.create_grammar_backend`."""
    if _init_loom_backend is None:
        raise ImportError(
            "grammar_backend='loom' requires the `loom` package with native bindings "
            "(build from the loom repository, e.g. ./build.sh) on PYTHONPATH."
        )
    return _init_loom_backend(server_args, tokenizer, vocab_size, eos_token_ids)


__all__ = ["LoomGrammarBackend", "create_loom_backend"]
