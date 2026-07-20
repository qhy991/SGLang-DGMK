"""Per-forward op tagging and forward-mode context."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Iterator, Optional

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

_op_name: ContextVar[Optional[str]] = ContextVar("glm52_op_name", default=None)
_forward_mode: ContextVar[Optional["ForwardMode"]] = ContextVar(
    "glm52_forward_mode", default=None
)


def get_op_name() -> Optional[str]:
    return _op_name.get()


def get_forward_mode() -> Optional["ForwardMode"]:
    return _forward_mode.get()


def set_forward_mode(mode: Optional["ForwardMode"]) -> None:
    _forward_mode.set(mode)


@contextmanager
def op_context(op_name: Optional[str]) -> Iterator[None]:
    token = _op_name.set(op_name)
    try:
        yield
    finally:
        _op_name.reset(token)


def prefix_to_op_name(prefix: Optional[str]) -> Optional[str]:
    if not prefix:
        return None
    leaf = prefix.rsplit(".", 1)[-1]
    mapping = {
        "q_a_proj": "fused_qkv_a_proj",
        "fused_qkv_a_proj": "fused_qkv_a_proj",
        "fused_qkv_a_proj_with_mqa": "fused_qkv_a_proj",
        "q_b_proj": "q_b_proj",
        "wq_b": "q_b_proj",
        "o_proj": "o_proj",
        "wk": "index_k_proj",
        "wk_proj": "index_k_proj",
        "index_k_proj": "index_k_proj",
        "q_up_proj": "index_q_upproj",
        "index_q_upproj": "index_q_upproj",
        "weights_proj": "index_weights_proj",
        "index_weights_proj": "index_weights_proj",
        "gate_proj": "moe_gate_proj",
        "gate_up_proj": "moe_gate_proj",
        "up_proj": "moe_up_proj",
        "down_proj": "moe_down_proj",
        "w2": "moe_down_proj",
        "w13": "moe_gate_proj",
        "wk_weights_proj": "index_weights_proj",
    }
    if leaf in mapping:
        return mapping[leaf]
    for key, op in mapping.items():
        if leaf.endswith(key) or key in prefix:
            return op
    return None
