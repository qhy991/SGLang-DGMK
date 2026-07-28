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
_forward_m: ContextVar[Optional[int]] = ContextVar("glm52_forward_m", default=None)
_attn_o_direct_nk_context: ContextVar[Optional[tuple["ForwardMode", int]]] = ContextVar(
    "glm52_attn_o_direct_nk_context",
    default=None,
)


def get_op_name() -> Optional[str]:
    return _op_name.get()


def get_forward_mode() -> Optional["ForwardMode"]:
    return _forward_mode.get()


def get_forward_m() -> Optional[int]:
    """Return the local token bucket of the current model forward."""
    return _forward_m.get()


def set_forward_mode(mode: Optional["ForwardMode"], m: Optional[int] = None) -> None:
    _forward_mode.set(mode)
    _forward_m.set(None if m is None else int(m))


def get_attn_o_direct_nk_context() -> Optional[tuple["ForwardMode", int]]:
    """Return task-local mode/M without exposing it to legacy GLM52 dispatch."""
    return _attn_o_direct_nk_context.get()


def is_attn_o_direct_nk_scope() -> bool:
    return get_attn_o_direct_nk_context() is not None


@contextmanager
def attn_o_direct_nk_context(
    mode: Optional["ForwardMode"], m: Optional[int] = None
) -> Iterator[None]:
    """Publish only supported task-local metadata; otherwise be a strict no-op."""
    local_m = None if m is None else int(m)
    if (
        mode is None
        or local_m not in (16, 32)
        or not mode.is_decode()
    ):
        yield
        return

    token = _attn_o_direct_nk_context.set((mode, local_m))
    try:
        yield
    finally:
        _attn_o_direct_nk_context.reset(token)


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
        # DSA indexer Q up-projection.  This is a replicated [2048, 4096]
        # linear and must not share the attention q_b_proj tag (which is TP
        # sharded and has a different production shape).
        "wq_b": "index_q_upproj",
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
        # Default CUDA fusion is one BF16 [hidden, head_dim + n_heads] linear.
        # Keep it distinct from the legacy standalone head-gate projection.
        "wk_weights_proj": "index_wk_weights_proj",
    }
    # LinearBase.prefix is the exact module prefix, so its final component is
    # sufficient.  Substring matching can silently route unrelated or fused
    # modules to a shape-incompatible kernel.
    return mapping.get(leaf)
