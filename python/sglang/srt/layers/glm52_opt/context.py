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
_layer_id: ContextVar[Optional[int]] = ContextVar("glm52_layer_id", default=None)


def get_op_name() -> Optional[str]:
    return _op_name.get()


def get_forward_mode() -> Optional["ForwardMode"]:
    return _forward_mode.get()


def get_forward_m() -> Optional[int]:
    """Return the local token bucket of the current model forward."""
    return _forward_m.get()


def get_layer_id() -> Optional[int]:
    """Return the model layer currently entering a GLM-5.2 dispatch hook."""
    return _layer_id.get()


def set_forward_mode(mode: Optional["ForwardMode"], m: Optional[int] = None) -> None:
    _forward_mode.set(mode)
    _forward_m.set(None if m is None else int(m))


@contextmanager
def op_context(op_name: Optional[str]) -> Iterator[None]:
    token = _op_name.set(op_name)
    try:
        yield
    finally:
        _op_name.reset(token)


@contextmanager
def layer_context(layer_id: Optional[int]) -> Iterator[None]:
    """Attach a layer id to hit/miss evidence without changing kernel ABIs."""
    token = _layer_id.set(None if layer_id is None else int(layer_id))
    try:
        yield
    finally:
        _layer_id.reset(token)


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
