"""Private, reset-on-exit production context for the GLM-5.2 W13 decode path.

The generic GLM52 phase ContextVars are populated by ``ModelRunner._forward_raw``.
Decode CUDA-graph capture calls the model directly, so the W13 selector must not
depend on those variables.  The real eager and graph forward call sites enter
this scope and always reset its token on exit.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@dataclass(frozen=True)
class W13DecodeForwardMarker:
    token_bucket: int
    graph_capture: bool


_w13_decode_forward: ContextVar[W13DecodeForwardMarker | None] = ContextVar(
    "glm52_w13_decode_forward", default=None
)


def get_w13_decode_forward_marker() -> W13DecodeForwardMarker | None:
    return _w13_decode_forward.get()


def _marker_for_forward(
    forward_batch: ForwardBatch,
    token_bucket: int,
    *,
    graph_capture: bool,
) -> W13DecodeForwardMarker | None:
    mode = getattr(forward_batch, "forward_mode", None)
    is_exact_decode = bool(
        mode is not None
        and getattr(mode, "is_decode", lambda: False)()
        and int(token_bucket) in (16, 32)
    )
    if not is_exact_decode:
        return None
    return W13DecodeForwardMarker(
        token_bucket=int(token_bucket),
        graph_capture=bool(graph_capture),
    )


@contextmanager
def w13_decode_forward_scope(
    forward_batch: ForwardBatch,
    token_bucket: int,
    *,
    graph_capture: bool,
) -> Iterator[None]:
    """Publish an exact-DECODE marker for one real model forward only."""

    token = _w13_decode_forward.set(
        _marker_for_forward(
            forward_batch,
            token_bucket,
            graph_capture=graph_capture,
        )
    )
    try:
        yield
    finally:
        _w13_decode_forward.reset(token)
