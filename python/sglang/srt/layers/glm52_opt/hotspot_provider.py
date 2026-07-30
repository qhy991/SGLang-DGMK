"""Out-of-tree provider for exact GLM-5.2 hotspot operator experiments.

The provider is loaded once after a worker has been assigned its CUDA device.
Hot-path calls do not import modules, inspect signatures, read files, allocate
adapters, or catch candidate launch failures.  A selected candidate either
launches exactly once or raises; it is never followed by a hidden stock launch.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import logging
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from sglang.srt.layers.glm52_opt import config

logger = logging.getLogger(__name__)

INFINI_KERNEL_API_VERSION = 1

_CALLBACK_BY_OP = {
    "dsa_decode_attn": "flashmla_sparse_decode",
    "dsa_prefill_attn": "flashmla_sparse_prefill",
    "moe_gate_proj": "moe_w13",
    "moe_down_proj": "moe_w2",
}


@dataclass(frozen=True)
class _ProviderState:
    ready: bool
    reason: str
    module_ref: str = ""
    module_name: str = ""
    gpu_id: int | None = None
    selected_ops: frozenset[str] = frozenset()
    callbacks: dict[str, Callable[..., Any]] = field(default_factory=dict, repr=False)
    provider_info: dict[str, Any] = field(default_factory=dict)


_STATE = _ProviderState(False, "not_initialized")
_LOCK = threading.Lock()


def _load_module(module_ref: str) -> ModuleType:
    path = Path(module_ref).expanduser()
    if path.suffix == ".py" or path.is_absolute():
        if not path.is_absolute():
            raise ValueError(
                "SGLANG_GLM52_HOTSPOT_MODULE file references must be absolute"
            )
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.sha256(str(path).encode()).hexdigest()[:16]
        module_name = f"sglang_glm52_hotspot_provider_{digest}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load GLM-5.2 hotspot provider from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
        return module
    return importlib.import_module(module_ref)


def initialize_hotspot_provider(gpu_id: int | None = None) -> bool:
    """Load and validate the explicitly requested provider.

    Returns ``False`` when the hotspot profile is not active.  When it is
    active, a missing or malformed provider is fatal so an A/B run cannot
    silently label stock kernels as candidates.
    """

    global _STATE
    if not config.is_enabled() or config.profile_name() != "hotspot_candidates":
        _STATE = _ProviderState(False, "profile_inactive")
        return False

    selected_ops = config.hotspot_candidate_ops()
    if not selected_ops:
        _STATE = _ProviderState(False, "no_selected_ops")
        return False

    module_ref = config.hotspot_module_ref()
    if not module_ref:
        _STATE = _ProviderState(
            False,
            "missing_module",
            gpu_id=gpu_id,
            selected_ops=selected_ops,
        )
        raise RuntimeError("hotspot_candidates requires SGLANG_GLM52_HOTSPOT_MODULE")

    with _LOCK:
        if _STATE.ready:
            if (
                _STATE.module_ref != module_ref
                or _STATE.gpu_id != gpu_id
                or _STATE.selected_ops != selected_ops
            ):
                raise RuntimeError(
                    "GLM-5.2 hotspot provider was initialized with different settings"
                )
            return True

        try:
            module = _load_module(module_ref)
            api_version = getattr(module, "INFINI_KERNEL_API_VERSION", None)
            if api_version != INFINI_KERNEL_API_VERSION:
                raise RuntimeError(
                    "hotspot provider API mismatch: "
                    f"got {api_version!r}, expected {INFINI_KERNEL_API_VERSION}"
                )

            callbacks: dict[str, Callable[..., Any]] = {}
            for op_name in selected_ops:
                callback_name = _CALLBACK_BY_OP[op_name]
                callback = getattr(module, callback_name, None)
                if not callable(callback):
                    raise RuntimeError(
                        f"hotspot provider has no callable {callback_name!r} "
                        f"for {op_name!r}"
                    )
                callbacks[op_name] = callback

            initializer = getattr(module, "initialize", None)
            if initializer is not None:
                if not callable(initializer):
                    raise RuntimeError("hotspot provider initialize is not callable")
                initializer(gpu_id=gpu_id)

            raw_info = getattr(module, "PROVIDER_INFO", {})
            provider_info = dict(raw_info) if isinstance(raw_info, dict) else {}
            _STATE = _ProviderState(
                True,
                "ready",
                module_ref=module_ref,
                module_name=module.__name__,
                gpu_id=gpu_id,
                selected_ops=selected_ops,
                callbacks=callbacks,
                provider_info=provider_info,
            )
            logger.warning(
                "GLM-5.2 hotspot provider ready: module=%s gpu=%s ops=%s",
                module.__name__,
                gpu_id,
                sorted(selected_ops),
            )
            return True
        except Exception as exc:
            _STATE = _ProviderState(
                False,
                f"initialization_failed:{type(exc).__name__}",
                module_ref=module_ref,
                gpu_id=gpu_id,
                selected_ops=selected_ops,
            )
            raise RuntimeError(
                "GLM-5.2 hotspot provider initialization failed"
            ) from exc


def _callback(op_name: str) -> Callable[..., Any]:
    state = _STATE
    if not state.ready:
        raise RuntimeError(
            "GLM-5.2 hotspot candidate selected before provider initialization: "
            f"{state.reason}"
        )
    try:
        return state.callbacks[op_name]
    except KeyError as exc:
        raise RuntimeError(
            f"GLM-5.2 hotspot provider did not register {op_name!r}"
        ) from exc


def run_flashmla_sparse_decode(**kwargs):
    """Call the provider's production-signature FlashMLA decode implementation."""
    return _callback("dsa_decode_attn")(**kwargs)


def run_flashmla_sparse_prefill(**kwargs):
    """Call the provider's production-signature FlashMLA prefill implementation."""
    return _callback("dsa_prefill_attn")(**kwargs)


def run_moe_masked(
    op_name: str,
    *,
    lhs,
    rhs,
    out,
    masked_m,
    expected_m: int,
):
    """Call one exact masked grouped-GEMM provider implementation."""
    return _callback(op_name)(
        lhs=lhs,
        rhs=rhs,
        out=out,
        masked_m=masked_m,
        expected_m=expected_m,
    )


def provider_state() -> dict[str, Any]:
    """Return read-only startup/debug metadata; not used on the hot path."""
    state = _STATE
    return {
        "ready": state.ready,
        "reason": state.reason,
        "module_ref": state.module_ref,
        "module_name": state.module_name,
        "gpu_id": state.gpu_id,
        "selected_ops": sorted(state.selected_ops),
        "provider_info": state.provider_info,
    }


def _reset_hotspot_provider_for_tests() -> None:
    global _STATE
    _STATE = _ProviderState(False, "not_initialized")
