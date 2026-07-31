"""Central dispatch for GLM-5.2 optimized kernels."""

from __future__ import annotations

import atexit
import json
import logging
import os
import subprocess
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
from sglang.srt.layers.glm52_opt import config
from sglang.srt.layers.glm52_opt.context import (
    get_forward_m,
    get_forward_mode,
    get_layer_id,
    get_op_name,
)
from sglang.srt.layers.glm52_opt.fp8_gemm import run_fp8_gemm
from sglang.srt.layers.glm52_opt.hotspot_provider import (
    provider_state,
    run_flashmla_sparse_decode,
    run_flashmla_sparse_prefill,
)
from sglang.srt.layers.glm52_opt.hotspot_provider import (
    run_moe_masked as run_hotspot_moe_masked,
)
from sglang.srt.layers.glm52_opt.moe_masked import run_moe_masked
from sglang.srt.layers.glm52_opt.phase import infer_glm52_phase
from sglang.srt.layers.glm52_opt.registry import KernelSpec, lookup

logger = logging.getLogger(__name__)

_HIT_LOCK = threading.Lock()
_HIT_COUNTS: dict[str, int] = {}
_MISS_COUNTS: dict[str, int] = {}
_SELECTED_SCOPE_COUNTS: dict[tuple[str, str, str, str, str], int] = {}
_MISS_SCOPE_COUNTS: dict[tuple[str, str, str, str, str], int] = {}
_HIT_FILE_RAW = os.environ.get(
    "SGLANG_GLM52_OPT_HIT_FILE",
    "/home/ubuntu/wwxq/cache/sglang/glm52_opt_hits.json",
).strip()
_STATIC_ARTIFACT_METADATA: dict[str, Any] | None = None


def _rank_from_env(*names: str) -> int | str | None:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            try:
                return int(value)
            except ValueError:
                return value
    return None


def _gpu_uuid_from_env(local_rank: int | str | None) -> str | None:
    """Read a UUID from launcher/container env without touching CUDA."""
    direct_names = (
        "SGLANG_GPU_UUID",
        "GPU_UUID",
        "GPU_DEVICE_UUID",
        "NVIDIA_GPU_UUID",
    )
    for name in direct_names:
        value = os.environ.get(name, "").strip()
        if value:
            return value

    for name in ("NVIDIA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        values = [
            item.strip() for item in os.environ.get(name, "").split(",") if item.strip()
        ]
        uuid_values = [item for item in values if item.startswith(("GPU-", "MIG-"))]
        if not uuid_values:
            continue
        try:
            local_index = int(local_rank)
        except (TypeError, ValueError):
            local_index = 0
        if 0 <= local_index < len(uuid_values):
            return uuid_values[local_index]
        return uuid_values[0]
    return None


def _process_identity() -> dict[str, Any]:
    global_rank = _rank_from_env(
        "RANK",
        "WORLD_RANK",
        "SLURM_PROCID",
        "OMPI_COMM_WORLD_RANK",
    )
    local_rank = _rank_from_env(
        "LOCAL_RANK",
        "SLURM_LOCALID",
        "OMPI_COMM_WORLD_LOCAL_RANK",
    )
    return {
        "pid": os.getpid(),
        "global_rank": global_rank,
        "local_rank": local_rank,
        "gpu_uuid": _gpu_uuid_from_env(local_rank),
    }


def _path_token(value: object) -> str:
    text = str(value)
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in text)


def _resolve_hit_file(raw_path: str, identity: dict[str, Any]) -> Path | None:
    """Expand a rank-aware path and guarantee a process-unique target."""
    if not raw_path:
        return None
    global_rank = identity["global_rank"]
    local_rank = identity["local_rank"]
    values = {
        "pid": _path_token(identity["pid"]),
        "rank": _path_token(global_rank if global_rank is not None else "unknown"),
        "global_rank": _path_token(
            global_rank if global_rank is not None else "unknown"
        ),
        "local_rank": _path_token(local_rank if local_rank is not None else "unknown"),
        "gpu_uuid": _path_token(identity["gpu_uuid"] or "unknown"),
    }
    expanded = raw_path
    used: set[str] = set()
    for name, value in values.items():
        marker = "{" + name + "}"
        if marker in expanded:
            expanded = expanded.replace(marker, value)
            used.add(name)

    # A template may choose its own layout.  Missing identity dimensions are
    # appended so a legacy fixed filename can never be shared by TP workers.
    tags: list[str] = []
    if not ({"rank", "global_rank"} & used):
        tags.append(f"rank{values['global_rank']}")
    if "local_rank" not in used:
        tags.append(f"local{values['local_rank']}")
    if "pid" not in used:
        tags.append(f"pid{values['pid']}")
    path = Path(expanded).expanduser()
    if tags:
        suffix = path.suffix
        stem = path.name[: -len(suffix)] if suffix else path.name
        path = path.with_name(f"{stem}.{'.'.join(tags)}{suffix}")
    return path


_PROCESS_IDENTITY = _process_identity()
_HIT_FILE = _resolve_hit_file(_HIT_FILE_RAW, _PROCESS_IDENTITY)


def _initialize_process_artifact() -> None:
    """Refresh launcher identity lazily after a multiprocessing fork."""
    global _PROCESS_IDENTITY, _HIT_FILE, _STATIC_ARTIFACT_METADATA
    _PROCESS_IDENTITY = _process_identity()
    _HIT_FILE = _resolve_hit_file(_HIT_FILE_RAW, _PROCESS_IDENTITY)
    _STATIC_ARTIFACT_METADATA = None


def _reset_artifact_after_fork() -> None:
    """Drop parent counters, paths, and locks in a forked worker."""
    global _HIT_FILE, _HIT_LOCK, _STATIC_ARTIFACT_METADATA
    _HIT_FILE = None
    _HIT_LOCK = threading.Lock()
    _STATIC_ARTIFACT_METADATA = None
    _HIT_COUNTS.clear()
    _MISS_COUNTS.clear()
    _SELECTED_SCOPE_COUNTS.clear()
    _MISS_SCOPE_COUNTS.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_artifact_after_fork)


def _find_repo_root() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            return parent
    return None


def _git_output(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ("git", "-C", str(repo_root), *args),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _sglang_source_identity() -> dict[str, Any]:
    repo_root = _find_repo_root()
    if repo_root is None:
        return {
            "repo_root": None,
            "commit": None,
            "branch": None,
            "dirty": None,
        }
    status = _git_output(repo_root, "status", "--porcelain", "--untracked-files=normal")
    return {
        "repo_root": str(repo_root),
        "commit": _git_output(repo_root, "rev-parse", "HEAD"),
        "branch": _git_output(repo_root, "branch", "--show-current") or "DETACHED",
        "dirty": None if status is None else bool(status),
    }


def _static_artifact_metadata() -> dict[str, Any]:
    """Cache launcher and git metadata; never run git from every hit."""
    global _STATIC_ARTIFACT_METADATA
    if _STATIC_ARTIFACT_METADATA is None:
        _STATIC_ARTIFACT_METADATA = {
            "process": dict(_PROCESS_IDENTITY),
            "sglang": _sglang_source_identity(),
        }
    return _STATIC_ARTIFACT_METADATA


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _dso_identities(provider: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract convenient DSO fingerprints while retaining full provider_info."""
    identities: list[dict[str, Any]] = []
    fingerprint_keys = (
        "module_name",
        "extension_file",
        "so_file",
        "sha256",
        "build_id",
        "main_variant",
        "combine_variant",
        "promotion_status",
        "scheduler_order",
        "scheduler_mapping_location",
        "scheduler_metadata_order",
        "scheduler_contract_version",
        "scheduler_permutation_sha256",
    )

    def visit(value: Any, location: str) -> None:
        if isinstance(value, dict):
            identity = {
                key: _json_safe(value[key]) for key in fingerprint_keys if key in value
            }
            if identity and any(
                key in identity
                for key in ("module_name", "extension_file", "so_file", "sha256")
            ):
                identities.append({"location": location, **identity})
            for key, item in value.items():
                visit(item, f"{location}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{location}[{index}]")

    visit(provider.get("provider_info", {}), "provider_info")
    return identities


def _scope_rows(
    counts: dict[tuple[str, str, str, str, str], int],
    detail_name: str,
) -> list[dict[str, Any]]:
    rows = []
    for (layer, op, phase, m, detail), count in sorted(counts.items()):
        rows.append(
            {
                "layer": None if layer == "unknown" else int(layer),
                "op": op,
                "phase": phase,
                "m": None if m == "unknown" else int(m),
                detail_name: detail,
                "count": count,
            }
        )
    return rows


def _flush_stats() -> None:
    if _HIT_FILE is None:
        return
    try:
        _HIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        state = _json_safe(provider_state())
        payload = {
            "schema_version": 2,
            **_static_artifact_metadata(),
            "manifest": _json_safe(config.load_manifest()),
            "provider_state": state,
            "provider_dso_identities": _dso_identities(state),
            "counts": {
                "selected": _scope_rows(_SELECTED_SCOPE_COUNTS, "kind"),
                "misses": _scope_rows(_MISS_SCOPE_COUNTS, "reason"),
            },
            # Keep the original flat maps for existing one-GPU harness readers.
            "hits": dict(_HIT_COUNTS),
            "misses": dict(_MISS_COUNTS),
        }
        temporary = _HIT_FILE.with_name(
            f".{_HIT_FILE.name}.tmp.{os.getpid()}.{threading.get_ident()}"
        )
        try:
            temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
            os.replace(temporary, _HIT_FILE)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    except Exception as exc:
        print(f"[glm52_opt] hit-file write failed: {exc}", flush=True)


def _record_hit(
    kind: str, op: Optional[str], phase: str, m: Optional[int] = None
) -> None:
    """Count successful glm52_opt dispatches; log first hit per key."""
    if _HIT_FILE is None and _HIT_FILE_RAW:
        _initialize_process_artifact()
    if _HIT_FILE is None:
        return
    key = f"{kind}:{op or 'untagged'}:{phase}"
    if m is not None:
        key += f":m{m}"
    layer_id = get_layer_id()
    scope_key = (
        str(layer_id) if layer_id is not None else "unknown",
        op or "untagged",
        phase,
        str(m) if m is not None else "unknown",
        kind,
    )
    with _HIT_LOCK:
        n = _HIT_COUNTS.get(key, 0) + 1
        _HIT_COUNTS[key] = n
        scope_n = _SELECTED_SCOPE_COUNTS.get(scope_key, 0) + 1
        _SELECTED_SCOPE_COUNTS[scope_key] = scope_n
        # Flush the first observation of every layer/op/bucket so an abrupt
        # worker exit cannot leave later CUDA-graph layers absent from G4.
        should_flush = scope_n == 1 or n in (10, 100) or n % 500 == 0
        if should_flush:
            _flush_stats()
    if n == 1:
        msg = f"glm52_opt HIT {key} (first)"
        logger.warning(msg)
        print(msg, flush=True)


def _record_miss(
    reason: str, op: Optional[str], phase: str, m: Optional[int] = None
) -> None:
    if _HIT_FILE is None and _HIT_FILE_RAW:
        _initialize_process_artifact()
    if _HIT_FILE is None:
        return
    key = f"{reason}:{op or 'untagged'}:{phase}"
    if m is not None:
        key += f":m{m}"
    layer_id = get_layer_id()
    scope_key = (
        str(layer_id) if layer_id is not None else "unknown",
        op or "untagged",
        phase,
        str(m) if m is not None else "unknown",
        reason,
    )
    with _HIT_LOCK:
        n = _MISS_COUNTS.get(key, 0) + 1
        _MISS_COUNTS[key] = n
        scope_n = _MISS_SCOPE_COUNTS.get(scope_key, 0) + 1
        _MISS_SCOPE_COUNTS[scope_key] = scope_n
        should_log = n == 1
        should_flush = scope_n == 1 or n in (10, 100) or n % 500 == 0
        if should_flush:
            _flush_stats()
    if should_log:
        msg = f"glm52_opt MISS {key} (first)"
        logger.warning(msg)
        print(msg, flush=True)


def _flush_stats_at_exit() -> None:
    if _HIT_FILE is None or not (_HIT_COUNTS or _MISS_COUNTS):
        return
    with _HIT_LOCK:
        _flush_stats()


if _HIT_FILE is not None:
    atexit.register(_flush_stats_at_exit)


def _current_phase(token_num: int) -> str:
    return infer_glm52_phase(get_forward_mode(), token_num)


def _nvtx_range(name: str):
    """Default-off profiler range; authoritative A/B emits no NVTX events."""

    @contextmanager
    def _cm():
        if not config.emit_infini_kernel_nvtx():
            yield
            return
        pushed = False
        try:
            torch.cuda.nvtx.range_push(name)
            pushed = True
        except Exception:
            pass
        try:
            yield
        finally:
            if pushed:
                try:
                    torch.cuda.nvtx.range_pop()
                except Exception:
                    pass

    return _cm()


def _is_cuda_graph_capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _graph_only_declines(spec: KernelSpec) -> bool:
    """Whether a ``graph_only`` spec must decline this eager call.

    Returns True only outside CUDA graph capture, so the caller can return the
    stock path before any provider launch.  Deliberately takes no hit/miss lock:
    this runs on every eager decode step.
    """
    return (
        spec.graph_only
        and config.graph_only_enabled(spec.op)
        and not _is_cuda_graph_capturing()
    )


def _profiler_range_name(spec: KernelSpec, m: int) -> str:
    name = spec.profiler_name or (
        f"infini_kernel_glm52_{spec.op}_{spec.phase}_{spec.implementation}"
    )
    if spec.n is not None and spec.k is not None:
        return f"{name}[M={m},N={spec.n},K={spec.k}]"
    return f"{name}[M={m}]"


def _fixed_nk_forward_mode_matches(spec: KernelSpec) -> bool:
    if spec.implementation != "fixed_nk":
        return True
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    mode = get_forward_mode()
    if spec.phase == "decode":
        return mode is ForwardMode.DECODE
    if spec.phase == "prefill":
        return mode is ForwardMode.EXTEND
    return False


def _fixed_nk_abi_matches(
    spec: KernelSpec,
    input_2d: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    block_size: List[int],
    output_dtype: torch.dtype,
    bias: Optional[torch.Tensor],
) -> bool:
    if spec.implementation != "fixed_nk":
        return True
    if spec.n is None or spec.k is None:
        return False
    m = int(input_2d.shape[0]) if input_2d.ndim == 2 else -1
    return (
        _fixed_nk_forward_mode_matches(spec)
        and input_2d.ndim == 2
        and tuple(input_2d.shape) == (m, spec.k)
        and tuple(weight.shape) == (spec.n, spec.k)
        and tuple(block_size) == (128, 128)
        and input_2d.dtype == torch.float8_e4m3fn
        and weight.dtype == torch.float8_e4m3fn
        and input_2d.is_cuda
        and weight.is_cuda
        and input_2d.is_contiguous()
        and weight.is_contiguous()
        and input_2d.device == weight.device
        and x_scale.dtype == torch.int32
        and weight_scale.dtype == torch.int32
        and x_scale.is_cuda
        and weight_scale.is_cuda
        and x_scale.device == input_2d.device
        and weight_scale.device == input_2d.device
        and tuple(x_scale.shape) == (m, spec.k // 128 // 4)
        and tuple(weight_scale.shape) == (spec.n, spec.k // 128 // 4)
        and tuple(x_scale.stride()) == (1, m)
        and tuple(weight_scale.stride()) == (1, spec.n)
        and output_dtype == torch.bfloat16
        and bias is None
    )


def _tensor_contract(
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device | None = None,
) -> bool:
    return bool(
        tensor.is_cuda
        and tensor.dtype == dtype
        and tuple(tensor.shape) == shape
        and tuple(tensor.stride()) == stride
        and tensor.storage_offset() == 0
        and (device is None or tensor.device == device)
    )


def _data_ptr_is_aligned(tensor: torch.Tensor, alignment: int = 16) -> bool:
    """Check the caller-owned alignment required by the FlashMLA loads/TMA."""
    return tensor.data_ptr() % alignment == 0


def _moe_hotspot_abi_matches(
    spec: KernelSpec,
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    forward_m: Optional[int],
) -> bool:
    if (
        spec.implementation != "hotspot_plugin"
        or spec.kind != "moe_masked"
        or spec.n is None
        or spec.k is None
        or spec.num_groups is None
        or spec.slab_m is None
        or forward_m not in (16, 32)
    ):
        return False
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    if get_forward_mode() is not ForwardMode.DECODE:
        return False
    allowed_expected_m = {16: (4, 5), 32: (8, 9)}[int(forward_m)]
    if (
        expected_m not in allowed_expected_m
        or spec.expected_m_values is None
        or expected_m not in spec.expected_m_values
    ):
        return False

    x, x_scale = lhs
    weight, weight_scale = rhs
    groups, slab_m, n, k = (
        spec.num_groups,
        spec.slab_m,
        spec.n,
        spec.k,
    )
    scale_k = k // 512
    device = x.device
    return bool(
        _tensor_contract(
            x,
            shape=(groups, slab_m, k),
            stride=(slab_m * k, k, 1),
            dtype=torch.float8_e4m3fn,
        )
        and _tensor_contract(
            x_scale,
            shape=(groups, slab_m, scale_k),
            stride=(slab_m * scale_k, 1, slab_m),
            dtype=torch.int32,
            device=device,
        )
        and _tensor_contract(
            weight,
            shape=(groups, n, k),
            stride=(n * k, k, 1),
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        and _tensor_contract(
            weight_scale,
            shape=(groups, n, scale_k),
            stride=(n * scale_k, 1, n),
            dtype=torch.int32,
            device=device,
        )
        and _tensor_contract(
            out,
            shape=(groups, slab_m, n),
            stride=(slab_m * n, n, 1),
            dtype=torch.bfloat16,
            device=device,
        )
        and _tensor_contract(
            masked_m,
            shape=(groups,),
            stride=(1,),
            dtype=torch.int32,
            device=device,
        )
    )


def _flashmla_hotspot_abi_matches(
    spec: KernelSpec,
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: float,
    indices: torch.Tensor,
    block_table: torch.Tensor,
    is_fp8_kvcache: bool,
) -> bool:
    if (
        spec.implementation != "hotspot_plugin"
        or spec.kind != "dsa"
        or None
        in (
            spec.topk,
            spec.q_heads,
            spec.qk_dim,
            spec.v_dim,
            spec.page_size,
            spec.kv_dim,
        )
    ):
        return False
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    if get_forward_mode() is not ForwardMode.DECODE:
        return False
    m = int(q.shape[0]) if q.ndim == 4 else -1
    num_pages = int(k_cache.shape[0]) if k_cache.ndim == 4 else 0
    max_tma_pages = (2**31 - 1) // int(spec.page_size)
    device = q.device
    return bool(
        m in (16, 32)
        and 0 < num_pages <= max_tma_pages
        and head_dim_v == spec.v_dim
        and is_fp8_kvcache is True
        and float(softmax_scale) == 0.0625
        and _tensor_contract(
            q,
            shape=(m, 1, int(spec.q_heads), int(spec.qk_dim)),
            stride=(
                int(spec.q_heads) * int(spec.qk_dim),
                int(spec.q_heads) * int(spec.qk_dim),
                int(spec.qk_dim),
                1,
            ),
            dtype=torch.bfloat16,
        )
        and _data_ptr_is_aligned(q)
        and _tensor_contract(
            k_cache,
            shape=(
                num_pages,
                int(spec.page_size),
                1,
                int(spec.kv_dim),
            ),
            stride=(
                int(spec.page_size) * int(spec.kv_dim),
                int(spec.kv_dim),
                int(spec.kv_dim),
                1,
            ),
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        and _data_ptr_is_aligned(k_cache)
        and _tensor_contract(
            cache_seqlens,
            shape=(m,),
            stride=(1,),
            dtype=torch.int32,
            device=device,
        )
        and _tensor_contract(
            tile_scheduler_metadata,
            shape=(148, 8),
            stride=(8, 1),
            dtype=torch.int32,
            device=device,
        )
        and _data_ptr_is_aligned(tile_scheduler_metadata, 32)
        and _tensor_contract(
            num_splits,
            shape=(m + 1,),
            stride=(1,),
            dtype=torch.int32,
            device=device,
        )
        and _data_ptr_is_aligned(num_splits)
        and _tensor_contract(
            indices,
            shape=(m, 1, int(spec.topk)),
            stride=(int(spec.topk), int(spec.topk), 1),
            dtype=torch.int32,
            device=device,
        )
        and _data_ptr_is_aligned(indices)
        and _tensor_contract(
            block_table,
            shape=(m, 0),
            stride=(1, 1),
            dtype=torch.int32,
            device=device,
        )
    )


def try_dispatch_flashmla_sparse_decode(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: float,
    indices: torch.Tensor,
    block_table: torch.Tensor,
    is_fp8_kvcache: bool,
) -> Optional[torch.Tensor]:
    """Run one exact FlashMLA provider call or return ``None`` before launch."""
    if not config.is_enabled():
        return None
    m = int(q.shape[0]) if q.ndim == 4 else -1
    phase = _current_phase(m)
    spec = lookup("dsa_decode_attn", phase, m=m)
    if spec is None or spec.kind != "dsa":
        _record_miss("flashmla_no_spec", "dsa_decode_attn", phase, m=m)
        return None
    # The provider is production-graph-bound.  Decline before the ABI guard
    # and launch so eager decode falls through to the stock FlashMLA path.
    if _graph_only_declines(spec):
        return None
    if not _flashmla_hotspot_abi_matches(
        spec,
        q=q,
        k_cache=k_cache,
        cache_seqlens=cache_seqlens,
        head_dim_v=head_dim_v,
        tile_scheduler_metadata=tile_scheduler_metadata,
        num_splits=num_splits,
        softmax_scale=softmax_scale,
        indices=indices,
        block_table=block_table,
        is_fp8_kvcache=is_fp8_kvcache,
    ):
        _record_miss("flashmla_abi", spec.op, phase, m=m)
        return None

    with _nvtx_range(_profiler_range_name(spec, m)):
        result = run_flashmla_sparse_decode(
            q=q,
            k_cache=k_cache,
            cache_seqlens=cache_seqlens,
            head_dim_v=head_dim_v,
            tile_scheduler_metadata=tile_scheduler_metadata,
            num_splits=num_splits,
            softmax_scale=softmax_scale,
            indices=indices,
            block_table=block_table,
            is_fp8_kvcache=is_fp8_kvcache,
        )
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise RuntimeError(
            "FlashMLA hotspot provider must return the stock (output, lse) pair"
        )
    candidate_out, candidate_lse = result
    if not isinstance(candidate_out, torch.Tensor) or not _tensor_contract(
        candidate_out,
        shape=(m, 1, int(spec.q_heads), int(spec.v_dim)),
        stride=(
            int(spec.q_heads) * int(spec.v_dim),
            int(spec.q_heads) * int(spec.v_dim),
            int(spec.v_dim),
            1,
        ),
        dtype=torch.bfloat16,
        device=q.device,
    ):
        raise RuntimeError("FlashMLA hotspot provider returned an invalid output")
    if not isinstance(candidate_lse, torch.Tensor) or not _tensor_contract(
        candidate_lse,
        shape=(m, int(spec.q_heads), 1),
        stride=(int(spec.q_heads), 1, int(spec.q_heads)),
        dtype=torch.float32,
        device=q.device,
    ):
        raise RuntimeError("FlashMLA hotspot provider returned an invalid LSE")
    _record_hit("hotspot_plugin/flashmla_sparse_decode", spec.op, phase, m=m)
    return candidate_out


def _flashmla_prefill_hotspot_abi_matches(
    spec: KernelSpec,
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: float,
    indices: torch.Tensor,
    block_table: torch.Tensor,
    is_fp8_kvcache: bool,
) -> bool:
    """Exact flashmla_kv PREFILL ABI gate.

    Identical tensor contract to the decode gate; the two differences are the
    forward mode (EXTEND, not DECODE) and the admissible batch extent, which is
    the extend token count and is taken from the spec rather than hardcoded so
    an unmeasured bucket can never be admitted.
    """
    if (
        spec.implementation != "hotspot_plugin"
        or spec.kind != "dsa"
        or spec.phase != "prefill"
        or spec.m_values is None
        or None
        in (
            spec.topk,
            spec.q_heads,
            spec.qk_dim,
            spec.v_dim,
            spec.page_size,
            spec.kv_dim,
        )
    ):
        return False
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    if get_forward_mode() is not ForwardMode.EXTEND:
        return False
    m = int(q.shape[0]) if q.ndim == 4 else -1
    device = q.device
    return bool(
        m in tuple(spec.m_values)
        and head_dim_v == spec.v_dim
        and is_fp8_kvcache is True
        and float(softmax_scale) == 0.0625
        and _tensor_contract(
            q,
            shape=(m, 1, int(spec.q_heads), int(spec.qk_dim)),
            stride=(
                int(spec.q_heads) * int(spec.qk_dim),
                int(spec.q_heads) * int(spec.qk_dim),
                int(spec.qk_dim),
                1,
            ),
            dtype=torch.bfloat16,
        )
        and k_cache.is_cuda
        and k_cache.dtype == torch.float8_e4m3fn
        and tuple(k_cache.shape[1:]) == (int(spec.page_size), 1, int(spec.kv_dim))
        and k_cache.is_contiguous()
        and k_cache.storage_offset() == 0
        and k_cache.device == device
        and _tensor_contract(
            cache_seqlens,
            shape=(m,),
            stride=(1,),
            dtype=torch.int32,
            device=device,
        )
        and _tensor_contract(
            tile_scheduler_metadata,
            shape=(148, 8),
            stride=(8, 1),
            dtype=torch.int32,
            device=device,
        )
        and _tensor_contract(
            num_splits,
            shape=(m + 1,),
            stride=(1,),
            dtype=torch.int32,
            device=device,
        )
        and _tensor_contract(
            indices,
            shape=(m, 1, int(spec.topk)),
            stride=(int(spec.topk), int(spec.topk), 1),
            dtype=torch.int32,
            device=device,
        )
        and block_table.is_cuda
        and block_table.dtype == torch.int32
        and tuple(block_table.shape) == (m, 0)
        and block_table.device == device
    )


def try_dispatch_flashmla_sparse_prefill(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: float,
    indices: torch.Tensor,
    block_table: torch.Tensor,
    is_fp8_kvcache: bool,
) -> Optional[torch.Tensor]:
    """Run one exact flashmla_kv prefill provider call, or return ``None``.

    Returning ``None`` leaves the caller on the stock path having launched
    nothing, which is the only fallback: once the provider is invoked the result
    is used.
    """
    if not config.is_enabled():
        return None
    m = int(q.shape[0]) if q.ndim == 4 else -1
    spec = lookup("dsa_prefill_attn", "prefill", m=m)
    if spec is None or spec.kind != "dsa":
        _record_miss("flashmla_prefill_no_spec", "dsa_prefill_attn", "prefill", m=m)
        return None
    if _graph_only_declines(spec):
        return None
    if not _flashmla_prefill_hotspot_abi_matches(
        spec,
        q=q,
        k_cache=k_cache,
        cache_seqlens=cache_seqlens,
        head_dim_v=head_dim_v,
        tile_scheduler_metadata=tile_scheduler_metadata,
        num_splits=num_splits,
        softmax_scale=softmax_scale,
        indices=indices,
        block_table=block_table,
        is_fp8_kvcache=is_fp8_kvcache,
    ):
        _record_miss("flashmla_prefill_abi", spec.op, "prefill", m=m)
        return None

    with _nvtx_range(_profiler_range_name(spec, m)):
        result = run_flashmla_sparse_prefill(
            q=q,
            k_cache=k_cache,
            cache_seqlens=cache_seqlens,
            head_dim_v=head_dim_v,
            tile_scheduler_metadata=tile_scheduler_metadata,
            num_splits=num_splits,
            softmax_scale=softmax_scale,
            indices=indices,
            block_table=block_table,
            is_fp8_kvcache=is_fp8_kvcache,
        )
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise RuntimeError(
            "FlashMLA prefill hotspot provider must return the stock (output, lse) pair"
        )
    candidate_out = result[0]
    if not isinstance(candidate_out, torch.Tensor) or not _tensor_contract(
        candidate_out,
        shape=(m, 1, int(spec.q_heads), int(spec.v_dim)),
        stride=(
            int(spec.q_heads) * int(spec.v_dim),
            int(spec.q_heads) * int(spec.v_dim),
            int(spec.v_dim),
            1,
        ),
        dtype=torch.bfloat16,
        device=q.device,
    ):
        raise RuntimeError(
            "FlashMLA prefill hotspot provider returned an invalid output"
        )
    _record_hit("hotspot_plugin/flashmla_sparse_prefill", spec.op, "prefill", m=m)
    return candidate_out


def record_psum_hit(op: Optional[str], m: Optional[int] = None) -> None:
    """Count contig PSUM layout applications (goals 08/09)."""
    phase = "prefill"
    try:
        phase = _current_phase(int(m) if m is not None else 1)
    except Exception:
        pass
    _record_hit("moe_contig_psum", op, phase, m=m)


def try_dispatch_fp8_gemm(
    input_2d: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    block_size: List[int],
    output_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    if not config.is_enabled():
        return None
    op = get_op_name()
    m = int(input_2d.shape[0])
    phase = _current_phase(m)
    spec = lookup(op, phase, m=m)
    if spec is None or spec.kind != "fp8_gemm":
        _record_miss("no_spec", op, phase, m=m)
        return None
    # A graph_only fp8_gemm spec (decode o_proj) declines outside CUDA-graph
    # capture before the ABI check and before the hit/miss lock, so eager decode
    # returns the stock path with zero provider launch and no glm52_opt tax.
    if _graph_only_declines(spec):
        return None
    if not _fixed_nk_abi_matches(
        spec,
        input_2d,
        weight,
        x_scale,
        weight_scale,
        block_size,
        output_dtype,
        bias,
    ):
        _record_miss("fixed_nk_abi", op, phase, m=m)
        return None
    out = input_2d.new_empty(input_2d.shape[0], weight.shape[0], dtype=output_dtype)
    with _nvtx_range(_profiler_range_name(spec, m)):
        ok, path = run_fp8_gemm(
            op,
            input_2d,
            weight,
            x_scale,
            weight_scale,
            out,
            block_size,
            spec.archive_ref,
            phase=phase,
            implementation=spec.implementation,
        )
    if not ok:
        _record_miss(f"run_skipped:{path}", op, phase, m=m)
        return None
    _record_hit(f"fp8_gemm/{path}", op, phase, m=m)
    if bias is not None:
        out = out + bias
    return out.view(*input_2d.shape[:-1], weight.shape[0])


def try_dispatch_moe_masked(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
) -> bool:
    """Run one exact MoE grouped-masked replacement or leave stock selected.

    The MoE W2 hotspot spec is ``graph_only``: outside CUDA graph capture the
    call returns ``False`` immediately so eager decode uses stock and avoids the
    API-v1 Python provider tax on the containing region.
    """
    if not config.is_enabled():
        return False
    op = get_op_name()
    # w13 is fused gate+up in SGLang; either tag should enable decode pack path.
    phase = _current_phase(lhs[0].shape[1] if lhs[0].ndim >= 2 else 1)
    # The grouped input's second dimension is the fixed expert slab (1024),
    # not the model-forward token bucket.  Use launch-time ForwardBatch M so
    # M16 and M32 can independently select a replacement.
    forward_m = get_forward_m()
    spec = lookup(op, phase, m=forward_m)
    if spec is None and op in ("moe_gate_proj", "moe_up_proj"):
        # Prefer gate's registered decode pack if only one is present.
        spec = lookup("moe_gate_proj", phase, m=forward_m) or lookup(
            "moe_up_proj", phase, m=forward_m
        )
    if spec is None or spec.kind != "moe_masked":
        _record_miss("moe_no_spec", op, phase, m=forward_m)
        return False
    # Prefill moe_gate: Graph regresses at large M — never swap.
    if phase == "prefill" and op == "moe_gate_proj":
        _record_miss("moe_prefill_skip", op, phase, m=forward_m)
        return False
    x_fp8, x_scale = lhs
    w_fp8, w_scale = rhs
    if spec.implementation == "hotspot_plugin":
        if _graph_only_declines(spec):
            return False
        if not _moe_hotspot_abi_matches(
            spec,
            lhs,
            rhs,
            out,
            masked_m,
            expected_m,
            forward_m,
        ):
            _record_miss("moe_hotspot_abi", spec.op, phase, m=forward_m)
            return False
        with _nvtx_range(
            _profiler_range_name(
                spec,
                int(forward_m) if forward_m is not None else -1,
            )
        ):
            returned = run_hotspot_moe_masked(
                spec.op,
                lhs=lhs,
                rhs=rhs,
                out=out,
                masked_m=masked_m,
                expected_m=expected_m,
            )
        if returned is not None:
            raise RuntimeError(
                f"{spec.op} hotspot provider violated the stock None return contract"
            )
        _record_hit("hotspot_plugin", spec.op, phase, m=forward_m)
        return True

    try:
        range_name = (
            f"infini_kernel_glm52_{op or spec.op}_{phase}_moe_masked"
            f"[M={forward_m if forward_m is not None else 'unknown'}]"
        )
        with _nvtx_range(range_name):
            run_moe_masked(x_fp8, w_fp8, x_scale, w_scale, out, masked_m, expected_m)
    except Exception as exc:
        _record_miss(
            f"moe_run_fail:{type(exc).__name__}",
            op or (spec.op if spec else None),
            phase,
            m=forward_m,
        )
        logger.warning("glm52_opt moe_masked failed: %s", exc)
        return False
    _record_hit("moe_masked", op or spec.op, phase, m=forward_m)
    return True
