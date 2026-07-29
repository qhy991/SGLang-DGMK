"""Default-off, fail-closed GLM-5.2 contiguous-prefill W2 PSUM dispatch.

Import is CPU-only. Explicit post-assignment initialization loads independent
same-base stock and candidate modules, warms the exact row-map/PSUM signatures,
and freezes their task-local JIT caches. The hot path performs only host tensor
metadata checks and one candidate launch using the endpoint retained by Task 28.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from sglang.srt.layers.glm52_opt import w13_prefill as task28

logger = logging.getLogger(__name__)

BASE_COMMIT = "edcf77b276965de8f03cdc47c23f01b08bf7c7ab"
CUTLASS_COMMIT = "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8"
FMT_COMMIT = "553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28"
PATCH_SHA256 = "8cccdd7135a04a532c96605743466471410fac1d5e612067fb1cda10be1bd53e"
STOCK_TREE_SHA256 = "4bfc233540d0478bf88860d924c53e105be29e01ddd039a68a8c5242addb2af5"
STAGE7_TREE_SHA256 = "c02885d8fac2549b34b66a25ed106c7d369cbd5cbbf043465b6c084056ed86f2"
BASE_BLOB_SHA256 = task28.BASE_BLOB_SHA256

REQUIRED_PDL = True
REQUIRED_NUM_SMS = 148
REQUIRED_TC_UTIL = 100
REQUIRED_TOPOLOGY = task28.REQUIRED_TOPOLOGY
EXPECTED_M = 1024
EXPERTS = 32
ALIGNED_ROWS = 35200
VALID_ROWS = 32982

_A_SHAPE = (35200, 2048)
_A_STRIDE = (2048, 1)
_AS_SHAPE = (35200, 4)
_AS_STRIDE = (1, 35200)
_B_SHAPE = (32, 6144, 2048)
_B_STRIDE = (6144 * 2048, 2048, 1)
_BS_SHAPE = (32, 6144, 4)
_BS_STRIDE = (6144 * 4, 1, 6144)
_OUT_SHAPE = (35200, 6144)
_OUT_STRIDE = (6144, 1)
_ROWMAP_SHAPE = (35200,)
_ENDPOINT_SHAPE = (32,)


def _manifest(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    expected_source = {
        "commit": BASE_COMMIT,
        "cutlass_commit": CUTLASS_COMMIT,
        "fmt_commit": FMT_COMMIT,
        "base_blob_sha256": BASE_BLOB_SHA256,
        "stock_source_tree_sha256": STOCK_TREE_SHA256,
        "psum_source_tree_sha256": STOCK_TREE_SHA256,
        "stage7_source_tree_sha256": STAGE7_TREE_SHA256,
        "complete_source_diff_sha256": PATCH_SHA256,
    }
    source = document.get("source", {})
    if (
        document.get("schema_version") != 3
        or document.get("task") != "30_moe_w2_prefill_psum"
        or {key: source.get(key) for key in expected_source} != expected_source
    ):
        raise ValueError("W2 prefill manifest source identity mismatch")
    required_build = {
        "cuda_arch": "10.0a",
        "variant_command_identical": True,
        "submodule_update": False,
        "compile_api": "tvm_ffi.cpp.build",
        "force_clean_build_directories": True,
        "jit_compiler": "nvcc",
        "max_jobs": "1",
        "cpp_files_template": ["<SOURCE>/csrc/tvm_ffi_api.cpp"],
        "build_directory_template": "<OUTPUT>/compile/<VARIANT>",
        "jit_cache_template": "<OUTPUT>/jit/<VARIANT>",
    }
    build = document.get("build", {})
    if {key: build.get(key) for key in required_build} != required_build:
        raise ValueError("W2 prefill manifest build contract mismatch")
    return document


def _variant_record(
    manifest_path: str | Path, variant: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    if variant not in ("stock", "psum", "stage7"):
        raise ValueError(f"unsupported W2 prefill variant: {variant}")
    path = Path(manifest_path).expanduser().resolve()
    document = _manifest(path)
    record = document.get("variants", {}).get(variant)
    if not isinstance(record, dict):
        raise TypeError(f"W2 prefill manifest has no {variant} variant")
    expected_tree = STAGE7_TREE_SHA256 if variant == "stage7" else STOCK_TREE_SHA256
    expected = {
        "source_tree_sha256": expected_tree,
        "patched": variant == "stage7",
        "pipeline_stages": 7 if variant == "stage7" else 8,
        "normalized_build_plan_sha256": document["build"][
            "normalized_build_plan_sha256"
        ],
    }
    if {key: record.get(key) for key in expected} != expected:
        raise ValueError(f"W2 prefill {variant} manifest record mismatch")
    package = Path(str(record.get("package", ""))).resolve()
    shared_object = Path(str(record.get("shared_object", ""))).resolve()
    jit_cache = Path(str(record.get("jit_cache", ""))).resolve()
    init_py = package / "__init__.py"
    if package / "_C.so" != shared_object:
        raise ValueError(f"W2 prefill {variant} package/DSO mismatch")
    if not shared_object.is_file() or task28._sha256(shared_object) != record.get(
        "shared_object_sha256"
    ):
        raise ValueError(f"W2 prefill {variant} DSO hash mismatch")
    if not init_py.is_file() or task28._sha256(init_py) != record.get(
        "package_init_sha256"
    ):
        raise ValueError(f"W2 prefill {variant} package hash mismatch")
    if not jit_cache.is_dir():
        raise FileNotFoundError(jit_cache)
    return record, document


def load_variant(
    manifest_path: str | Path,
    variant: str,
    *,
    module_name: str | None = None,
) -> tuple[ModuleType, dict[str, Any], dict[str, Any]]:
    record, document = _variant_record(manifest_path, variant)
    package = Path(record["package"]).resolve()
    name = module_name or f"deep_gemm_w2_prefill_{variant}"
    spec = importlib.util.spec_from_file_location(
        name,
        package / "__init__.py",
        submodule_search_locations=[str(package)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load W2 prefill module from {package}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    module.__path__ = [str(package)]  # type: ignore[attr-defined]
    module.__package__ = name
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module, record, document


def _set_required_runtime_state(module: ModuleType, label: str) -> dict[str, Any]:
    module.set_pdl(REQUIRED_PDL)
    module.set_num_sms(REQUIRED_NUM_SMS)
    module.set_tc_util(REQUIRED_TC_UTIL)
    actual = task28._read_runtime_state(module)
    expected = {
        "pdl": REQUIRED_PDL,
        "num_sms": REQUIRED_NUM_SMS,
        "tc_util": REQUIRED_TC_UTIL,
    }
    if actual != expected:
        raise RuntimeError(
            f"W2 prefill {label} runtime state mismatch: "
            f"actual={actual}, expected={expected}"
        )
    return actual


def _allocate_warm_inputs(device: torch.device) -> dict[str, torch.Tensor]:
    # This fixed startup-only fixture compiles the signature. Production calls
    # consume DeepGemmRunnerInput.expert_start_loc from the live Task 28 scatter.
    tensors = task28._allocate_warm_inputs(device)
    return {
        "a": tensors["w2_a"],
        "a_scale": tensors["w2_a_scale"],
        "b": tensors["w2_b"],
        "b_scale": tensors["w2_b_scale"],
        "out": tensors["w2_out"],
        "rowmap": tensors["rowmap"],
        "endpoint": tensors["endpoint"],
        "w13_a": tensors["a"],
        "w13_a_scale": tensors["a_scale"],
        "w13_b": tensors["w13_b"],
        "w13_b_scale": tensors["w13_b_scale"],
        "w13_out": tensors["w13_out"],
    }


def _launch_w2(
    module: ModuleType,
    tensors: dict[str, torch.Tensor],
    *,
    use_psum: bool,
) -> None:
    returned = module.m_grouped_fp8_fp4_gemm_nt_contiguous(
        (tensors["a"], tensors["a_scale"]),
        (tensors["b"], tensors["b_scale"]),
        tensors["out"],
        tensors["endpoint"] if use_psum else tensors["rowmap"],
        compiled_dims="nk",
        disable_ue8m0_cast=True,
        use_psum_layout=use_psum,
        ensure_zero_padding=not use_psum,
        expected_m_for_psum_layout=EXPECTED_M if use_psum else None,
    )
    if returned is not None:
        raise RuntimeError("W2 prefill launch changed the stock None return ABI")


def _launch_w13_stock(module: ModuleType, tensors: dict[str, torch.Tensor]) -> None:
    returned = module.m_grouped_fp8_fp4_gemm_nt_contiguous(
        (tensors["w13_a"], tensors["w13_a_scale"]),
        (tensors["w13_b"], tensors["w13_b_scale"]),
        tensors["w13_out"],
        tensors["rowmap"],
        compiled_dims="nk",
        disable_ue8m0_cast=True,
    )
    if returned is not None:
        raise RuntimeError("stock W13 warmup changed the None return ABI")


@dataclass(frozen=True)
class _DispatchState:
    enabled: bool
    reason: str
    variant: str = ""
    gpu_id: int | None = None
    stock_module: ModuleType | None = field(default=None, repr=False)
    candidate_module: ModuleType | None = field(default=None, repr=False)
    manifest: str = ""
    modules: dict[str, dict[str, Any]] = field(default_factory=dict)
    runtime_state: dict[str, dict[str, Any]] = field(default_factory=dict)
    state_independence: dict[str, Any] = field(default_factory=dict)
    jit_use_nvrtc: bool | None = None
    topology: dict[str, Any] = field(default_factory=dict)


_STATE = _DispatchState(False, "not_initialized")
_INITIALIZE_LOCK = threading.Lock()


def requested_variant() -> str:
    return os.environ.get("SGLANG_GLM52_W2_PREFILL_VARIANT", "").strip().lower()


def initialization_requested() -> bool:
    return requested_variant() not in ("", "0", "off", "false")


def _validate_startup_topology(server_args: Any) -> dict[str, Any]:
    actual = {name: getattr(server_args, name, None) for name in REQUIRED_TOPOLOGY}
    if actual != REQUIRED_TOPOLOGY:
        raise RuntimeError(
            "selected W2 prefill candidate requires exact TP8/DP8/EP8 "
            f"AUTO-normal DeepEP topology: actual={actual}, "
            f"required={REQUIRED_TOPOLOGY}"
        )
    return actual


def initialize_w2_prefill_after_assignment(
    gpu_id: int,
    server_args: Any,
    *,
    compile_utils_loader: Callable[[], ModuleType],
) -> bool:
    """Initialize exact modules after worker GPU assignment."""

    global _STATE
    variant = requested_variant()
    if variant in ("", "0", "off", "false"):
        _STATE = _DispatchState(False, "default_off")
        return False
    if variant not in ("psum", "stage7"):
        _STATE = _DispatchState(False, f"unsupported_variant:{variant}")
        raise RuntimeError(f"unsupported requested W2 prefill variant: {variant}")
    manifest_text = os.environ.get("SGLANG_GLM52_W2_PREFILL_MANIFEST", "").strip()
    if not manifest_text:
        _STATE = _DispatchState(False, "missing_manifest", variant=variant)
        raise RuntimeError(
            "requested W2 prefill variant requires an exact build manifest"
        )
    topology = _validate_startup_topology(server_args)

    with _INITIALIZE_LOCK:
        if _STATE.enabled:
            if _STATE.gpu_id != int(gpu_id) or _STATE.variant != variant:
                raise RuntimeError(
                    "W2 prefill runtime was initialized for another worker"
                )
            return True
        saved = {
            name: os.environ.get(name)
            for name in (
                "DG_JIT_CACHE_DIR",
                "SGLANG_DG_CACHE_DIR",
                "DG_JIT_USE_NVRTC",
                "SGL_DG_USE_NVRTC",
            )
        }
        tensors = None
        try:
            if int(torch.cuda.current_device()) != int(gpu_id):
                raise RuntimeError("W2 prefill initialized before GPU assignment")
            if torch.cuda.get_device_capability(gpu_id) != (10, 0):
                raise RuntimeError("W2 prefill candidate requires an sm_100 B200")
            manifest_path = Path(manifest_text).expanduser().resolve()
            stock_record, _ = _variant_record(manifest_path, "stock")
            candidate_record, _ = _variant_record(manifest_path, variant)
            stock_cache = Path(stock_record["jit_cache"]).resolve()
            candidate_cache = Path(candidate_record["jit_cache"]).resolve()
            if stock_cache == candidate_cache:
                raise RuntimeError("W2 prefill stock/candidate JIT caches alias")

            os.environ["DG_JIT_USE_NVRTC"] = "0"
            os.environ["SGL_DG_USE_NVRTC"] = "0"
            os.environ["DG_JIT_CACHE_DIR"] = str(stock_cache)
            os.environ["SGLANG_DG_CACHE_DIR"] = str(stock_cache)
            stock, stock_record, _ = load_variant(
                manifest_path,
                "stock",
                module_name=f"deep_gemm_w2_prefill_stock_production_{os.getpid()}",
            )
            stock_state = _set_required_runtime_state(stock, "stock")
            tensors = _allocate_warm_inputs(torch.device("cuda", gpu_id))
            _launch_w13_stock(stock, tensors)
            _launch_w2(stock, tensors, use_psum=False)
            torch.cuda.synchronize(gpu_id)
            stock_snapshot = task28._cache_snapshot(stock_cache)
            if not stock_snapshot:
                raise RuntimeError("W2 prefill stock warmup produced no JIT files")

            compile_utils = compile_utils_loader()
            compile_utils._ENABLE_JIT_DEEPGEMM_PRECOMPILE = False
            compile_utils.update_deep_gemm_config(gpu_id, server_args)

            os.environ["DG_JIT_CACHE_DIR"] = str(candidate_cache)
            os.environ["SGLANG_DG_CACHE_DIR"] = str(candidate_cache)
            candidate, candidate_record, _ = load_variant(
                manifest_path,
                variant,
                module_name=(
                    f"deep_gemm_w2_prefill_{variant}_production_{os.getpid()}"
                ),
            )
            candidate_state = _set_required_runtime_state(candidate, variant)
            independence = task28._prove_runtime_state_independence(stock, candidate)
            _launch_w2(candidate, tensors, use_psum=True)
            torch.cuda.synchronize(gpu_id)
            candidate_snapshot = task28._cache_snapshot(candidate_cache)
            if not candidate_snapshot:
                raise RuntimeError("W2 prefill candidate warmup produced no JIT files")

            probe_cache = (
                manifest_path.parent / f"unbound-w2-prefill-cache-probe-{os.getpid()}"
            )
            if probe_cache.exists():
                raise RuntimeError(f"cache-owner probe already exists: {probe_cache}")
            os.environ["DG_JIT_CACHE_DIR"] = str(probe_cache)
            os.environ["SGLANG_DG_CACHE_DIR"] = str(probe_cache)
            _launch_w13_stock(stock, tensors)
            _launch_w2(stock, tensors, use_psum=False)
            _launch_w2(candidate, tensors, use_psum=True)
            torch.cuda.synchronize(gpu_id)
            if probe_cache.exists():
                raise RuntimeError("a W2 prefill compiler escaped its cache owner")
            if task28._cache_snapshot(stock_cache) != stock_snapshot:
                raise RuntimeError("W2 prefill stock cache changed after freeze")
            if task28._cache_snapshot(candidate_cache) != candidate_snapshot:
                raise RuntimeError("W2 prefill candidate cache changed after freeze")

            _STATE = _DispatchState(
                True,
                "ready",
                variant=variant,
                gpu_id=int(gpu_id),
                stock_module=stock,
                candidate_module=candidate,
                manifest=str(manifest_path),
                modules={
                    "stock": task28._module_identity(stock_record, stock_snapshot),
                    variant: task28._module_identity(
                        candidate_record, candidate_snapshot
                    ),
                },
                runtime_state={"stock": stock_state, variant: candidate_state},
                state_independence=independence,
                jit_use_nvrtc=False,
                topology=topology,
            )
            logger.info(
                "GLM-5.2 W2 prefill candidate ready: variant=%s gpu=%d",
                variant,
                gpu_id,
            )
            return True
        except Exception as exc:
            _STATE = _DispatchState(
                False,
                f"initialization_failed:{type(exc).__name__}",
                variant=variant,
                gpu_id=int(gpu_id),
                manifest=manifest_text,
            )
            logger.error("W2 prefill initialization failed: %s", exc)
            raise RuntimeError(
                "requested W2 prefill candidate failed post-assignment initialization"
            ) from exc
        finally:
            if tensors is not None:
                del tensors
                torch.cuda.empty_cache()
            task28._restore_environment(saved)


def _shape_target(
    lhs: tuple[torch.Tensor, torch.Tensor],
    rhs: tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
) -> bool:
    return bool(
        tuple(lhs[0].shape) == _A_SHAPE
        and tuple(rhs[0].shape) == _B_SHAPE
        and tuple(out.shape) == _OUT_SHAPE
    )


def _exact_contract(
    lhs: tuple[torch.Tensor, torch.Tensor],
    rhs: tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    rowmap: torch.Tensor,
    endpoint: torch.Tensor,
) -> bool:
    a, a_scale = lhs
    b, b_scale = rhs
    return bool(
        task28._tensor_contract(
            a, shape=_A_SHAPE, stride=_A_STRIDE, dtype=torch.float8_e4m3fn
        )
        and task28._tensor_contract(
            a_scale, shape=_AS_SHAPE, stride=_AS_STRIDE, dtype=torch.int32
        )
        and task28._tensor_contract(
            b, shape=_B_SHAPE, stride=_B_STRIDE, dtype=torch.float8_e4m3fn
        )
        and task28._tensor_contract(
            b_scale, shape=_BS_SHAPE, stride=_BS_STRIDE, dtype=torch.int32
        )
        and task28._tensor_contract(
            out, shape=_OUT_SHAPE, stride=_OUT_STRIDE, dtype=torch.bfloat16
        )
        and task28._tensor_contract(
            rowmap, shape=_ROWMAP_SHAPE, stride=(1,), dtype=torch.int32
        )
        and task28._tensor_contract(
            endpoint, shape=_ENDPOINT_SHAPE, stride=(1,), dtype=torch.int32
        )
        and task28._same_device(a, a_scale, b, b_scale, out, rowmap, endpoint)
    )


def try_dispatch_w2_prefill(
    lhs: tuple[torch.Tensor, torch.Tensor],
    rhs: tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    rowmap: torch.Tensor,
    endpoint: torch.Tensor | None,
    *,
    recipe_a: tuple[int, int] | None,
    recipe_b: tuple[int, int] | None,
) -> bool:
    """Launch once for the exact W2 ABI; abort malformed selected calls."""

    state = _STATE
    if not state.enabled:
        return False
    if not _shape_target(lhs, rhs, out):
        return False
    if endpoint is None:
        raise RuntimeError("selected W2 prefill PSUM call has no Task 28 endpoint")
    if recipe_a is not None or recipe_b is not None:
        raise RuntimeError("selected W2 prefill PSUM call changed the FP8 recipe")
    if not _exact_contract(lhs, rhs, out, rowmap, endpoint):
        raise RuntimeError("selected W2 prefill PSUM call violates the packed ABI")
    if lhs[0].device.index != state.gpu_id:
        raise RuntimeError("selected W2 prefill PSUM call reached the wrong GPU")
    assert state.candidate_module is not None
    returned = state.candidate_module.m_grouped_fp8_fp4_gemm_nt_contiguous(
        lhs,
        rhs,
        out,
        endpoint,
        compiled_dims="nk",
        disable_ue8m0_cast=True,
        use_psum_layout=True,
        ensure_zero_padding=False,
        expected_m_for_psum_layout=EXPECTED_M,
    )
    if returned is not None:
        raise RuntimeError("selected W2 prefill launch violated the None return ABI")
    return True


def dispatch_state() -> dict[str, Any]:
    state = _STATE
    return {
        "enabled": state.enabled,
        "reason": state.reason,
        "variant": state.variant,
        "gpu_id": state.gpu_id,
        "manifest": state.manifest,
        "modules": state.modules,
        "runtime_state": state.runtime_state,
        "state_independence": state.state_independence,
        "jit_use_nvrtc": state.jit_use_nvrtc,
        "topology": state.topology,
    }
