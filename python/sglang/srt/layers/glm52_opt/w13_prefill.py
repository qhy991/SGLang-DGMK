"""Default-off, fail-closed GLM-5.2 prefill W13 PSUM dispatch.

Import is CPU-only. The explicit initializer runs after worker GPU assignment,
loads independent same-base stock/candidate DeepGEMM modules, binds their lazy
JIT compilers to distinct cache roots, and warms the exact W13/W2 ABI. The
selected hot path performs host tensor-metadata checks and exactly one launch.
"""

from __future__ import annotations

import hashlib
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

logger = logging.getLogger(__name__)

BASE_COMMIT = "edcf77b276965de8f03cdc47c23f01b08bf7c7ab"
CUTLASS_COMMIT = "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8"
FMT_COMMIT = "553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28"
PATCH_SHA256 = "e5d75cf8116a3497e5677ab28e2800dda3af2a78b977092417c33b039735e134"
BASE_BLOB_SHA256 = {
    "csrc/apis/gemm.hpp": (
        "0840d64249e2a5a4a994d495e8320a0fff26bad9ca107426a1a1226e7d621186"
    ),
    "csrc/jit_kernels/heuristics/sm100.hpp": (
        "487cac2ff19027c781b08e9a0391836e77c03cdffcb7ceb3346d8633c8eb0884"
    ),
    "csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp": (
        "cca1ddb5b5787942c31b39a9d5618929ee609c6c3b57b877fe636df39540366b"
    ),
    "csrc/tvm_ffi_api.cpp": (
        "c09aeec8187a2e29a3ebfc61c9ce1168a89fea775040a47bcf73739131ea57c0"
    ),
    "sgl_deep_gemm/__init__.py": (
        "b33e89deacdce241f01f5070d321918f5e5480e3e6d3af569678d4192db4f2a7"
    ),
    "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh": (
        "9c1e70677ede6ba09ab98e629482da7874182f8227907382efe0a81658da5a37"
    ),
}
SOURCE_TREE_SHA256 = {
    "stock_source_tree_sha256": (
        "4bfc233540d0478bf88860d924c53e105be29e01ddd039a68a8c5242addb2af5"
    ),
    "psum_source_tree_sha256": (
        "4bfc233540d0478bf88860d924c53e105be29e01ddd039a68a8c5242addb2af5"
    ),
    "xor_source_tree_sha256": (
        "8b2cce5a251e9903d67a6d13f761fa75b8ad03bd54dddf03819bc6f2793411ab"
    ),
    "complete_source_diff_sha256": PATCH_SHA256,
}

REQUIRED_PDL = True
REQUIRED_NUM_SMS = 148
REQUIRED_TC_UTIL = 100
REQUIRED_TOPOLOGY = {
    "tp_size": 8,
    "dp_size": 8,
    "ep_size": 8,
    "pp_size": 1,
    "moe_dp_size": 1,
    "enable_dp_attention": True,
    "moe_a2a_backend": "deepep",
    "deepep_mode": "auto",
    "moe_runner_backend": "deep_gemm",
    "ep_num_redundant_experts": 0,
}
EXPECTED_M = 1024
EXPERTS = 32
ALIGNED_ROWS = 35200
VALID_ROWS = 32982

RAW_COUNTS = (
    1061,
    975,
    985,
    1046,
    1044,
    1049,
    1016,
    1027,
    1043,
    1059,
    1065,
    1038,
    1045,
    1014,
    995,
    1015,
    1104,
    1051,
    1047,
    996,
    1023,
    999,
    995,
    984,
    1073,
    1029,
    1053,
    997,
    1021,
    1050,
    1030,
    1053,
)
ALIGNED_COUNTS = (
    1152,
    1024,
    1024,
    1152,
    1152,
    1152,
    1024,
    1152,
    1152,
    1152,
    1152,
    1152,
    1152,
    1024,
    1024,
    1024,
    1152,
    1152,
    1152,
    1024,
    1024,
    1024,
    1024,
    1024,
    1152,
    1152,
    1152,
    1024,
    1024,
    1152,
    1152,
    1152,
)
assert sum(RAW_COUNTS) == VALID_ROWS
assert sum(ALIGNED_COUNTS) == ALIGNED_ROWS

_A_SHAPE = (ALIGNED_ROWS, 6144)
_A_STRIDE = (6144, 1)
_AS_SHAPE = (ALIGNED_ROWS, 12)
_AS_STRIDE = (1, ALIGNED_ROWS)
_W13_B_SHAPE = (EXPERTS, 4096, 6144)
_W13_B_STRIDE = (4096 * 6144, 6144, 1)
_W13_BS_SHAPE = (EXPERTS, 4096, 12)
_W13_BS_STRIDE = (4096 * 12, 1, 4096)
_W13_OUT_SHAPE = (ALIGNED_ROWS, 4096)
_W13_OUT_STRIDE = (4096, 1)
_W2_A_SHAPE = (ALIGNED_ROWS, 2048)
_W2_A_STRIDE = (2048, 1)
_W2_AS_SHAPE = (ALIGNED_ROWS, 4)
_W2_AS_STRIDE = (1, ALIGNED_ROWS)
_W2_B_SHAPE = (EXPERTS, 6144, 2048)
_W2_B_STRIDE = (6144 * 2048, 2048, 1)
_W2_BS_SHAPE = (EXPERTS, 6144, 4)
_W2_BS_STRIDE = (6144 * 4, 1, 6144)
_W2_OUT_SHAPE = (ALIGNED_ROWS, 6144)
_W2_OUT_STRIDE = (6144, 1)
_LAYOUT_SHAPE = (ALIGNED_ROWS,)
_ENDPOINT_SHAPE = (EXPERTS,)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_snapshot(path: Path) -> dict[str, str]:
    if not path.is_dir():
        raise FileNotFoundError(f"W13 prefill JIT cache does not exist: {path}")
    return {
        item.relative_to(path).as_posix(): _sha256(item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def _restore_environment(saved: dict[str, str | None]) -> None:
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _validate_manifest(
    manifest: dict[str, Any], manifest_path: Path
) -> dict[str, Any]:
    if manifest.get("schema_version") != 3:
        raise ValueError(
            f"W13 prefill manifest is not reproducibility schema 3: {manifest_path}"
        )
    source = manifest.get("source")
    build = manifest.get("build")
    variants = manifest.get("variants")
    if not all(isinstance(value, dict) for value in (source, build, variants)):
        raise ValueError(f"malformed W13 prefill manifest: {manifest_path}")
    expected_source = {
        "commit": BASE_COMMIT,
        "cutlass_commit": CUTLASS_COMMIT,
        "fmt_commit": FMT_COMMIT,
        "base_blob_sha256": BASE_BLOB_SHA256,
        **SOURCE_TREE_SHA256,
    }
    actual_source = {key: source.get(key) for key in expected_source}
    if actual_source != expected_source:
        raise ValueError(
            "W13 prefill source identity mismatch: "
            f"actual={actual_source}, expected={expected_source}"
        )
    if (
        build.get("torch") != torch.__version__
        or build.get("torch_cuda") != torch.version.cuda
    ):
        raise ValueError(
            "W13 prefill build/runtime mismatch: "
            f"built={build.get('torch')}/{build.get('torch_cuda')}, "
            f"runtime={torch.__version__}/{torch.version.cuda}"
        )
    required_build = {
        "cuda_arch": "10.0a",
        "jit_compiler": "nvcc",
        "compile_api": "tvm_ffi.cpp.build",
        "submodule_update": False,
        "force_clean_build_directories": True,
        "max_jobs": "1",
        "variant_command_identical": True,
        "cpp_files_template": ["<SOURCE>/csrc/tvm_ffi_api.cpp"],
        "build_directory_template": "<OUTPUT>/compile/<VARIANT>",
        "jit_cache_template": "<OUTPUT>/jit/<VARIANT>",
    }
    actual_build = {key: build.get(key) for key in required_build}
    if actual_build != required_build:
        raise ValueError(
            "W13 prefill build contract mismatch: "
            f"actual={actual_build}, expected={required_build}"
        )
    plan_sha = build.get("normalized_build_plan_sha256")
    if not isinstance(plan_sha, str) or len(plan_sha) != 64:
        raise ValueError("W13 prefill normalized compiler plan is missing")
    for compiler in ("cxx", "nvcc"):
        identity = build.get(f"{compiler}_identity")
        if not isinstance(identity, dict):
            raise ValueError(f"W13 prefill {compiler} identity is missing")
        path = Path(str(identity.get("path", ""))).resolve()
        if not path.is_file() or _sha256(path) != identity.get("sha256"):
            raise ValueError(f"W13 prefill {compiler} identity mismatch")
    for name in ("stock", "psum", "xor"):
        record = variants.get(name)
        if not isinstance(record, dict):
            raise ValueError(f"W13 prefill manifest has no {name} record")
        ninja = Path(str(record.get("build_ninja", ""))).resolve()
        if (
            record.get("normalized_build_plan_sha256") != plan_sha
            or not ninja.is_file()
            or _sha256(ninja) != record.get("build_ninja_sha256")
        ):
            raise ValueError(f"W13 prefill {name} compiler-plan mismatch")
    return variants


def _variant_record(
    manifest_path: str | Path, variant: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(path.read_text())
    variants = _validate_manifest(manifest, path)
    record = variants.get(variant)
    if not isinstance(record, dict):
        raise ValueError(f"manifest has no W13 prefill variant {variant!r}")
    expected_tree = SOURCE_TREE_SHA256[f"{variant}_source_tree_sha256"]
    expected_patched = variant == "xor"
    if (
        record.get("patched") is not expected_patched
        or record.get("source_tree_sha256") != expected_tree
    ):
        raise ValueError(f"W13 prefill {variant} source contract mismatch")
    package = Path(str(record.get("package", ""))).resolve()
    shared_object = Path(str(record.get("shared_object", ""))).resolve()
    jit_cache = Path(str(record.get("jit_cache", ""))).resolve()
    init_py = package / "__init__.py"
    if package / "_C.so" != shared_object:
        raise ValueError(f"W13 prefill {variant} package/DSO mismatch")
    if not shared_object.is_file() or _sha256(shared_object) != record.get(
        "shared_object_sha256"
    ):
        raise ValueError(f"W13 prefill {variant} DSO hash mismatch")
    if not init_py.is_file() or _sha256(init_py) != record.get(
        "package_init_sha256"
    ):
        raise ValueError(f"W13 prefill {variant} package hash mismatch")
    if not jit_cache.is_dir():
        raise FileNotFoundError(jit_cache)
    return record, manifest


def load_variant(
    manifest_path: str | Path,
    variant: str,
    *,
    module_name: str | None = None,
) -> tuple[ModuleType, dict[str, Any], dict[str, Any]]:
    """Load one validated variant without rebinding its cache environment."""

    record, manifest = _variant_record(manifest_path, variant)
    package = Path(record["package"]).resolve()
    name = module_name or f"deep_gemm_w13_prefill_{variant}"
    spec = importlib.util.spec_from_file_location(
        name,
        package / "__init__.py",
        submodule_search_locations=[str(package)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load W13 prefill module from {package}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    module.__path__ = [str(package)]  # type: ignore[attr-defined]
    module.__package__ = name
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module, record, manifest


def _read_runtime_state(module: ModuleType) -> dict[str, Any]:
    return {
        "pdl": bool(module.get_pdl()),
        "num_sms": int(module.get_num_sms()),
        "tc_util": int(module.get_tc_util()),
    }


def _set_required_runtime_state(module: ModuleType, label: str) -> dict[str, Any]:
    module.set_pdl(REQUIRED_PDL)
    module.set_num_sms(REQUIRED_NUM_SMS)
    module.set_tc_util(REQUIRED_TC_UTIL)
    actual = _read_runtime_state(module)
    expected = {
        "pdl": REQUIRED_PDL,
        "num_sms": REQUIRED_NUM_SMS,
        "tc_util": REQUIRED_TC_UTIL,
    }
    if actual != expected:
        raise RuntimeError(
            f"W13 prefill {label} runtime state mismatch: "
            f"actual={actual}, expected={expected}"
        )
    return actual


def _prove_runtime_state_independence(
    stock: ModuleType, candidate: ModuleType
) -> dict[str, Any]:
    required = {
        "pdl": REQUIRED_PDL,
        "num_sms": REQUIRED_NUM_SMS,
        "tc_util": REQUIRED_TC_UTIL,
    }
    mutations = {
        "pdl": ("set_pdl", False),
        "num_sms": ("set_num_sms", REQUIRED_NUM_SMS - 1),
        "tc_util": ("set_tc_util", REQUIRED_TC_UTIL - 1),
    }
    proof: dict[str, Any] = {}
    for field_name, (setter_name, mutation) in mutations.items():
        field_proof: dict[str, Any] = {}
        for changed_name, changed, other_name, other in (
            ("stock", stock, "candidate", candidate),
            ("candidate", candidate, "stock", stock),
        ):
            getattr(changed, setter_name)(mutation)
            changed_value = _read_runtime_state(changed)[field_name]
            other_value = _read_runtime_state(other)[field_name]
            if changed_value != mutation or other_value != required[field_name]:
                raise RuntimeError(
                    "W13 prefill module runtime globals alias: "
                    f"{field_name} {changed_name}={changed_value}, "
                    f"{other_name}={other_value}"
                )
            getattr(changed, setter_name)(required[field_name])
            field_proof[f"mutate_{changed_name}"] = {
                "mutated_value": changed_value,
                f"{other_name}_unchanged": other_value,
                "restored": _read_runtime_state(changed)[field_name],
            }
        proof[field_name] = field_proof
    return proof


def _module_identity(
    record: dict[str, Any], cache_snapshot: dict[str, str]
) -> dict[str, Any]:
    return {
        "package": str(Path(record["package"]).resolve()),
        "package_init_sha256": record["package_init_sha256"],
        "shared_object": str(Path(record["shared_object"]).resolve()),
        "shared_object_sha256": record["shared_object_sha256"],
        "jit_cache": str(Path(record["jit_cache"]).resolve()),
        "jit_artifacts": cache_snapshot,
    }


def _zeros_strided(
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    value = torch.empty_strided(shape, stride, dtype=dtype, device=device)
    value.zero_()
    return value


def _layout_values() -> tuple[list[int], list[int]]:
    rowmap: list[int] = []
    endpoints: list[int] = []
    start = 0
    for expert, (raw, aligned) in enumerate(zip(RAW_COUNTS, ALIGNED_COUNTS)):
        rowmap.extend([expert] * aligned)
        endpoints.append(start + raw)
        start += aligned
    if len(rowmap) != ALIGNED_ROWS or endpoints[-1] != ALIGNED_ROWS - (
        ALIGNED_COUNTS[-1] - RAW_COUNTS[-1]
    ):
        raise RuntimeError("W13 prefill fixed layout construction failed")
    return rowmap, endpoints


def _allocate_warm_inputs(device: torch.device) -> dict[str, torch.Tensor]:
    rowmap, endpoints = _layout_values()
    return {
        "a": _zeros_strided(_A_SHAPE, _A_STRIDE, torch.float8_e4m3fn, device),
        "a_scale": _zeros_strided(_AS_SHAPE, _AS_STRIDE, torch.int32, device),
        "w13_b": _zeros_strided(
            _W13_B_SHAPE, _W13_B_STRIDE, torch.float8_e4m3fn, device
        ),
        "w13_b_scale": _zeros_strided(
            _W13_BS_SHAPE, _W13_BS_STRIDE, torch.int32, device
        ),
        "w13_out": _zeros_strided(
            _W13_OUT_SHAPE, _W13_OUT_STRIDE, torch.bfloat16, device
        ),
        "w2_a": _zeros_strided(
            _W2_A_SHAPE, _W2_A_STRIDE, torch.float8_e4m3fn, device
        ),
        "w2_a_scale": _zeros_strided(
            _W2_AS_SHAPE, _W2_AS_STRIDE, torch.int32, device
        ),
        "w2_b": _zeros_strided(
            _W2_B_SHAPE, _W2_B_STRIDE, torch.float8_e4m3fn, device
        ),
        "w2_b_scale": _zeros_strided(
            _W2_BS_SHAPE, _W2_BS_STRIDE, torch.int32, device
        ),
        "w2_out": _zeros_strided(
            _W2_OUT_SHAPE, _W2_OUT_STRIDE, torch.bfloat16, device
        ),
        "rowmap": torch.tensor(rowmap, dtype=torch.int32, device=device),
        "endpoint": torch.tensor(endpoints, dtype=torch.int32, device=device),
    }


def _launch_w13(
    module: ModuleType,
    tensors: dict[str, torch.Tensor],
    *,
    use_psum: bool,
) -> None:
    layout = tensors["endpoint"] if use_psum else tensors["rowmap"]
    returned = module.m_grouped_fp8_fp4_gemm_nt_contiguous(
        (tensors["a"], tensors["a_scale"]),
        (tensors["w13_b"], tensors["w13_b_scale"]),
        tensors["w13_out"],
        layout,
        compiled_dims="nk",
        disable_ue8m0_cast=True,
        use_psum_layout=use_psum,
        ensure_zero_padding=not use_psum,
        expected_m_for_psum_layout=EXPECTED_M if use_psum else None,
    )
    if returned is not None:
        raise RuntimeError("W13 prefill launch changed the stock None return ABI")


def _launch_w2_stock(
    module: ModuleType, tensors: dict[str, torch.Tensor]
) -> None:
    returned = module.m_grouped_fp8_fp4_gemm_nt_contiguous(
        (tensors["w2_a"], tensors["w2_a_scale"]),
        (tensors["w2_b"], tensors["w2_b_scale"]),
        tensors["w2_out"],
        tensors["rowmap"],
        compiled_dims="nk",
        disable_ue8m0_cast=True,
    )
    if returned is not None:
        raise RuntimeError("W2 stock launch changed the None return ABI")


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
    return os.environ.get("SGLANG_GLM52_W13_PREFILL_VARIANT", "").strip().lower()


def initialization_requested() -> bool:
    return requested_variant() not in ("", "0", "off", "false")


def _validate_startup_topology(server_args: Any) -> dict[str, Any]:
    actual = {
        name: getattr(server_args, name, None) for name in REQUIRED_TOPOLOGY
    }
    if actual != REQUIRED_TOPOLOGY:
        raise RuntimeError(
            "selected W13 prefill candidate requires exact TP8/DP8/EP8 "
            f"AUTO-normal DeepEP topology: actual={actual}, "
            f"required={REQUIRED_TOPOLOGY}"
        )
    return actual


def initialize_w13_prefill_after_assignment(
    gpu_id: int,
    server_args: Any,
    *,
    compile_utils_loader: Callable[[], ModuleType],
) -> bool:
    """Initialize selected modules after GPU assignment; return compile status."""

    global _STATE
    variant = requested_variant()
    if variant in ("", "0", "off", "false"):
        _STATE = _DispatchState(False, "default_off")
        return False
    if variant not in ("psum", "xor"):
        _STATE = _DispatchState(False, f"unsupported_variant:{variant}")
        raise RuntimeError(f"unsupported requested W13 prefill variant: {variant}")
    manifest_text = os.environ.get("SGLANG_GLM52_W13_PREFILL_MANIFEST", "").strip()
    if not manifest_text:
        _STATE = _DispatchState(False, "missing_manifest", variant=variant)
        raise RuntimeError(
            "requested W13 prefill variant requires an exact build manifest"
        )
    try:
        topology = _validate_startup_topology(server_args)
    except RuntimeError:
        _STATE = _DispatchState(
            False,
            "topology_mismatch",
            variant=variant,
            gpu_id=int(gpu_id),
            manifest=manifest_text,
        )
        raise

    with _INITIALIZE_LOCK:
        if _STATE.enabled:
            if _STATE.gpu_id != int(gpu_id) or _STATE.variant != variant:
                raise RuntimeError(
                    "W13 prefill runtime was initialized for another worker"
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
        try:
            if int(torch.cuda.current_device()) != int(gpu_id):
                raise RuntimeError("W13 prefill initialized before GPU assignment")
            if torch.cuda.get_device_capability(gpu_id) != (10, 0):
                raise RuntimeError("W13 prefill candidate requires an sm_100 B200")
            manifest_path = Path(manifest_text).expanduser().resolve()
            stock_record, _ = _variant_record(manifest_path, "stock")
            candidate_record, _ = _variant_record(manifest_path, variant)
            stock_cache = Path(stock_record["jit_cache"]).resolve()
            candidate_cache = Path(candidate_record["jit_cache"]).resolve()
            if stock_cache == candidate_cache:
                raise RuntimeError("W13 prefill stock/candidate JIT caches alias")

            os.environ["DG_JIT_USE_NVRTC"] = "0"
            os.environ["SGL_DG_USE_NVRTC"] = "0"
            os.environ["DG_JIT_CACHE_DIR"] = str(stock_cache)
            os.environ["SGLANG_DG_CACHE_DIR"] = str(stock_cache)
            stock, stock_record, _ = load_variant(
                manifest_path,
                "stock",
                module_name="deep_gemm_w13_prefill_stock_production",
            )
            stock_state = _set_required_runtime_state(stock, "stock")
            tensors = _allocate_warm_inputs(torch.device("cuda", gpu_id))
            _launch_w13(stock, tensors, use_psum=False)
            _launch_w2_stock(stock, tensors)
            torch.cuda.synchronize(gpu_id)
            stock_snapshot = _cache_snapshot(stock_cache)
            if not stock_snapshot:
                raise RuntimeError("W13 prefill stock warmup produced no JIT files")

            compile_utils = compile_utils_loader()
            compile_utils._ENABLE_JIT_DEEPGEMM_PRECOMPILE = False
            compile_utils.update_deep_gemm_config(gpu_id, server_args)

            os.environ["DG_JIT_CACHE_DIR"] = str(candidate_cache)
            os.environ["SGLANG_DG_CACHE_DIR"] = str(candidate_cache)
            candidate, candidate_record, _ = load_variant(
                manifest_path,
                variant,
                module_name=f"deep_gemm_w13_prefill_{variant}_production",
            )
            candidate_state = _set_required_runtime_state(candidate, variant)
            independence = _prove_runtime_state_independence(stock, candidate)
            _launch_w13(candidate, tensors, use_psum=True)
            torch.cuda.synchronize(gpu_id)
            candidate_snapshot = _cache_snapshot(candidate_cache)
            if not candidate_snapshot:
                raise RuntimeError(
                    "W13 prefill candidate warmup produced no JIT files"
                )

            probe_cache = (
                manifest_path.parent
                / f"unbound-w13-prefill-cache-probe-{os.getpid()}"
            )
            if probe_cache.exists():
                raise RuntimeError(f"cache-owner probe already exists: {probe_cache}")
            os.environ["DG_JIT_CACHE_DIR"] = str(probe_cache)
            os.environ["SGLANG_DG_CACHE_DIR"] = str(probe_cache)
            _launch_w13(stock, tensors, use_psum=False)
            _launch_w13(candidate, tensors, use_psum=True)
            _launch_w2_stock(stock, tensors)
            torch.cuda.synchronize(gpu_id)
            if probe_cache.exists():
                raise RuntimeError("a W13 prefill compiler escaped its cache owner")
            if _cache_snapshot(stock_cache) != stock_snapshot:
                raise RuntimeError("W13 prefill stock cache changed after freeze")
            if _cache_snapshot(candidate_cache) != candidate_snapshot:
                raise RuntimeError("W13 prefill candidate cache changed after freeze")

            _STATE = _DispatchState(
                True,
                "ready",
                variant=variant,
                gpu_id=int(gpu_id),
                stock_module=stock,
                candidate_module=candidate,
                manifest=str(manifest_path),
                modules={
                    "stock": _module_identity(stock_record, stock_snapshot),
                    variant: _module_identity(candidate_record, candidate_snapshot),
                },
                runtime_state={"stock": stock_state, variant: candidate_state},
                state_independence=independence,
                jit_use_nvrtc=False,
                topology=topology,
            )
            logger.info(
                "GLM-5.2 W13 prefill candidate ready: variant=%s gpu=%d",
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
            logger.error("W13 prefill initialization failed: %s", exc)
            raise RuntimeError(
                "requested W13 prefill candidate failed post-assignment initialization"
            ) from exc
        finally:
            if "tensors" in locals():
                del tensors
                torch.cuda.empty_cache()
            _restore_environment(saved)


def _tensor_contract(
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    dtype: torch.dtype,
) -> bool:
    return bool(
        tensor.is_cuda
        and tensor.dtype == dtype
        and tuple(tensor.shape) == shape
        and tuple(tensor.stride()) == stride
        and tensor.storage_offset() == 0
    )


def _same_device(first: torch.Tensor, *others: torch.Tensor) -> bool:
    return all(value.device == first.device for value in others)


def _w13_shape_target(
    lhs: tuple[torch.Tensor, torch.Tensor],
    rhs: tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
) -> bool:
    return bool(
        tuple(lhs[0].shape) == _A_SHAPE
        and tuple(rhs[0].shape) == _W13_B_SHAPE
        and tuple(out.shape) == _W13_OUT_SHAPE
    )


def _exact_w13_contract(
    lhs: tuple[torch.Tensor, torch.Tensor],
    rhs: tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    rowmap: torch.Tensor,
    endpoint: torch.Tensor,
) -> bool:
    a, a_scale = lhs
    b, b_scale = rhs
    return bool(
        _tensor_contract(a, shape=_A_SHAPE, stride=_A_STRIDE, dtype=torch.float8_e4m3fn)
        and _tensor_contract(
            a_scale, shape=_AS_SHAPE, stride=_AS_STRIDE, dtype=torch.int32
        )
        and _tensor_contract(
            b, shape=_W13_B_SHAPE, stride=_W13_B_STRIDE, dtype=torch.float8_e4m3fn
        )
        and _tensor_contract(
            b_scale, shape=_W13_BS_SHAPE, stride=_W13_BS_STRIDE, dtype=torch.int32
        )
        and _tensor_contract(
            out, shape=_W13_OUT_SHAPE, stride=_W13_OUT_STRIDE, dtype=torch.bfloat16
        )
        and _tensor_contract(
            rowmap, shape=_LAYOUT_SHAPE, stride=(1,), dtype=torch.int32
        )
        and _tensor_contract(
            endpoint, shape=_ENDPOINT_SHAPE, stride=(1,), dtype=torch.int32
        )
        and _same_device(a, a_scale, b, b_scale, out, rowmap, endpoint)
    )


def try_dispatch_w13_prefill(
    lhs: tuple[torch.Tensor, torch.Tensor],
    rhs: tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    rowmap: torch.Tensor,
    endpoint: torch.Tensor | None,
    *,
    recipe_a: tuple[int, int] | None,
    recipe_b: tuple[int, int] | None,
) -> bool:
    """Launch once for the exact selected W13 ABI; abort malformed target calls."""

    state = _STATE
    if not state.enabled:
        return False
    if not _w13_shape_target(lhs, rhs, out):
        return False
    if endpoint is None:
        raise RuntimeError("selected W13 prefill PSUM call has no scatter endpoint")
    if recipe_a is not None or recipe_b is not None:
        raise RuntimeError("selected W13 prefill PSUM call changed the FP8 recipe")
    if not _exact_w13_contract(lhs, rhs, out, rowmap, endpoint):
        raise RuntimeError("selected W13 prefill PSUM call violates the packed ABI")
    if lhs[0].device.index != state.gpu_id:
        raise RuntimeError("selected W13 prefill PSUM call reached the wrong GPU")
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
        raise RuntimeError("selected W13 prefill launch violated the None return ABI")
    return True


def _w2_shape_target(
    lhs: tuple[torch.Tensor, torch.Tensor],
    rhs: tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
) -> bool:
    return bool(
        tuple(lhs[0].shape) == _W2_A_SHAPE
        and tuple(rhs[0].shape) == _W2_B_SHAPE
        and tuple(out.shape) == _W2_OUT_SHAPE
    )


def _exact_w2_contract(
    lhs: tuple[torch.Tensor, torch.Tensor],
    rhs: tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    rowmap: torch.Tensor,
) -> bool:
    a, a_scale = lhs
    b, b_scale = rhs
    return bool(
        _tensor_contract(
            a, shape=_W2_A_SHAPE, stride=_W2_A_STRIDE, dtype=torch.float8_e4m3fn
        )
        and _tensor_contract(
            a_scale, shape=_W2_AS_SHAPE, stride=_W2_AS_STRIDE, dtype=torch.int32
        )
        and _tensor_contract(
            b, shape=_W2_B_SHAPE, stride=_W2_B_STRIDE, dtype=torch.float8_e4m3fn
        )
        and _tensor_contract(
            b_scale, shape=_W2_BS_SHAPE, stride=_W2_BS_STRIDE, dtype=torch.int32
        )
        and _tensor_contract(
            out, shape=_W2_OUT_SHAPE, stride=_W2_OUT_STRIDE, dtype=torch.bfloat16
        )
        and _tensor_contract(
            rowmap, shape=_LAYOUT_SHAPE, stride=(1,), dtype=torch.int32
        )
        and _same_device(a, a_scale, b, b_scale, out, rowmap)
    )


def try_dispatch_w2_prefill_stock(
    lhs: tuple[torch.Tensor, torch.Tensor],
    rhs: tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    rowmap: torch.Tensor,
    *,
    recipe_a: tuple[int, int] | None,
    recipe_b: tuple[int, int] | None,
) -> bool:
    """Keep exact W2 on the pinned stock row-map module while W13 is selected."""

    state = _STATE
    if not state.enabled:
        return False
    if not _w2_shape_target(lhs, rhs, out):
        return False
    if recipe_a is not None or recipe_b is not None:
        raise RuntimeError("selected W13 experiment reached a non-stock W2 recipe")
    if not _exact_w2_contract(lhs, rhs, out, rowmap):
        raise RuntimeError("selected W13 experiment reached a malformed stock W2 ABI")
    if lhs[0].device.index != state.gpu_id:
        raise RuntimeError("selected stock W2 call reached the wrong GPU")
    assert state.stock_module is not None
    returned = state.stock_module.m_grouped_fp8_fp4_gemm_nt_contiguous(
        lhs,
        rhs,
        out,
        rowmap,
        compiled_dims="nk",
        disable_ue8m0_cast=True,
    )
    if returned is not None:
        raise RuntimeError("selected stock W2 launch violated the None return ABI")
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
