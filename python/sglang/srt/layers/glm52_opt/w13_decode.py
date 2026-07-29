"""Default-off, fail-closed GLM-5.2 decode W13 DeepGEMM dispatch.

Importing this module performs no CUDA query, DSO load, cache mutation, or
manifest I/O.  The opt-in runtime is initialized explicitly after worker GPU
assignment.  Once selected, the hot path performs host-metadata guards and one
candidate launch; it never catches a launch failure or launches stock second.
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

from sglang.srt.layers.glm52_opt.w13_context import (
    W13DecodeForwardMarker,
    get_w13_decode_forward_marker,
)

logger = logging.getLogger(__name__)

BASE_COMMIT = "731e7c7a97d269e4b9f482ea18d0e709a948f293"
CUTLASS_COMMIT = "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8"
FMT_COMMIT = "553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28"
PATCH_SHA256 = "997348b6498aa18a7d70a5b1d36249b356b508cdc71e2f514a979818c48490a5"
BASE_BLOB_SHA256 = {
    "csrc/apis/gemm.hpp": "0840d64249e2a5a4a994d495e8320a0fff26bad9ca107426a1a1226e7d621186",
    "csrc/jit_kernels/heuristics/sm100.hpp": (
        "487cac2ff19027c781b08e9a0391836e77c03cdffcb7ceb3346d8633c8eb0884"
    ),
    "csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp": (
        "cca1ddb5b5787942c31b39a9d5618929ee609c6c3b57b877fe636df39540366b"
    ),
    "csrc/tvm_ffi_api.cpp": (
        "d1e5dbd833f257d2c4be516772404c02f1747247eef5075315ff2d1220a64c1f"
    ),
    "sgl_deep_gemm/__init__.py": (
        "243eeaa71fa65cecaddd7298245438cb371ca765d7bf914a9427e132be8d5f26"
    ),
}
SOURCE_TREE_SHA256 = {
    "stock_source_tree_sha256": (
        "917592ab68ea0608c9be33208c2c609bc7f20bd9b1603f32743dd0d1ae03d0ed"
    ),
    "candidate_source_tree_sha256": (
        "d38d8bf9a2118a2506be0fd71827568e70a20839505238a36a9c0325415332ef"
    ),
    "complete_source_diff_sha256": PATCH_SHA256,
}

VARIANT_CONFIGS = {
    "bm32_2sm": (32, 128, 128, 11, 2),
    "bm32_1sm": (32, 128, 128, 10, 1),
}
EXPECTED_M_BY_TOKEN_BUCKET = {
    16: frozenset((4, 5)),
    32: frozenset((8, 9)),
}
REQUIRED_PDL = True
REQUIRED_NUM_SMS = 148
REQUIRED_TC_UTIL = 100

_A_SHAPE = (32, 1024, 6144)
_A_STRIDE = (1024 * 6144, 6144, 1)
_AS_SHAPE = (32, 1024, 12)
_AS_STRIDE = (12288, 1, 1024)
_B_SHAPE = (32, 4096, 6144)
_B_STRIDE = (4096 * 6144, 6144, 1)
_BS_SHAPE = (32, 4096, 12)
_BS_STRIDE = (49152, 1, 4096)
_OUT_SHAPE = (32, 1024, 4096)
_OUT_STRIDE = (1024 * 4096, 4096, 1)
_MASK_SHAPE = (32,)
_MASK_STRIDE = (1,)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_snapshot(path: Path) -> dict[str, str]:
    if not path.is_dir():
        raise FileNotFoundError(f"W13 JIT cache does not exist: {path}")
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
    manifest: dict[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    if manifest.get("schema_version") != 2:
        raise ValueError(
            f"W13 manifest is not reproducibility schema 2: {manifest_path}"
        )
    source = manifest.get("source")
    build = manifest.get("build")
    variants = manifest.get("variants")
    if (
        not isinstance(source, dict)
        or not isinstance(build, dict)
        or not isinstance(variants, dict)
    ):
        raise ValueError(f"malformed W13 manifest: {manifest_path}")
    expected = {
        "commit": BASE_COMMIT,
        "cutlass_commit": CUTLASS_COMMIT,
        "fmt_commit": FMT_COMMIT,
        "candidate_patch_sha256": PATCH_SHA256,
        "base_blob_sha256": BASE_BLOB_SHA256,
        **SOURCE_TREE_SHA256,
    }
    actual = {key: source.get(key) for key in expected}
    if actual != expected:
        raise ValueError(
            f"W13 source identity mismatch: actual={actual}, expected={expected}"
        )
    if (
        build.get("torch") != torch.__version__
        or build.get("torch_cuda") != torch.version.cuda
    ):
        raise ValueError(
            "W13 build/runtime mismatch: "
            f"built torch={build.get('torch')}/cuda={build.get('torch_cuda')}, "
            f"runtime torch={torch.__version__}/cuda={torch.version.cuda}"
        )
    if build.get("stock_candidate_command_identical") is not True:
        raise ValueError("W13 stock/candidate build command identity is not attested")
    required_build_contract = {
        "cuda_arch": "10.0a",
        "submodule_update": False,
        "compile_api": "tvm_ffi.cpp.build",
        "force_clean_build_directories": True,
        "jit_compiler": "nvcc",
        "max_jobs": "1",
        "cpp_files_template": ["<SOURCE>/csrc/tvm_ffi_api.cpp"],
        "build_directory_template": "<OUTPUT>/compile/<VARIANT>",
        "jit_cache_template": "<OUTPUT>/jit/<VARIANT>",
    }
    actual_build_contract = {key: build.get(key) for key in required_build_contract}
    if actual_build_contract != required_build_contract:
        raise ValueError(
            "W13 build reproducibility contract mismatch: "
            f"actual={actual_build_contract}, expected={required_build_contract}"
        )
    build_plan_sha = build.get("normalized_build_plan_sha256")
    if not isinstance(build_plan_sha, str) or len(build_plan_sha) != 64:
        raise ValueError("W13 normalized build-plan hash is missing")
    for compiler in ("cxx", "nvcc"):
        compiler_path = Path(str(build.get(f"{compiler}_path", ""))).resolve()
        compiler_sha = build.get(f"{compiler}_sha256")
        if (
            not compiler_path.is_file()
            or not isinstance(compiler_sha, str)
            or _sha256(compiler_path) != compiler_sha
        ):
            raise ValueError(f"W13 {compiler} compiler identity mismatch")
    for name in ("stock", "candidate"):
        record = variants.get(name)
        if not isinstance(record, dict):
            raise ValueError(f"W13 manifest has no {name} build record")
        build_ninja = Path(str(record.get("build_ninja", ""))).resolve()
        if (
            record.get("normalized_build_plan_sha256") != build_plan_sha
            or not build_ninja.is_file()
            or _sha256(build_ninja) != record.get("build_ninja_sha256")
        ):
            raise ValueError(f"W13 {name} generated build-plan identity mismatch")
    return variants


def _variant_record(
    manifest_path: Path,
    build_variant: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    variants = _validate_manifest(manifest, manifest_path)
    record = variants.get(build_variant)
    if not isinstance(record, dict):
        raise ValueError(f"manifest has no build variant {build_variant!r}")
    expected_patched = build_variant == "candidate"
    expected_source_tree = SOURCE_TREE_SHA256[f"{build_variant}_source_tree_sha256"]
    if (
        record.get("patched") is not expected_patched
        or record.get("source_tree_sha256") != expected_source_tree
    ):
        raise ValueError(
            f"W13 {build_variant} variant source contract mismatch: {record}"
        )
    package = Path(record["package"]).resolve()
    shared_object = Path(record["shared_object"]).resolve()
    jit_cache = Path(record["jit_cache"]).resolve()
    if package / "_C.so" != shared_object:
        raise ValueError("manifest package/shared-object mismatch")
    if _sha256(shared_object) != record.get("shared_object_sha256"):
        raise ValueError(f"W13 shared-object hash mismatch: {shared_object}")
    init_py = package / "__init__.py"
    if not init_py.is_file():
        raise FileNotFoundError(init_py)
    if _sha256(init_py) != record.get("package_init_sha256"):
        raise ValueError(f"W13 package init hash mismatch: {init_py}")
    if not jit_cache.is_dir():
        raise FileNotFoundError(jit_cache)
    return record, manifest


def load_variant(
    manifest_path: str | Path,
    build_variant: str,
    *,
    module_name: str | None = None,
) -> tuple[ModuleType, dict[str, Any], dict[str, Any]]:
    """Load one validated module without changing cache environment variables."""

    path = Path(manifest_path).expanduser().resolve()
    record, manifest = _variant_record(path, build_variant)
    package = Path(record["package"]).resolve()
    init_py = package / "__init__.py"
    name = module_name or f"deep_gemm_w13_{build_variant}"
    spec = importlib.util.spec_from_file_location(
        name,
        init_py,
        submodule_search_locations=[str(package)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load W13 module from {init_py}")
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


@dataclass(frozen=True)
class _DispatchState:
    enabled: bool
    reason: str
    variant: str = ""
    config: tuple[int, int, int, int, int] = (0, 0, 0, 0, 0)
    gpu_id: int | None = None
    stock_module: ModuleType | None = field(default=None, repr=False)
    candidate_module: ModuleType | None = field(default=None, repr=False)
    manifest: str = ""
    modules: dict[str, dict[str, Any]] = field(default_factory=dict)
    runtime_state: dict[str, dict[str, Any]] = field(default_factory=dict)
    state_independence: dict[str, Any] = field(default_factory=dict)
    jit_use_nvrtc: bool | None = None


_STATE = _DispatchState(False, "not_initialized")
_INITIALIZE_LOCK = threading.Lock()


def requested_variant() -> str:
    return os.environ.get("SGLANG_GLM52_W13_DECODE_VARIANT", "").strip().lower()


def initialization_requested() -> bool:
    return requested_variant() not in ("", "0", "off", "false")


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
            f"W13 {label} runtime state mismatch: actual={actual}, expected={expected}"
        )
    return actual


def _prove_runtime_state_independence(
    stock: ModuleType,
    candidate: ModuleType,
) -> dict[str, Any]:
    required = {
        "pdl": REQUIRED_PDL,
        "num_sms": REQUIRED_NUM_SMS,
        "tc_util": REQUIRED_TC_UTIL,
    }
    setters = {
        "pdl": ("set_pdl", False),
        "num_sms": ("set_num_sms", REQUIRED_NUM_SMS - 1),
        "tc_util": ("set_tc_util", REQUIRED_TC_UTIL - 1),
    }
    proof: dict[str, Any] = {}
    for field_name, (setter_name, mutation) in setters.items():
        field_proof: dict[str, Any] = {}
        for mutated_name, mutated, other_name, other in (
            ("stock", stock, "candidate", candidate),
            ("candidate", candidate, "stock", stock),
        ):
            getattr(mutated, setter_name)(mutation)
            mutated_value = _read_runtime_state(mutated)[field_name]
            other_value = _read_runtime_state(other)[field_name]
            if mutated_value != mutation or other_value != required[field_name]:
                raise RuntimeError(
                    "W13 DeepGEMM runtime globals are not independent: "
                    f"field={field_name}, mutated={mutated_name}:{mutated_value}, "
                    f"other={other_name}:{other_value}"
                )
            getattr(mutated, setter_name)(required[field_name])
            field_proof[f"mutate_{mutated_name}"] = {
                "mutated_value": mutated_value,
                f"{other_name}_unchanged": other_value,
                "restored": _read_runtime_state(mutated)[field_name],
            }
        proof[field_name] = field_proof
    if (
        _read_runtime_state(stock) != required
        or _read_runtime_state(candidate) != required
    ):
        raise RuntimeError("W13 runtime-state independence restore failed")
    return proof


def _allocate_warm_inputs(device: torch.device) -> dict[str, torch.Tensor]:
    def zeros_strided(
        shape: tuple[int, ...],
        stride: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        value = torch.empty_strided(shape, stride, device=device, dtype=dtype)
        value.zero_()
        return value

    return {
        "a": zeros_strided(_A_SHAPE, _A_STRIDE, torch.float8_e4m3fn),
        "a_scale": zeros_strided(_AS_SHAPE, _AS_STRIDE, torch.int32),
        "b": zeros_strided(_B_SHAPE, _B_STRIDE, torch.float8_e4m3fn),
        "b_scale": zeros_strided(_BS_SHAPE, _BS_STRIDE, torch.int32),
        "out": zeros_strided(_OUT_SHAPE, _OUT_STRIDE, torch.bfloat16),
        "masked_m": zeros_strided(_MASK_SHAPE, _MASK_STRIDE, torch.int32),
    }


def _launch_named_config(
    module: ModuleType,
    tensors: dict[str, torch.Tensor],
    expected_m: int,
    config: tuple[int, int, int, int, int] | None,
) -> None:
    tensors["masked_m"].fill_(expected_m)
    tensors["out"].fill_(float("nan"))
    kwargs: dict[str, Any] = {
        "compiled_dims": "nk",
        "disable_ue8m0_cast": True,
    }
    if config is not None:
        kwargs["w13_config"] = config
    returned = module.fp8_m_grouped_gemm_nt_masked(
        (tensors["a"], tensors["a_scale"]),
        (tensors["b"], tensors["b_scale"]),
        tensors["out"],
        tensors["masked_m"],
        expected_m,
        **kwargs,
    )
    if returned is not None:
        raise RuntimeError(
            "W13 same-source DeepGEMM no-overlap call changed its None return contract"
        )


def _module_identity(
    record: dict[str, Any],
    cache_snapshot: dict[str, str],
) -> dict[str, Any]:
    return {
        "package": str(Path(record["package"]).resolve()),
        "package_init_sha256": str(record["package_init_sha256"]),
        "shared_object": str(Path(record["shared_object"]).resolve()),
        "shared_object_sha256": str(record["shared_object_sha256"]),
        "jit_cache": str(Path(record["jit_cache"]).resolve()),
        "jit_artifacts": cache_snapshot,
    }


def initialize_w13_decode_after_assignment(
    gpu_id: int,
    server_args: Any,
    *,
    compile_utils_loader: Callable[[], ModuleType],
) -> bool:
    """Initialize opt-in W13 modules after stock config and GPU assignment.

    Returns whether this function imported and configured SGLang compile_utils.
    The caller performs the normal compile_utils setup when it returns False.
    """

    global _STATE
    variant = requested_variant()
    if variant in ("", "0", "off", "false"):
        _STATE = _DispatchState(False, "default_off")
        return False
    if variant not in VARIANT_CONFIGS:
        _STATE = _DispatchState(False, f"unsupported_variant:{variant}")
        raise RuntimeError(f"unsupported requested W13 variant: {variant}")

    manifest_text = os.environ.get("SGLANG_GLM52_W13_DECODE_MANIFEST", "").strip()
    if not manifest_text:
        _STATE = _DispatchState(False, "missing_manifest")
        raise RuntimeError("requested W13 variant requires an exact build manifest")

    with _INITIALIZE_LOCK:
        if _STATE.enabled:
            if _STATE.gpu_id != int(gpu_id) or _STATE.variant != variant:
                raise RuntimeError("W13 runtime was initialized for a different worker")
            return True

        saved_environment = {
            name: os.environ.get(name)
            for name in (
                "DG_JIT_CACHE_DIR",
                "SGLANG_DG_CACHE_DIR",
                "DG_JIT_USE_NVRTC",
                "SGL_DG_USE_NVRTC",
            )
        }
        compile_utils_configured = False
        try:
            current_device = int(torch.cuda.current_device())
            if current_device != int(gpu_id):
                raise RuntimeError(
                    f"W13 initializer current CUDA device {current_device} != assigned {gpu_id}"
                )
            if torch.cuda.get_device_capability(gpu_id) != (10, 0):
                raise RuntimeError("W13 candidate requires an sm_100 worker")

            manifest_path = Path(manifest_text).expanduser().resolve()
            stock_record, _ = _variant_record(manifest_path, "stock")
            candidate_record, _ = _variant_record(manifest_path, "candidate")
            stock_cache = Path(stock_record["jit_cache"]).resolve()
            candidate_cache = Path(candidate_record["jit_cache"]).resolve()
            if stock_cache == candidate_cache:
                raise RuntimeError("W13 stock and candidate JIT caches alias")

            # Deterministic ownership order: bind/warm exact same-source stock
            # before importing compile_utils, whose import otherwise rewrites
            # DG_JIT_CACHE_DIR.  Only then bind/warm the candidate compiler.
            os.environ["DG_JIT_USE_NVRTC"] = "0"
            os.environ["SGL_DG_USE_NVRTC"] = "0"
            os.environ["DG_JIT_CACHE_DIR"] = str(stock_cache)
            os.environ["SGLANG_DG_CACHE_DIR"] = str(stock_cache)
            stock, stock_record, _ = load_variant(
                manifest_path,
                "stock",
                module_name="deep_gemm_w13_stock_production",
            )
            stock_runtime = _set_required_runtime_state(stock, "stock")

            # The first named stock launch binds its lazy Compiler to stock_cache.
            tensors = _allocate_warm_inputs(torch.device("cuda", gpu_id))
            for expected_m in (4, 5, 8, 9):
                _launch_named_config(stock, tensors, expected_m, None)
            torch.cuda.synchronize(gpu_id)
            stock_snapshot = _cache_snapshot(stock_cache)
            if not stock_snapshot:
                raise RuntimeError("W13 stock named warmup produced no JIT artifact")
            del tensors

            compile_utils = compile_utils_loader()
            compile_utils._ENABLE_JIT_DEEPGEMM_PRECOMPILE = False
            compile_utils.update_deep_gemm_config(gpu_id, server_args)
            compile_utils_configured = True

            os.environ["DG_JIT_CACHE_DIR"] = str(candidate_cache)
            os.environ["SGLANG_DG_CACHE_DIR"] = str(candidate_cache)
            candidate, candidate_record, _ = load_variant(
                manifest_path,
                "candidate",
                module_name="deep_gemm_w13_candidate_production",
            )
            candidate_runtime = _set_required_runtime_state(candidate, "candidate")
            independence = _prove_runtime_state_independence(stock, candidate)

            # Reuse one bounded tensor set for candidate warm and post-bind
            # probes.  No broad precompile hook is involved.
            tensors = _allocate_warm_inputs(torch.device("cuda", gpu_id))
            for expected_m in (4, 5, 8, 9):
                _launch_named_config(
                    candidate,
                    tensors,
                    expected_m,
                    VARIANT_CONFIGS[variant],
                )
            torch.cuda.synchronize(gpu_id)
            candidate_snapshot = _cache_snapshot(candidate_cache)
            if not candidate_snapshot:
                raise RuntimeError(
                    "W13 candidate named warmup produced no JIT artifact"
                )

            probe_cache = manifest_path.parent / f"unbound-cache-probe-{os.getpid()}"
            if probe_cache.exists():
                raise RuntimeError(
                    f"W13 cache-bind probe path already exists: {probe_cache}"
                )
            os.environ["DG_JIT_CACHE_DIR"] = str(probe_cache)
            os.environ["SGLANG_DG_CACHE_DIR"] = str(probe_cache)
            _launch_named_config(stock, tensors, 4, None)
            _launch_named_config(
                candidate,
                tensors,
                4,
                VARIANT_CONFIGS[variant],
            )
            torch.cuda.synchronize(gpu_id)
            if probe_cache.exists():
                raise RuntimeError(
                    f"W13 compiler escaped its frozen cache owner: {probe_cache}"
                )
            if _cache_snapshot(stock_cache) != stock_snapshot:
                raise RuntimeError("W13 stock cache changed after freeze probe")
            if _cache_snapshot(candidate_cache) != candidate_snapshot:
                raise RuntimeError("W13 candidate cache changed after freeze probe")
            del tensors

            _STATE = _DispatchState(
                True,
                "ready",
                variant=variant,
                config=VARIANT_CONFIGS[variant],
                gpu_id=int(gpu_id),
                stock_module=stock,
                candidate_module=candidate,
                manifest=str(manifest_path),
                modules={
                    "stock": _module_identity(stock_record, stock_snapshot),
                    "candidate": _module_identity(candidate_record, candidate_snapshot),
                },
                runtime_state={
                    "stock": stock_runtime,
                    "candidate": candidate_runtime,
                },
                state_independence=independence,
                jit_use_nvrtc=bool(int(os.environ["DG_JIT_USE_NVRTC"])),
            )
            logger.info(
                "GLM-5.2 W13 decode candidate ready: variant=%s gpu=%d",
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
                manifest=str(Path(manifest_text).expanduser()),
            )
            logger.error("W13 candidate post-assignment initialization failed: %s", exc)
            raise RuntimeError(
                "requested W13 candidate failed post-assignment initialization"
            ) from exc
        finally:
            _restore_environment(saved_environment)


def _tensor_contract(
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    dtype: torch.dtype,
) -> bool:
    return (
        tensor.is_cuda
        and tensor.dtype == dtype
        and tuple(tensor.shape) == shape
        and tuple(tensor.stride()) == stride
        and tensor.storage_offset() == 0
    )


def _exact_tensor_contract(
    lhs: tuple[torch.Tensor, torch.Tensor],
    rhs: tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
) -> bool:
    a, a_scale = lhs
    b, b_scale = rhs
    if not _tensor_contract(
        a, shape=_A_SHAPE, stride=_A_STRIDE, dtype=torch.float8_e4m3fn
    ):
        return False
    if not _tensor_contract(
        a_scale, shape=_AS_SHAPE, stride=_AS_STRIDE, dtype=torch.int32
    ):
        return False
    if not _tensor_contract(
        b, shape=_B_SHAPE, stride=_B_STRIDE, dtype=torch.float8_e4m3fn
    ):
        return False
    if not _tensor_contract(
        b_scale, shape=_BS_SHAPE, stride=_BS_STRIDE, dtype=torch.int32
    ):
        return False
    if not _tensor_contract(
        out, shape=_OUT_SHAPE, stride=_OUT_STRIDE, dtype=torch.bfloat16
    ):
        return False
    if not _tensor_contract(
        masked_m, shape=_MASK_SHAPE, stride=_MASK_STRIDE, dtype=torch.int32
    ):
        return False
    device = a.device
    return all(
        tensor.device == device for tensor in (a_scale, b, b_scale, out, masked_m)
    )


def is_exact_w13_tensor_call(
    lhs: tuple[torch.Tensor, torch.Tensor],
    rhs: tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
) -> bool:
    """Identify W13 tensors so generic overlays cannot intercept this ABI."""

    return _exact_tensor_contract(lhs, rhs, out, masked_m)


def _marker_matches(
    marker: W13DecodeForwardMarker | None,
    expected_m: int,
) -> bool:
    return bool(
        marker is not None
        and expected_m
        in EXPECTED_M_BY_TOKEN_BUCKET.get(marker.token_bucket, frozenset())
    )


def _contract_matches(
    lhs: tuple[torch.Tensor, torch.Tensor],
    rhs: tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    overlap_args: Any | None,
    max_block_n: int,
    recipe_a: tuple[int, int] | None,
    recipe_b: tuple[int, int] | None,
) -> bool:
    return bool(
        overlap_args is None
        and recipe_a is None
        and recipe_b is None
        and expected_m in (4, 5, 8, 9)
        and max_block_n == 256
        and _exact_tensor_contract(lhs, rhs, out, masked_m)
    )


def try_dispatch_w13_decode(
    lhs: tuple[torch.Tensor, torch.Tensor],
    rhs: tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    *,
    overlap_args: Any | None,
    max_block_n: int,
    recipe_a: tuple[int, int] | None,
    recipe_b: tuple[int, int] | None,
) -> bool:
    """Launch exactly once only under initialized, exact production metadata."""

    state = _STATE
    if not state.enabled:
        return False
    marker = get_w13_decode_forward_marker()
    if not _marker_matches(marker, expected_m):
        return False
    if not _contract_matches(
        lhs,
        rhs,
        out,
        masked_m,
        expected_m,
        overlap_args,
        max_block_n,
        recipe_a,
        recipe_b,
    ):
        return False
    if lhs[0].device.index != state.gpu_id:
        return False
    assert state.candidate_module is not None
    returned = state.candidate_module.fp8_m_grouped_gemm_nt_masked(
        lhs,
        rhs,
        out,
        masked_m,
        expected_m,
        compiled_dims="nk",
        disable_ue8m0_cast=True,
        w13_config=state.config,
    )
    if returned is not None:
        raise RuntimeError("W13 candidate violated the stock None return contract")
    return True


def dispatch_state() -> dict[str, Any]:
    """Read-only startup/debug identity; never called from the selected hot path."""

    state = _STATE
    return {
        "enabled": state.enabled,
        "reason": state.reason,
        "variant": state.variant,
        "config": list(state.config),
        "gpu_id": state.gpu_id,
        "manifest": state.manifest,
        "modules": state.modules,
        "runtime_state": state.runtime_state,
        "state_independence": state.state_independence,
        "jit_use_nvrtc": state.jit_use_nvrtc,
    }
