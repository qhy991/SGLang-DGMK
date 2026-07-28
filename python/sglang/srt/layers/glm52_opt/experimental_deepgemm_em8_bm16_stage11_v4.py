"""Task26 exact-post1 W2 em8/BM16/stage11 side-by-side runtime."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from packaging.version import Version
from sglang.srt.model_executor.forward_batch_info import ForwardMode

_REPO_ROOT = Path(__file__).resolve().parents[5]
_OVERLAY_DIR = _REPO_ROOT / "third_party" / "deepgemm_w2_em8_bm16_stage11_v4"
_READY_TOOL = _OVERLAY_DIR / "ready_bundle.py"
_KERNEL_HARNESS_ROOT = Path(
    os.environ.get(
        "TASK26_V4_KERNEL_HARNESS_ROOT",
        _REPO_ROOT.parent / "kernel-harness",
    )
).resolve()
BASE_COMMIT = "edcf77b276965de8f03cdc47c23f01b08bf7c7ab"
BASE_VERSION = Version("0.1.4.post1")
VARIANT_NAME = "em8_bm16_stage11"
VARIANT_VERSION = 4
PREDECLARED_FALLBACK = "em8_bm16_stage10"
CANDIDATE_IMPORT_NAME = "deep_gemm_glm52_w2_em8_bm16_stage11_v4"
BUILD_ID = (
    "glm52-task26-em8-bm16-stage11-v4:"
    "sgl-deep-gemm-0.1.4.post1@"
    "edcf77b276965de8f03cdc47c23f01b08bf7c7ab:"
    "sm100:e32:m1024:k2048:n6144:expected-m8:"
    "bm16:stages11:pdl1:sms148:packed-ue8m0:"
    "no-recipe:no-overlap"
)
JIT_IDENTITY = "sm100_m_grouped_fp8_fp4_gemm_masked_1d1d_glm52_w2_em8_bm16_stage11_v4"
FORWARD_BUCKET = (32, 8)

_FORWARD_STATE: ContextVar[tuple[ForwardMode, int] | None] = ContextVar(
    "glm52_w2_em8_bm16_stage11_forward_state",
    default=None,
)


def _strict_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-bool int")
    return value


@contextmanager
def forward_context(mode: ForwardMode, local_m: int) -> Iterator[None]:
    token = _FORWARD_STATE.set((mode, _strict_int(local_m, "local_m")))
    try:
        yield
    finally:
        _FORWARD_STATE.reset(token)


def current_forward_state() -> tuple[ForwardMode, int] | None:
    return _FORWARD_STATE.get()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_VERIFIED_READY: dict[str, Any] = {}


def expected_ready_path() -> Path:
    completed = subprocess.run(
        [
            sys.executable,
            str(_READY_TOOL),
            "locate",
            "--sglang-root",
            str(_REPO_ROOT),
            "--print",
            "ready",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"stage11-v4 READY lookup failed: {detail}")
    path = Path(completed.stdout.strip()).resolve()
    override = os.environ.get("SGLANG_GLM52_W2_EM8_BM16_STAGE11_V4_READY", "").strip()
    if override and Path(override).resolve() != path:
        raise RuntimeError(
            "em8/BM16/stage11-v4 READY override is not the canonical bundle: "
            f"{Path(override).resolve()} != {path}"
        )
    return path


def expected_manifest_path() -> Path:
    return (expected_ready_path().parent / "manifest.json").resolve()


def _manifest_path() -> Path:
    expected = expected_manifest_path()
    override = os.environ.get(
        "SGLANG_GLM52_W2_EM8_BM16_STAGE11_V4_MANIFEST", ""
    ).strip()
    if override and Path(override).resolve() != expected:
        raise RuntimeError(
            "em8/BM16/stage11 manifest override is not the exact task "
            f"artifact: {Path(override).resolve()} != {expected}"
        )
    return expected


def _verify_manifest() -> Path:
    global _VERIFIED_READY
    path = _manifest_path()
    ready_path = (path.parent / "READY").resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"em8/BM16/stage11-v4 manifest missing from READY bundle: {path}"
        )
    completed = subprocess.run(
        [
            sys.executable,
            str(_READY_TOOL),
            "verify",
            "--ready",
            str(ready_path),
            "--sglang-root",
            str(_REPO_ROOT),
            "--kernel-harness-root",
            str(_KERNEL_HARNESS_ROOT),
            "--check-env",
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"em8/BM16/stage11-v4 READY verification failed: {detail}")
    try:
        evidence = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("stage11-v4 READY verifier returned malformed JSON") from exc
    if (
        not isinstance(evidence, dict)
        or Path(str(evidence.get("manifest_path", ""))).resolve() != path
    ):
        raise RuntimeError("stage11-v4 READY verifier returned the wrong manifest")
    _VERIFIED_READY = evidence
    return path


def _verify_cache_contract(manifest: dict[str, Any]) -> None:
    cache_paths = manifest["runtime_contract"]["cache_paths"]
    for name, expected_text in cache_paths.items():
        value = os.environ.get(name)
        if not value:
            raise RuntimeError(f"{name} is not exported")
        actual = Path(value).resolve()
        expected = Path(expected_text).resolve()
        if actual != expected:
            raise RuntimeError(f"{name} drifted: {actual} != {expected}")
    if os.environ["SGLANG_DG_CACHE_DIR"] != os.environ["DG_JIT_CACHE_DIR"]:
        raise RuntimeError(
            "SGLANG_DG_CACHE_DIR and DG_JIT_CACHE_DIR must remain identical"
        )


def _verify_module_package(
    module: Any,
    manifest: dict[str, Any],
    *,
    role: str,
) -> None:
    record = manifest[role]
    bundle_dir = Path(str(manifest.get("_bundle_dir", ""))).resolve()
    package = (bundle_dir / str(record["package_relpath"])).resolve()
    if bundle_dir not in package.parents:
        raise RuntimeError(f"{role} DeepGEMM package escapes the READY bundle")
    actual_init = Path(getattr(module, "__file__", "") or "").resolve()
    expected_init = package / "__init__.py"
    if actual_init != expected_init:
        raise RuntimeError(
            f"{role} DeepGEMM module path mismatch: {actual_init} != {expected_init}"
        )
    try:
        actual_version = Version(str(module.__version__))
    except Exception as exc:
        raise RuntimeError(f"{role} DeepGEMM VERSION is unavailable") from exc
    if actual_version != BASE_VERSION:
        raise RuntimeError(
            f"{role} DeepGEMM VERSION mismatch: {actual_version} != {BASE_VERSION}"
        )
    version_file = package / "VERSION"
    extension = package / "_C.so"
    if _sha256(expected_init) != record["init_sha256"]:
        raise RuntimeError(f"{role} DeepGEMM __init__.py hash drifted")
    if _sha256(version_file) != record["version_sha256"]:
        raise RuntimeError(f"{role} DeepGEMM VERSION hash drifted")
    if _sha256(extension) != record["extension_sha256"]:
        raise RuntimeError(f"{role} DeepGEMM _C.so hash drifted")
    if getattr(module, "_C", None) is None:
        raise RuntimeError(f"{role} DeepGEMM _C module is unavailable")


def _ensure_stock(
    manifest: dict[str, Any],
    *,
    verify_consumers: bool,
) -> Any:
    stock = importlib.import_module("deep_gemm")
    _verify_module_package(stock, manifest, role="stock")
    _verify_cache_contract(manifest)
    if verify_consumers:
        for name in (
            "sglang.srt.layers.deep_gemm_wrapper.entrypoint",
            "sglang.srt.layers.deep_gemm_wrapper.compile_utils",
        ):
            consumer = sys.modules.get(name)
            if consumer is None:
                raise RuntimeError(f"required stock consumer is not imported: {name}")
            if getattr(consumer, "deep_gemm", None) is not stock:
                raise RuntimeError(f"{name} is not bound to manifest exact-post1 stock")
    return stock


_CANDIDATE_MODULE: Any = None


def _load_candidate(manifest: dict[str, Any], stock: Any) -> Any:
    global _CANDIDATE_MODULE
    bundle_dir = Path(str(manifest.get("_bundle_dir", ""))).resolve()
    package = (bundle_dir / str(manifest["candidate"]["package_relpath"])).resolve()
    if bundle_dir not in package.parents:
        raise RuntimeError("candidate DeepGEMM package escapes the READY bundle")
    init_py = package / "__init__.py"
    module = _CANDIDATE_MODULE or sys.modules.get(CANDIDATE_IMPORT_NAME)
    if module is None:
        spec = importlib.util.spec_from_file_location(
            CANDIDATE_IMPORT_NAME,
            init_py,
            submodule_search_locations=[str(package)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load stage11 DeepGEMM from {init_py}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[CANDIDATE_IMPORT_NAME] = module
        module.__path__ = [str(package)]  # type: ignore[attr-defined]
        module.__package__ = CANDIDATE_IMPORT_NAME
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(CANDIDATE_IMPORT_NAME, None)
            raise
        _CANDIDATE_MODULE = module

    _verify_module_package(module, manifest, role="candidate")
    actual_build_id = getattr(
        module,
        "GLM52_W2_EM8_BM16_STAGE11_V4_BUILD_ID",
        None,
    )
    if actual_build_id != BUILD_ID:
        raise RuntimeError("loaded stage11 module has the wrong build identity")
    launch = getattr(module, "fp8_m_grouped_gemm_nt_masked", None)
    if not callable(launch):
        raise TypeError("loaded stage11 module lacks the masked API")
    parameters = inspect.signature(launch).parameters
    for keyword in (
        "masked_block_m_override",
        "masked_num_stages_override",
    ):
        parameter = parameters.get(keyword)
        if parameter is None or parameter.default != 0:
            raise RuntimeError(f"loaded stage11 masked API lacks optional {keyword}=0")
    if (
        module is stock
        or Path(module.__file__).resolve() == Path(stock.__file__).resolve()
    ):
        raise RuntimeError("stage11 candidate aliased the stock package")
    return module


def _require_runtime_api(module: Any, role: str) -> None:
    required = (
        "get_num_sms",
        "set_num_sms",
        "get_tc_util",
        "set_tc_util",
        "get_pdl",
        "set_pdl",
    )
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        raise RuntimeError(f"{role} DeepGEMM runtime API is incomplete: {missing}")


@dataclass(frozen=True)
class PreparedContract:
    candidate_module: Any = field(repr=False, compare=False)
    launch: Callable[..., Any] = field(repr=False, compare=False)
    device_index: int
    compute_capability: tuple[int, int]
    physical_num_sms: int
    stock_initial_num_sms: int
    candidate_initial_num_sms: int
    stock_num_sms: int
    candidate_num_sms: int
    stock_initial_tc_util: int
    candidate_initial_tc_util: int
    stock_tc_util: int
    candidate_tc_util: int
    stock_initial_pdl: bool
    candidate_initial_pdl: bool
    stock_pdl: bool
    candidate_pdl: bool
    runtime_modules_distinct: bool
    runtime_extension_modules_distinct: bool
    independence_probe_num_sms: int
    independence_probe_tc_util: int
    stock_module_path: str
    candidate_module_path: str
    stock_extension_sha256: str
    candidate_extension_sha256: str
    manifest_path: str
    manifest_sha256: str
    dg_jit_cache_dir: str
    sglang_dg_cache_dir: str
    base_commit: str
    base_version: str
    cutlass_commit: str
    fmt_commit: str
    ready_path: str = ""
    ready_sha256: str = ""
    ready_contract_sha256: str = ""
    ready_bundle_digest: str = ""
    source_replay_path: str = ""
    source_replay_sha256: str = ""
    build_provenance_path: str = ""
    build_provenance_sha256: str = ""
    stock_package_tree_sha256: str = ""
    candidate_package_tree_sha256: str = ""
    ready_verified_before_runtime: bool = True
    bundle_contract: str = "content-addressed-ready-v1"
    build_phase: str = "cpu-only-before-gpu-lease"
    variant_name: str = VARIANT_NAME
    variant_version: int = VARIANT_VERSION
    predeclared_fallback: str = PREDECLARED_FALLBACK
    fallback_eligible: bool = False
    decode_m: int = 32
    expected_m: int = 8
    masked_block_m_override: int = 16
    masked_num_stages_override: int = 11
    candidate_jit_identity: str = JIT_IDENTITY
    build_id: str = BUILD_ID
    pipeline_smem_per_stage_bytes: int = 18432
    pipeline_fixed_bytes: int = 9004
    stock_pipeline_num_stages: int = 12
    stock_pipeline_smem_bytes: int = 230188
    candidate_pipeline_num_stages: int = 11
    candidate_pipeline_smem_bytes: int = 211756
    two_ctas_per_sm_enabled: bool = False
    performance_hypothesis: str = "reduced-pipeline-pressure-falsifiable"

    def evidence(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("candidate_module", "launch")
        }

    def forward_context(self, mode: ForwardMode, local_m: int):
        return forward_context(mode, local_m)

    def current_forward_state(self) -> tuple[ForwardMode, int] | None:
        return current_forward_state()


_REQUESTED = False
_PREPARED: PreparedContract | None = None
_PREPARE_ERROR: str | None = None


def request() -> None:
    global _REQUESTED, _PREPARED, _PREPARE_ERROR, _VERIFIED_READY
    _REQUESTED = True
    _PREPARED = None
    _PREPARE_ERROR = None
    _VERIFIED_READY = {}


def prepared_contract() -> PreparedContract | None:
    return _PREPARED


def prepare_error() -> str | None:
    return _PREPARE_ERROR


def prepare_deep_gemm(gpu_id: int) -> PreparedContract:
    """Establish the exact independent PDL/148-SM/no-overlap contract."""
    global _PREPARED, _PREPARE_ERROR
    request()
    try:
        manifest_path = _verify_manifest()
        manifest = json.loads(manifest_path.read_text())
        manifest["_bundle_dir"] = str(manifest_path.parent.resolve())
        for role in ("stock", "candidate"):
            expected_tree = (
                manifest.get(role, {})
                .get("package_tree", {})
                .get("tree_sha256")
            )
            observed_tree = _VERIFIED_READY.get(
                f"{role}_package_tree_sha256"
            )
            if (
                not isinstance(expected_tree, str)
                or observed_tree != expected_tree
            ):
                raise RuntimeError(
                    f"stage11-v4 {role} package-tree identity mismatch"
                )
        if manifest.get("variant") != {
            "name": VARIANT_NAME,
            "version": VARIANT_VERSION,
            "predeclared_fallback": PREDECLARED_FALLBACK,
            "fallback_eligible": False,
        }:
            raise RuntimeError("stage11 manifest variant identity mismatch")
        if manifest["candidate_api"].get("jit_identity") != JIT_IDENTITY:
            raise RuntimeError("stage11 manifest JIT identity mismatch")
        stock = _ensure_stock(manifest, verify_consumers=True)
        candidate = _load_candidate(manifest, stock)
        _verify_cache_contract(manifest)
        _require_runtime_api(stock, "stock")
        _require_runtime_api(candidate, "candidate")
        if stock is candidate or stock._C is candidate._C:
            raise RuntimeError(
                "stock and stage11 candidate must own independent runtimes"
            )

        current_device = int(torch.cuda.current_device())
        if current_device != int(gpu_id):
            raise RuntimeError(
                "assigned CUDA device mismatch: "
                f"current={current_device}, gpu_id={gpu_id}"
            )
        capability = tuple(torch.cuda.get_device_capability(gpu_id))
        properties = torch.cuda.get_device_properties(gpu_id)
        physical_num_sms = int(properties.multi_processor_count)
        if capability != (10, 0) or physical_num_sms != 148:
            raise RuntimeError(
                "stage11 requires exact B200 SM100/148-SM readiness, got "
                f"capability={capability}, num_sms={physical_num_sms}"
            )

        stock_initial_pdl = stock.get_pdl()
        stock_initial_num_sms = int(stock.get_num_sms())
        stock_initial_tc_util = int(stock.get_tc_util())
        candidate_initial_pdl = candidate.get_pdl()
        candidate_initial_num_sms = int(candidate.get_num_sms())
        candidate_initial_tc_util = int(candidate.get_tc_util())

        stock.set_pdl(True)
        stock.set_num_sms(148)
        stock.set_tc_util(stock_initial_tc_util)
        stock_pdl = stock.get_pdl()
        stock_num_sms = int(stock.get_num_sms())
        stock_tc_util = int(stock.get_tc_util())
        if (
            stock_pdl is not True
            or stock_num_sms != 148
            or stock_tc_util != stock_initial_tc_util
        ):
            raise RuntimeError("stock DeepGEMM runtime contract readback failed")

        independence_probe_num_sms = 146
        independence_probe_tc_util = (
            stock_tc_util + 1 if stock_tc_util < 100 else stock_tc_util - 1
        )
        candidate.set_pdl(False)
        candidate.set_num_sms(independence_probe_num_sms)
        candidate.set_tc_util(independence_probe_tc_util)
        if (
            candidate.get_pdl() is not False
            or int(candidate.get_num_sms()) != independence_probe_num_sms
            or int(candidate.get_tc_util()) != independence_probe_tc_util
        ):
            raise RuntimeError("candidate DeviceRuntime independence probe failed")
        if (
            stock.get_pdl() is not True
            or int(stock.get_num_sms()) != stock_num_sms
            or int(stock.get_tc_util()) != stock_tc_util
        ):
            raise RuntimeError("candidate DeviceRuntime probe mutated stock state")

        candidate.set_num_sms(stock_num_sms)
        candidate.set_tc_util(stock_tc_util)
        candidate.set_pdl(stock_pdl)
        candidate_num_sms = int(candidate.get_num_sms())
        candidate_tc_util = int(candidate.get_tc_util())
        candidate_pdl = candidate.get_pdl()
        if (
            candidate_num_sms != 148
            or candidate_tc_util != stock_tc_util
            or candidate_pdl is not True
        ):
            raise RuntimeError("candidate runtime synchronization failed")
        if (
            stock.get_pdl() is not True
            or int(stock.get_num_sms()) != stock_num_sms
            or int(stock.get_tc_util()) != stock_tc_util
        ):
            raise RuntimeError("candidate synchronization mutated stock state")

        contract = PreparedContract(
            candidate_module=candidate,
            launch=candidate.fp8_m_grouped_gemm_nt_masked,
            device_index=current_device,
            compute_capability=(10, 0),
            physical_num_sms=physical_num_sms,
            stock_initial_num_sms=stock_initial_num_sms,
            candidate_initial_num_sms=candidate_initial_num_sms,
            stock_num_sms=stock_num_sms,
            candidate_num_sms=candidate_num_sms,
            stock_initial_tc_util=stock_initial_tc_util,
            candidate_initial_tc_util=candidate_initial_tc_util,
            stock_tc_util=stock_tc_util,
            candidate_tc_util=candidate_tc_util,
            stock_initial_pdl=bool(stock_initial_pdl),
            candidate_initial_pdl=bool(candidate_initial_pdl),
            stock_pdl=True,
            candidate_pdl=True,
            runtime_modules_distinct=True,
            runtime_extension_modules_distinct=True,
            independence_probe_num_sms=independence_probe_num_sms,
            independence_probe_tc_util=independence_probe_tc_util,
            stock_module_path=str(Path(stock.__file__).resolve()),
            candidate_module_path=str(Path(candidate.__file__).resolve()),
            stock_extension_sha256=manifest["stock"]["extension_sha256"],
            candidate_extension_sha256=manifest["candidate"]["extension_sha256"],
            manifest_path=str(manifest_path),
            manifest_sha256=_sha256(manifest_path),
            dg_jit_cache_dir=os.environ["DG_JIT_CACHE_DIR"],
            sglang_dg_cache_dir=os.environ["SGLANG_DG_CACHE_DIR"],
            base_commit=manifest["base"]["commit"],
            base_version=manifest["base"]["version"],
            cutlass_commit=manifest["base"]["submodules"]["third-party/cutlass"],
            fmt_commit=manifest["base"]["submodules"]["third-party/fmt"],
            ready_path=str(_VERIFIED_READY.get("ready_path", "")),
            ready_sha256=str(_VERIFIED_READY.get("ready_sha256", "")),
            ready_contract_sha256=str(_VERIFIED_READY.get("contract_sha256", "")),
            ready_bundle_digest=str(_VERIFIED_READY.get("bundle_digest", "")),
            source_replay_path=str(_VERIFIED_READY.get("source_replay_path", "")),
            source_replay_sha256=str(_VERIFIED_READY.get("source_replay_sha256", "")),
            build_provenance_path=str(_VERIFIED_READY.get("build_provenance_path", "")),
            build_provenance_sha256=str(
                _VERIFIED_READY.get("build_provenance_sha256", "")
            ),
            stock_package_tree_sha256=str(
                _VERIFIED_READY.get("stock_package_tree_sha256", "")
            ),
            candidate_package_tree_sha256=str(
                _VERIFIED_READY.get("candidate_package_tree_sha256", "")
            ),
        )
        _PREPARED = contract
        return contract
    except Exception as exc:
        _PREPARED = None
        _PREPARE_ERROR = f"{type(exc).__name__}: {exc}"
        raise


def _tensor_meta_matches(
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    stride: tuple[int, ...],
    contiguous: bool,
) -> bool:
    return bool(
        isinstance(tensor, torch.Tensor)
        and tensor.layout == torch.strided
        and tuple(tensor.shape) == shape
        and tensor.dtype == dtype
        and tuple(tensor.stride()) == stride
        and tensor.is_contiguous() is contiguous
    )


@dataclass
class LayerContract:
    runtime: PreparedContract | None
    w2_weight: torch.Tensor
    w2_scale: torch.Tensor
    static_eligible: bool
    static_reason: str
    callsite_checked: bool = False
    callsite_eligible: bool = False
    callsite_reason: str = "not-checked"


def create_layer_contract(
    *,
    w2_weight: torch.Tensor,
    w2_scale: torch.Tensor,
    block_shape: list[int] | None,
    deep_gemm_backend: bool,
    is_fp4_experts: bool,
    use_mxfp8: bool,
) -> LayerContract | None:
    if not _REQUESTED:
        return None
    runtime = _PREPARED
    eligible = all(
        (
            runtime is not None,
            deep_gemm_backend,
            block_shape == [128, 128],
            not is_fp4_experts,
            not use_mxfp8,
            _tensor_meta_matches(
                w2_weight,
                shape=(32, 6144, 2048),
                dtype=torch.float8_e4m3fn,
                stride=(12582912, 2048, 1),
                contiguous=True,
            ),
            _tensor_meta_matches(
                w2_scale,
                shape=(32, 6144, 4),
                dtype=torch.int32,
                stride=(24576, 1, 6144),
                contiguous=False,
            ),
            getattr(w2_scale, "format_ue8m0", False) is True,
        )
    )
    if not eligible:
        raise RuntimeError(
            "explicit em8/BM16/stage11-v4 profile rejected the W2 layer/runtime ABI"
        )
    return LayerContract(
        runtime=runtime,
        w2_weight=w2_weight,
        w2_scale=w2_scale,
        static_eligible=eligible,
        static_reason="ready" if eligible else "static-abi-or-runtime",
    )


def prepare_callsite_contract(
    contract: LayerContract | None,
    lhs: tuple[torch.Tensor, torch.Tensor],
    rhs: tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
    *,
    expected_m: int,
    recipe_a: tuple[int, int] | None,
    recipe_b: tuple[int, int] | None,
    overlap_args: Any | None,
) -> bool:
    """Latch exact immutable ABI once; unsupported states remain stock."""
    if contract is None:
        raise RuntimeError("stage11-v4 layer contract is missing")
    expected_m = _strict_int(expected_m, "expected_m")
    if expected_m != 8:
        return False
    runtime = contract.runtime
    if runtime is None or runtime is not _PREPARED:
        raise RuntimeError("stage11-v4 prepared runtime token is missing or stale")
    state = runtime.current_forward_state()
    if state is None:
        raise RuntimeError("stage11-v4 forward context is missing")
    local_m = _strict_int(state[1], "local_m")
    if state[0] is not ForwardMode.DECODE or (local_m, expected_m) != FORWARD_BUCKET:
        return False
    if recipe_a is not None or recipe_b is not None or overlap_args is not None:
        raise RuntimeError(
            "exact M32/em8 stage11-v4 route does not support recipe or overlap"
        )
    if contract.callsite_checked:
        if not contract.callsite_eligible:
            raise RuntimeError("exact M32/em8 stage11-v4 ABI was previously rejected")
        return True

    x_fp8, x_scale = lhs
    devices = (
        x_fp8.device,
        x_scale.device,
        rhs[0].device,
        rhs[1].device,
        out.device,
        masked_m.device,
    )
    eligible = bool(
        contract.static_eligible
        and runtime is not None
        and runtime is _PREPARED
        and rhs[0] is contract.w2_weight
        and rhs[1] is contract.w2_scale
        and _tensor_meta_matches(
            x_fp8,
            shape=(32, 1024, 2048),
            dtype=torch.float8_e4m3fn,
            stride=(2097152, 2048, 1),
            contiguous=True,
        )
        and _tensor_meta_matches(
            x_scale,
            shape=(32, 1024, 4),
            dtype=torch.int32,
            stride=(4096, 1, 1024),
            contiguous=False,
        )
        and _tensor_meta_matches(
            out,
            shape=(32, 1024, 6144),
            dtype=torch.bfloat16,
            stride=(6291456, 6144, 1),
            contiguous=True,
        )
        and _tensor_meta_matches(
            masked_m,
            shape=(32,),
            dtype=torch.int32,
            stride=(1,),
            contiguous=True,
        )
        and all(device == devices[0] for device in devices[1:])
        and devices[0].type == "cuda"
        and devices[0].index == runtime.device_index
    )
    contract.callsite_checked = True
    contract.callsite_eligible = eligible
    contract.callsite_reason = "ready" if eligible else "callsite-abi"
    if not eligible:
        raise RuntimeError("explicit M32/em8 stage11-v4 callsite ABI rejected")
    return True
