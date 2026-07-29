"""Load source-scoped DeepGEMM overlays alongside the stock package."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Tuple

import torch
from packaging.version import Version

from sglang.srt.layers.glm52_opt.config import deepgemm_overlay_path, deepgemm_variant
from sglang.srt.model_executor.forward_batch_info import ForwardMode

_REPO_ROOT = Path(__file__).resolve().parents[5]
_W2_BM16_DIR = _REPO_ROOT / "third_party" / "deepgemm_w2_bm16"
_W2_BM16_BASE_COMMIT = "edcf77b276965de8f03cdc47c23f01b08bf7c7ab"
_W2_BM16_VERSION = Version("0.1.4.post1")
_W2_BM16_IMPORT_NAME = "deep_gemm_glm52_w2_bm16"
W2_BM16_BUILD_ID = (
    "glm52-w2-bm16-v2:sgl-deep-gemm-0.1.4.post1@"
    "edcf77b276965de8f03cdc47c23f01b08bf7c7ab:"
    "sm100:e32:m1024:k2048:n6144:bm16:pdl1:sms148:"
    "no-recipe:no-overlap"
)
W2_BM16_EXPECTED_M = frozenset((4, 5, 8, 9))
W2_BM16_FORWARD_BUCKETS = frozenset(
    ((16, 4), (16, 5), (32, 8), (32, 9))
)
_W2_BM16_FORWARD_STATE: ContextVar[Optional[Tuple[ForwardMode, int]]] = (
    ContextVar("glm52_w2_bm16_forward_state", default=None)
)


@contextmanager
def w2_bm16_forward_context(
    mode: ForwardMode, local_m: int
) -> Iterator[None]:
    """Publish task-private W2 scope only while the exact profile is armed."""
    token = _W2_BM16_FORWARD_STATE.set((mode, int(local_m)))
    try:
        yield
    finally:
        _W2_BM16_FORWARD_STATE.reset(token)


def get_w2_bm16_forward_state() -> Optional[Tuple[ForwardMode, int]]:
    return _W2_BM16_FORWARD_STATE.get()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Existing general-purpose experimental-overlay support.
def _manifest_path() -> Path:
    overlay = deepgemm_overlay_path()
    env = os.environ.get("SGLANG_GLM52_DEEPGEMM_MANIFEST")
    if env:
        return Path(env)
    return overlay / "manifest.json"


def _read_manifest() -> dict[str, Any]:
    path = _manifest_path()
    if not path.is_file():
        raise FileNotFoundError(f"DeepGEMM overlay manifest missing: {path}")
    return json.loads(path.read_text())


def _expected_w2_bm16_manifest_path() -> Path:
    source_sha = _sha256(_W2_BM16_DIR / "source.patch")
    build_tool_sha = _sha256(_W2_BM16_DIR / "build_tool.patch")
    key = (
        f"{_W2_BM16_BASE_COMMIT[:12]}-"
        f"{source_sha[:12]}-{build_tool_sha[:12]}"
    )
    return (
        _REPO_ROOT
        / "build"
        / "deepgemm-w2-bm16-overlays"
        / key
        / "manifest.json"
    ).resolve()


def _w2_bm16_manifest_path() -> Path:
    expected = _expected_w2_bm16_manifest_path()
    override = os.environ.get("SGLANG_GLM52_W2_BM16_MANIFEST", "").strip()
    if override and Path(override).resolve() != expected:
        raise RuntimeError(
            "W2/BM16 manifest override is not the exact task artifact: "
            f"{Path(override).resolve()} != {expected}"
        )
    return expected


def _verify_w2_bm16_manifest() -> Path:
    """Run the CPU-only artifact/source/cache verifier during worker setup."""
    path = _w2_bm16_manifest_path()
    if not path.is_file():
        raise FileNotFoundError(
            "W2/BM16 overlay manifest missing; run "
            f"{_W2_BM16_DIR / 'build_overlay.sh'}: {path}"
        )
    command = [
        sys.executable,
        str(_W2_BM16_DIR / "overlay_manifest.py"),
        "verify",
        "--manifest",
        str(path),
        "--check-env",
        "--check-provenance",
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"W2/BM16 overlay verification failed: {detail}")
    return path


def _read_w2_bm16_manifest() -> dict[str, Any]:
    return json.loads(_verify_w2_bm16_manifest().read_text())


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
    package = Path(record["package_dir"]).resolve()
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
    if actual_version != _W2_BM16_VERSION:
        raise RuntimeError(
            f"{role} DeepGEMM VERSION mismatch: {actual_version} "
            f"!= {_W2_BM16_VERSION}"
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


def ensure_stock_deep_gemm(
    manifest: Optional[dict[str, Any]] = None,
    *,
    verify_consumers: bool = False,
):
    """Verify the already-bound normal import; never mutate import state."""
    if manifest is None:
        manifest = _read_w2_bm16_manifest()
    stock = importlib.import_module("deep_gemm")
    _verify_module_package(stock, manifest, role="stock")
    _verify_cache_contract(manifest)

    if verify_consumers:
        consumers = (
            "sglang.srt.layers.deep_gemm_wrapper.entrypoint",
            "sglang.srt.layers.deep_gemm_wrapper.compile_utils",
        )
        for name in consumers:
            consumer = sys.modules.get(name)
            if consumer is None:
                raise RuntimeError(f"required stock consumer is not imported: {name}")
            if getattr(consumer, "deep_gemm", None) is not stock:
                raise RuntimeError(
                    f"{name} is not bound to the manifest stock DeepGEMM module"
                )
    return stock


_W2_BM16_CANDIDATE_MODULE: Any = None


def _load_w2_bm16_candidate(
    manifest: dict[str, Any],
    stock: Any,
):
    global _W2_BM16_CANDIDATE_MODULE
    package = Path(manifest["candidate"]["package_dir"]).resolve()
    init_py = package / "__init__.py"

    module = _W2_BM16_CANDIDATE_MODULE
    if module is None:
        module = sys.modules.get(_W2_BM16_IMPORT_NAME)
    if module is None:
        spec = importlib.util.spec_from_file_location(
            _W2_BM16_IMPORT_NAME,
            init_py,
            submodule_search_locations=[str(package)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load W2/BM16 DeepGEMM from {init_py}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[_W2_BM16_IMPORT_NAME] = module
        module.__path__ = [str(package)]  # type: ignore[attr-defined]
        module.__package__ = _W2_BM16_IMPORT_NAME
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(_W2_BM16_IMPORT_NAME, None)
            raise
        _W2_BM16_CANDIDATE_MODULE = module

    _verify_module_package(module, manifest, role="candidate")
    if getattr(module, "GLM52_W2_BM16_BUILD_ID", None) != W2_BM16_BUILD_ID:
        raise RuntimeError("loaded W2/BM16 module has the wrong build identity")
    launch = getattr(module, "fp8_m_grouped_gemm_nt_masked", None)
    if not callable(launch):
        raise RuntimeError("loaded W2/BM16 module lacks the masked API")
    override = inspect.signature(launch).parameters.get("masked_block_m_override")
    if override is None or override.default != 0:
        raise RuntimeError(
            "loaded W2/BM16 masked API lacks optional "
            "masked_block_m_override=0"
        )
    if module is stock or Path(module.__file__).resolve() == Path(stock.__file__).resolve():
        raise RuntimeError("W2/BM16 candidate aliased the stock package")
    return module


def get_w2_bm16_deep_gemm():
    """Load the candidate beside an already verified exact-post1 stock module."""
    manifest = _read_w2_bm16_manifest()
    stock = ensure_stock_deep_gemm(manifest)
    return _load_w2_bm16_candidate(manifest, stock)


@dataclass(frozen=True)
class W2BM16PreparedContract:
    """Immutable startup proof consumed by the launch hot path."""

    candidate_module: Any = field(repr=False, compare=False)
    launch: Callable[..., Any] = field(repr=False, compare=False)
    device_index: int
    compute_capability: Tuple[int, int]
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
    build_id: str = W2_BM16_BUILD_ID

    def evidence(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("candidate_module", "launch")
        }

    def forward_context(
        self, mode: ForwardMode, local_m: int
    ):
        return w2_bm16_forward_context(mode, local_m)

    def current_forward_state(self) -> Optional[Tuple[ForwardMode, int]]:
        return get_w2_bm16_forward_state()


_W2_BM16_REQUESTED = False
_W2_BM16_PREPARED: Optional[W2BM16PreparedContract] = None
_W2_BM16_PREPARE_ERROR: Optional[str] = None


def w2_bm16_requested() -> bool:
    return _W2_BM16_REQUESTED


def get_w2_bm16_prepared_contract() -> Optional[W2BM16PreparedContract]:
    return _W2_BM16_PREPARED


def get_w2_bm16_prepare_error() -> Optional[str]:
    return _W2_BM16_PREPARE_ERROR


def request_w2_bm16_deep_gemm() -> None:
    """Mark the profile requested before any fallible preparation work."""
    global _W2_BM16_REQUESTED
    global _W2_BM16_PREPARED
    global _W2_BM16_PREPARE_ERROR
    _W2_BM16_REQUESTED = True
    _W2_BM16_PREPARED = None
    _W2_BM16_PREPARE_ERROR = None


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


def prepare_w2_bm16_deep_gemm(gpu_id: int) -> W2BM16PreparedContract:
    """Prepare and freeze the exact default 148-SM/no-overlap runtime contract."""
    global _W2_BM16_PREPARED
    global _W2_BM16_PREPARE_ERROR
    request_w2_bm16_deep_gemm()
    try:
        manifest_path = _verify_w2_bm16_manifest()
        manifest = json.loads(manifest_path.read_text())
        stock = ensure_stock_deep_gemm(manifest, verify_consumers=True)
        candidate = _load_w2_bm16_candidate(manifest, stock)
        _verify_cache_contract(manifest)
        _require_runtime_api(stock, "stock")
        _require_runtime_api(candidate, "candidate")
        if stock is candidate or stock._C is candidate._C:
            raise RuntimeError(
                "stock and candidate must own independent Python and _C runtimes"
            )

        current_device = int(torch.cuda.current_device())
        if current_device != int(gpu_id):
            raise RuntimeError(
                f"assigned CUDA device mismatch: current={current_device}, gpu_id={gpu_id}"
            )
        capability = tuple(torch.cuda.get_device_capability(gpu_id))
        properties = torch.cuda.get_device_properties(gpu_id)
        physical_num_sms = int(properties.multi_processor_count)
        if capability != (10, 0) or physical_num_sms != 148:
            raise RuntimeError(
                "W2/BM16 requires exact B200 SM100/148-SM readiness, got "
                f"capability={capability}, num_sms={physical_num_sms}"
            )

        stock_initial_pdl = stock.get_pdl()
        stock_initial_num_sms = int(stock.get_num_sms())
        stock_initial_tc_util = int(stock.get_tc_util())
        candidate_initial_pdl = candidate.get_pdl()
        candidate_initial_num_sms = int(candidate.get_num_sms())
        candidate_initial_tc_util = int(candidate.get_tc_util())

        # DeviceRuntime defaults are not a contract. Explicitly establish and
        # read back the production reference state after GPU assignment and
        # before either runtime is allowed to JIT a GEMM.
        stock.set_pdl(True)
        stock.set_num_sms(148)
        stock.set_tc_util(stock_initial_tc_util)
        stock_pdl = stock.get_pdl()
        stock_num_sms = int(stock.get_num_sms())
        stock_tc_util = int(stock.get_tc_util())
        if stock_pdl is not True:
            raise RuntimeError(
                f"stock DeepGEMM PDL set/readback failed: {stock_pdl!r}"
            )
        if stock_num_sms != 148:
            raise RuntimeError(
                f"stock DeepGEMM num_sms set/readback failed: {stock_num_sms}"
            )
        if stock_tc_util != stock_initial_tc_util:
            raise RuntimeError(
                "stock DeepGEMM tc_util set/readback failed: "
                f"{stock_tc_util} != {stock_initial_tc_util}"
            )

        # Prove the overlay owns an independent DeviceRuntime by first placing
        # it in a deliberately different valid state and re-reading stock.
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

        # Copy the now-proven production state into the independent candidate.
        candidate.set_num_sms(stock_num_sms)
        candidate.set_tc_util(stock_tc_util)
        candidate.set_pdl(stock_pdl)
        candidate_num_sms = int(candidate.get_num_sms())
        candidate_tc_util = int(candidate.get_tc_util())
        candidate_pdl = candidate.get_pdl()
        if candidate_num_sms != 148:
            raise RuntimeError(
                f"candidate DeepGEMM num_sms is not 148: {candidate_num_sms}"
            )
        if candidate_tc_util != stock_tc_util:
            raise RuntimeError(
                "candidate DeepGEMM tc_util does not match stock: "
                f"{candidate_tc_util} != {stock_tc_util}"
            )
        if candidate_pdl is not True:
            raise RuntimeError(
                f"candidate DeepGEMM PDL is not true: {candidate_pdl!r}"
            )
        # Re-read stock once after candidate mutation to prove independent state
        # did not disturb the authoritative reference runtime.
        if (
            stock.get_pdl() is not True
            or int(stock.get_num_sms()) != stock_num_sms
            or int(stock.get_tc_util()) != stock_tc_util
        ):
            raise RuntimeError("candidate runtime synchronization mutated stock state")

        launch = candidate.fp8_m_grouped_gemm_nt_masked
        contract = W2BM16PreparedContract(
            candidate_module=candidate,
            launch=launch,
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
            cutlass_commit=manifest["base"]["submodules"][
                "third-party/cutlass"
            ],
            fmt_commit=manifest["base"]["submodules"]["third-party/fmt"],
        )
        _W2_BM16_PREPARED = contract
        return contract
    except Exception as exc:
        _W2_BM16_PREPARED = None
        _W2_BM16_PREPARE_ERROR = f"{type(exc).__name__}: {exc}"
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
class W2BM16LayerContract:
    """Persistent per-layer ABI proof; dynamic buffers are checked once."""

    runtime: Optional[W2BM16PreparedContract]
    w2_weight: torch.Tensor
    w2_scale: torch.Tensor
    static_eligible: bool
    static_reason: str
    callsite_checked: bool = False
    callsite_eligible: bool = False
    callsite_reason: str = "not-checked"


def create_w2_bm16_layer_contract(
    *,
    w2_weight: torch.Tensor,
    w2_scale: torch.Tensor,
    block_shape: Optional[list[int]],
    deep_gemm_backend: bool,
    is_fp4_experts: bool,
    use_mxfp8: bool,
) -> Optional[W2BM16LayerContract]:
    """Prevalidate immutable W2 weight ABI once after weight preparation."""
    if not _W2_BM16_REQUESTED:
        return None
    runtime = _W2_BM16_PREPARED
    checks = (
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
    eligible = all(checks)
    return W2BM16LayerContract(
        runtime=runtime,
        w2_weight=w2_weight,
        w2_scale=w2_scale,
        static_eligible=eligible,
        static_reason="ready" if eligible else "static-abi-or-runtime",
    )


def prepare_w2_bm16_callsite_contract(
    contract: Optional[W2BM16LayerContract],
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
    *,
    expected_m: int,
    recipe_a: Optional[Tuple[int, int]],
    recipe_b: Optional[Tuple[int, int]],
    overlap_args: Optional[Any],
) -> bool:
    """Validate exact decode/no-overlap callsite metadata once during warmup."""
    if contract is None:
        return False
    runtime = contract.runtime
    forward_state = (
        runtime.current_forward_state() if runtime is not None else None
    )
    if (
        forward_state is None
        or forward_state[0] is not ForwardMode.DECODE
        or (int(forward_state[1]), int(expected_m))
        not in W2_BM16_FORWARD_BUCKETS
        or recipe_a is not None
        or recipe_b is not None
        or overlap_args is not None
    ):
        # An unrelated prefill, TARGET_VERIFY, recipe, or overlap call must not
        # permanently disable a later exact decode warmup for this layer.
        return False
    if contract.callsite_checked:
        return contract.callsite_eligible

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
        and runtime is _W2_BM16_PREPARED
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
    return eligible


@lru_cache(maxsize=1)
def get_experimental_deep_gemm():
    if not deepgemm_variant():
        return None
    # This legacy overlay retains its own manifest/cache contract. It is not
    # used by the W2/BM16 profile.
    importlib.import_module("deep_gemm")
    manifest = _read_manifest()
    pkg_dir = Path(manifest["package_dir"]).resolve()
    jit_cache = Path(manifest["jit_cache_dir"]).resolve()
    init_py = pkg_dir / "__init__.py"
    if not init_py.is_file():
        raise FileNotFoundError(f"overlay package missing: {init_py}")
    jit_cache.mkdir(parents=True, exist_ok=True)
    os.environ["DG_JIT_CACHE_DIR"] = str(jit_cache)

    if (
        "deep_gemm_experimental" in sys.modules
        and getattr(sys.modules["deep_gemm_experimental"], "_C", None) is not None
    ):
        return sys.modules["deep_gemm_experimental"]

    spec = importlib.util.spec_from_file_location(
        "deep_gemm_experimental",
        init_py,
        submodule_search_locations=[str(pkg_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load experimental deep_gemm from {init_py}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["deep_gemm_experimental"] = module
    module.__path__ = [str(pkg_dir)]  # type: ignore[attr-defined]
    module.__package__ = "deep_gemm_experimental"
    spec.loader.exec_module(module)
    return module


def has_fused_fp8_gemm_nt() -> bool:
    mod = get_experimental_deep_gemm()
    return mod is not None and hasattr(mod, "fp8_gemm_nt_fused")
