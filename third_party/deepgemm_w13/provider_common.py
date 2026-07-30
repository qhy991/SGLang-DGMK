"""API-v1 implementation shared by the two exact GLM-5.2 W13 providers."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Any

BASE_COMMIT = "731e7c7a97d269e4b9f482ea18d0e709a948f293"
CANDIDATE_COMMIT = "87e0359edbb461181d3bba218442132007b9a738"
CANDIDATE_DIFF_SHA256 = (
    "465c8373c0a37970225a0e93267b6c399431b23e22cf35b4511db2308df98092"
)
STOCK_TREE_SHA256 = (
    "917592ab68ea0608c9be33208c2c609bc7f20bd9b1603f32743dd0d1ae03d0ed"
)
CANDIDATE_TREE_SHA256 = (
    "d682daa65b8ba0ac3846d766910b8c751e0568fe62087084271bb354e46c49e4"
)
_REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(
    os.environ.get(
        "SGLANG_GLM52_W13_MANIFEST",
        str(_REPO_ROOT / ".cache" / "glm52_w13_variants" / "manifest.json"),
    )
).expanduser().resolve()
REQUIRED_PDL = True
REQUIRED_NUM_SMS = 148
REQUIRED_TC_UTIL = 100
EXPECTED_M_VALUES = (4, 5, 8, 9)

_A_SHAPE = (32, 1024, 6144)
_A_STRIDE = (6291456, 6144, 1)
_AS_SHAPE = (32, 1024, 12)
_AS_STRIDE = (12288, 1, 1024)
_B_SHAPE = (32, 4096, 6144)
_B_STRIDE = (25165824, 6144, 1)
_BS_SHAPE = (32, 4096, 12)
_BS_STRIDE = (49152, 1, 4096)
_OUT_SHAPE = (32, 1024, 4096)
_OUT_STRIDE = (4194304, 4096, 1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _load_candidate(package: Path, module_name: str) -> ModuleType:
    init_py = package / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        module_name,
        init_py,
        submodule_search_locations=[str(package)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load W13 candidate package from {init_py}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    module.__path__ = [str(package)]  # type: ignore[attr-defined]
    module.__package__ = module_name
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _validate_manifest(torch: Any) -> tuple[dict[str, Any], Path]:
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get("schema_version") != 3:
        raise RuntimeError("W13 build manifest schema mismatch")
    source = manifest.get("source", {})
    expected_source = {
        "base_commit": BASE_COMMIT,
        "candidate_commit": CANDIDATE_COMMIT,
        "candidate_diff_sha256": CANDIDATE_DIFF_SHA256,
        "stock_source_tree_sha256": STOCK_TREE_SHA256,
        "candidate_source_tree_sha256": CANDIDATE_TREE_SHA256,
    }
    actual_source = {key: source.get(key) for key in expected_source}
    if actual_source != expected_source:
        raise RuntimeError(
            f"W13 manifest source mismatch: {actual_source} != {expected_source}"
        )
    build = manifest.get("build", {})
    if (
        build.get("torch") != torch.__version__
        or build.get("torch_cuda") != torch.version.cuda
        or build.get("cuda_arch") != "10.0a"
        or build.get("jit_compiler") != "nvcc"
        or build.get("stock_candidate_command_identical") is not True
        or build.get("elf_symbol_binding") != "Bsymbolic"
        or build.get("elf_symbol_visibility") != "hidden"
    ):
        raise RuntimeError("W13 build/runtime contract mismatch")
    record = manifest.get("variants", {}).get("candidate")
    if not isinstance(record, dict) or record.get("commit") != CANDIDATE_COMMIT:
        raise RuntimeError("W13 candidate build record is missing")
    package = Path(str(record.get("package", ""))).resolve()
    shared_object = Path(str(record.get("shared_object", ""))).resolve()
    jit_cache = Path(str(record.get("jit_cache", ""))).resolve()
    if (
        package / "_C.so" != shared_object
        or not shared_object.is_file()
        or _sha256(shared_object) != record.get("shared_object_sha256")
        or not jit_cache.is_dir()
    ):
        raise RuntimeError("W13 candidate artifact identity mismatch")
    if _sha256(package / "__init__.py") != record.get("package_init_sha256"):
        raise RuntimeError("W13 candidate Python package identity mismatch")
    return record, jit_cache


class Provider:
    """One startup-bound, one-launch-hot-path DeepGEMM provider."""

    def __init__(self, *, name: str, config: tuple[int, int, int, int, int]):
        self.name = name
        self.config = config
        self._lock = threading.Lock()
        self._module: ModuleType | None = None
        self._launcher: Any = None
        self.identity: dict[str, Any] = {}

    def initialize(self, *, gpu_id: int | None) -> None:
        with self._lock:
            if self._launcher is not None:
                if self.identity.get("gpu_id") != gpu_id:
                    raise RuntimeError("W13 provider already belongs to another GPU")
                return

            import torch

            current = int(torch.cuda.current_device())
            expected = current if gpu_id is None else int(gpu_id)
            if current != expected:
                raise RuntimeError(
                    f"W13 provider current device {current} != assigned {expected}"
                )
            if torch.cuda.get_device_capability(current) != (10, 0):
                raise RuntimeError("W13 provider requires sm_100")
            record, jit_cache = _validate_manifest(torch)
            saved = {
                name: os.environ.get(name)
                for name in (
                    "DG_JIT_CACHE_DIR",
                    "SGLANG_DG_CACHE_DIR",
                    "DG_JIT_USE_NVRTC",
                    "SGL_DG_USE_NVRTC",
                    "DG_JIT_DUMP_PTX",
                    "DG_JIT_DUMP_SASS",
                    "DG_JIT_PTXAS_VERBOSE",
                    "DG_JIT_PTXAS_CHECK",
                )
            }
            tensors = None
            try:
                os.environ.update(
                    {
                        "DG_JIT_CACHE_DIR": str(jit_cache),
                        "SGLANG_DG_CACHE_DIR": str(jit_cache),
                        "DG_JIT_USE_NVRTC": "0",
                        "SGL_DG_USE_NVRTC": "0",
                        "DG_JIT_DUMP_PTX": "1",
                        "DG_JIT_DUMP_SASS": "1",
                        "DG_JIT_PTXAS_VERBOSE": "1",
                        # The upstream check rejects any "Local memory used"
                        # diagnostic before reporting its size. Retain verbose
                        # ptxas output and audit stack/spills from the CUBIN.
                        "DG_JIT_PTXAS_CHECK": "0",
                    }
                )
                module = _load_candidate(
                    Path(record["package"]).resolve(),
                    f"deep_gemm_w13_{self.name}_{os.getpid()}",
                )
                module.set_pdl(REQUIRED_PDL)
                module.set_num_sms(REQUIRED_NUM_SMS)
                module.set_tc_util(REQUIRED_TC_UTIL)
                runtime_state = {
                    "pdl": bool(module.get_pdl()),
                    "num_sms": int(module.get_num_sms()),
                    "tc_util": int(module.get_tc_util()),
                }
                required_state = {
                    "pdl": REQUIRED_PDL,
                    "num_sms": REQUIRED_NUM_SMS,
                    "tc_util": REQUIRED_TC_UTIL,
                }
                if runtime_state != required_state:
                    raise RuntimeError(
                        f"W13 runtime state mismatch: {runtime_state}"
                    )

                device = torch.device("cuda", current)

                def empty_strided(shape, stride, dtype):
                    value = torch.empty_strided(
                        shape, stride, device=device, dtype=dtype
                    )
                    value.zero_()
                    return value

                tensors = {
                    "a": empty_strided(
                        _A_SHAPE, _A_STRIDE, torch.float8_e4m3fn
                    ),
                    "a_scale": empty_strided(
                        _AS_SHAPE, _AS_STRIDE, torch.int32
                    ),
                    "b": empty_strided(
                        _B_SHAPE, _B_STRIDE, torch.float8_e4m3fn
                    ),
                    "b_scale": empty_strided(
                        _BS_SHAPE, _BS_STRIDE, torch.int32
                    ),
                    "out": empty_strided(
                        _OUT_SHAPE, _OUT_STRIDE, torch.bfloat16
                    ),
                    "masked_m": torch.zeros(
                        (32,), device=device, dtype=torch.int32
                    ),
                }
                launcher = module.fp8_m_grouped_gemm_nt_masked
                for expected_m in EXPECTED_M_VALUES:
                    tensors["masked_m"].fill_(expected_m)
                    tensors["out"].fill_(float("nan"))
                    returned = launcher(
                        (tensors["a"], tensors["a_scale"]),
                        (tensors["b"], tensors["b_scale"]),
                        tensors["out"],
                        tensors["masked_m"],
                        expected_m,
                        compiled_dims="nk",
                        disable_ue8m0_cast=True,
                        w13_config=self.config,
                    )
                    if returned is not None:
                        raise RuntimeError("W13 warmup changed the None contract")
                torch.cuda.synchronize(device)
                frozen = _snapshot(jit_cache)
                if not frozen:
                    raise RuntimeError("W13 candidate warmup produced no JIT files")

                probe = MANIFEST.parent / f"unbound-provider-probe-{os.getpid()}"
                if probe.exists():
                    raise RuntimeError(f"W13 cache probe already exists: {probe}")
                os.environ["DG_JIT_CACHE_DIR"] = str(probe)
                os.environ["SGLANG_DG_CACHE_DIR"] = str(probe)
                launcher(
                    (tensors["a"], tensors["a_scale"]),
                    (tensors["b"], tensors["b_scale"]),
                    tensors["out"],
                    tensors["masked_m"],
                    EXPECTED_M_VALUES[0],
                    compiled_dims="nk",
                    disable_ue8m0_cast=True,
                    w13_config=self.config,
                )
                torch.cuda.synchronize(device)
                if probe.exists() or _snapshot(jit_cache) != frozen:
                    raise RuntimeError("W13 JIT cache ownership is not frozen")

                self._module = module
                self._launcher = launcher
                self.identity = {
                    "name": self.name,
                    "gpu_id": gpu_id,
                    "config": list(self.config),
                    "runtime_state": runtime_state,
                    "manifest": str(MANIFEST),
                    "manifest_sha256": _sha256(MANIFEST),
                    "shared_object": record["shared_object"],
                    "shared_object_sha256": record["shared_object_sha256"],
                    "jit_cache": str(jit_cache),
                    "jit_artifacts": frozen,
                }
            finally:
                if tensors is not None:
                    del tensors
                    torch.cuda.empty_cache()
                for name, value in saved.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value

    def moe_w13(self, *, lhs, rhs, out, masked_m, expected_m):
        launcher = self._launcher
        if launcher is None:
            raise RuntimeError("W13 provider used before initialize")
        returned = launcher(
            lhs,
            rhs,
            out,
            masked_m,
            expected_m,
            compiled_dims="nk",
            disable_ue8m0_cast=True,
            w13_config=self.config,
        )
        if returned is not None:
            raise RuntimeError("W13 candidate violated the None return contract")
        return None
