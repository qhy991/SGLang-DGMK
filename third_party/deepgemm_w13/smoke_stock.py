#!/usr/bin/env python3
"""Leased-GPU startup/JIT smoke for the exact same-source stock W13 module."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from provider_common import (
    EXPECTED_M_VALUES,
    MANIFEST,
    REQUIRED_NUM_SMS,
    REQUIRED_PDL,
    REQUIRED_TC_UTIL,
    _load_candidate,
    _sha256,
    _snapshot,
)


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    record = manifest["variants"]["stock"]
    package = Path(record["package"]).resolve()
    shared_object = Path(record["shared_object"]).resolve()
    jit_cache = Path(record["jit_cache"]).resolve()
    if (
        package / "_C.so" != shared_object
        or not shared_object.is_file()
        or _sha256(shared_object) != record["shared_object_sha256"]
        or _sha256(package / "__init__.py") != record["package_init_sha256"]
    ):
        raise RuntimeError("same-source stock artifact identity mismatch")

    os.environ.update(
        {
            "DG_JIT_CACHE_DIR": str(jit_cache),
            "SGLANG_DG_CACHE_DIR": str(jit_cache),
            "DG_JIT_USE_NVRTC": "0",
            "SGL_DG_USE_NVRTC": "0",
            "DG_JIT_DUMP_PTX": "1",
            "DG_JIT_DUMP_SASS": "1",
            "DG_JIT_PTXAS_VERBOSE": "1",
            "DG_JIT_PTXAS_CHECK": "0",
        }
    )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("stock smoke requires a leased CUDA device")
    torch.cuda.set_device(0)
    if torch.cuda.get_device_capability(0) != (10, 0):
        raise RuntimeError("stock smoke requires sm_100")

    module = _load_candidate(package, f"deep_gemm_w13_stock_smoke_{os.getpid()}")
    module.set_pdl(REQUIRED_PDL)
    module.set_num_sms(REQUIRED_NUM_SMS)
    module.set_tc_util(REQUIRED_TC_UTIL)
    state = {
        "pdl": bool(module.get_pdl()),
        "num_sms": int(module.get_num_sms()),
        "tc_util": int(module.get_tc_util()),
    }
    required_state = {
        "pdl": REQUIRED_PDL,
        "num_sms": REQUIRED_NUM_SMS,
        "tc_util": REQUIRED_TC_UTIL,
    }
    if state != required_state:
        raise RuntimeError(f"stock runtime state mismatch: {state}")

    device = torch.device("cuda", 0)

    def empty_strided(shape, stride, dtype):
        value = torch.empty_strided(shape, stride, device=device, dtype=dtype)
        value.zero_()
        return value

    tensors = {
        "a": empty_strided(
            (32, 1024, 6144), (6291456, 6144, 1), torch.float8_e4m3fn
        ),
        "a_scale": empty_strided(
            (32, 1024, 12), (12288, 1, 1024), torch.int32
        ),
        "b": empty_strided(
            (32, 4096, 6144), (25165824, 6144, 1), torch.float8_e4m3fn
        ),
        "b_scale": empty_strided(
            (32, 4096, 12), (49152, 1, 4096), torch.int32
        ),
        "out": empty_strided(
            (32, 1024, 4096), (4194304, 4096, 1), torch.bfloat16
        ),
        "masked_m": torch.zeros((32,), device=device, dtype=torch.int32),
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
        )
        if returned is not None:
            raise RuntimeError("stock warmup changed the None contract")
    torch.cuda.synchronize(device)
    frozen = _snapshot(jit_cache)
    if not frozen:
        raise RuntimeError("stock warmup produced no JIT files")

    probe = MANIFEST.parent / f"unbound-stock-probe-{os.getpid()}"
    if probe.exists():
        raise RuntimeError(f"stock cache probe already exists: {probe}")
    os.environ["DG_JIT_CACHE_DIR"] = str(probe)
    os.environ["SGLANG_DG_CACHE_DIR"] = str(probe)
    returned = launcher(
        (tensors["a"], tensors["a_scale"]),
        (tensors["b"], tensors["b_scale"]),
        tensors["out"],
        tensors["masked_m"],
        EXPECTED_M_VALUES[0],
        compiled_dims="nk",
        disable_ue8m0_cast=True,
    )
    torch.cuda.synchronize(device)
    if returned is not None or probe.exists() or _snapshot(jit_cache) != frozen:
        raise RuntimeError("stock JIT cache ownership is not frozen")

    result = {
        "manifest": str(MANIFEST),
        "manifest_sha256": _sha256(MANIFEST),
        "shared_object": str(shared_object),
        "shared_object_sha256": record["shared_object_sha256"],
        "jit_cache": str(jit_cache),
        "jit_artifacts": frozen,
        "expected_m_values": list(EXPECTED_M_VALUES),
        "runtime_state": state,
        "physical_gpu": os.environ.get("GLM52_PHYSICAL_GPU"),
        "physical_gpu_uuid": os.environ.get("GLM52_PHYSICAL_GPU_UUID"),
        "logical_device": int(torch.cuda.current_device()),
        "device_name": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
