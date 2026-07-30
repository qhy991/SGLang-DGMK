#!/usr/bin/env python3
"""Profile one exact W13 launch of a chosen build variant under the frozen ABI.

Diagnostic only. This is not a performance harness: it exists so Nsight Compute
can answer one concrete device-code question about a single launch of the exact
`infini_kernel_glm52_moe_w13_decode_em*` symbol (or the stock masked symbol)
with production tensors, packed UE8M0 scales, PDL, SM budget, and stream.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location(
    "w13_profile_provider_common", _HERE / "provider_common.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_COMMON = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_COMMON)

VARIANT_CONFIGS = {
    "bm16_2sm": (16, 128, 128, 12, 2, 0),
    "bm16_1sm": (16, 128, 128, 11, 1, 0),
    "bm16_2sm_sfbypass": (16, 128, 128, 12, 2, 1),
    "bm16_1sm_sfbypass": (16, 128, 128, 11, 1, 1),
}


def _load_package(package: Path, name: str):
    init_py = package / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        name, init_py, submodule_search_locations=[str(package)]
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    module.__path__ = [str(package)]
    module.__package__ = name
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--arm", choices=("stock", "candidate"), required=True)
    parser.add_argument(
        "--variant",
        default="bm16_2sm",
        help="candidate template identity; ignored for the stock arm",
    )
    parser.add_argument("--expected-m", type=int, default=4)
    parser.add_argument("--masked-m", type=int, default=None)
    parser.add_argument(
        "--active-experts",
        type=int,
        default=32,
        help="zero masked_m beyond this many experts to shrink sanitizer runs",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--profiled-launches", type=int, default=1)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    import torch

    manifest = json.loads(args.manifest.read_text())
    record = manifest["variants"][args.arm]
    jit_cache = Path(record["jit_cache"]).resolve()
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

    device = torch.device("cuda", int(torch.cuda.current_device()))
    if torch.cuda.get_device_capability(device.index) != (10, 0):
        raise RuntimeError("profile requires sm_100")

    module = _load_package(
        Path(record["package"]).resolve(), f"deep_gemm_w13_profile_{args.arm}"
    )
    module.set_pdl(_COMMON.REQUIRED_PDL)
    module.set_num_sms(_COMMON.REQUIRED_NUM_SMS)
    module.set_tc_util(_COMMON.REQUIRED_TC_UTIL)

    def empty_strided(shape, stride, dtype):
        value = torch.empty_strided(shape, stride, device=device, dtype=dtype)
        value.zero_()
        return value

    generator = torch.Generator(device="cpu").manual_seed(20260730)
    a = empty_strided(_COMMON._A_SHAPE, _COMMON._A_STRIDE, torch.float8_e4m3fn)
    b = empty_strided(_COMMON._B_SHAPE, _COMMON._B_STRIDE, torch.float8_e4m3fn)
    a.view(torch.uint8).copy_(
        torch.randint(
            1, 120, _COMMON._A_SHAPE, generator=generator, dtype=torch.uint8
        ).to(device)
    )
    b.view(torch.uint8).copy_(
        torch.randint(
            1, 120, _COMMON._B_SHAPE, generator=generator, dtype=torch.uint8
        ).to(device)
    )
    a_scale = empty_strided(_COMMON._AS_SHAPE, _COMMON._AS_STRIDE, torch.int32)
    b_scale = empty_strided(_COMMON._BS_SHAPE, _COMMON._BS_STRIDE, torch.int32)
    # Packed int32 UE8M0: four biased exponents per int32 word, all 127 (2^0).
    packed_one = int.from_bytes(bytes([127, 127, 127, 127]), "little", signed=True)
    a_scale.fill_(packed_one)
    b_scale.fill_(packed_one)
    out = empty_strided(_COMMON._OUT_SHAPE, _COMMON._OUT_STRIDE, torch.bfloat16)
    masked_value = args.expected_m if args.masked_m is None else args.masked_m
    masked_m = torch.full((32,), masked_value, device=device, dtype=torch.int32)
    if args.active_experts < 32:
        masked_m[args.active_experts :] = 0

    kwargs = dict(compiled_dims="nk", disable_ue8m0_cast=True)
    if args.arm == "candidate":
        kwargs["w13_config"] = VARIANT_CONFIGS[args.variant]
    launcher = module.fp8_m_grouped_gemm_nt_masked

    def launch():
        returned = launcher(
            (a, a_scale), (b, b_scale), out, masked_m, args.expected_m, **kwargs
        )
        if returned is not None:
            raise RuntimeError("W13 launch violated the None return contract")

    for _ in range(args.warmup):
        launch()
    torch.cuda.synchronize(device)

    torch.cuda.nvtx.range_push(f"w13_profile_{args.arm}_{args.variant}")
    try:
        for _ in range(args.profiled_launches):
            launch()
    finally:
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize(device)

    finite = bool(torch.isfinite(out[:, :masked_value, :]).all().item())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "arm": args.arm,
                "variant": args.variant if args.arm == "candidate" else "stock",
                "expected_m": args.expected_m,
                "masked_m": masked_value,
                "active_experts": args.active_experts,
                "commit": record["commit"],
                "shared_object_sha256": record["shared_object_sha256"],
                "jit_cache": str(jit_cache),
                "pdl": bool(module.get_pdl()),
                "num_sms": int(module.get_num_sms()),
                "tc_util": int(module.get_tc_util()),
                "stream": int(torch.cuda.current_stream(device).cuda_stream),
                "physical_gpu": os.environ.get("GLM52_PHYSICAL_GPU"),
                "physical_gpu_uuid": os.environ.get("GLM52_PHYSICAL_GPU_UUID"),
                "sm_clock_mhz": None,
                "valid_region_finite": finite,
                "output_pointer": int(out.data_ptr()),
                "profiled_launches": args.profiled_launches,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
