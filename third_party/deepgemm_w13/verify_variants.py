#!/usr/bin/env python3
"""Identity and exact-numerics gate for round-2 W13 candidate identities.

Loads the stock and candidate DeepGEMM builds side by side from one round-2
manifest and, for every requested `w13_config` identity and expected-M point,
compares the candidate output against the production stock output on identical
input bytes.

Round-1 established that stock BM128 and candidate BM16 agree bit-exactly: the
K reduction order is unchanged by M/N tiling, so any deviation is a
synchronization defect rather than a numerics difference. This gate therefore
requires exact equality and treats any mismatch as a failed attempt.
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
    "w13_verify_provider_common", _HERE / "provider_common.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_COMMON = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_COMMON)

IDENTITIES = {
    "bm16_2sm_control": (16, 128, 128, 12, 2, 0),
    "bm16_1sm_control": (16, 128, 128, 11, 1, 0),
    "bm16_2sm_sfbypass": (16, 128, 128, 12, 2, 1),
    "bm16_1sm_sfbypass": (16, 128, 128, 11, 1, 1),
}
EXPECTED_M_VALUES = (4, 5, 8, 9)


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


def _bind(record, name):
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
    module = _load_package(Path(record["package"]).resolve(), name)
    module.set_pdl(_COMMON.REQUIRED_PDL)
    module.set_num_sms(_COMMON.REQUIRED_NUM_SMS)
    module.set_tc_util(_COMMON.REQUIRED_TC_UTIL)
    return module, jit_cache


def _mask_patterns(torch, device, expected_m):
    """Exact-ABI masked_m patterns: uniform, empty, boundary, skewed, maximum."""
    base = torch.full((32,), expected_m, device=device, dtype=torch.int32)
    patterns = {"uniform": base.clone()}

    empty = base.clone()
    empty[0] = 0
    empty[17] = 0
    patterns["empty_experts"] = empty

    boundary = base.clone()
    # exactly on and around the BM16 tile boundary
    boundary[1] = 16
    boundary[2] = 15
    boundary[3] = 17
    boundary[4] = 1
    patterns["tile_boundary"] = boundary

    skewed = torch.zeros((32,), device=device, dtype=torch.int32)
    skewed[0] = 1024
    skewed[1] = 512
    skewed[2] = 3
    patterns["skewed"] = skewed

    maximum = torch.full((32,), 1024, device=device, dtype=torch.int32)
    patterns["maximum"] = maximum
    return patterns


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--identity", required=True, choices=list(IDENTITIES))
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    import torch

    manifest = json.loads(args.manifest.read_text())
    device = torch.device("cuda", int(torch.cuda.current_device()))
    if torch.cuda.get_device_capability(device.index) != (10, 0):
        raise RuntimeError("verification requires sm_100")

    stock, stock_cache = _bind(manifest["variants"]["stock"], "deep_gemm_w13_verify_stock")
    cand, cand_cache = _bind(
        manifest["variants"]["candidate"], "deep_gemm_w13_verify_candidate"
    )
    if stock_cache == cand_cache:
        raise RuntimeError("stock and candidate JIT caches alias")

    def empty_strided(shape, stride, dtype):
        value = torch.empty_strided(shape, stride, device=device, dtype=dtype)
        value.zero_()
        return value

    generator = torch.Generator(device="cpu").manual_seed(20260730)
    a = empty_strided(_COMMON._A_SHAPE, _COMMON._A_STRIDE, torch.float8_e4m3fn)
    b = empty_strided(_COMMON._B_SHAPE, _COMMON._B_STRIDE, torch.float8_e4m3fn)
    # FP8 E4M3FN encodes NaN as 0x7F / 0xFF. Draw only finite byte patterns so a
    # mismatch can only mean a real numerical divergence, never NaN != NaN.
    def finite_fp8(shape):
        raw = torch.randint(0, 252, shape, generator=generator, dtype=torch.int16)
        # map [0,126) -> 0x00..0x7E and [126,252) -> 0x80..0xFE
        raw = torch.where(raw < 126, raw, raw - 126 + 128)
        return raw.to(torch.uint8)

    a.view(torch.uint8).copy_(finite_fp8(_COMMON._A_SHAPE).to(device))
    b.view(torch.uint8).copy_(finite_fp8(_COMMON._B_SHAPE).to(device))
    a_scale = empty_strided(_COMMON._AS_SHAPE, _COMMON._AS_STRIDE, torch.int32)
    b_scale = empty_strided(_COMMON._BS_SHAPE, _COMMON._BS_STRIDE, torch.int32)
    # Packed int32 UE8M0: four biased exponents per word, exercising 2^-1..2^2.
    exps = bytes([126, 127, 128, 129])
    packed = int.from_bytes(exps, "little", signed=True)
    a_scale.fill_(packed)
    b_scale.fill_(packed)
    ref = empty_strided(_COMMON._OUT_SHAPE, _COMMON._OUT_STRIDE, torch.bfloat16)
    got = empty_strided(_COMMON._OUT_SHAPE, _COMMON._OUT_STRIDE, torch.bfloat16)

    POISON = float("nan")
    results = []
    patterns_cache = {}
    for expected_m in EXPECTED_M_VALUES:
        patterns_cache[expected_m] = _mask_patterns(torch, device, expected_m)

    for identity in (args.identity,):
        config = IDENTITIES[identity]
        record = {"identity": identity, "config": list(config), "cases": []}
        for expected_m in EXPECTED_M_VALUES:
            for pattern_name, masked_m in patterns_cache[expected_m].items():
                worst = {"max_abs_err": 0.0, "mismatches": 0, "masked_touched": 0}
                for repeat in range(args.repeats):
                    ref.fill_(POISON)
                    got.fill_(POISON)
                    stock.fp8_m_grouped_gemm_nt_masked(
                        (a, a_scale), (b, b_scale), ref, masked_m, expected_m,
                        compiled_dims="nk", disable_ue8m0_cast=True,
                    )
                    cand.fp8_m_grouped_gemm_nt_masked(
                        (a, a_scale), (b, b_scale), got, masked_m, expected_m,
                        compiled_dims="nk", disable_ue8m0_cast=True,
                        w13_config=config,
                    )
                    torch.cuda.synchronize(device)
                    rows = masked_m.to("cpu").tolist()
                    mismatches = 0
                    max_err = 0.0
                    masked_touched = 0
                    for expert, valid in enumerate(rows):
                        if valid:
                            r = ref[expert, :valid].float()
                            g = got[expert, :valid].float()
                            bad = int((r != g).sum().item())
                            mismatches += bad
                            if bad:
                                max_err = max(
                                    max_err, float((r - g).abs().max().item())
                                )
                        # The candidate stores whole BM-row tiles, so the
                        # untouched envelope starts at the tile-aligned bound.
                        envelope = min(
                            1024, -(-valid // config[0]) * config[0]
                        )
                        if envelope < 1024:
                            tail = got[expert, envelope:]
                            masked_touched += int(
                                (~torch.isnan(tail.float())).sum().item()
                            )
                    worst["mismatches"] = max(worst["mismatches"], mismatches)
                    worst["max_abs_err"] = max(worst["max_abs_err"], max_err)
                    worst["masked_touched"] = max(
                        worst["masked_touched"], masked_touched
                    )
                record["cases"].append(
                    {
                        "expected_m": expected_m,
                        "mask_pattern": pattern_name,
                        "repeats": args.repeats,
                        **worst,
                        "exact": worst["mismatches"] == 0,
                        "masked_region_untouched": worst["masked_touched"] == 0,
                    }
                )
        record["all_exact"] = all(c["exact"] for c in record["cases"])
        record["all_masked_untouched"] = all(
            c["masked_region_untouched"] for c in record["cases"]
        )
        record["status"] = (
            "pass" if record["all_exact"] and record["all_masked_untouched"] else "fail"
        )
        results.append(record)
        print(
            f"{identity}: {record['status']}"
            f" exact={record['all_exact']}"
            f" masked_untouched={record['all_masked_untouched']}",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "candidate_commit": manifest["source"]["candidate_commit"],
                "physical_gpu": os.environ.get("GLM52_PHYSICAL_GPU"),
                "physical_gpu_uuid": os.environ.get("GLM52_PHYSICAL_GPU_UUID"),
                "reference": "stock production masked grouped GEMM, same input bytes",
                "criterion": "bitwise equality on valid rows; masked rows untouched",
                "identities": results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return 0 if all(r["status"] == "pass" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
