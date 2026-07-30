#!/usr/bin/env python3
"""Leased-GPU startup/JIT smoke for one exact API-v1 W13 provider."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", type=Path, required=True)
    args = parser.parse_args()
    provider = args.provider.expanduser().resolve()
    os.environ.update(
        {
            "SGLANG_GLM52_OPT": "1",
            "SGLANG_GLM52_OPT_PROFILE": "hotspot_candidates",
            "SGLANG_GLM52_OPT_OPS": "moe_w13",
            "SGLANG_GLM52_OPT_M_BUCKETS": "moe_gate_proj:16|32",
            "SGLANG_GLM52_HOTSPOT_MODULE": str(provider),
        }
    )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("provider smoke requires a leased CUDA device")
    torch.cuda.set_device(0)
    # Mirror current SGLang worker startup: the installed stock DeepGEMM
    # wrapper is imported before the out-of-tree provider initializes.  This
    # fixes the JIT include-tree cache key to the production-context identity.
    from sglang.srt.layers.deep_gemm_wrapper import entrypoint as _entrypoint

    del _entrypoint
    from sglang.srt.layers.glm52_opt import hotspot_provider

    hotspot_provider.initialize_hotspot_provider(gpu_id=0)
    state = hotspot_provider.provider_state()
    module = sys.modules[state["module_name"]]
    identity = dict(module._PROVIDER.identity)
    result = {
        "provider_state": state,
        "provider_identity": identity,
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
