#!/usr/bin/env bash
# Smoke test for GLM-5.2 optimized kernel scaffolding.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/python:${PYTHONPATH:-}"
export SGLANG_GLM52_OPT=1
export SGLANG_GLM52_OPT_PROFILE=decode_max
export SGLANG_GLM52_DEEPGEMM_VARIANT=41c6235

# Prefer Kernel-Harness venv when present (has torch/deep_gemm for overlay test).
PY="${KERNEL_HARNESS_VENV:-/home/qinhaiyan/Kernel-Harness/.venv}/bin/python"
if [[ ! -x "$PY" ]]; then
  PY=python3
fi

"$PY" - <<'PY'
import importlib.util
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1] if "__file__" in dir() else Path.cwd()
# When run via heredoc, resolve from env
import os
root = Path(os.environ.get("ROOT", ".")).resolve() if "ROOT" in os.environ else Path("/home/qinhaiyan/sglang")
sys.path.insert(0, str(root / "python"))

from sglang.srt.layers.glm52_opt.config import is_enabled, load_manifest, profile_name
from sglang.srt.layers.glm52_opt.registry import list_enabled, lookup

assert is_enabled()
manifest = load_manifest()
assert manifest.get("archive_tag") == "0720-Best-GLM-52"
assert profile_name() == "decode_max"
assert lookup("q_b_proj", "decode") is not None
assert lookup("moe_gate_proj", "decode") is not None
assert lookup("fused_qkv_a_proj", "prefill") is None
decode_ops = [s.op for s in list_enabled("decode")]
assert "o_proj" in decode_ops
print("glm52_opt smoke OK:", len(decode_ops), "decode ops registered")
PY

echo "Optional: load DeepGEMM experimental overlay"
ROOT="$ROOT" "$PY" - <<'PY' || echo "WARN: DeepGEMM overlay not built (run build_overlay.sh)"
import os, sys
from pathlib import Path
root = Path(os.environ["ROOT"])
sys.path.insert(0, str(root / "python"))
from sglang.srt.layers.glm52_opt.experimental_deepgemm import get_experimental_deep_gemm
mod = get_experimental_deep_gemm()
print("experimental deep_gemm:", getattr(mod, "__file__", mod))
PY
