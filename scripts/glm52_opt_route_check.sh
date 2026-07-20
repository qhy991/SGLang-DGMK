#!/usr/bin/env bash
# Verify glm52_opt routes to the real winning backends.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/python:${PYTHONPATH:-}"
export SGLANG_GLM52_OPT=1
export SGLANG_GLM52_OPT_PROFILE=full
export SGLANG_GLM52_DEEPGEMM_VARIANT=41c6235
PY="${KERNEL_HARNESS_VENV:-/home/qinhaiyan/Kernel-Harness/.venv}/bin/python"
[[ -x "$PY" ]] || PY=python3

ROOT="$ROOT" "$PY" - <<'PY'
import os, sys
from pathlib import Path
root = Path(os.environ["ROOT"])
sys.path.insert(0, str(root / "python"))

from sglang.srt.layers.glm52_opt.resolve import resolve_backend
from sglang.srt.layers.glm52_opt.archive_loader import load_run_fn
from sglang.srt.layers.glm52_opt.config import archive_root

expect = {
    ("fused_qkv_a_proj", "decode"): "archive:best-hechenxi-0720/fused_qkv_a_decode",
    ("index_q_upproj", "decode"): "archive:best-hechenxi-0720/index_q_upproj_decode",
    ("index_k_proj", "decode"): "archive:best/index_k_proj_decode",
    ("q_b_proj", "decode"): "deepgemm_fork_fused",
    ("o_proj", "decode"): "native_packed_ue8m0",
    ("moe_gate_proj", "decode"): "native_moe_pack_pdl",
    ("dsa_decode_attn", "decode"): "archive:best-hechenxi-0720/dsa_decode_attn",
    ("fused_qkv_a_proj", "prefill"): "archive:fused_qkv_a_prefill.py",
    ("q_b_proj", "prefill"): "archive:q_b_prefill.py",
    ("o_proj", "prefill"): "native_packed_ue8m0",
    ("index_q_upproj", "prefill"): "archive:index_q_upproj_prefill.py",
    ("index_k_proj", "prefill"): "archive:best/index_k_proj_decode",
    ("index_weights_proj", "prefill"): "native_bf16_cuda_graph_mm",
    ("moe_up_proj", "prefill"): "native_moe_pack_pdl",
    ("moe_down_proj", "prefill"): "native_moe_pack_pdl",
    ("moe_gate_proj", "prefill"): None,  # intentional stock
}

for (op, phase), want in expect.items():
    got = resolve_backend(op, phase)
    assert got == want, f"{op}/{phase}: got {got!r}, want {want!r}"
    print(f"OK  {phase:7s} {op:22s} -> {got}")

for ref in (
    "best-hechenxi-0720/fused_qkv_a_decode",
    "best-hechenxi-0720/index_q_upproj_decode",
    "best/index_k_proj_decode",
    "best-hechenxi-0720/dsa_decode_attn",
    "fused_qkv_a_prefill.py",
):
    assert callable(load_run_fn(ref)), ref
    print(f"OK  load {ref}")

print("glm52_opt route check PASSED")
print("archive_root =", archive_root())
PY
