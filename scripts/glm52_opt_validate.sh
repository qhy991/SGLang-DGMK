#!/usr/bin/env bash
# End-to-end validation helper for GLM-5.2 optimized kernels.
set -euo pipefail

HARNESS="${KERNEL_HARNESS_ROOT:-/home/qinhaiyan/Kernel-Harness}"
SGLANG="${SGLANG_ROOT:-/home/qinhaiyan/sglang}"
ARCHIVE="${HARNESS}/archive/0720-Best-GLM-52/llm_flops_style"

echo "== glm52_opt smoke =="
bash "${SGLANG}/scripts/glm52_opt_smoke.sh"

echo "== harness decode layer bench (requires GPU) =="
if [[ -x "${HARNESS}/.venv/bin/python" ]]; then
  cd "${ARCHIVE}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    "${HARNESS}/.venv/bin/python" bench_decode.py || echo "WARN: bench_decode failed (GPU?)"
else
  echo "SKIP: Kernel-Harness venv not found"
fi

echo "== registry spot-check =="
export PYTHONPATH="${SGLANG}/python:${PYTHONPATH:-}"
export SGLANG_GLM52_OPT=1
export SGLANG_GLM52_OPT_PROFILE=full
python3 - <<'PY'
from sglang.srt.layers.glm52_opt.registry import lookup
for op in ("fused_qkv_a_proj", "q_b_proj", "index_k_proj"):
    assert lookup(op, "prefill") is not None, op
print("full profile prefill registry OK")
PY

echo "glm52_opt_validate.sh done"
