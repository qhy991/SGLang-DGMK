#!/usr/bin/env bash
# Start a fresh process with exact post1 bound as the normal `deep_gemm` import.
set -euo pipefail

if (( $# == 0 )); then
  echo "usage: $0 <command> [args ...]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HARNESS_PYTHON="${HARNESS_PYTHON:-${REPO_ROOT}/../kernel-harness/.venv/bin/python}"
KERNEL_HARNESS_ROOT="${TASK26_V4_KERNEL_HARNESS_ROOT:-${REPO_ROOT}/../kernel-harness}"
READY_TOOL="${SCRIPT_DIR}/ready_bundle.py"
TASK_CACHE_ROOT="/home/qinhaiyan/glm52-v2-goal-runs/cache/26-moe_w2_decode_scoped_bm16/em8_bm16_stage11_v4"
TASK_DEEPGEMM_CACHE="${TASK_CACHE_ROOT}/deepgemm"

export DG_JIT_CACHE_DIR="$TASK_DEEPGEMM_CACHE"
export SGLANG_DG_CACHE_DIR="$TASK_DEEPGEMM_CACHE"
export TRITON_CACHE_DIR="${TASK_CACHE_ROOT}/triton"
export TORCH_EXTENSIONS_DIR="${TASK_CACHE_ROOT}/torch_extensions"

LOCATED_READY="$("$HARNESS_PYTHON" "$READY_TOOL" locate \
  --sglang-root "$REPO_ROOT" \
  --print ready)"
READY_OVERRIDE="${SGLANG_GLM52_W2_EM8_BM16_STAGE11_V4_READY:-}"
if [[ -n "$READY_OVERRIDE" && "$(readlink -m "$READY_OVERRIDE")" != "$LOCATED_READY" ]]; then
  echo "ERROR: v4 READY override does not name the canonical bundle" >&2
  exit 1
fi
"$HARNESS_PYTHON" "$READY_TOOL" verify \
  --ready "$LOCATED_READY" \
  --sglang-root "$REPO_ROOT" \
  --kernel-harness-root "$KERNEL_HARNESS_ROOT" \
  --check-env \
  >/dev/null
MANIFEST_PATH="$("$HARNESS_PYTHON" "$READY_TOOL" locate \
  --sglang-root "$REPO_ROOT" \
  --print manifest)"
STOCK_SITE="$("$HARNESS_PYTHON" "$READY_TOOL" locate \
  --sglang-root "$REPO_ROOT" \
  --print stock-site)"

export SGLANG_GLM52_W2_EM8_BM16_STAGE11_V4_READY="$LOCATED_READY"
export SGLANG_GLM52_W2_EM8_BM16_STAGE11_V4_MANIFEST="$MANIFEST_PATH"
export PYTHONPATH="${STOCK_SITE}:${REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
exec "$@"
