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
BASE_COMMIT="edcf77b276965de8f03cdc47c23f01b08bf7c7ab"
SOURCE_SHA="$(sha256sum "${SCRIPT_DIR}/source.patch" | awk '{print $1}')"
BUILD_TOOL_SHA="$(sha256sum "${SCRIPT_DIR}/build_tool.patch" | awk '{print $1}')"
BUILD_KEY="${BASE_COMMIT:0:12}-${SOURCE_SHA:0:12}-${BUILD_TOOL_SHA:0:12}"
OVERLAY_DIR="${REPO_ROOT}/build/deepgemm-w2-bm16-overlays/${BUILD_KEY}"
MANIFEST_PATH="${OVERLAY_DIR}/manifest.json"
STOCK_SITE="${OVERLAY_DIR}/stock/site"
TASK_CACHE_ROOT="/home/qinhaiyan/glm52-v2-goal-runs/cache/26-moe_w2_decode_scoped_bm16"
TASK_DEEPGEMM_CACHE="${TASK_CACHE_ROOT}/deepgemm"

export DG_JIT_CACHE_DIR="$TASK_DEEPGEMM_CACHE"
export SGLANG_DG_CACHE_DIR="$TASK_DEEPGEMM_CACHE"
export TRITON_CACHE_DIR="${TASK_CACHE_ROOT}/triton"
export TORCH_EXTENSIONS_DIR="${TASK_CACHE_ROOT}/torch_extensions"
"$HARNESS_PYTHON" "${SCRIPT_DIR}/overlay_manifest.py" verify \
  --manifest "$MANIFEST_PATH" \
  --check-env \
  --check-provenance

export SGLANG_GLM52_W2_BM16_MANIFEST="$MANIFEST_PATH"
export PYTHONPATH="${STOCK_SITE}:${REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
exec "$@"
