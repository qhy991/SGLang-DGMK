#!/usr/bin/env bash
# Publish READY only after generated provenance is committed in clean repos.
set -euo pipefail

readonly TASK_CACHE_ROOT="/home/qinhaiyan/glm52-v2-goal-runs/cache/26-moe_w2_decode_scoped_bm16/em8_bm16_stage11_v4"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
KERNEL_HARNESS_ROOT="${TASK26_V4_KERNEL_HARNESS_ROOT:-${REPO_ROOT}/../kernel-harness}"
HARNESS_PYTHON="${HARNESS_PYTHON:-${KERNEL_HARNESS_ROOT}/.venv/bin/python}"
READY_TOOL="${SCRIPT_DIR}/ready_bundle.py"

if [[ "${CUDA_VISIBLE_DEVICES+x}" != "x" || -n "${CUDA_VISIBLE_DEVICES}" ]]; then
  echo "ERROR: READY publication requires explicit CUDA_VISIBLE_DEVICES=''" >&2
  exit 1
fi
for fd_link in /proc/$$/fd/[0-9]*; do
  [[ -e "$fd_link" ]] || continue
  fd_target="$(readlink -f -- "$fd_link" 2>/dev/null || true)"
  if [[ "$fd_target" =~ /glm52-goal-runs/locks/gpu[0-9]+\.lock$ ]]; then
    echo "ERROR: refusing READY publication under a GPU scheduler lease" >&2
    exit 1
  fi
done
[[ -x "$HARNESS_PYTHON" ]] \
  || { echo "ERROR: harness Python is not executable: $HARNESS_PYTHON" >&2; exit 1; }
[[ -f "$READY_TOOL" ]] \
  || { echo "ERROR: READY verifier is missing: $READY_TOOL" >&2; exit 1; }

require_exact_cache() {
  local name="$1"
  local expected="$2"
  local actual="${!name:-}"
  if [[ -z "$actual" || "$(readlink -m "$actual")" != "$expected" ]]; then
    echo "ERROR: $name must be exactly $expected (got ${actual:-<unset>})" >&2
    exit 1
  fi
}
require_exact_cache DG_JIT_CACHE_DIR "${TASK_CACHE_ROOT}/deepgemm"
require_exact_cache SGLANG_DG_CACHE_DIR "${TASK_CACHE_ROOT}/deepgemm"
require_exact_cache TRITON_CACHE_DIR "${TASK_CACHE_ROOT}/triton"
require_exact_cache TORCH_EXTENSIONS_DIR "${TASK_CACHE_ROOT}/torch_extensions"

BUNDLE="$("$HARNESS_PYTHON" "$READY_TOOL" locate-bundle \
  --sglang-root "$REPO_ROOT" \
  --ready-state unready)"
READY="$("$HARNESS_PYTHON" "$READY_TOOL" write-ready \
  --manifest "${BUNDLE}/manifest.json" \
  --sglang-root "$REPO_ROOT" \
  --kernel-harness-root "$KERNEL_HARNESS_ROOT" \
  --check-env)"
"$HARNESS_PYTHON" "$READY_TOOL" verify \
  --ready "$READY" \
  --sglang-root "$REPO_ROOT" \
  --kernel-harness-root "$KERNEL_HARNESS_ROOT" \
  --check-env
echo "READY_PUBLISHED=${READY}"
