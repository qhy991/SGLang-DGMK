#!/usr/bin/env bash
# Fresh-check the exact post1 em8/BM16/stage11 source delta and identity.
set -euo pipefail

readonly BASE_COMMIT="edcf77b276965de8f03cdc47c23f01b08bf7c7ab"
readonly CUTLASS_COMMIT="f3fde58372d33e9a5650ba7b80fc48b3b49d40c8"
readonly FMT_COMMIT="553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28"
readonly TASK_CACHE_ROOT="/home/qinhaiyan/glm52-v2-goal-runs/cache/26-moe_w2_decode_scoped_bm16/em8_bm16_stage11_v3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASE_REPO="${DEEPGEMM_W2_EM8_BM16_STAGE11_BASE_REPO:-/home/qinhaiyan/DeepGEMM-GLM52}"
SOURCE_PATCH="${SCRIPT_DIR}/source.patch"
BUILD_TOOL_PATCH="${SCRIPT_DIR}/build_tool.patch"
CORE_HASHES="${SCRIPT_DIR}/core_source_hashes.sha256"
MANIFEST_TOOL="${SCRIPT_DIR}/overlay_manifest.py"
HARNESS_PYTHON="${HARNESS_PYTHON:-${REPO_ROOT}/../kernel-harness/.venv/bin/python}"

SOURCE_SHA="$(sha256sum "$SOURCE_PATCH" | awk '{print $1}')"
BUILD_TOOL_SHA="$(sha256sum "$BUILD_TOOL_PATCH" | awk '{print $1}')"
BUILD_KEY="${BASE_COMMIT:0:12}-${SOURCE_SHA:0:12}-${BUILD_TOOL_SHA:0:12}"
OVERLAY_DIR="${REPO_ROOT}/build/deepgemm-w2-em8-bm16-stage11-v3-overlays/${BUILD_KEY}"
MANIFEST_PATH="${1:-${OVERLAY_DIR}/manifest.json}"

VERIFY_ROOT="$(mktemp -d "${REPO_ROOT}/build/deepgemm-w2-em8-bm16-stage11-v3-verify.XXXXXX")"
cleanup() {
  rm -rf -- "$VERIFY_ROOT"
}
trap cleanup EXIT

available_kib="$(df -Pk "$REPO_ROOT" | awk 'NR == 2 {print $4}')"
if (( available_kib < 8 * 1024 * 1024 )); then
  echo "ERROR: fewer than 8 GiB remain; refusing fresh source expansion" >&2
  exit 1
fi

clone_exact() {
  local destination="$1"
  git clone --quiet --shared "$BASE_REPO" "$destination"
  git -C "$destination" checkout --quiet --detach "$BASE_COMMIT"
  git -C "$destination" config submodule.third-party/cutlass.url \
    "${BASE_REPO}/third-party/cutlass"
  git -C "$destination" config submodule.third-party/fmt.url \
    "${BASE_REPO}/third-party/fmt"
  git -c protocol.file.allow=always -C "$destination" \
    submodule update --init --recursive
  [[ "$(git -C "$destination" rev-parse HEAD)" == "$BASE_COMMIT" ]]
  [[ "$(git -C "$destination/third-party/cutlass" rev-parse HEAD)" == "$CUTLASS_COMMIT" ]]
  [[ "$(git -C "$destination/third-party/fmt" rev-parse HEAD)" == "$FMT_COMMIT" ]]
  [[ -z "$(git -C "$destination" status --porcelain=v1 --untracked-files=no)" ]]
}

STOCK_DIR="${VERIFY_ROOT}/stock-source"
SOURCE_DIR="${VERIFY_ROOT}/candidate-source"
clone_exact "$STOCK_DIR"
clone_exact "$SOURCE_DIR"

git -C "$STOCK_DIR" apply --check "$BUILD_TOOL_PATCH"
git -C "$STOCK_DIR" apply "$BUILD_TOOL_PATCH"
git -C "$STOCK_DIR" diff --check
STOCK_BUILD_DIFF="${VERIFY_ROOT}/stock-build.patch"
git -C "$STOCK_DIR" diff --binary > "$STOCK_BUILD_DIFF"
cmp "$BUILD_TOOL_PATCH" "$STOCK_BUILD_DIFF"

git -C "$SOURCE_DIR" apply --check "$SOURCE_PATCH"
git -C "$SOURCE_DIR" apply "$SOURCE_PATCH"
git -C "$SOURCE_DIR" diff --check
(
  cd "$SOURCE_DIR"
  sha256sum --check "$CORE_HASHES"
)

REAPPLIED_PATCH="${VERIFY_ROOT}/reapplied.patch"
git -C "$SOURCE_DIR" diff --binary > "$REAPPLIED_PATCH"
cmp "$SOURCE_PATCH" "$REAPPLIED_PATCH"

git -C "$SOURCE_DIR" apply --check "$BUILD_TOOL_PATCH"
git -C "$SOURCE_DIR" apply "$BUILD_TOOL_PATCH"
git -C "$SOURCE_DIR" diff --check
[[ "$(git -C "$SOURCE_DIR" diff --name-only | sort | tr '\n' ' ')" == \
  "build_sgl_deep_gemm.sh csrc/apis/gemm.hpp csrc/jit_kernels/heuristics/config.hpp csrc/jit_kernels/heuristics/sm100.hpp csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp csrc/tvm_ffi_api.cpp sgl_deep_gemm/__init__.py " ]]

if [[ -d "${OVERLAY_DIR}/candidate/core_source" ]]; then
  while read -r expected_hash relative; do
    cmp "$SOURCE_DIR/$relative" \
      "${OVERLAY_DIR}/candidate/core_source/$relative"
  done < "$CORE_HASHES"
  (
    cd "${OVERLAY_DIR}/candidate/core_source"
    sha256sum --check "$CORE_HASHES"
  )
fi

if [[ -f "$MANIFEST_PATH" ]]; then
  FRESH_MANIFEST="${VERIFY_ROOT}/fresh-manifest.json"
  env CUDA_VISIBLE_DEVICES= "$HARNESS_PYTHON" "$MANIFEST_TOOL" write \
    --artifact-root "$OVERLAY_DIR" \
    --final-overlay-dir "$OVERLAY_DIR" \
    --stock-source "$STOCK_DIR" \
    --candidate-source "$SOURCE_DIR" \
    --base-repo "$BASE_REPO" \
    --output "$FRESH_MANIFEST"
  "$HARNESS_PYTHON" - "$MANIFEST_PATH" "$FRESH_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

actual = json.loads(Path(sys.argv[1]).read_text())
fresh = json.loads(Path(sys.argv[2]).read_text())
for key in (
    "base",
    "patches",
    "source_identity",
    "candidate_api",
    "runtime_contract",
    "stock",
    "candidate",
):
    assert actual[key] == fresh[key], key
PY
  "$HARNESS_PYTHON" "$MANIFEST_TOOL" verify \
    --manifest "$MANIFEST_PATH" \
    --check-env
fi

echo "PASS exact-post1 em8/BM16/stage11 source reproducibility"
echo "base=${BASE_COMMIT}"
echo "cutlass=${CUTLASS_COMMIT}"
echo "fmt=${FMT_COMMIT}"
echo "source_patch_sha256=${SOURCE_SHA}"
echo "build_tool_patch_sha256=${BUILD_TOOL_SHA}"
echo "overlay=${OVERLAY_DIR}"
