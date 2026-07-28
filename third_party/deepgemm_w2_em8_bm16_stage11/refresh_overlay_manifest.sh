#!/usr/bin/env bash
# Recreate manifest provenance from fresh exact-post1 source trees without rebuilding.
set -euo pipefail

readonly BASE_COMMIT="edcf77b276965de8f03cdc47c23f01b08bf7c7ab"
readonly CUTLASS_COMMIT="f3fde58372d33e9a5650ba7b80fc48b3b49d40c8"
readonly FMT_COMMIT="553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28"
readonly TASK_CACHE_ROOT="/home/qinhaiyan/glm52-v2-goal-runs/cache/26-moe_w2_decode_scoped_bm16/em8_bm16_stage11_v3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASE_REPO="${DEEPGEMM_W2_EM8_BM16_STAGE11_BASE_REPO:-/home/qinhaiyan/DeepGEMM-GLM52}"
HARNESS_PYTHON="${HARNESS_PYTHON:-${REPO_ROOT}/../kernel-harness/.venv/bin/python}"
SOURCE_PATCH="${SCRIPT_DIR}/source.patch"
BUILD_TOOL_PATCH="${SCRIPT_DIR}/build_tool.patch"

SOURCE_SHA="$(sha256sum "$SOURCE_PATCH" | awk '{print $1}')"
BUILD_TOOL_SHA="$(sha256sum "$BUILD_TOOL_PATCH" | awk '{print $1}')"
BUILD_KEY="${BASE_COMMIT:0:12}-${SOURCE_SHA:0:12}-${BUILD_TOOL_SHA:0:12}"
OVERLAY_DIR="${REPO_ROOT}/build/deepgemm-w2-em8-bm16-stage11-v3-overlays/${BUILD_KEY}"
MANIFEST_PATH="${OVERLAY_DIR}/manifest.json"

available_kib="$(df -Pk "$REPO_ROOT" | awk 'NR == 2 {print $4}')"
if (( available_kib < 8 * 1024 * 1024 )); then
  echo "ERROR: fewer than 8 GiB remain; refusing fresh source expansion" >&2
  exit 1
fi
for name in DG_JIT_CACHE_DIR SGLANG_DG_CACHE_DIR TRITON_CACHE_DIR TORCH_EXTENSIONS_DIR; do
  case "$name" in
    DG_JIT_CACHE_DIR) expected="${TASK_CACHE_ROOT}/deepgemm" ;;
    SGLANG_DG_CACHE_DIR) expected="${TASK_CACHE_ROOT}/deepgemm" ;;
    TRITON_CACHE_DIR) expected="${TASK_CACHE_ROOT}/triton" ;;
    TORCH_EXTENSIONS_DIR) expected="${TASK_CACHE_ROOT}/torch_extensions" ;;
  esac
  actual="${!name:-}"
  if [[ -z "$actual" || "$(readlink -m "$actual")" != "$expected" ]]; then
    echo "ERROR: $name must be exactly $expected (got ${actual:-<unset>})" >&2
    exit 1
  fi
done
if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "ERROR: existing overlay artifact is missing: ${MANIFEST_PATH}" >&2
  exit 1
fi

VERIFY_PARENT="${REPO_ROOT}/build/deepgemm-w2-em8-bm16-stage11-v3-manifest-refresh"
mkdir -p "$VERIFY_PARENT"
VERIFY_ROOT="$(mktemp -d "${VERIFY_PARENT}/refresh.${BUILD_KEY}.XXXXXX")"
cleanup() {
  rm -rf -- "$VERIFY_ROOT"
}
trap cleanup EXIT

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
}

STOCK_SOURCE="${VERIFY_ROOT}/stock-source"
CANDIDATE_SOURCE="${VERIFY_ROOT}/candidate-source"
clone_exact "$STOCK_SOURCE"
clone_exact "$CANDIDATE_SOURCE"
git -C "$STOCK_SOURCE" apply --check "$BUILD_TOOL_PATCH"
git -C "$STOCK_SOURCE" apply "$BUILD_TOOL_PATCH"
git -C "$CANDIDATE_SOURCE" apply --check "$SOURCE_PATCH"
git -C "$CANDIDATE_SOURCE" apply "$SOURCE_PATCH"
git -C "$CANDIDATE_SOURCE" apply --check "$BUILD_TOOL_PATCH"
git -C "$CANDIDATE_SOURCE" apply "$BUILD_TOOL_PATCH"

REFRESHED="${VERIFY_ROOT}/manifest.json"
env CUDA_VISIBLE_DEVICES= "$HARNESS_PYTHON" "${SCRIPT_DIR}/overlay_manifest.py" write \
  --artifact-root "$OVERLAY_DIR" \
  --final-overlay-dir "$OVERLAY_DIR" \
  --stock-source "$STOCK_SOURCE" \
  --candidate-source "$CANDIDATE_SOURCE" \
  --base-repo "$BASE_REPO" \
  --output "$REFRESHED"
mv "$REFRESHED" "$MANIFEST_PATH"
"$HARNESS_PYTHON" "${SCRIPT_DIR}/overlay_manifest.py" write-build-provenance \
  --manifest "$MANIFEST_PATH" \
  --output "${SCRIPT_DIR}/build_provenance.json"
"$HARNESS_PYTHON" "${SCRIPT_DIR}/overlay_manifest.py" verify \
  --manifest "$MANIFEST_PATH" \
  --check-env \
  --check-provenance
echo "Refreshed exact-post1 manifest without rebuilding: $MANIFEST_PATH"
