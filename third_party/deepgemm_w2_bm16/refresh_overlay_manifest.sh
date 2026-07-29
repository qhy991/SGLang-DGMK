#!/usr/bin/env bash
# Recreate manifest provenance from fresh exact-post1 source trees without rebuilding.
set -euo pipefail

readonly BASE_COMMIT="edcf77b276965de8f03cdc47c23f01b08bf7c7ab"
readonly CUTLASS_COMMIT="f3fde58372d33e9a5650ba7b80fc48b3b49d40c8"
readonly FMT_COMMIT="553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASE_REPO="${DEEPGEMM_W2_BM16_BASE_REPO:-${REPO_ROOT}/../DeepGEMM-GLM52}"
HARNESS_PYTHON="${HARNESS_PYTHON:-python3}"
if [[ "$HARNESS_PYTHON" != */* ]]; then
  HARNESS_PYTHON="$(command -v "$HARNESS_PYTHON" || true)"
fi
if [[ ! -x "$HARNESS_PYTHON" ]]; then
  echo "ERROR: HARNESS_PYTHON is not executable: ${HARNESS_PYTHON:-<unset>}" >&2
  exit 1
fi
SOURCE_PATCH="${SCRIPT_DIR}/source.patch"
BUILD_TOOL_PATCH="${SCRIPT_DIR}/build_tool.patch"

SOURCE_SHA="$(sha256sum "$SOURCE_PATCH" | awk '{print $1}')"
BUILD_TOOL_SHA="$(sha256sum "$BUILD_TOOL_PATCH" | awk '{print $1}')"
BUILD_KEY="${BASE_COMMIT:0:12}-${SOURCE_SHA:0:12}-${BUILD_TOOL_SHA:0:12}"
OVERLAY_DIR="${REPO_ROOT}/build/deepgemm-w2-bm16-overlays/${BUILD_KEY}"
MANIFEST_PATH="${OVERLAY_DIR}/manifest.json"

available_kib="$(df -Pk "$REPO_ROOT" | awk 'NR == 2 {print $4}')"
if (( available_kib < 8 * 1024 * 1024 )); then
  echo "ERROR: fewer than 8 GiB remain; refusing fresh source expansion" >&2
  exit 1
fi
if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "ERROR: existing overlay artifact is missing: ${MANIFEST_PATH}" >&2
  exit 1
fi
mapfile -t CACHE_PATHS < <(
  "$HARNESS_PYTHON" - "$MANIFEST_PATH" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
for name in (
    "DG_JIT_CACHE_DIR",
    "SGLANG_DG_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "TORCH_EXTENSIONS_DIR",
):
    print(manifest["runtime_contract"]["cache_paths"][name])
PY
)
if (( ${#CACHE_PATHS[@]} != 4 )); then
  echo "ERROR: malformed cache paths in ${MANIFEST_PATH}" >&2
  exit 1
fi
export DG_JIT_CACHE_DIR="${CACHE_PATHS[0]}"
export SGLANG_DG_CACHE_DIR="${CACHE_PATHS[1]}"
export TRITON_CACHE_DIR="${CACHE_PATHS[2]}"
export TORCH_EXTENSIONS_DIR="${CACHE_PATHS[3]}"
CACHE_ROOT="$(dirname "$DG_JIT_CACHE_DIR")"

VERIFY_PARENT="${REPO_ROOT}/build/deepgemm-w2-manifest-refresh"
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
  --cache-root "$CACHE_ROOT" \
  --output "$REFRESHED"
mv "$REFRESHED" "$MANIFEST_PATH"
"$HARNESS_PYTHON" "${SCRIPT_DIR}/overlay_manifest.py" verify \
  --manifest "$MANIFEST_PATH" \
  --check-env
echo "Refreshed exact-post1 manifest without rebuilding: $MANIFEST_PATH"
