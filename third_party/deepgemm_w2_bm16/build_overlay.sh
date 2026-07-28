#!/usr/bin/env bash
# Build exact-post1 stock and exact-post1-plus-BM16 side by side, without CUDA.
set -euo pipefail

readonly BASE_COMMIT="edcf77b276965de8f03cdc47c23f01b08bf7c7ab"
readonly CUTLASS_COMMIT="f3fde58372d33e9a5650ba7b80fc48b3b49d40c8"
readonly FMT_COMMIT="553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28"
readonly BUILD_ID="glm52-w2-bm16-v2:sgl-deep-gemm-0.1.4.post1@${BASE_COMMIT}:sm100:e32:m1024:k2048:n6144:bm16:pdl1:sms148:no-recipe:no-overlap"
readonly TASK_CACHE_ROOT="/home/qinhaiyan/glm52-v2-goal-runs/cache/26-moe_w2_decode_scoped_bm16"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SOURCE_PATCH="${SCRIPT_DIR}/source.patch"
BUILD_TOOL_PATCH="${SCRIPT_DIR}/build_tool.patch"
CORE_HASHES="${SCRIPT_DIR}/core_source_hashes.sha256"
BASE_LOCK="${SCRIPT_DIR}/base_lock.json"
MANIFEST_TOOL="${SCRIPT_DIR}/overlay_manifest.py"
REPRO_TOOL="${SCRIPT_DIR}/verify_source_reproducibility.sh"
BASE_REPO="${DEEPGEMM_W2_BM16_BASE_REPO:-/home/qinhaiyan/DeepGEMM-GLM52}"
HARNESS_PYTHON="${HARNESS_PYTHON:-${REPO_ROOT}/../kernel-harness/.venv/bin/python}"

for required in \
  "$SOURCE_PATCH" "$BUILD_TOOL_PATCH" "$CORE_HASHES" "$BASE_LOCK" \
  "$MANIFEST_TOOL" "$REPRO_TOOL" "$HARNESS_PYTHON"; do
  if [[ ! -e "$required" ]]; then
    echo "ERROR: required build input is missing: $required" >&2
    exit 1
  fi
done
if [[ ! -x "$HARNESS_PYTHON" ]]; then
  echo "ERROR: repo-local harness Python is not executable: $HARNESS_PYTHON" >&2
  exit 1
fi
if [[ ! -d "$BASE_REPO/.git" && ! -f "$BASE_REPO/.git" ]]; then
  echo "ERROR: DeepGEMM base repository is missing: $BASE_REPO" >&2
  exit 1
fi
if [[ "$(git -C "$BASE_REPO" cat-file -t "$BASE_COMMIT" 2>/dev/null || true)" != "commit" ]]; then
  echo "ERROR: exact post1 commit is unavailable: $BASE_COMMIT" >&2
  exit 1
fi

require_disk_headroom() {
  local available_kib
  available_kib="$(df -Pk "$REPO_ROOT" | awk 'NR == 2 {print $4}')"
  if (( available_kib < 8 * 1024 * 1024 )); then
    echo "ERROR: fewer than 8 GiB remain; refusing to expand build caches" >&2
    exit 1
  fi
}
require_disk_headroom

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

SOURCE_SHA="$(sha256sum "$SOURCE_PATCH" | awk '{print $1}')"
BUILD_TOOL_SHA="$(sha256sum "$BUILD_TOOL_PATCH" | awk '{print $1}')"
CORE_HASHES_SHA="$(sha256sum "$CORE_HASHES" | awk '{print $1}')"
BUILD_KEY="${BASE_COMMIT:0:12}-${SOURCE_SHA:0:12}-${BUILD_TOOL_SHA:0:12}"
OVERLAY_ROOT="${REPO_ROOT}/build/deepgemm-w2-bm16-overlays"
OVERLAY_DIR="${OVERLAY_ROOT}/${BUILD_KEY}"
MANIFEST_PATH="${OVERLAY_DIR}/manifest.json"
STOCK_PACKAGE="${OVERLAY_DIR}/stock/site/deep_gemm"
CANDIDATE_PACKAGE="${OVERLAY_DIR}/candidate/site/deep_gemm_glm52_w2_bm16"
BUILD_TMP_PARENT="${REPO_ROOT}/build/deepgemm-w2-bm16-tmp"
TASK_TMP="${DG_JIT_CACHE_DIR:?DG_JIT_CACHE_DIR must point at the task-local cache}/build_tmp"

if [[ -f "$MANIFEST_PATH" ]]; then
  "$HARNESS_PYTHON" "$MANIFEST_TOOL" verify \
    --manifest "$MANIFEST_PATH" \
    --check-env \
    --check-provenance
  "$REPRO_TOOL" "$MANIFEST_PATH"
  echo "Reusing verified exact-post1 overlays: $OVERLAY_DIR"
  exit 0
fi
if [[ -e "$OVERLAY_DIR" ]]; then
  echo "ERROR: incomplete overlay exists; inspect without overwriting: $OVERLAY_DIR" >&2
  exit 1
fi

mkdir -p "$OVERLAY_ROOT" "$BUILD_TMP_PARENT" "$TASK_TMP"
BUILD_TMP="$(mktemp -d "${BUILD_TMP_PARENT}/build.${BUILD_KEY}.XXXXXX")"
cleanup() {
  rm -rf -- "$BUILD_TMP"
}
trap cleanup EXIT

clone_exact_post1() {
  local destination="$1"
  git clone --quiet --no-hardlinks "$BASE_REPO" "$destination"
  git -C "$destination" checkout --quiet --detach "$BASE_COMMIT"
  git -C "$destination" config submodule.third-party/cutlass.url \
    "${BASE_REPO}/third-party/cutlass"
  git -C "$destination" config submodule.third-party/fmt.url \
    "${BASE_REPO}/third-party/fmt"
  git -c protocol.file.allow=always -C "$destination" \
    submodule update --init --recursive
  [[ "$(git -C "$destination/third-party/cutlass" rev-parse HEAD)" == "$CUTLASS_COMMIT" ]]
  [[ "$(git -C "$destination/third-party/fmt" rev-parse HEAD)" == "$FMT_COMMIT" ]]
}

build_one() {
  local role="$1"
  local source_dir="${BUILD_TMP}/${role}-source"
  clone_exact_post1 "$source_dir"

  if [[ "$role" == "candidate" ]]; then
    git -C "$source_dir" apply --check "$SOURCE_PATCH"
    git -C "$source_dir" apply "$SOURCE_PATCH"
    (
      cd "$source_dir"
      sha256sum --check "$CORE_HASHES"
    )
  fi
  git -C "$source_dir" apply --check "$BUILD_TOOL_PATCH"
  git -C "$source_dir" apply "$BUILD_TOOL_PATCH"
  git -C "$source_dir" diff --check

  require_disk_headroom
  env \
    CUDA_VISIBLE_DEVICES= \
    TVM_FFI_CUDA_ARCH_LIST=100 \
    PATH="$(dirname "$HARNESS_PYTHON"):/usr/local/cuda/bin:/usr/bin:/bin" \
    TMPDIR="$TASK_TMP" \
    SGLANG_DEEPGEMM_BUILD_WHEEL=0 \
    bash "${source_dir}/build_sgl_deep_gemm.sh"
}

build_one stock
build_one candidate

STAGED_OVERLAY="${BUILD_TMP}/overlay"
mkdir -p \
  "${STAGED_OVERLAY}/stock/site" \
  "${STAGED_OVERLAY}/candidate/site" \
  "${STAGED_OVERLAY}/candidate/core_source"
cp -a "${BUILD_TMP}/stock-source/build/deep_gemm" \
  "${STAGED_OVERLAY}/stock/site/deep_gemm"
cp -a "${BUILD_TMP}/candidate-source/build/deep_gemm" \
  "${STAGED_OVERLAY}/candidate/site/deep_gemm_glm52_w2_bm16"

while read -r expected_hash relative; do
  [[ -n "$expected_hash" && -n "$relative" ]]
  mkdir -p "${STAGED_OVERLAY}/candidate/core_source/$(dirname "$relative")"
  cp "${BUILD_TMP}/candidate-source/${relative}" \
    "${STAGED_OVERLAY}/candidate/core_source/${relative}"
  cmp "${BUILD_TMP}/candidate-source/${relative}" \
    "${STAGED_OVERLAY}/candidate/core_source/${relative}"
done < "$CORE_HASHES"
(
  cd "${STAGED_OVERLAY}/candidate/core_source"
  sha256sum --check "$CORE_HASHES"
)

env CUDA_VISIBLE_DEVICES= "$HARNESS_PYTHON" "$MANIFEST_TOOL" write \
  --artifact-root "$STAGED_OVERLAY" \
  --final-overlay-dir "$OVERLAY_DIR" \
  --stock-source "${BUILD_TMP}/stock-source" \
  --candidate-source "${BUILD_TMP}/candidate-source" \
  --base-repo "$BASE_REPO" \
  --output "${STAGED_OVERLAY}/manifest.json"

mv "$STAGED_OVERLAY" "$OVERLAY_DIR"
echo "Exact-post1 stock and W2/BM16 candidate ready: $MANIFEST_PATH"
echo "Stock package: $STOCK_PACKAGE"
echo "Candidate package: $CANDIDATE_PACKAGE"
