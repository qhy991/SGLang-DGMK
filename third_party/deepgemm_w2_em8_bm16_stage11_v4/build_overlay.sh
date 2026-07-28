#!/usr/bin/env bash
# Build exact-post1 stock and exact-post1-plus-em8/BM16/stage11 side by side.
set -euo pipefail

readonly BASE_COMMIT="edcf77b276965de8f03cdc47c23f01b08bf7c7ab"
readonly CUTLASS_COMMIT="f3fde58372d33e9a5650ba7b80fc48b3b49d40c8"
readonly FMT_COMMIT="553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28"
readonly BUILD_ID="glm52-task26-em8-bm16-stage11-v4:sgl-deep-gemm-0.1.4.post1@${BASE_COMMIT}:sm100:e32:m1024:k2048:n6144:expected-m8:bm16:stages11:pdl1:sms148:packed-ue8m0:no-recipe:no-overlap"
readonly TASK_CACHE_ROOT="/home/qinhaiyan/glm52-v2-goal-runs/cache/26-moe_w2_decode_scoped_bm16/em8_bm16_stage11_v4"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SOURCE_PATCH="${SCRIPT_DIR}/source.patch"
BUILD_TOOL_PATCH="${SCRIPT_DIR}/build_tool.patch"
CORE_HASHES="${SCRIPT_DIR}/core_source_hashes.sha256"
BASE_LOCK="${SCRIPT_DIR}/base_lock.json"
MANIFEST_TOOL="${SCRIPT_DIR}/overlay_manifest.py"
REPRO_TOOL="${SCRIPT_DIR}/verify_source_reproducibility.sh"
READY_TOOL="${SCRIPT_DIR}/ready_bundle.py"
BASE_REPO="${DEEPGEMM_W2_EM8_BM16_STAGE11_V4_BASE_REPO:-/home/qinhaiyan/DeepGEMM-GLM52}"
HARNESS_PYTHON="${HARNESS_PYTHON:-${REPO_ROOT}/../kernel-harness/.venv/bin/python}"

for required in \
  "$SOURCE_PATCH" "$BUILD_TOOL_PATCH" "$CORE_HASHES" "$BASE_LOCK" \
  "$MANIFEST_TOOL" "$REPRO_TOOL" "$READY_TOOL" "$HARNESS_PYTHON"; do
  if [[ ! -e "$required" ]]; then
    echo "ERROR: required build input is missing: $required" >&2
    exit 1
  fi
done
if [[ ! -x "$HARNESS_PYTHON" ]]; then
  echo "ERROR: repo-local harness Python is not executable: $HARNESS_PYTHON" >&2
  exit 1
fi

if [[ "${CUDA_VISIBLE_DEVICES+x}" != "x" || -n "${CUDA_VISIBLE_DEVICES}" ]]; then
  echo "ERROR: the v4 build requires explicit CUDA_VISIBLE_DEVICES=''" >&2
  exit 1
fi
for fd_link in /proc/$$/fd/[0-9]*; do
  [[ -e "$fd_link" ]] || continue
  fd_target="$(readlink -f -- "$fd_link" 2>/dev/null || true)"
  if [[ "$fd_target" =~ /glm52-goal-runs/locks/gpu[0-9]+\.lock$ ]]; then
    echo "ERROR: refusing to build while inheriting a GPU scheduler lease" >&2
    exit 1
  fi
done
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
BUNDLE_ROOT="${REPO_ROOT}/build/deepgemm-w2-em8-bm16-stage11-v4-ready-bundles"
BUILD_TMP_PARENT="${REPO_ROOT}/build/deepgemm-w2-em8-bm16-stage11-v4-tmp"
TASK_TMP="${DG_JIT_CACHE_DIR:?DG_JIT_CACHE_DIR must point at the task-local cache}/build_tmp"
TRACKED_PROVENANCE="${SCRIPT_DIR}/build_provenance.json"

if [[ -e "$TRACKED_PROVENANCE" ]]; then
  echo "ERROR: tracked v4 provenance already exists; never overwrite a build identity" >&2
  exit 1
fi

mkdir -p "$BUNDLE_ROOT" "$BUILD_TMP_PARENT" "$TASK_TMP"
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

STAGED_OVERLAY="${BUILD_TMP}/bundle"
mkdir -p \
  "${STAGED_OVERLAY}/stock/site" \
  "${STAGED_OVERLAY}/candidate/site" \
  "${STAGED_OVERLAY}/candidate/core_source"
cp -a "${BUILD_TMP}/stock-source/build/deep_gemm" \
  "${STAGED_OVERLAY}/stock/site/deep_gemm"
cp -a "${BUILD_TMP}/candidate-source/build/deep_gemm" \
  "${STAGED_OVERLAY}/candidate/site/deep_gemm_glm52_w2_em8_bm16_stage11_v4"

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
  --final-overlay-dir "$STAGED_OVERLAY" \
  --stock-source "${BUILD_TMP}/stock-source" \
  --candidate-source "${BUILD_TMP}/candidate-source" \
  --base-repo "$BASE_REPO" \
  --output "${STAGED_OVERLAY}/manifest.json"

"$HARNESS_PYTHON" "$MANIFEST_TOOL" write-build-provenance \
  --manifest "${STAGED_OVERLAY}/manifest.json" \
  --output "$TRACKED_PROVENANCE"
"$HARNESS_PYTHON" "$MANIFEST_TOOL" verify \
  --manifest "${STAGED_OVERLAY}/manifest.json" \
  --check-env \
  --check-provenance
"$REPRO_TOOL" \
  --manifest "${STAGED_OVERLAY}/manifest.json" \
  --record "${STAGED_OVERLAY}/source_replay.json"
BUNDLE_DIGEST="$("$HARNESS_PYTHON" "$READY_TOOL" compute-digest \
  --manifest "${STAGED_OVERLAY}/manifest.json" \
  --sglang-root "$REPO_ROOT" \
  --check-env)"
[[ "$BUNDLE_DIGEST" =~ ^[0-9a-f]{64}$ ]] \
  || { echo "ERROR: invalid content digest: $BUNDLE_DIGEST" >&2; exit 1; }
FINAL_BUNDLE="${BUNDLE_ROOT}/${BUNDLE_DIGEST}"
if [[ -e "$FINAL_BUNDLE" ]]; then
  echo "ERROR: v4 content-addressed bundle already exists: $FINAL_BUNDLE" >&2
  exit 1
fi
mv "$STAGED_OVERLAY" "$FINAL_BUNDLE"
[[ "$("$HARNESS_PYTHON" "$READY_TOOL" compute-digest \
  --manifest "${FINAL_BUNDLE}/manifest.json" \
  --sglang-root "$REPO_ROOT" \
  --check-env)" == "$BUNDLE_DIGEST" ]] \
  || { echo "ERROR: v4 bundle digest changed after publication" >&2; exit 1; }

echo "BUILD_COMPLETE (not READY): ${FINAL_BUNDLE}"
echo "Tracked provenance generated: ${TRACKED_PROVENANCE}"
echo "Commit the generated provenance, then run publish_ready.sh outside a GPU lease."
