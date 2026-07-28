#!/usr/bin/env bash
# Fresh-check the exact post1 em8/BM16/stage11 source delta and identity.
set -euo pipefail

readonly BASE_COMMIT="edcf77b276965de8f03cdc47c23f01b08bf7c7ab"
readonly CUTLASS_COMMIT="f3fde58372d33e9a5650ba7b80fc48b3b49d40c8"
readonly FMT_COMMIT="553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28"
readonly TASK_CACHE_ROOT="/home/qinhaiyan/glm52-v2-goal-runs/cache/26-moe_w2_decode_scoped_bm16/em8_bm16_stage11_v4"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASE_REPO="${DEEPGEMM_W2_EM8_BM16_STAGE11_V4_BASE_REPO:-/home/qinhaiyan/DeepGEMM-GLM52}"
SOURCE_PATCH="${SCRIPT_DIR}/source.patch"
BUILD_TOOL_PATCH="${SCRIPT_DIR}/build_tool.patch"
CORE_HASHES="${SCRIPT_DIR}/core_source_hashes.sha256"
MANIFEST_TOOL="${SCRIPT_DIR}/overlay_manifest.py"
READY_TOOL="${SCRIPT_DIR}/ready_bundle.py"
HARNESS_PYTHON="${HARNESS_PYTHON:-${REPO_ROOT}/../kernel-harness/.venv/bin/python}"

if (( $# != 4 )) || [[ "$1" != "--manifest" || "$3" != "--record" ]]; then
  echo "usage: $0 --manifest MANIFEST --record SOURCE_REPLAY_JSON" >&2
  exit 2
fi
MANIFEST_PATH="$(readlink -m "$2")"
REPLAY_RECORD="$(readlink -m "$4")"
OVERLAY_DIR="$(dirname "$MANIFEST_PATH")"
[[ "$(basename "$MANIFEST_PATH")" == "manifest.json" ]] \
  || { echo "ERROR: manifest must be named manifest.json" >&2; exit 1; }
[[ ! -e "$REPLAY_RECORD" ]] \
  || { echo "ERROR: refusing to overwrite source replay record: $REPLAY_RECORD" >&2; exit 1; }

VERIFY_PARENT="${REPO_ROOT}/build/deepgemm-w2-em8-bm16-stage11-v4-replay"
mkdir -p "$VERIFY_PARENT"
VERIFY_ROOT="$(mktemp -d "${VERIFY_PARENT}/replay.XXXXXX")"
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
assert actual == fresh
PY
  "$HARNESS_PYTHON" "$MANIFEST_TOOL" verify \
    --manifest "$MANIFEST_PATH" \
    --check-env \
    --check-provenance
else
  echo "ERROR: manifest is missing: $MANIFEST_PATH" >&2
  exit 1
fi

"$HARNESS_PYTHON" "$READY_TOOL" write-source-replay \
  --manifest "$MANIFEST_PATH" \
  --output "$REPLAY_RECORD" \
  --sglang-root "$REPO_ROOT"

echo "PASS exact-post1 em8/BM16/stage11 source reproducibility"
echo "base=${BASE_COMMIT}"
echo "cutlass=${CUTLASS_COMMIT}"
echo "fmt=${FMT_COMMIT}"
echo "source_patch_sha256=$(sha256sum "$SOURCE_PATCH" | awk '{print $1}')"
echo "build_tool_patch_sha256=$(sha256sum "$BUILD_TOOL_PATCH" | awk '{print $1}')"
echo "overlay=${OVERLAY_DIR}"
echo "source_replay_record=${REPLAY_RECORD}"
