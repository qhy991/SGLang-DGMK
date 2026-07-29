#!/usr/bin/env bash
# Start a fresh process with exact post1 bound as the normal `deep_gemm` import.
set -euo pipefail

if (( $# == 0 )); then
  echo "usage: $0 <command> [args ...]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HARNESS_PYTHON="${HARNESS_PYTHON:-python3}"
if [[ "$HARNESS_PYTHON" != */* ]]; then
  HARNESS_PYTHON="$(command -v "$HARNESS_PYTHON" || true)"
fi
if [[ ! -x "$HARNESS_PYTHON" ]]; then
  echo "ERROR: HARNESS_PYTHON is not executable: ${HARNESS_PYTHON:-<unset>}" >&2
  exit 1
fi
BASE_COMMIT="edcf77b276965de8f03cdc47c23f01b08bf7c7ab"
SOURCE_SHA="$(sha256sum "${SCRIPT_DIR}/source.patch" | awk '{print $1}')"
BUILD_TOOL_SHA="$(sha256sum "${SCRIPT_DIR}/build_tool.patch" | awk '{print $1}')"
BUILD_KEY="${BASE_COMMIT:0:12}-${SOURCE_SHA:0:12}-${BUILD_TOOL_SHA:0:12}"
OVERLAY_DIR="${REPO_ROOT}/build/deepgemm-w2-bm16-overlays/${BUILD_KEY}"
MANIFEST_PATH="$(
  readlink -m "${SGLANG_GLM52_W2_BM16_MANIFEST:-${OVERLAY_DIR}/manifest.json}"
)"
if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "ERROR: W2/BM16 manifest is missing: $MANIFEST_PATH" >&2
  exit 1
fi

mapfile -t ARTIFACT_PATHS < <(
  "$HARNESS_PYTHON" - "$MANIFEST_PATH" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
print(manifest["stock"]["package_dir"])
for name in (
    "DG_JIT_CACHE_DIR",
    "SGLANG_DG_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "TORCH_EXTENSIONS_DIR",
):
    print(manifest["runtime_contract"]["cache_paths"][name])
PY
)
if (( ${#ARTIFACT_PATHS[@]} != 5 )); then
  echo "ERROR: malformed W2/BM16 artifact paths in $MANIFEST_PATH" >&2
  exit 1
fi
STOCK_SITE="$(dirname "${ARTIFACT_PATHS[0]}")"
export DG_JIT_CACHE_DIR="${ARTIFACT_PATHS[1]}"
export SGLANG_DG_CACHE_DIR="${ARTIFACT_PATHS[2]}"
export TRITON_CACHE_DIR="${ARTIFACT_PATHS[3]}"
export TORCH_EXTENSIONS_DIR="${ARTIFACT_PATHS[4]}"
"$HARNESS_PYTHON" "${SCRIPT_DIR}/overlay_manifest.py" verify \
  --manifest "$MANIFEST_PATH" \
  --check-env

export SGLANG_GLM52_W2_BM16_MANIFEST="$MANIFEST_PATH"
export PYTHONPATH="${STOCK_SITE}:${REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
exec "$@"
