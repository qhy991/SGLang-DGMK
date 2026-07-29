#!/usr/bin/env bash
# Bind the manifest's same-source post1 stock module before SGLang imports.
set -euo pipefail

if (( $# == 0 )); then
  echo "usage: $0 <command> [args ...]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ "$PYTHON_BIN" != */* ]]; then
  PYTHON_BIN="$(command -v "$PYTHON_BIN" || true)"
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: PYTHON_BIN is not executable: ${PYTHON_BIN:-<unset>}" >&2
  exit 1
fi

MANIFEST_PATH="$(
  readlink -m "${SGLANG_GLM52_W13_DECODE_MANIFEST:-${REPO_ROOT}/build/deepgemm-w13-variants/manifest.json}"
)"
if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "ERROR: W13 manifest is missing: $MANIFEST_PATH" >&2
  exit 1
fi

mapfile -t STOCK_PATHS < <(
  "$PYTHON_BIN" - "$MANIFEST_PATH" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


manifest_path = Path(sys.argv[1]).resolve()
manifest = json.loads(manifest_path.read_text())
expected_source = {
    "commit": "edcf77b276965de8f03cdc47c23f01b08bf7c7ab",
    "cutlass_commit": "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8",
    "fmt_commit": "553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28",
    "candidate_patch_sha256": (
        "056c90d416f2278c23bcb495d41ecf28f82e7047f220f4acc8321e8f1436a458"
    ),
    "stock_source_tree_sha256": (
        "4bfc233540d0478bf88860d924c53e105be29e01ddd039a68a8c5242addb2af5"
    ),
    "candidate_source_tree_sha256": (
        "1e23f011428ca83bcc3fe1a2e990b62ed82abbba65a291a61db9cf4a729cf657"
    ),
}
if manifest.get("schema_version") != 2:
    raise RuntimeError(f"W13 manifest schema mismatch: {manifest_path}")
source = manifest.get("source", {})
for name, expected in expected_source.items():
    if source.get(name) != expected:
        raise RuntimeError(f"W13 manifest source mismatch: {name}")
build = manifest.get("build", {})
if build.get("stock_candidate_command_identical") is not True:
    raise RuntimeError("W13 stock/candidate build commands are not attested equal")
plan_sha256 = build.get("normalized_build_plan_sha256")
if not isinstance(plan_sha256, str) or len(plan_sha256) != 64:
    raise RuntimeError("W13 normalized build-plan identity is missing")

variants = manifest.get("variants", {})
for role, package_name in (
    ("stock", "deep_gemm"),
    ("candidate", "deep_gemm_w13_candidate"),
):
    record = variants.get(role, {})
    package = Path(record.get("package", "")).resolve()
    shared_object = Path(record.get("shared_object", "")).resolve()
    cache = Path(record.get("jit_cache", "")).resolve()
    init_py = package / "__init__.py"
    build_ninja = Path(record.get("build_ninja", "")).resolve()
    if (
        package.name != package_name
        or shared_object != package / "_C.so"
        or not init_py.is_file()
        or not shared_object.is_file()
        or not cache.is_dir()
        or not build_ninja.is_file()
    ):
        raise RuntimeError(f"W13 {role} artifact is incomplete")
    if sha256(init_py) != record.get("package_init_sha256"):
        raise RuntimeError(f"W13 {role} package hash mismatch")
    if sha256(shared_object) != record.get("shared_object_sha256"):
        raise RuntimeError(f"W13 {role} shared-object hash mismatch")
    if (
        sha256(build_ninja) != record.get("build_ninja_sha256")
        or record.get("normalized_build_plan_sha256") != plan_sha256
    ):
        raise RuntimeError(f"W13 {role} build-plan identity mismatch")

stock = manifest["variants"]["stock"]
print(stock["package"])
print(stock["jit_cache"])
PY
)
if (( ${#STOCK_PATHS[@]} != 2 )); then
  echo "ERROR: malformed W13 stock paths in $MANIFEST_PATH" >&2
  exit 1
fi
STOCK_PACKAGE="$(readlink -m "${STOCK_PATHS[0]}")"
STOCK_CACHE="$(readlink -m "${STOCK_PATHS[1]}")"
if [[ "$(basename "$STOCK_PACKAGE")" != "deep_gemm" ]]; then
  echo "ERROR: W13 stock package is not importable as deep_gemm: $STOCK_PACKAGE" >&2
  exit 1
fi
if [[ ! -f "${STOCK_PACKAGE}/__init__.py" || ! -f "${STOCK_PACKAGE}/_C.so" ]]; then
  echo "ERROR: W13 stock package is incomplete: $STOCK_PACKAGE" >&2
  exit 1
fi
if [[ ! -d "$STOCK_CACHE" ]]; then
  echo "ERROR: W13 stock JIT cache is missing: $STOCK_CACHE" >&2
  exit 1
fi

export DG_JIT_USE_NVRTC=0
export SGLANG_DG_USE_NVRTC=0
export DG_JIT_CACHE_DIR="$STOCK_CACHE"
export SGLANG_DG_CACHE_DIR="$STOCK_CACHE"
export SGLANG_GLM52_W13_DECODE_MANIFEST="$MANIFEST_PATH"
export PYTHONPATH="$(dirname "$STOCK_PACKAGE"):${REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
exec "$@"
