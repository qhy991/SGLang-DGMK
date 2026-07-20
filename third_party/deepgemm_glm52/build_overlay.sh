#!/usr/bin/env bash
# Build DeepGEMM-GLM52 into a commit-partitioned overlay without touching
# stock sgl-deep-gemm in any venv site-packages.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORK_ROOT="${DEEPGEMM_GLM52_ROOT:-${SCRIPT_DIR}/../DeepGEMM-GLM52}"
FORK_ROOT="$(cd "$FORK_ROOT" && pwd)"
HPYTHON="${HARNESS_PYTHON:-python3}"
OVERLAY_ROOT="${FORK_ROOT}/overlays"

if [[ -x "$HPYTHON" ]]; then
  :
elif command -v "$HPYTHON" >/dev/null 2>&1; then
  HPYTHON="$(command -v "$HPYTHON")"
else
  echo "ERROR: python missing: $HPYTHON (set HARNESS_PYTHON)" >&2
  exit 1
fi
if [[ ! -f "$FORK_ROOT/build_sgl_deep_gemm.sh" ]]; then
  echo "ERROR: fork root missing build_sgl_deep_gemm.sh: $FORK_ROOT" >&2
  exit 1
fi

# Resolve commit id: nested git > GLM52_OPT_COMMIT.txt > env > "unknown"
if [[ -d "$FORK_ROOT/.git" ]] || [[ -f "$FORK_ROOT/.git" ]]; then
  COMMIT="$(git -C "$FORK_ROOT" rev-parse HEAD)"
elif [[ -n "${DEEPGEMM_GLM52_COMMIT:-}" ]]; then
  COMMIT="$DEEPGEMM_GLM52_COMMIT"
elif [[ -f "$FORK_ROOT/GLM52_OPT_COMMIT.txt" ]]; then
  COMMIT="$(tr -d '[:space:]' < "$FORK_ROOT/GLM52_OPT_COMMIT.txt")"
else
  echo "ERROR: cannot resolve DeepGEMM commit (no .git / GLM52_OPT_COMMIT.txt / DEEPGEMM_GLM52_COMMIT)" >&2
  exit 1
fi
SHORT="${COMMIT:0:7}"

cd "$FORK_ROOT"
OVERLAY_DIR="${OVERLAY_ROOT}/${COMMIT}"
SITE_DIR="${OVERLAY_DIR}/site"
PKG_DIR="${SITE_DIR}/deep_gemm_experimental"
JIT_CACHE_DIR="${OVERLAY_DIR}/jit_cache"
MANIFEST_OUT="${OVERLAY_DIR}/provenance.json"
WHEEL_STAGE="${OVERLAY_DIR}/wheel_stage"
SHARED_MANIFEST="${SCRIPT_DIR}/manifest.json"

echo "=== DeepGEMM-GLM52 overlay build ==="
echo "fork:    $FORK_ROOT"
echo "commit:  $COMMIT ($SHORT)"
echo "python:  $HPYTHON"
echo "overlay: $OVERLAY_DIR"

# Never install into the harness/site venv.
export PYTHONNOUSERSITE=1
unset VIRTUAL_ENV || true

# Stage package via upstream sgl build script.
export PATH="$(dirname "$HPYTHON"):$PATH"
rm -rf "${FORK_ROOT}/build" "${FORK_ROOT}/dist"
mkdir -p "${OVERLAY_DIR}"
(
  cd "$FORK_ROOT"
  PYTHON_EXE="$HPYTHON" bash -c '
    set -euo pipefail
    export PYTHON_EXE
    bash ./build_sgl_deep_gemm.sh
  '
)

# Locate staged build package (build/deep_gemm from build_sgl_deep_gemm.sh)
STAGED="${FORK_ROOT}/build/deep_gemm"
if [[ ! -f "${STAGED}/_C.so" ]]; then
  echo "ERROR: staged _C.so missing at ${STAGED}" >&2
  ls -la "${FORK_ROOT}/build" || true
  exit 1
fi

rm -rf "$SITE_DIR" "$WHEEL_STAGE"
mkdir -p "$SITE_DIR" "$JIT_CACHE_DIR" "$WHEEL_STAGE"
cp -a "$STAGED" "$PKG_DIR"

# Preserve wheel artifact if present (audit only; not installed into venv).
if compgen -G "${FORK_ROOT}/dist/sgl_deep_gemm"*.whl > /dev/null \
  || compgen -G "${FORK_ROOT}/dist/sgl"*.whl > /dev/null; then
  cp -a "${FORK_ROOT}/dist/"*.whl "$WHEEL_STAGE/" 2>/dev/null || true
fi

# Write provenance + shared loader manifest (absolute paths for runtime).
"$HPYTHON" - "$FORK_ROOT" "$COMMIT" "$OVERLAY_DIR" "$PKG_DIR" "$JIT_CACHE_DIR" "$MANIFEST_OUT" "$SHARED_MANIFEST" <<'PY'
import json, sys, hashlib, platform, subprocess
from pathlib import Path

fork, commit, overlay, pkg, jit, out, shared = sys.argv[1:8]
cso = Path(pkg) / "_C.so"
h = hashlib.sha256(cso.read_bytes()).hexdigest() if cso.exists() else None
fork = str(Path(fork).resolve())
overlay = str(Path(overlay).resolve())
pkg = str(Path(pkg).resolve())
jit = str(Path(jit).resolve())
out = str(Path(out).resolve())

def _git_rev(path: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None

def _git_branch(path: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "--abbrev-ref", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None

def _git_dirty(path: str) -> bool | None:
    try:
        return bool(
            subprocess.check_output(
                ["git", "-C", path, "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
    except Exception:
        return None

try:
    import torch
    torch_ver = torch.__version__
    torch_cuda = torch.version.cuda
    cxx11 = bool(torch.compiled_with_cxx11_abi())
except Exception as e:
    torch_ver = f"unavailable: {e}"
    torch_cuda = None
    cxx11 = None

info = {
    "remote": "https://github.com/sgl-project/DeepGEMM",
    "tag": "v0.1.4",
    "upstream_commit": "731e7c7a97d269e4b9f482ea18d0e709a948f293",
    "branch": _git_branch(fork),
    "commit": commit,
    "dirty": _git_dirty(fork),
    "vendored_without_nested_git": not (Path(fork) / ".git").exists(),
    "submodules": {
        "third-party/cutlass": _git_rev(f"{fork}/third-party/cutlass"),
        "third-party/fmt": _git_rev(f"{fork}/third-party/fmt"),
    },
    "python_exe": sys.executable,
    "python": sys.version.split()[0],
    "torch": torch_ver,
    "torch_cuda": torch_cuda,
    "cxx11_abi": cxx11,
    "platform": platform.platform(),
    "sgl_version_file": Path(fork, "sgl_deep_gemm/VERSION").read_text().strip(),
    "overlay_dir": overlay,
    "package_dir": pkg,
    "import_name": "deep_gemm_experimental",
    "jit_cache_dir": jit,
    "_C.so_sha256": h,
    "stock_untouched": True,
}
Path(out).write_text(json.dumps(info, indent=2) + "\n")
# Shared manifest next to this tooling (absolute paths for the loader).
Path(shared).write_text(json.dumps({
    "active_commit": commit,
    "overlay_dir": overlay,
    "package_dir": pkg,
    "jit_cache_dir": jit,
    "import_name": "deep_gemm_experimental",
    "fork_root": fork,
    "provenance": out,
}, indent=2) + "\n")
print(json.dumps(info, indent=2))
PY

# Symlink "current" for convenience
ln -sfn "$OVERLAY_DIR" "${OVERLAY_ROOT}/current"

echo "=== overlay ready ==="
echo "package: $PKG_DIR"
echo "jit:     $JIT_CACHE_DIR"
echo "manifest:${SHARED_MANIFEST}"
ls -la "$PKG_DIR" | head
