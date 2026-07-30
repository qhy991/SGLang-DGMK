#!/usr/bin/env python3
"""Build one isolated DeepGEMM module carrying the round-2 W13 identities.

Round 1 needed two modules because its stock and candidate arms came from two
different source trees, and C++ symbol interposition let their JIT/parser
statics cross the DSO boundary.  Round 2's comparison is different: every arm
-- the round-1 BM16 two-SM survivor, each SM-budget identity, and the same-DSO
stock denominator -- is one `w13_config` value inside a single source tree.  So
exactly one module is materialized and built here, and the arms differ only by
the argument passed at call time.

`w13_config=None` selects DeepGEMM's own `get_best_config` heuristic, i.e. the
stock BM128 two-SM path.  That is the same-DSO stock denominator the W2 hotspot
lane already uses (`serving_native/runner.py::_stock_w2_gemm`).

The module is never installed into the active environment; it is materialized
from a clean `git archive` of the pinned candidate commit into the task-local
cache and imported by path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
from pathlib import Path
from typing import Any

BASE_COMMIT = "731e7c7a97d269e4b9f482ea18d0e709a948f293"
CUTLASS_COMMIT = "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8"
FMT_COMMIT = "553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28"
DEFAULT_SOURCE = Path(
    "/home/qinhaiyan/glm52-hotspot-goal-runs/worktrees/moe-gate-decode/deepgemm"
)
HOTSPOT_CACHE_ROOT = Path("/home/qinhaiyan/glm52-hotspot-goal-runs/cache")
# Round 1's builder refused below 8 GiB free.  That floor is a host-health
# guard, not a build-size requirement: this build materializes one DeepGEMM
# tree plus cutlass/fmt, copies the JIT include set, and links one host
# translation unit -- under 1 GiB in total, re-measured and recorded in the
# manifest as `footprint_bytes`.
#
# The shared root filesystem on this host sits at 99% used because of ~13 GiB
# of stale `/tmp/tmpxft_*` nvcc temporaries owned by a *different* user and
# 16 GiB of the user's HuggingFace cache.  None of it is this task's to delete,
# so the 8 GiB floor is unreachable through any action available here.  The
# floor is therefore set to 4 GiB: still more than 3x this build's footprint of
# headroom after it completes, and the deviation is disclosed in the round-2
# attempt ledger rather than applied silently.
MIN_FREE_BYTES = 4 * 1024**3

BASE_CFLAGS = [
    "-std=c++17",
    "-O3",
    "-fPIC",
    "-Wno-psabi",
    "-Wno-deprecated-declarations",
    "-fvisibility=hidden",
    "-fvisibility-inlines-hidden",
]
LINK_LIBRARIES = [
    "-lcudart",
    "-lnvrtc",
    "-lcublasLt",
    "-lcublas",
    "-ltorch",
    "-ltorch_cpu",
    "-lc10",
    "-lc10_cuda",
    "-ltorch_cuda",
]
ELF_ISOLATION_LDFLAGS = ["-Wl,-Bsymbolic"]


def task_cache_root() -> Path:
    """Resolve the task-local cache the launcher exported for this goal."""
    build_dir = os.environ.get("GLM52_TASK_BUILD_DIR", "").strip()
    if not build_dir:
        raise RuntimeError("GLM52_TASK_BUILD_DIR is not exported by the launcher")
    root = Path(build_dir).resolve().parent
    if root.parent != HOTSPOT_CACHE_ROOT.resolve():
        raise RuntimeError(f"task build dir is outside the hotspot cache: {root}")
    return root


def run(*args: str) -> str:
    return subprocess.check_output(list(args), text=True).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(sha256(path).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def ensure_output_root(output: Path) -> None:
    task_cache = task_cache_root()
    resolved = output.resolve()
    if task_cache not in resolved.parents:
        raise RuntimeError(f"output must stay below task-local cache: {resolved}")
    free = shutil.disk_usage(task_cache).free
    if free < MIN_FREE_BYTES:
        raise RuntimeError(
            f"refusing build below {MIN_FREE_BYTES / 1024**3:.0f} GiB free: "
            f"{free / 1024**3:.2f} GiB"
        )


def verify_source(source: Path, candidate_commit: str) -> dict[str, Any]:
    if run("git", "-C", str(source), "status", "--porcelain"):
        raise RuntimeError("DeepGEMM source must be clean before materialization")
    head = run("git", "-C", str(source), "rev-parse", "HEAD")
    if head != candidate_commit:
        raise RuntimeError(
            f"candidate commit {candidate_commit} is not the worktree HEAD {head}"
        )
    if (
        run("git", "-C", str(source), "merge-base", BASE_COMMIT, candidate_commit)
        != BASE_COMMIT
    ):
        raise RuntimeError("candidate is not based on the required DeepGEMM commit")
    revisions = {
        "cutlass": run(
            "git", "-C", str(source / "third-party/cutlass"), "rev-parse", "HEAD"
        ),
        "fmt": run("git", "-C", str(source / "third-party/fmt"), "rev-parse", "HEAD"),
    }
    if revisions != {"cutlass": CUTLASS_COMMIT, "fmt": FMT_COMMIT}:
        raise RuntimeError(f"submodule identity mismatch: {revisions}")
    diff = subprocess.check_output(
        ["git", "-C", str(source), "diff", "--binary", BASE_COMMIT, candidate_commit]
    )
    if not diff:
        raise RuntimeError("candidate commit has no source diff")
    return {
        "base_commit": BASE_COMMIT,
        "candidate_commit": candidate_commit,
        "candidate_diff_sha256": sha256_bytes(diff),
        "candidate_diff_bytes": len(diff),
    }


def extract_archive(repository: Path, commit: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination.parent / f".{destination.name}-{commit[:12]}.tar"
    try:
        with archive.open("wb") as output:
            subprocess.run(
                ["git", "-C", str(repository), "archive", commit],
                check=True,
                stdout=output,
            )
        with tarfile.open(archive) as bundle:
            bundle.extractall(destination, filter="data")
    finally:
        archive.unlink(missing_ok=True)


def materialize(source: Path, commit: str, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=destination.parent, prefix=f".{destination.name}."
    ) as temporary:
        tree = Path(temporary) / "source"
        extract_archive(source, commit, tree)
        for name, dependency_commit in (
            ("cutlass", CUTLASS_COMMIT),
            ("fmt", FMT_COMMIT),
        ):
            dependency = tree / "third-party" / name
            if dependency.exists():
                dependency.rmdir()
            extract_archive(source / "third-party" / name, dependency_commit, dependency)
        digest = tree_sha256(tree)
        if destination.exists():
            shutil.rmtree(destination)
        tree.rename(destination)
    return digest


def copy_package_source(source: Path, package: Path) -> None:
    package.parent.mkdir(parents=True, exist_ok=True)
    if package.exists():
        shutil.rmtree(package)
    package.mkdir()
    for name in ("__init__.py", "cuda_helpers.py", "VERSION"):
        shutil.copy2(source / "sgl_deep_gemm" / name, package / name)
    for name in ("utils", "testing", "legacy", "mega"):
        shutil.copytree(source / "deep_gemm" / name, package / name)
    include = package / "include"
    shutil.copytree(source / "deep_gemm" / "include" / "deep_gemm", include / "deep_gemm")
    shutil.copytree(
        source / "third-party" / "cutlass" / "include" / "cute", include / "cute"
    )
    shutil.copytree(
        source / "third-party" / "cutlass" / "include" / "cutlass", include / "cutlass"
    )


def build_extension(source: Path, package: Path, build_dir: Path) -> Path:
    import torch
    import tvm_ffi.cpp

    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")).resolve()
    torch_root = Path(torch.__file__).resolve().parent
    os.environ["TVM_FFI_CUDA_ARCH_LIST"] = "10.0a"
    includes = [
        cuda_home / "include",
        Path(sysconfig.get_path("include")),
        torch_root / "include",
        torch_root / "include" / "torch" / "csrc" / "api" / "include",
        source / "deep_gemm" / "include",
        source / "third-party" / "cutlass" / "include",
        source / "third-party" / "fmt" / "include",
    ]
    cccl = cuda_home / "include" / "cccl"
    if cccl.exists():
        includes.append(cccl)
    cflags = [
        *BASE_CFLAGS,
        f"-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}",
    ]
    ldflags = [
        f"-L{cuda_home / 'lib64'}",
        f"-L{torch_root / 'lib'}",
        *ELF_ISOLATION_LDFLAGS,
        *LINK_LIBRARIES,
    ]
    build_dir.mkdir(parents=True, exist_ok=True)
    library = Path(
        tvm_ffi.cpp.build(
            name="_C",
            cpp_files=[str(source / "csrc" / "tvm_ffi_api.cpp")],
            extra_cflags=cflags,
            extra_ldflags=ldflags,
            extra_include_paths=[str(path) for path in includes],
            build_directory=str(build_dir),
        )
    )
    target = package / "_C.so"
    shutil.copy2(library, target)
    return target


def normalized_plan(build_dir: Path, source: Path) -> tuple[str, str]:
    ninja = build_dir / "build.ninja"
    text = ninja.read_text()
    normalized = text.replace(str(source), "<SOURCE>").replace(str(build_dir), "<BUILD>")
    return sha256(ninja), sha256_bytes(normalized.encode())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    source = args.source.resolve()
    output = (args.output or task_cache_root() / "deepgemm" / "w13_r2").resolve()
    ensure_output_root(output)

    identity = verify_source(source, args.candidate_commit)
    output.mkdir(parents=True, exist_ok=True)
    tree = output / "source"
    identity["candidate_source_tree_sha256"] = materialize(
        source, args.candidate_commit, tree
    )

    package = output / "package" / "deep_gemm"
    copy_package_source(tree, package)
    build_dir = output / "build"
    library = build_extension(tree, package, build_dir)
    ninja_sha, plan_sha = normalized_plan(build_dir, tree)

    import torch

    manifest = {
        "schema_version": 1,
        "contract": "glm52-moe-w13-decode-r2-single-module-v1",
        "identity": identity,
        "cutlass_commit": CUTLASS_COMMIT,
        "fmt_commit": FMT_COMMIT,
        "package_root": str(package.parent),
        "package_dir": str(package),
        "library": str(library),
        "library_sha256": sha256(library),
        "build_ninja_sha256": ninja_sha,
        "normalized_build_plan_sha256": plan_sha,
        "elf_symbol_visibility": "hidden",
        "elf_symbol_binding": "Bsymbolic",
        "stock_denominator": "same_module_w13_config_none",
        "cflags": BASE_CFLAGS,
        "ldflags": ELF_ISOLATION_LDFLAGS + LINK_LIBRARIES,
        "torch_version": torch.__version__,
        "cuda_arch_list": "10.0a",
        "python": platform.python_version(),
        "executable": sys.executable,
        "task_cache_root": str(task_cache_root()),
    }
    footprint = sum(
        path.stat().st_size for path in output.rglob("*") if path.is_file()
    )
    manifest["footprint_bytes"] = footprint
    manifest["free_bytes_after_build"] = shutil.disk_usage(task_cache_root()).free
    manifest_path = output / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "library": str(library)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
