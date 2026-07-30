#!/usr/bin/env python3
"""Build isolated same-base stock and BM16 W13 DeepGEMM modules.

The stock arm is materialized from the pinned DeepGEMM v0.1.4 commit.  The
candidate arm is materialized from the dedicated task branch.  Both use the
same dependency commits, compiler function, flags, and clean build procedure;
neither module is installed into the active Python environment.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
from pathlib import Path
from typing import Any

BASE_COMMIT = "731e7c7a97d269e4b9f482ea18d0e709a948f293"
CANDIDATE_COMMIT = "87e0359edbb461181d3bba218442132007b9a738"
CANDIDATE_DIFF_SHA256 = (
    "465c8373c0a37970225a0e93267b6c399431b23e22cf35b4511db2308df98092"
)
CUTLASS_COMMIT = "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8"
FMT_COMMIT = "553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28"
EXPECTED_BASE_BLOBS = {
    "csrc/apis/gemm.hpp": "0840d64249e2a5a4a994d495e8320a0fff26bad9ca107426a1a1226e7d621186",
    "csrc/jit_kernels/heuristics/sm100.hpp": (
        "487cac2ff19027c781b08e9a0391836e77c03cdffcb7ceb3346d8633c8eb0884"
    ),
    "csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp": (
        "cca1ddb5b5787942c31b39a9d5618929ee609c6c3b57b877fe636df39540366b"
    ),
    "csrc/tvm_ffi_api.cpp": (
        "d1e5dbd833f257d2c4be516772404c02f1747247eef5075315ff2d1220a64c1f"
    ),
    "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh": (
        "9c1e70677ede6ba09ab98e629482da7874182f8227907382efe0a81658da5a37"
    ),
    "sgl_deep_gemm/__init__.py": (
        "243eeaa71fa65cecaddd7298245438cb371ca765d7bf914a9427e132be8d5f26"
    ),
}
_REPO_ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_PATCH = Path(__file__).resolve().with_name(
    "deepgemm_w13_bm16_87e0359.patch.b64"
)
DEFAULT_SOURCE = os.environ.get("SGLANG_GLM52_W13_DEEPGEMM_SOURCE")
DEFAULT_OUTPUT = _REPO_ROOT / ".cache" / "glm52_w13_variants"
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


def run(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
        relative = path.relative_to(root).as_posix().encode()
        if path.is_symlink():
            digest.update(
                b"L\0" + relative + b"\0" + os.readlink(path).encode() + b"\0"
            )
        elif path.is_file():
            mode = stat.S_IMODE(path.stat().st_mode)
            digest.update(b"F\0" + relative + b"\0" + f"{mode:o}".encode() + b"\0")
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            digest.update(b"\0")
    return digest.hexdigest()


def ensure_output_root(output: Path) -> None:
    resolved = output.resolve()
    forbidden = {
        Path("/").resolve(),
        Path.home().resolve(),
        _REPO_ROOT.resolve(),
    }
    if resolved in forbidden or len(resolved.parts) < 3:
        raise RuntimeError(f"refusing unsafe W13 output root: {resolved}")
    existing = resolved
    while not existing.exists():
        existing = existing.parent
    free = shutil.disk_usage(existing).free
    if free < 8 * 1024**3:
        raise RuntimeError(
            f"refusing build below 8 GiB free: {free / 1024**3:.2f} GiB"
        )


def verify_source(source: Path) -> dict[str, Any]:
    if run("git", "-C", str(source), "status", "--porcelain"):
        raise RuntimeError("DeepGEMM source must be clean before materialization")
    subprocess.run(
        ["git", "-C", str(source), "cat-file", "-e", f"{BASE_COMMIT}^{{commit}}"],
        check=True,
    )
    revisions = {
        "cutlass": run("git", "-C", str(source / "third-party/cutlass"), "rev-parse", "HEAD"),
        "fmt": run("git", "-C", str(source / "third-party/fmt"), "rev-parse", "HEAD"),
    }
    if revisions != {"cutlass": CUTLASS_COMMIT, "fmt": FMT_COMMIT}:
        raise RuntimeError(f"submodule identity mismatch: {revisions}")
    base_blobs = {
        relative: sha256_bytes(
            subprocess.check_output(
                ["git", "-C", str(source), "show", f"{BASE_COMMIT}:{relative}"]
            )
        )
        for relative in EXPECTED_BASE_BLOBS
    }
    if base_blobs != EXPECTED_BASE_BLOBS:
        raise RuntimeError(f"base blob identity mismatch: {base_blobs}")
    diff = base64.b64decode(
        "".join(CANDIDATE_PATCH.read_text().splitlines()),
        validate=True,
    )
    if sha256_bytes(diff) != CANDIDATE_DIFF_SHA256:
        raise RuntimeError("bundled W13 candidate patch identity mismatch")
    return {
        "candidate_commit": CANDIDATE_COMMIT,
        "candidate_diff_sha256": sha256_bytes(diff),
        "candidate_diff_bytes": len(diff),
        "base_blob_sha256": base_blobs,
        "candidate_diff": diff,
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


def materialize(
    source: Path,
    commit: str,
    destination: Path,
    *,
    candidate_patch: bytes | None = None,
) -> str:
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
            extract_archive(
                source / "third-party" / name,
                dependency_commit,
                dependency,
            )
        if candidate_patch is not None:
            patch_path = Path(temporary) / "candidate.patch"
            patch_path.write_bytes(candidate_patch)
            subprocess.run(
                [
                    "git",
                    "apply",
                    "--unsafe-paths",
                    "--directory",
                    str(tree),
                    str(patch_path),
                ],
                cwd=_REPO_ROOT,
                check=True,
            )
            # ``git apply`` recreates modified files under the caller's umask.
            # The measured candidate commit records these six files as 0644,
            # and file mode participates in the signed source-tree identity.
            for relative in EXPECTED_BASE_BLOBS:
                (tree / relative).chmod(0o644)
        digest = tree_sha256(tree)
        if destination.exists():
            shutil.rmtree(destination)
        tree.rename(destination)
    return digest


def audit_materialization(
    source: Path, candidate_patch: bytes, scratch_parent: Path
) -> dict[str, str]:
    scratch_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=scratch_parent, prefix=".w13-materialization-audit."
    ) as temporary:
        root = Path(temporary)
        first = {
            "stock_source_tree_sha256": materialize(
                source, BASE_COMMIT, root / "stock-a"
            ),
            "candidate_source_tree_sha256": materialize(
                source,
                BASE_COMMIT,
                root / "candidate-a",
                candidate_patch=candidate_patch,
            ),
        }
        second = {
            "stock_source_tree_sha256": materialize(
                source, BASE_COMMIT, root / "stock-b"
            ),
            "candidate_source_tree_sha256": materialize(
                source,
                BASE_COMMIT,
                root / "candidate-b",
                candidate_patch=candidate_patch,
            ),
        }
        if first != second:
            raise RuntimeError("source materialization is not byte-identical")
        if first["stock_source_tree_sha256"] == first["candidate_source_tree_sha256"]:
            raise RuntimeError("candidate tree aliases stock")
        return first


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
    shutil.copytree(
        source / "deep_gemm" / "include" / "deep_gemm",
        include / "deep_gemm",
    )
    shutil.copytree(
        source / "third-party" / "cutlass" / "include" / "cute",
        include / "cute",
    )
    shutil.copytree(
        source / "third-party" / "cutlass" / "include" / "cutlass",
        include / "cutlass",
    )


def build_extension(source: Path, package: Path, build_dir: Path) -> Path:
    import torch
    import tvm_ffi.cpp

    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda-13.2")).resolve()
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
    normalized = text.replace(str(source), "<SOURCE>").replace(
        str(build_dir), "<BUILD>"
    )
    return sha256(ninja), sha256_bytes(normalized.encode())


def build_manifest(
    source: Path,
    output: Path,
    identity: dict[str, Any],
    reconstruction: dict[str, str],
) -> dict[str, Any]:
    import torch
    import tvm_ffi

    variants: dict[str, dict[str, Any]] = {}
    commits = {
        "stock": BASE_COMMIT,
        "candidate": CANDIDATE_COMMIT,
    }
    for variant, commit in commits.items():
        tree = output / "sources" / variant
        package = output / "artifacts" / variant / "site" / f"deep_gemm_w13_{variant}"
        build_dir = output / "compile" / variant
        if build_dir.exists():
            shutil.rmtree(build_dir)
        source_digest = materialize(
            source,
            BASE_COMMIT if variant == "candidate" else commit,
            tree,
            candidate_patch=(
                identity["candidate_diff"] if variant == "candidate" else None
            ),
        )
        if source_digest != reconstruction[f"{variant}_source_tree_sha256"]:
            raise RuntimeError(f"{variant} source tree changed after audit")
        copy_package_source(tree, package)
        shared_object = build_extension(tree, package, build_dir)
        jit_cache = output / "jit" / variant
        if jit_cache.exists():
            shutil.rmtree(jit_cache)
        jit_cache.mkdir(parents=True)
        ninja_sha, plan_sha = normalized_plan(build_dir, tree)
        variants[variant] = {
            "commit": commit,
            "source": str(tree),
            "source_tree_sha256": source_digest,
            "package": str(package),
            "package_init_sha256": sha256(package / "__init__.py"),
            "shared_object": str(shared_object),
            "shared_object_sha256": sha256(shared_object),
            "build_ninja": str(build_dir / "build.ninja"),
            "build_ninja_sha256": ninja_sha,
            "normalized_build_plan_sha256": plan_sha,
            "jit_cache": str(jit_cache),
        }
    plan_hashes = {
        record["normalized_build_plan_sha256"] for record in variants.values()
    }
    if len(plan_hashes) != 1:
        raise RuntimeError(f"stock/candidate build plans differ: {plan_hashes}")
    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda-13.2")).resolve()
    cxx = Path(shutil.which("c++") or "c++").resolve()
    nvcc = (cuda_home / "bin" / "nvcc").resolve()
    manifest = {
        "schema_version": 3,
        "source": {
            "repository": str(source),
            "base_commit": BASE_COMMIT,
            "candidate_commit": identity["candidate_commit"],
            "cutlass_commit": CUTLASS_COMMIT,
            "fmt_commit": FMT_COMMIT,
            "candidate_diff_sha256": identity["candidate_diff_sha256"],
            "candidate_diff_bytes": identity["candidate_diff_bytes"],
            "base_blob_sha256": identity["base_blob_sha256"],
            **reconstruction,
        },
        "build": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "tvm_ffi": getattr(tvm_ffi, "__version__", "unknown"),
            "cuda_home": str(cuda_home),
            "cuda_arch": "10.0a",
            "cxx_path": str(cxx),
            "cxx_sha256": sha256(cxx),
            "nvcc_path": str(nvcc),
            "nvcc_sha256": sha256(nvcc),
            "nvcc_version": run(str(nvcc), "--version").splitlines()[-1],
            "jit_compiler": "nvcc",
            "stock_candidate_command_identical": True,
            "elf_symbol_binding": "Bsymbolic",
            "elf_symbol_visibility": "hidden",
            "normalized_build_plan_sha256": plan_hashes.pop(),
            "force_clean_build_directories": True,
            "max_jobs": os.environ.get("MAX_JOBS"),
            "compile_api": "tvm_ffi.cpp.build",
        },
        "variants": variants,
    }
    diff_path = output / "candidate.diff"
    diff_path.write_bytes(identity["candidate_diff"])
    manifest["source"]["candidate_diff_path"] = str(diff_path)
    manifest["source"]["candidate_diff_file_sha256"] = sha256(diff_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(DEFAULT_SOURCE) if DEFAULT_SOURCE else None,
        help=(
            "clean DeepGEMM checkout containing the pinned base and dependency "
            "commits (or set SGLANG_GLM52_W13_DEEPGEMM_SOURCE)"
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate-commit")
    parser.add_argument("--audit-materialization", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.source is None:
        parser.error(
            "--source or SGLANG_GLM52_W13_DEEPGEMM_SOURCE is required"
        )
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    ensure_output_root(output)
    if args.candidate_commit not in (None, CANDIDATE_COMMIT):
        parser.error(
            f"only the measured candidate commit {CANDIDATE_COMMIT} is supported"
        )
    identity = verify_source(source)
    output.mkdir(parents=True, exist_ok=True)
    reconstruction = audit_materialization(
        source, identity["candidate_diff"], output
    )
    audit = {
        **reconstruction,
        "candidate_commit": CANDIDATE_COMMIT,
        "candidate_diff_sha256": identity["candidate_diff_sha256"],
        "candidate_diff_bytes": identity["candidate_diff_bytes"],
    }
    if args.audit_materialization:
        print(json.dumps(audit, indent=2, sort_keys=True))
        return 0
    if not args.force:
        raise RuntimeError("use --force for two clean, comparable extension builds")
    manifest = build_manifest(source, output, identity, reconstruction)
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
