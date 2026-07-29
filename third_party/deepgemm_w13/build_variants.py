#!/usr/bin/env python3
"""Build same-base stock and explicit-config W13 DeepGEMM modules.

The immutable source denominator is SGL DeepGEMM v0.1.4.post1 at commit
edcf77b276965de8f03cdc47c23f01b08bf7c7ab. The candidate differs only by the
checked-in patch in ``patches/``. Neither module is installed into the active
Python environment; both are staged under the selected output for side-by-side
loading.
"""

from __future__ import annotations

import argparse
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

BASE_COMMIT = "edcf77b276965de8f03cdc47c23f01b08bf7c7ab"
CUTLASS_COMMIT = "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8"
FMT_COMMIT = "553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28"
BUILD_MAX_JOBS = "1"
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_UPSTREAM = Path(
    os.environ.get(
        "DEEPGEMM_W13_BASE_REPO",
        str(REPO_ROOT.parent / "DeepGEMM-GLM52"),
    )
)
DEFAULT_OUTPUT = Path(
    os.environ.get(
        "DEEPGEMM_W13_OUTPUT",
        str(REPO_ROOT / "build" / "deepgemm-w13-variants"),
    )
)
PATCH = HERE / "patches" / "0001-explicit-w13-config.patch"
BASE_BLOB_SHA256 = {
    "csrc/apis/gemm.hpp": "0840d64249e2a5a4a994d495e8320a0fff26bad9ca107426a1a1226e7d621186",
    "csrc/jit_kernels/heuristics/sm100.hpp": (
        "487cac2ff19027c781b08e9a0391836e77c03cdffcb7ceb3346d8633c8eb0884"
    ),
    "csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp": (
        "cca1ddb5b5787942c31b39a9d5618929ee609c6c3b57b877fe636df39540366b"
    ),
    "csrc/tvm_ffi_api.cpp": (
        "c09aeec8187a2e29a3ebfc61c9ce1168a89fea775040a47bcf73739131ea57c0"
    ),
    "sgl_deep_gemm/__init__.py": (
        "b33e89deacdce241f01f5070d321918f5e5480e3e6d3af569678d4192db4f2a7"
    ),
}
EXPECTED_SOURCE_TREE_SHA256 = {
    "stock_source_tree_sha256": (
        "4bfc233540d0478bf88860d924c53e105be29e01ddd039a68a8c5242addb2af5"
    ),
    "candidate_source_tree_sha256": (
        "1e23f011428ca83bcc3fe1a2e990b62ed82abbba65a291a61db9cf4a729cf657"
    ),
    "complete_source_diff_sha256": (
        "056c90d416f2278c23bcb495d41ecf28f82e7047f220f4acc8321e8f1436a458"
    ),
}
BASE_CFLAGS = [
    "-std=c++17",
    "-O3",
    "-fPIC",
    "-Wno-psabi",
    "-Wno-deprecated-declarations",
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
    """Hash relative path, executable mode, symlink target, and file bytes."""

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


def ensure_disk(path: Path) -> None:
    free = shutil.disk_usage(path.parent if path.parent.exists() else HERE).free
    if free < 8 * 1024**3:
        raise RuntimeError(
            f"refusing build with less than 8 GiB free: {free / 1024**3:.2f} GiB"
        )


def verify_source(upstream: Path) -> None:
    if run("git", "-C", str(upstream), "cat-file", "-t", BASE_COMMIT) != "commit":
        raise RuntimeError(f"missing DeepGEMM base commit {BASE_COMMIT} in {upstream}")
    revisions = {
        "cutlass": run(
            "git", "-C", str(upstream / "third-party/cutlass"), "rev-parse", "HEAD"
        ),
        "fmt": run("git", "-C", str(upstream / "third-party/fmt"), "rev-parse", "HEAD"),
    }
    expected = {"cutlass": CUTLASS_COMMIT, "fmt": FMT_COMMIT}
    if revisions != expected:
        raise RuntimeError(
            f"submodule revision drift: actual={revisions}, expected={expected}"
        )
    if not PATCH.is_file():
        raise RuntimeError(f"candidate patch is missing: {PATCH}")
    actual_blobs = {
        relative: sha256_bytes(
            subprocess.check_output(
                ["git", "-C", str(upstream), "show", f"{BASE_COMMIT}:{relative}"]
            )
        )
        for relative in BASE_BLOB_SHA256
    }
    if actual_blobs != BASE_BLOB_SHA256:
        raise RuntimeError(
            f"DeepGEMM base blob drift: actual={actual_blobs}, "
            f"expected={BASE_BLOB_SHA256}"
        )


def extract_git_archive(repository: Path, commit: str, destination: Path) -> None:
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
        # `git archive` tar umasks can vary with repository configuration.
        # Normalize to canonical Git file modes so source-tree identities are
        # reproducible across users and so the POSIX `patch` tool cannot
        # change only the five touched files from 0664 to 0644.
        for path in destination.rglob("*"):
            if path.is_symlink():
                continue
            if path.is_dir():
                path.chmod(0o755)
            elif path.is_file():
                executable = bool(stat.S_IMODE(path.stat().st_mode) & 0o111)
                path.chmod(0o755 if executable else 0o644)
    finally:
        archive.unlink(missing_ok=True)


def materialize_source(upstream: Path, destination: Path, patched: bool) -> str:
    """Reconstruct from three pinned git archives plus the tracked patch."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=destination.parent, prefix=f".{destination.name}."
    ) as tmp:
        source = Path(tmp) / "source"
        extract_git_archive(upstream, BASE_COMMIT, source)
        for dependency, commit in (
            ("cutlass", CUTLASS_COMMIT),
            ("fmt", FMT_COMMIT),
        ):
            dependency_path = source / "third-party" / dependency
            if dependency_path.exists():
                dependency_path.rmdir()
            extract_git_archive(
                upstream / "third-party" / dependency,
                commit,
                dependency_path,
            )
        if patched:
            subprocess.run(
                [
                    "patch",
                    f"--directory={source}",
                    "--strip=1",
                    "--batch",
                    "--forward",
                    "--dry-run",
                    f"--input={PATCH}",
                ],
                check=True,
            )
            subprocess.run(
                [
                    "patch",
                    f"--directory={source}",
                    "--strip=1",
                    "--batch",
                    "--forward",
                    f"--input={PATCH}",
                ],
                check=True,
            )
        digest = tree_sha256(source)
        if destination.exists():
            shutil.rmtree(destination)
        source.rename(destination)
    return digest


def audit_materialization(upstream: Path, scratch_parent: Path) -> dict[str, str]:
    """CPU-only proof that tracked inputs reconstruct byte-identical trees."""

    scratch_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=scratch_parent, prefix=".w13-materialization-audit."
    ) as tmp:
        root = Path(tmp)
        stock_a = materialize_source(upstream, root / "stock-a", False)
        candidate_a = materialize_source(upstream, root / "candidate-a", True)
        stock_b = materialize_source(upstream, root / "stock-b", False)
        candidate_b = materialize_source(upstream, root / "candidate-b", True)
        if stock_a != stock_b or candidate_a != candidate_b:
            raise RuntimeError(
                "W13 source reconstruction is not byte-for-byte deterministic"
            )
        if stock_a == candidate_a:
            raise RuntimeError("W13 candidate patch did not change the source tree")
        result = {
            "stock_source_tree_sha256": stock_a,
            "candidate_source_tree_sha256": candidate_a,
            "complete_source_diff_sha256": sha256(PATCH),
        }
        if result != EXPECTED_SOURCE_TREE_SHA256:
            raise RuntimeError(
                f"W13 reconstructed tree identity drift: actual={result}, "
                f"expected={EXPECTED_SOURCE_TREE_SHA256}"
            )
        return result


def copy_package_source(source: Path, package: Path) -> None:
    package.parent.mkdir(parents=True, exist_ok=True)
    if package.exists():
        shutil.rmtree(package)
    package.mkdir()
    for name in ("__init__.py", "cuda_helpers.py"):
        shutil.copy2(source / "sgl_deep_gemm" / name, package / name)
    version = source / "sgl_deep_gemm" / "VERSION"
    shutil.copy2(version, package / "VERSION")
    for name in ("utils", "testing", "legacy", "mega"):
        shutil.copytree(source / "deep_gemm" / name, package / name)
    include = package / "include"
    shutil.copytree(
        source / "deep_gemm" / "include" / "deep_gemm", include / "deep_gemm"
    )
    shutil.copytree(
        source / "third-party" / "cutlass" / "include" / "cute", include / "cute"
    )
    shutil.copytree(
        source / "third-party" / "cutlass" / "include" / "cutlass", include / "cutlass"
    )


def normalized_build_plan(build_directory: Path, source: Path) -> tuple[str, str]:
    """Return the complete Ninja plan with only arm-specific paths normalized."""

    ninja = build_directory / "build.ninja"
    text = ninja.read_text()
    normalized = text.replace(str(source), "<SOURCE>").replace(
        str(build_directory), "<BUILD>"
    )
    return sha256(ninja), sha256_bytes(normalized.encode())


def build_extension(source: Path, package: Path, build_directory: Path) -> Path:
    import torch
    import tvm_ffi.cpp

    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda-13.2")).resolve()
    if not (cuda_home / "include" / "cuda.h").is_file():
        raise RuntimeError(f"CUDA headers not found under {cuda_home}")
    os.environ["TVM_FFI_CUDA_ARCH_LIST"] = "10.0a"
    torch_root = Path(torch.__file__).resolve().parent
    cxx_abi = int(torch.compiled_with_cxx11_abi())
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
        f"-D_GLIBCXX_USE_CXX11_ABI={cxx_abi}",
    ]
    ldflags = [
        f"-L{cuda_home / 'lib64'}",
        f"-L{torch_root / 'lib'}",
        *LINK_LIBRARIES,
    ]
    build_directory.mkdir(parents=True, exist_ok=True)
    library = Path(
        tvm_ffi.cpp.build(
            name="_C",
            cpp_files=[str(source / "csrc" / "tvm_ffi_api.cpp")],
            extra_cflags=cflags,
            extra_ldflags=ldflags,
            extra_include_paths=[str(path) for path in includes],
            build_directory=str(build_directory),
        )
    )
    target = package / "_C.so"
    shutil.copy2(library, target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--audit-materialization",
        action="store_true",
        help="CPU-only reconstruction audit; never compiles or imports CUDA",
    )
    args = parser.parse_args()
    upstream = args.upstream.resolve()
    output = args.output.resolve()
    ensure_disk(output)
    verify_source(upstream)
    output.mkdir(parents=True, exist_ok=True)
    reconstruction = audit_materialization(upstream, output)
    if args.audit_materialization:
        print(json.dumps(reconstruction, indent=2, sort_keys=True))
        return 0
    if not args.force:
        raise RuntimeError(
            "reproducible W13 builds require --force so no prior object or DSO "
            "can be reused"
        )
    # Do not let a caller's shell change the compilation schedule recorded in
    # the runtime-attested build contract.
    os.environ["MAX_JOBS"] = BUILD_MAX_JOBS

    variants: dict[str, dict[str, object]] = {}
    for variant, patched in (("stock", False), ("candidate", True)):
        source = output / "sources" / variant
        package_name = "deep_gemm" if variant == "stock" else "deep_gemm_w13_candidate"
        package = output / "artifacts" / variant / "site" / package_name
        shared_object = package / "_C.so"
        build_directory = output / "compile" / variant
        if build_directory.exists():
            shutil.rmtree(build_directory)
        source_digest = materialize_source(upstream, source, patched)
        copy_package_source(source, package)
        build_extension(source, package, build_directory)
        jit_cache = output / "jit" / variant
        if jit_cache.exists():
            shutil.rmtree(jit_cache)
        jit_cache.mkdir(parents=True, exist_ok=True)
        expected_source_digest = reconstruction[f"{variant}_source_tree_sha256"]
        if source_digest != expected_source_digest:
            raise RuntimeError(
                f"{variant} materialized source digest mismatch: "
                f"{source_digest} != {expected_source_digest}"
            )
        build_ninja_sha256, normalized_plan_sha256 = normalized_build_plan(
            build_directory, source
        )
        variants[variant] = {
            "source": str(source),
            "source_tree_sha256": source_digest,
            "package": str(package),
            "package_init_sha256": sha256(package / "__init__.py"),
            "shared_object": str(shared_object),
            "shared_object_sha256": sha256(shared_object),
            "jit_cache": str(jit_cache),
            "patched": patched,
            "build_ninja": str(build_directory / "build.ninja"),
            "build_ninja_sha256": build_ninja_sha256,
            "normalized_build_plan_sha256": normalized_plan_sha256,
        }

    import torch
    import tvm_ffi

    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda-13.2")).resolve()
    torch_root = Path(torch.__file__).resolve().parent
    cxx_abi = int(torch.compiled_with_cxx11_abi())
    cxx_path = Path(shutil.which("c++") or "c++").resolve()
    nvcc_path = (cuda_home / "bin" / "nvcc").resolve()
    normalized_plans = {
        str(record["normalized_build_plan_sha256"]) for record in variants.values()
    }
    if len(normalized_plans) != 1:
        raise RuntimeError(
            "stock/candidate generated Ninja commands differ after path "
            f"normalization: {normalized_plans}"
        )
    normalized_build_plan_sha256 = normalized_plans.pop()
    include_path_template = [
        f"{cuda_home}/include",
        sysconfig.get_path("include"),
        f"{torch_root}/include",
        f"{torch_root}/include/torch/csrc/api/include",
        "<SOURCE>/deep_gemm/include",
        "<SOURCE>/third-party/cutlass/include",
        "<SOURCE>/third-party/fmt/include",
    ]
    if (cuda_home / "include" / "cccl").exists():
        include_path_template.append(f"{cuda_home}/include/cccl")

    manifest = {
        "schema_version": 2,
        "source": {
            "repository": str(upstream),
            "remote": "https://github.com/sgl-project/DeepGEMM",
            "tag": "v0.1.4.post1",
            "commit": BASE_COMMIT,
            "cutlass_commit": CUTLASS_COMMIT,
            "fmt_commit": FMT_COMMIT,
            "candidate_patch": str(PATCH),
            "candidate_patch_sha256": sha256(PATCH),
            "base_blob_sha256": BASE_BLOB_SHA256,
            **reconstruction,
        },
        "build": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "tvm_ffi": getattr(tvm_ffi, "__version__", "unknown"),
            "cxx11_abi": bool(torch.compiled_with_cxx11_abi()),
            "cuda_home": str(cuda_home),
            "cuda_arch": "10.0a",
            "cxx": run("c++", "--version").splitlines()[0],
            "cxx_path": str(cxx_path),
            "cxx_sha256": sha256(cxx_path),
            "nvcc": run(str(nvcc_path), "--version").splitlines()[-1],
            "nvcc_path": str(nvcc_path),
            "nvcc_sha256": sha256(nvcc_path),
            "jit_compiler": "nvcc",
            "stock_candidate_command_identical": True,
            "normalized_build_plan_sha256": normalized_build_plan_sha256,
            "force_clean_build_directories": True,
            "max_jobs": BUILD_MAX_JOBS,
            "source_materialization": (
                "git archive(base) + git archive(CUTLASS) + "
                "git archive(fmt) + candidate-only tracked patch"
            ),
            "submodule_update": False,
            "compile_api": "tvm_ffi.cpp.build",
            "extra_cflags": [
                *BASE_CFLAGS,
                f"-D_GLIBCXX_USE_CXX11_ABI={cxx_abi}",
            ],
            "extra_ldflags": [
                f"-L{cuda_home / 'lib64'}",
                f"-L{torch_root / 'lib'}",
                *LINK_LIBRARIES,
            ],
            "extra_include_path_template": include_path_template,
            "cpp_files_template": ["<SOURCE>/csrc/tvm_ffi_api.cpp"],
            "build_directory_template": "<OUTPUT>/compile/<VARIANT>",
            "package_template": (
                "<OUTPUT>/artifacts/stock/site/deep_gemm or "
                "<OUTPUT>/artifacts/candidate/site/deep_gemm_w13_candidate"
            ),
            "jit_cache_template": "<OUTPUT>/jit/<VARIANT>",
        },
        "variants": variants,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
