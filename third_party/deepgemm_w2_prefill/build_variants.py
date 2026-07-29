#!/usr/bin/env python3
"""Build pinned stock, stock-source PSUM, and exact W2 stage-7 modules."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path
from types import ModuleType

BASE_COMMIT = "edcf77b276965de8f03cdc47c23f01b08bf7c7ab"
CUTLASS_COMMIT = "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8"
FMT_COMMIT = "553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28"
DEFAULT_UPSTREAM = Path("/home/qinhaiyan/DeepGEMM-GLM52")
DEFAULT_OUTPUT = Path(
    "/home/qinhaiyan/glm52-v2-goal-runs/cache/"
    "30_moe_w2_prefill_psum/deepgemm/w2_prefill_variants"
)
HERE = Path(__file__).resolve().parent
PATCH = HERE / "patches" / "0001-exact-w2-psum-stage7.patch"
HELPER_PATH = HERE.parent / "deepgemm_w13" / "build_variants.py"

BASE_BLOB_SHA256 = {
    "csrc/apis/gemm.hpp": (
        "0840d64249e2a5a4a994d495e8320a0fff26bad9ca107426a1a1226e7d621186"
    ),
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
    "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh": (
        "9c1e70677ede6ba09ab98e629482da7874182f8227907382efe0a81658da5a37"
    ),
}

# Filled fail-closed after two independent archive reconstructions.
EXPECTED_SOURCE_IDENTITY = {
    "stock_source_tree_sha256": (
        "4bfc233540d0478bf88860d924c53e105be29e01ddd039a68a8c5242addb2af5"
    ),
    "psum_source_tree_sha256": (
        "4bfc233540d0478bf88860d924c53e105be29e01ddd039a68a8c5242addb2af5"
    ),
    "stage7_source_tree_sha256": (
        "c02885d8fac2549b34b66a25ed106c7d369cbd5cbbf043465b6c084056ed86f2"
    ),
    "complete_source_diff_sha256": (
        "8cccdd7135a04a532c96605743466471410fac1d5e612067fb1cda10be1bd53e"
    ),
}


def _load_helper() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_glm52_w2_deepgemm_build_helper", HELPER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load build helper: {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


H = _load_helper()


def run(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def _git_blob_sha256(upstream: Path, relative: str) -> str:
    value = subprocess.check_output(
        ["git", "-C", str(upstream), "show", f"{BASE_COMMIT}:{relative}"]
    )
    return H.sha256_bytes(value)


def verify_source(upstream: Path) -> None:
    if run("git", "-C", str(upstream), "cat-file", "-t", BASE_COMMIT) != "commit":
        raise RuntimeError(f"missing DeepGEMM base commit {BASE_COMMIT}")
    revisions = {
        "cutlass": run(
            "git", "-C", str(upstream / "third-party/cutlass"), "rev-parse", "HEAD"
        ),
        "fmt": run("git", "-C", str(upstream / "third-party/fmt"), "rev-parse", "HEAD"),
    }
    expected = {"cutlass": CUTLASS_COMMIT, "fmt": FMT_COMMIT}
    if revisions != expected:
        raise RuntimeError(
            f"DeepGEMM dependency drift: actual={revisions}, expected={expected}"
        )
    actual_blobs = {
        relative: _git_blob_sha256(upstream, relative) for relative in BASE_BLOB_SHA256
    }
    if actual_blobs != BASE_BLOB_SHA256:
        raise RuntimeError(
            f"DeepGEMM base blob drift: actual={actual_blobs}, "
            f"expected={BASE_BLOB_SHA256}"
        )
    if not PATCH.is_file():
        raise RuntimeError(f"missing tracked stage-7 patch: {PATCH}")


def materialize_source(
    upstream: Path,
    destination: Path,
    *,
    apply_stage7_patch: bool,
) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=destination.parent, prefix=f".{destination.name}."
    ) as temporary:
        source = Path(temporary) / "source"
        H.extract_git_archive(upstream, BASE_COMMIT, source)
        for dependency, commit in (
            ("cutlass", CUTLASS_COMMIT),
            ("fmt", FMT_COMMIT),
        ):
            dependency_path = source / "third-party" / dependency
            if dependency_path.exists():
                dependency_path.rmdir()
            H.extract_git_archive(
                upstream / "third-party" / dependency, commit, dependency_path
            )
        if apply_stage7_patch:
            subprocess.run(
                ["git", "apply", "--check", str(PATCH)], cwd=source, check=True
            )
            subprocess.run(["git", "apply", str(PATCH)], cwd=source, check=True)
        digest = H.tree_sha256(source)
        if destination.exists():
            shutil.rmtree(destination)
        source.rename(destination)
    return digest


def reconstruct_identity(upstream: Path, scratch_parent: Path) -> dict[str, str]:
    scratch_parent.mkdir(parents=True, exist_ok=True)
    passes: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory(
        dir=scratch_parent, prefix=".w2-prefill-materialization."
    ) as temporary:
        root = Path(temporary)
        for pass_index in range(2):
            stock = materialize_source(
                upstream, root / f"stock-{pass_index}", apply_stage7_patch=False
            )
            psum = materialize_source(
                upstream, root / f"psum-{pass_index}", apply_stage7_patch=False
            )
            stage7 = materialize_source(
                upstream, root / f"stage7-{pass_index}", apply_stage7_patch=True
            )
            if stock != psum:
                raise RuntimeError(
                    "stock and PSUM stage-8 source trees are not byte-identical"
                )
            if stage7 == stock:
                raise RuntimeError("stage-7 patch did not alter the source tree")
            passes.append(
                {
                    "stock_source_tree_sha256": stock,
                    "psum_source_tree_sha256": psum,
                    "stage7_source_tree_sha256": stage7,
                    "complete_source_diff_sha256": H.sha256(PATCH),
                }
            )
    if passes[0] != passes[1]:
        raise RuntimeError(f"source reconstruction is not deterministic: {passes}")
    return passes[0]


def assert_pinned_identity(identity: dict[str, str]) -> None:
    if identity != EXPECTED_SOURCE_IDENTITY:
        raise RuntimeError(
            "W2 prefill source identity mismatch: "
            f"actual={identity}, expected={EXPECTED_SOURCE_IDENTITY}"
        )


def _compiler_identity(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": H.sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--audit-materialization", action="store_true")
    parser.add_argument(
        "--print-reconstructed-identity",
        action="store_true",
        help="print two-pass identity before applying the checked-in pin",
    )
    args = parser.parse_args()
    upstream = args.upstream.expanduser().resolve()
    output = args.output.expanduser().resolve()
    H.ensure_disk(output)
    verify_source(upstream)
    output.mkdir(parents=True, exist_ok=True)
    identity = reconstruct_identity(upstream, output)
    if args.print_reconstructed_identity:
        print(json.dumps(identity, indent=2, sort_keys=True))
        return 0
    assert_pinned_identity(identity)
    if args.audit_materialization:
        print(json.dumps(identity, indent=2, sort_keys=True))
        return 0
    if not args.force:
        raise RuntimeError(
            "reproducible builds require --force; prior objects may not be reused"
        )

    os.environ["MAX_JOBS"] = "1"
    variants: dict[str, dict[str, object]] = {}
    for name, patched in (("stock", False), ("psum", False), ("stage7", True)):
        source = output / "sources" / name
        package = output / "artifacts" / name / "site" / f"deep_gemm_w2_prefill_{name}"
        build_directory = output / "compile" / name
        if build_directory.exists():
            shutil.rmtree(build_directory)
        source_digest = materialize_source(upstream, source, apply_stage7_patch=patched)
        H.copy_package_source(source, package)
        shared_object = H.build_extension(source, package, build_directory).resolve()
        jit_cache = (output / "jit" / name).resolve()
        if jit_cache.exists():
            shutil.rmtree(jit_cache)
        jit_cache.mkdir(parents=True)
        ninja_sha, normalized_plan_sha = H.normalized_build_plan(
            build_directory, source
        )
        expected_digest = identity[f"{name}_source_tree_sha256"]
        if source_digest != expected_digest:
            raise RuntimeError(
                f"{name} source digest changed during build: "
                f"{source_digest} != {expected_digest}"
            )
        variants[name] = {
            "source": str(source.resolve()),
            "source_tree_sha256": source_digest,
            "patched": patched,
            "pipeline_stages": 7 if name == "stage7" else 8,
            "package": str(package.resolve()),
            "package_init_sha256": H.sha256(package / "__init__.py"),
            "shared_object": str(shared_object),
            "shared_object_sha256": H.sha256(shared_object),
            "jit_cache": str(jit_cache),
            "build_ninja": str((build_directory / "build.ninja").resolve()),
            "build_ninja_sha256": ninja_sha,
            "normalized_build_plan_sha256": normalized_plan_sha,
        }

    normalized_plans = {
        str(record["normalized_build_plan_sha256"]) for record in variants.values()
    }
    if len(normalized_plans) != 1:
        raise RuntimeError(
            "variant compiler commands differ after path normalization: "
            f"{normalized_plans}"
        )

    import torch
    import tvm_ffi

    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")).resolve()
    torch_root = Path(torch.__file__).resolve().parent
    cxx_path = Path(shutil.which("c++") or "c++").resolve()
    nvcc_path = (cuda_home / "bin" / "nvcc").resolve()
    cxx_abi = int(torch.compiled_with_cxx11_abi())
    include_template = [
        f"{cuda_home}/include",
        sysconfig.get_path("include"),
        f"{torch_root}/include",
        f"{torch_root}/include/torch/csrc/api/include",
        "<SOURCE>/deep_gemm/include",
        "<SOURCE>/third-party/cutlass/include",
        "<SOURCE>/third-party/fmt/include",
    ]
    if (cuda_home / "include" / "cccl").exists():
        include_template.append(f"{cuda_home}/include/cccl")
    manifest = {
        "schema_version": 3,
        "task": "30_moe_w2_prefill_psum",
        "source": {
            "repository": str(upstream),
            "remote": "https://github.com/sgl-project/DeepGEMM",
            "tag": "v0.1.4.post1",
            "commit": BASE_COMMIT,
            "cutlass_commit": CUTLASS_COMMIT,
            "fmt_commit": FMT_COMMIT,
            "stage7_patch": str(PATCH.resolve()),
            "base_blob_sha256": BASE_BLOB_SHA256,
            **identity,
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
            "cxx": run(str(cxx_path), "--version").splitlines()[0],
            "cxx_identity": _compiler_identity(cxx_path),
            "nvcc": run(str(nvcc_path), "--version").splitlines()[-1],
            "nvcc_identity": _compiler_identity(nvcc_path),
            "jit_compiler": "nvcc",
            "compile_api": "tvm_ffi.cpp.build",
            "submodule_update": False,
            "force_clean_build_directories": True,
            "max_jobs": "1",
            "variant_command_identical": True,
            "normalized_build_plan_sha256": normalized_plans.pop(),
            "source_materialization": (
                "git archive(base) + git archive(CUTLASS) + git archive(fmt) "
                "+ stage7-only tracked patch"
            ),
            "extra_cflags": [
                *H.BASE_CFLAGS,
                f"-D_GLIBCXX_USE_CXX11_ABI={cxx_abi}",
            ],
            "extra_ldflags": [
                f"-L{cuda_home / 'lib64'}",
                f"-L{torch_root / 'lib'}",
                *H.LINK_LIBRARIES,
            ],
            "extra_include_path_template": include_template,
            "cpp_files_template": ["<SOURCE>/csrc/tvm_ffi_api.cpp"],
            "build_directory_template": "<OUTPUT>/compile/<VARIANT>",
            "package_template": (
                "<OUTPUT>/artifacts/<VARIANT>/site/deep_gemm_w2_prefill_<VARIANT>"
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
