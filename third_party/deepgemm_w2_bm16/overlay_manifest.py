#!/usr/bin/env python3
"""Write and verify the exact-post1 stock/candidate overlay manifest.

This tool is deliberately CPU-only: it hashes files and inspects Git objects,
but never imports DeepGEMM or queries a CUDA device.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from packaging.version import Version

BASE_COMMIT = "edcf77b276965de8f03cdc47c23f01b08bf7c7ab"
CUTLASS_COMMIT = "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8"
FMT_COMMIT = "553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28"
VERSION = "0.1.4.post1"
VERSION_LITERAL = "v0.1.4.post1"
BUILD_ID = (
    "glm52-w2-bm16-v2:sgl-deep-gemm-0.1.4.post1@"
    f"{BASE_COMMIT}:sm100:e32:m1024:k2048:n6144:"
    "bm16:pdl1:sms148:no-recipe:no-overlap"
)
STOCK_BUILD_ID = f"sgl-deep-gemm-0.1.4.post1@{BASE_COMMIT}"
STOCK_IMPORT_NAME = "deep_gemm"
CANDIDATE_IMPORT_NAME = "deep_gemm_glm52_w2_bm16"
JIT_IDENTITY_TEMPLATE = (
    "sm100_m_grouped_fp8_fp4_gemm_masked_1d1d_"
    "glm52_w2_bm16_v2_em{expected_m}"
)
TASK_CACHE_ROOT = Path(
    "/home/qinhaiyan/glm52-v2-goal-runs/cache/26-moe_w2_decode_scoped_bm16"
)
EXPECTED_CACHE_PATHS = {
    "DG_JIT_CACHE_DIR": TASK_CACHE_ROOT / "deepgemm",
    "SGLANG_DG_CACHE_DIR": TASK_CACHE_ROOT / "deepgemm",
    "TRITON_CACHE_DIR": TASK_CACHE_ROOT / "triton",
    "TORCH_EXTENSIONS_DIR": TASK_CACHE_ROOT / "torch_extensions",
}

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SOURCE_PATCH = SCRIPT_DIR / "source.patch"
BUILD_TOOL_PATCH = SCRIPT_DIR / "build_tool.patch"
CORE_HASHES = SCRIPT_DIR / "core_source_hashes.sha256"
BASE_LOCK = SCRIPT_DIR / "base_lock.json"
BUILD_PROVENANCE = SCRIPT_DIR / "build_provenance.json"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(repo: Path, *args: str, binary: bool = False) -> str | bytes:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=not binary,
    )


def _core_hash_entries() -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for line in CORE_HASHES.read_text().splitlines():
        digest, relative = line.split(maxsplit=1)
        entries.append((digest, relative))
    if not entries:
        raise RuntimeError(f"empty core hash list: {CORE_HASHES}")
    return entries


def _source_state(source: Path, *, role: str) -> dict[str, Any]:
    core_files = [relative for _, relative in _core_hash_entries()]
    runtime_diff = _git(
        source, "diff", "--binary", "--", *core_files, binary=True
    )
    build_diff = _git(source, "diff", "--binary", binary=True)
    status = str(
        _git(source, "status", "--porcelain=v1", "--untracked-files=no")
    ).splitlines()
    runtime_files = str(
        _git(source, "diff", "--name-only", "--", *core_files)
    ).splitlines()
    build_files = str(_git(source, "diff", "--name-only")).splitlines()

    expected_runtime = b"" if role == "stock" else SOURCE_PATCH.read_bytes()
    expected_build = (
        BUILD_TOOL_PATCH.read_bytes() if role == "stock" else build_diff
    )
    if runtime_diff != expected_runtime:
        raise RuntimeError(f"{role} runtime diff is not the exact expected delta")
    if role == "stock" and build_diff != expected_build:
        raise RuntimeError("stock build-input diff is not the exact build-tool patch")

    expected_runtime_files = [] if role == "stock" else core_files
    expected_build_files = sorted(
        expected_runtime_files + ["build_sgl_deep_gemm.sh"]
    )
    if runtime_files != expected_runtime_files:
        raise RuntimeError(
            f"{role} runtime diff files mismatch: {runtime_files} "
            f"!= {expected_runtime_files}"
        )
    if build_files != expected_build_files:
        raise RuntimeError(
            f"{role} build-input diff files mismatch: {build_files} "
            f"!= {expected_build_files}"
        )

    head = str(_git(source, "rev-parse", "HEAD")).strip()
    cutlass = str(_git(source / "third-party/cutlass", "rev-parse", "HEAD")).strip()
    fmt = str(_git(source / "third-party/fmt", "rev-parse", "HEAD")).strip()
    if (head, cutlass, fmt) != (BASE_COMMIT, CUTLASS_COMMIT, FMT_COMMIT):
        raise RuntimeError(
            f"{role} source base mismatch: {(head, cutlass, fmt)}"
        )

    return {
        "head": head,
        "submodules": {
            "third-party/cutlass": cutlass,
            "third-party/fmt": fmt,
        },
        "tracked_status_porcelain_v1": status,
        "runtime_diff": {
            "format": "git-diff-binary",
            "bytes": len(runtime_diff),
            "sha256": _sha256_bytes(runtime_diff),
            "files": runtime_files,
        },
        "build_input_diff": {
            "format": "git-diff-binary",
            "bytes": len(build_diff),
            "sha256": _sha256_bytes(build_diff),
            "files": build_files,
        },
    }


def _package_record(
    package_at_artifact_root: Path,
    package_at_final_root: Path,
    *,
    role: str,
) -> dict[str, Any]:
    init_py = package_at_artifact_root / "__init__.py"
    version_file = package_at_artifact_root / "VERSION"
    extension = package_at_artifact_root / "_C.so"
    for required in (init_py, version_file, extension):
        if not required.is_file():
            raise FileNotFoundError(required)
    literal = version_file.read_text().strip()
    if Version(literal) != Version(VERSION):
        raise RuntimeError(f"{role} VERSION mismatch: {literal!r}")
    return {
        "build_id": STOCK_BUILD_ID if role == "stock" else BUILD_ID,
        "package_dir": str(package_at_final_root.resolve()),
        "import_name": (
            STOCK_IMPORT_NAME if role == "stock" else CANDIDATE_IMPORT_NAME
        ),
        "version_literal": literal,
        "version_sha256": _sha256(version_file),
        "init_sha256": _sha256(init_py),
        "extension_sha256": _sha256(extension),
        "extension_bytes": extension.stat().st_size,
    }


def _toolchain() -> dict[str, Any]:
    import torch

    nvcc = subprocess.check_output(
        ["/usr/local/cuda/bin/nvcc", "--version"], text=True
    ).splitlines()[-1]
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "platform": platform.platform(),
        "nvcc": nvcc,
        "tvm_ffi_cuda_arch_list": "100",
    }


def write_manifest(args: argparse.Namespace) -> None:
    artifact_root = args.artifact_root.resolve()
    final_root = args.final_overlay_dir.resolve()
    stock_package_artifact = artifact_root / "stock/site/deep_gemm"
    candidate_package_artifact = (
        artifact_root / "candidate/site/deep_gemm_glm52_w2_bm16"
    )
    stock_package_final = final_root / "stock/site/deep_gemm"
    candidate_package_final = (
        final_root / "candidate/site/deep_gemm_glm52_w2_bm16"
    )
    source_sha = _sha256(SOURCE_PATCH)
    build_tool_sha = _sha256(BUILD_TOOL_PATCH)
    core_hashes_sha = _sha256(CORE_HASHES)
    build_key = (
        f"{BASE_COMMIT[:12]}-{source_sha[:12]}-{build_tool_sha[:12]}"
    )
    if final_root.name != build_key:
        raise RuntimeError(f"overlay key mismatch: {final_root.name} != {build_key}")

    manifest = {
        "schema_version": 3,
        "build_key": build_key,
        "base": {
            "distribution": "sgl-deep-gemm",
            "version": VERSION,
            "tag": VERSION_LITERAL,
            "commit": BASE_COMMIT,
            "repository": str(args.base_repo.resolve()),
            "submodules": {
                "third-party/cutlass": CUTLASS_COMMIT,
                "third-party/fmt": FMT_COMMIT,
            },
        },
        "patches": {
            "source": {
                "path": str(SOURCE_PATCH),
                "sha256": source_sha,
                "runtime_source_delta": True,
            },
            "build_tool": {
                "path": str(BUILD_TOOL_PATCH),
                "sha256": build_tool_sha,
                "runtime_source_delta": False,
            },
        },
        "source_identity": {
            "stock": _source_state(args.stock_source.resolve(), role="stock"),
            "candidate": _source_state(
                args.candidate_source.resolve(), role="candidate"
            ),
        },
        "core_source_hashes_sha256": core_hashes_sha,
        "candidate_core_source_dir": str(
            (final_root / "candidate/core_source").resolve()
        ),
        "stock": _package_record(
            stock_package_artifact,
            stock_package_final,
            role="stock",
        ),
        "candidate": _package_record(
            candidate_package_artifact,
            candidate_package_final,
            role="candidate",
        ),
        "candidate_api": {
            "function": "fp8_m_grouped_gemm_nt_masked",
            "keyword": "masked_block_m_override",
            "value": 16,
            "expected_m": [4, 5, 8, 9],
            "jit_identity_template": JIT_IDENTITY_TEMPLATE,
        },
        "runtime_contract": {
            "architecture": "sm100",
            "pdl": True,
            "num_sms": 148,
            "tc_util": "stock-candidate-equal",
            "recipe": None,
            "overlap": None,
            "global_alignment_setter_used": False,
            "stock_and_candidate_side_by_side": True,
            "stock_binding": "process-start-PYTHONPATH",
            "cache_paths": {
                name: str(path.resolve())
                for name, path in EXPECTED_CACHE_PATHS.items()
            },
            "build_tmp_dir": str(
                (EXPECTED_CACHE_PATHS["DG_JIT_CACHE_DIR"] / "build_tmp").resolve()
            ),
        },
        "toolchain": _toolchain(),
    }
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label}: {actual!r} != {expected!r}")


def _gitlink(base_repo: Path, path: str) -> str:
    line = str(_git(base_repo, "ls-tree", BASE_COMMIT, "--", path)).strip()
    if not line:
        raise RuntimeError(f"missing gitlink {path} at {BASE_COMMIT}")
    metadata, listed_path = line.split("\t", 1)
    mode, kind, commit = metadata.split()
    if (mode, kind, listed_path) != ("160000", "commit", path):
        raise RuntimeError(f"invalid gitlink for {path}: {line}")
    return commit


def _verify_package(
    manifest: dict[str, Any],
    overlay_dir: Path,
    *,
    role: str,
) -> None:
    expected_relative = (
        Path("stock/site/deep_gemm")
        if role == "stock"
        else Path("candidate/site/deep_gemm_glm52_w2_bm16")
    )
    expected_import = STOCK_IMPORT_NAME if role == "stock" else CANDIDATE_IMPORT_NAME
    expected_build = STOCK_BUILD_ID if role == "stock" else BUILD_ID
    record = manifest[role]
    package = (overlay_dir / expected_relative).resolve()
    _assert_equal(Path(record["package_dir"]).resolve(), package, f"{role}.package_dir")
    _assert_equal(record["import_name"], expected_import, f"{role}.import_name")
    _assert_equal(record["build_id"], expected_build, f"{role}.build_id")

    init_py = package / "__init__.py"
    version_file = package / "VERSION"
    extension = package / "_C.so"
    for required in (init_py, version_file, extension):
        if not required.is_file():
            raise FileNotFoundError(required)
    literal = version_file.read_text().strip()
    _assert_equal(Version(literal), Version(VERSION), f"{role}.VERSION")
    _assert_equal(record["version_literal"], literal, f"{role}.version_literal")
    _assert_equal(
        record["version_sha256"], _sha256(version_file), f"{role}.version_sha256"
    )
    _assert_equal(record["init_sha256"], _sha256(init_py), f"{role}.init_sha256")
    _assert_equal(
        record["extension_sha256"],
        _sha256(extension),
        f"{role}.extension_sha256",
    )
    _assert_equal(
        record["extension_bytes"], extension.stat().st_size, f"{role}.extension_bytes"
    )


def verify_manifest(args: argparse.Namespace) -> None:
    manifest_path = args.manifest.resolve()
    overlay_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    lock = json.loads(BASE_LOCK.read_text())
    source_sha = _sha256(SOURCE_PATCH)
    build_tool_sha = _sha256(BUILD_TOOL_PATCH)
    core_hashes_sha = _sha256(CORE_HASHES)
    build_key = (
        f"{BASE_COMMIT[:12]}-{source_sha[:12]}-{build_tool_sha[:12]}"
    )

    _assert_equal(manifest["schema_version"], 3, "schema_version")
    _assert_equal(manifest["build_key"], build_key, "build_key")
    _assert_equal(overlay_dir.name, build_key, "overlay directory")
    _assert_equal(lock["base_commit"], BASE_COMMIT, "lock.base_commit")
    _assert_equal(lock["version"], VERSION, "lock.version")
    _assert_equal(lock["source_patch_sha256"], source_sha, "lock.source_patch")
    _assert_equal(
        lock["build_tool_patch_sha256"], build_tool_sha, "lock.build_tool_patch"
    )
    _assert_equal(
        lock["core_source_hashes_sha256"], core_hashes_sha, "lock.core_hashes"
    )
    _assert_equal(lock["candidate_build_id"], BUILD_ID, "lock.candidate_build_id")
    _assert_equal(
        lock["submodules"]["third-party/cutlass"],
        CUTLASS_COMMIT,
        "lock.cutlass",
    )
    _assert_equal(
        lock["submodules"]["third-party/fmt"], FMT_COMMIT, "lock.fmt"
    )
    _assert_equal(
        lock["task_cache_paths"],
        {name: str(path.resolve()) for name, path in EXPECTED_CACHE_PATHS.items()},
        "lock.task_cache_paths",
    )

    base = manifest["base"]
    _assert_equal(base["commit"], BASE_COMMIT, "base.commit")
    _assert_equal(Version(base["version"]), Version(VERSION), "base.version")
    _assert_equal(Version(base["tag"]), Version(VERSION), "base.tag")
    _assert_equal(
        base["submodules"],
        {
            "third-party/cutlass": CUTLASS_COMMIT,
            "third-party/fmt": FMT_COMMIT,
        },
        "base.submodules",
    )
    base_repo = Path(base["repository"]).resolve()
    if str(_git(base_repo, "cat-file", "-t", BASE_COMMIT)).strip() != "commit":
        raise RuntimeError(f"base commit unavailable in {base_repo}")
    _assert_equal(
        _gitlink(base_repo, "third-party/cutlass"),
        CUTLASS_COMMIT,
        "base CUTLASS gitlink",
    )
    _assert_equal(
        _gitlink(base_repo, "third-party/fmt"), FMT_COMMIT, "base fmt gitlink"
    )

    patches = manifest["patches"]
    _assert_equal(patches["source"]["sha256"], source_sha, "patch.source")
    _assert_equal(
        patches["source"]["runtime_source_delta"], True, "patch.source.runtime"
    )
    _assert_equal(
        patches["build_tool"]["sha256"], build_tool_sha, "patch.build_tool"
    )
    _assert_equal(
        patches["build_tool"]["runtime_source_delta"],
        False,
        "patch.build_tool.runtime",
    )
    _assert_equal(
        manifest["core_source_hashes_sha256"],
        core_hashes_sha,
        "core_source_hashes_sha256",
    )

    core_dir = (overlay_dir / "candidate/core_source").resolve()
    _assert_equal(
        Path(manifest["candidate_core_source_dir"]).resolve(),
        core_dir,
        "candidate_core_source_dir",
    )
    expected_core_files: set[str] = set()
    for expected, relative in _core_hash_entries():
        path = core_dir / relative
        expected_core_files.add(relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        _assert_equal(_sha256(path), expected, f"core source {relative}")
    actual_core_files = {
        str(path.relative_to(core_dir))
        for path in core_dir.rglob("*")
        if path.is_file()
    }
    _assert_equal(actual_core_files, expected_core_files, "core source file set")

    _verify_package(manifest, overlay_dir, role="stock")
    _verify_package(manifest, overlay_dir, role="candidate")

    source_identity = manifest["source_identity"]
    core_files = [relative for _, relative in _core_hash_entries()]
    expected_build_files = sorted(core_files + ["build_sgl_deep_gemm.sh"])
    stock = source_identity["stock"]
    candidate = source_identity["candidate"]
    for role, state in (("stock", stock), ("candidate", candidate)):
        _assert_equal(state["head"], BASE_COMMIT, f"{role}.source.head")
        _assert_equal(
            state["submodules"],
            {
                "third-party/cutlass": CUTLASS_COMMIT,
                "third-party/fmt": FMT_COMMIT,
            },
            f"{role}.source.submodules",
        )
    _assert_equal(stock["runtime_diff"]["sha256"], EMPTY_SHA256, "stock runtime diff")
    _assert_equal(stock["runtime_diff"]["bytes"], 0, "stock runtime diff bytes")
    _assert_equal(stock["runtime_diff"]["files"], [], "stock runtime diff files")
    _assert_equal(
        stock["build_input_diff"]["sha256"],
        build_tool_sha,
        "stock build-input diff",
    )
    _assert_equal(
        stock["build_input_diff"]["bytes"],
        BUILD_TOOL_PATCH.stat().st_size,
        "stock build-input diff bytes",
    )
    _assert_equal(
        stock["build_input_diff"]["files"],
        ["build_sgl_deep_gemm.sh"],
        "stock build-input files",
    )
    _assert_equal(
        candidate["runtime_diff"]["sha256"], source_sha, "candidate runtime diff"
    )
    _assert_equal(
        candidate["runtime_diff"]["bytes"],
        SOURCE_PATCH.stat().st_size,
        "candidate runtime diff bytes",
    )
    _assert_equal(
        candidate["runtime_diff"]["files"], core_files, "candidate runtime files"
    )
    _assert_equal(
        candidate["build_input_diff"]["files"],
        expected_build_files,
        "candidate build-input files",
    )
    if len(candidate["build_input_diff"]["sha256"]) != 64:
        raise RuntimeError("candidate build-input diff hash is malformed")
    _assert_equal(
        stock["tracked_status_porcelain_v1"],
        [" M build_sgl_deep_gemm.sh"],
        "stock tracked status",
    )
    _assert_equal(
        candidate["tracked_status_porcelain_v1"],
        [f" M {path}" for path in expected_build_files],
        "candidate tracked status",
    )

    _assert_equal(
        manifest["candidate_api"],
        {
            "function": "fp8_m_grouped_gemm_nt_masked",
            "keyword": "masked_block_m_override",
            "value": 16,
            "expected_m": [4, 5, 8, 9],
            "jit_identity_template": JIT_IDENTITY_TEMPLATE,
        },
        "candidate_api",
    )
    expected_runtime = {
        "architecture": "sm100",
        "pdl": True,
        "num_sms": 148,
        "tc_util": "stock-candidate-equal",
        "recipe": None,
        "overlap": None,
        "global_alignment_setter_used": False,
        "stock_and_candidate_side_by_side": True,
        "stock_binding": "process-start-PYTHONPATH",
        "cache_paths": {
            name: str(path.resolve()) for name, path in EXPECTED_CACHE_PATHS.items()
        },
        "build_tmp_dir": str(
            (EXPECTED_CACHE_PATHS["DG_JIT_CACHE_DIR"] / "build_tmp").resolve()
        ),
    }
    _assert_equal(manifest["runtime_contract"], expected_runtime, "runtime_contract")

    if args.check_env:
        for name, expected in EXPECTED_CACHE_PATHS.items():
            value = os.environ.get(name)
            if not value:
                raise RuntimeError(f"{name} is not exported")
            _assert_equal(Path(value).resolve(), expected.resolve(), name)

    if args.check_provenance:
        provenance = json.loads(BUILD_PROVENANCE.read_text())
        _assert_equal(provenance["schema_version"], 2, "provenance.schema_version")
        _assert_equal(
            provenance["generated_manifest_sha256"],
            _sha256(manifest_path),
            "provenance manifest hash",
        )
        _assert_equal(provenance["build_key"], build_key, "provenance.build_key")
        for role in ("stock", "candidate"):
            for field in (
                "build_id",
                "import_name",
                "version_literal",
                "version_sha256",
                "init_sha256",
                "extension_sha256",
                "extension_bytes",
            ):
                _assert_equal(
                    provenance[role][field],
                    manifest[role][field],
                    f"provenance.{role}.{field}",
                )
        _assert_equal(
            provenance["source_identity"],
            manifest["source_identity"],
            "provenance.source_identity",
        )
        _assert_equal(
            provenance["runtime_contract"],
            manifest["runtime_contract"],
            "provenance.runtime_contract",
        )

    print(f"PASS exact-post1 overlay manifest: {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    write = subparsers.add_parser("write")
    write.add_argument("--artifact-root", type=Path, required=True)
    write.add_argument("--final-overlay-dir", type=Path, required=True)
    write.add_argument("--stock-source", type=Path, required=True)
    write.add_argument("--candidate-source", type=Path, required=True)
    write.add_argument("--base-repo", type=Path, required=True)
    write.add_argument("--output", type=Path, required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--check-env", action="store_true")
    verify.add_argument("--check-provenance", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "write":
        write_manifest(args)
    else:
        verify_manifest(args)


if __name__ == "__main__":
    main()
