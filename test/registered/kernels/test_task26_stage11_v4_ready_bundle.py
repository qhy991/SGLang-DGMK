"""CPU-only tests for the Task26 stage11-v4 READY bundle contract."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO / "third_party" / "deepgemm_w2_em8_bm16_stage11_v4" / "ready_bundle.py"
MANIFEST_TOOL_PATH = TOOL_PATH.with_name("overlay_manifest.py")


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "task26_stage11_v4_ready_bundle",
        TOOL_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


READY = _load_tool()


def _load_manifest_tool():
    spec = importlib.util.spec_from_file_location(
        "task26_stage11_v4_overlay_manifest",
        MANIFEST_TOOL_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MANIFEST = _load_manifest_tool()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _commit(repo: Path, message: str) -> None:
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Task26",
            "-c",
            "user.email=task26-ready@example.invalid",
            "commit",
            "-qm",
            message,
        ],
        check=True,
    )


def _package_record(
    bundle: Path,
    relative: str,
    *,
    import_name: str,
    build_id: str,
) -> dict:
    package = bundle / relative
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("__version__ = '0.1.4.post1'\n")
    (package / "VERSION").write_text("0.1.4.post1\n")
    (package / "_C.so").write_bytes(f"fixture:{import_name}".encode())
    (package / "utils").mkdir()
    (package / "utils/jit_runtime.py").write_text("RUNTIME_INPUT = True\n")
    (package / "include/deep_gemm").mkdir(parents=True)
    (package / "include/deep_gemm/kernel.hpp").write_text(
        "#pragma once\n#define TASK26_FIXTURE 1\n"
    )
    record = {
        "package_relpath": relative,
        "import_name": import_name,
        "build_id": build_id,
        "version_literal": "0.1.4.post1",
        "version_sha256": _sha256(package / "VERSION"),
        "init_sha256": _sha256(package / "__init__.py"),
        "extension_sha256": _sha256(package / "_C.so"),
        "extension_bytes": (package / "_C.so").stat().st_size,
    }
    record["package_tree"] = READY._package_tree_record(
        package,
        role=import_name,
    )
    return record


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    sglang = tmp_path / "sglang"
    harness = tmp_path / "kernel-harness"
    overlay = sglang / "third_party" / "deepgemm_w2_em8_bm16_stage11_v4"
    overlay.mkdir(parents=True)
    harness.mkdir()
    source_overlay = TOOL_PATH.parent
    for name in (
        "source.patch",
        "build_tool.patch",
        "core_source_hashes.sha256",
        "verify_source_reproducibility.sh",
    ):
        shutil.copy2(source_overlay / name, overlay / name)
    (harness / "README.md").write_text("fixture harness\n")

    staging = tmp_path / "staging"
    staging.mkdir()
    stock = _package_record(
        staging,
        "stock/site/deep_gemm",
        import_name="deep_gemm",
        build_id="stock-post1",
    )
    candidate = _package_record(
        staging,
        "candidate/site/deep_gemm_glm52_w2_em8_bm16_stage11_v4",
        import_name="deep_gemm_glm52_w2_em8_bm16_stage11_v4",
        build_id="stage11-v4",
    )
    build_key = READY.expected_build_key(sglang)
    source_identity = {"stock": {"head": "base"}, "candidate": {"head": "base+v4"}}
    manifest = {
        "schema_version": READY.MANIFEST_SCHEMA_VERSION,
        "variant": READY.VARIANT,
        "build_key": build_key,
        "stock": stock,
        "candidate": candidate,
        "source_identity": source_identity,
    }
    manifest_path = staging / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    provenance = {
        "schema_version": READY.PROVENANCE_SCHEMA_VERSION,
        "variant": READY.VARIANT,
        "build_key": build_key,
        "generated_manifest_sha256": _sha256(manifest_path),
        "stock": stock,
        "candidate": candidate,
        "source_identity": source_identity,
    }
    provenance_path = overlay / "build_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    READY.write_source_replay(
        manifest_path,
        staging / "source_replay.json",
        sglang_root=sglang,
    )
    _commit(sglang, "fixture v4 provenance")
    _commit(harness, "fixture harness")

    no_op_verifier = lambda *args, **kwargs: None
    digest = READY.bundle_digest(
        manifest_path,
        sglang_root=sglang,
        manifest_verifier=no_op_verifier,
    )
    bundle = tmp_path / "bundles" / digest
    bundle.parent.mkdir()
    staging.rename(bundle)
    ready = READY.write_ready(
        bundle / "manifest.json",
        sglang_root=sglang,
        kernel_harness_root=harness,
        manifest_verifier=no_op_verifier,
    )
    return ready, sglang, harness


def _verify(ready: Path, sglang: Path, harness: Path) -> dict:
    return READY.verify_ready(
        ready,
        sglang_root=sglang,
        kernel_harness_root=harness,
        manifest_verifier=lambda *args, **kwargs: None,
    )


def test_ready_binds_content_packages_replay_provenance_and_clean_heads(
    tmp_path,
):
    ready, sglang, harness = _fixture(tmp_path)
    evidence = _verify(ready, sglang, harness)
    assert ready.parent.name == evidence["bundle_digest"]
    assert Path(evidence["ready_path"]) == ready
    assert Path(evidence["manifest_path"]).parent == ready.parent
    assert Path(evidence["source_replay_path"]).parent == ready.parent
    assert len(evidence["stock_package_tree_sha256"]) == 64
    assert len(evidence["candidate_package_tree_sha256"]) == 64
    assert (
        evidence["stock_package_tree_sha256"]
        != evidence["candidate_package_tree_sha256"]
    )
    document = json.loads(ready.read_text())
    assert document["contract"]["release_policy"]["required_lanes"] == [
        "leaf_eager",
        "leaf_cuda_graph",
        "containing_region_eager",
        "containing_region_cuda_graph",
    ]
    assert document["contract"]["release_policy"]["gpu_driver_may_build"] is False


def test_manifest_and_ready_tools_hash_the_same_complete_package_tree(tmp_path):
    package = tmp_path / "deep_gemm"
    _package_record(
        tmp_path,
        "deep_gemm",
        import_name="deep_gemm",
        build_id="fixture",
    )
    ready_record = READY._package_tree_record(package, role="stock")
    manifest_record = MANIFEST._package_tree_record(package, role="stock")
    assert manifest_record == ready_record
    assert ready_record["file_count"] == 5
    assert ready_record["directory_count"] == 4
    paths = {entry["path"] for entry in ready_record["entries"]}
    assert "utils/jit_runtime.py" in paths
    assert "include/deep_gemm/kernel.hpp" in paths


@pytest.mark.parametrize(
    "mutation",
    (
        "ready",
        "manifest",
        "package",
        "jit_python",
        "header",
        "added_file",
        "deleted_file",
        "file_mode",
        "directory_mode",
        "symlink",
        "hardlink",
        "source_replay",
        "provenance",
        "source_input",
        "dirty_repo",
    ),
)
def test_corrupt_or_missing_ready_input_fails_closed(tmp_path, mutation):
    ready, sglang, harness = _fixture(tmp_path)
    if mutation == "ready":
        document = json.loads(ready.read_text())
        document["status"] = "CORRUPT"
        ready.write_text(json.dumps(document))
    elif mutation == "manifest":
        (ready.parent / "manifest.json").write_text("{}\n")
    elif mutation == "package":
        (
            ready.parent / "candidate/site/deep_gemm_glm52_w2_em8_bm16_stage11_v4/_C.so"
        ).write_bytes(b"corrupt")
    elif mutation == "jit_python":
        (
            ready.parent
            / "candidate/site/deep_gemm_glm52_w2_em8_bm16_stage11_v4"
            / "utils/jit_runtime.py"
        ).write_text("RUNTIME_INPUT = False\n")
    elif mutation == "header":
        (
            ready.parent
            / "candidate/site/deep_gemm_glm52_w2_em8_bm16_stage11_v4"
            / "include/deep_gemm/kernel.hpp"
        ).write_text("#pragma once\n#define TASK26_FIXTURE 0\n")
    elif mutation == "added_file":
        (
            ready.parent
            / "candidate/site/deep_gemm_glm52_w2_em8_bm16_stage11_v4"
            / "utils/untracked_runtime.py"
        ).write_text("UNBOUND = True\n")
    elif mutation == "deleted_file":
        (
            ready.parent
            / "candidate/site/deep_gemm_glm52_w2_em8_bm16_stage11_v4"
            / "utils/jit_runtime.py"
        ).unlink()
    elif mutation == "file_mode":
        (
            ready.parent
            / "candidate/site/deep_gemm_glm52_w2_em8_bm16_stage11_v4"
            / "utils/jit_runtime.py"
        ).chmod(0o755)
    elif mutation == "directory_mode":
        (
            ready.parent
            / "candidate/site/deep_gemm_glm52_w2_em8_bm16_stage11_v4"
            / "include/deep_gemm"
        ).chmod(0o700)
    elif mutation == "symlink":
        target = (
            ready.parent
            / "candidate/site/deep_gemm_glm52_w2_em8_bm16_stage11_v4"
            / "utils/jit_runtime.py"
        )
        target.unlink()
        target.symlink_to("../__init__.py")
    elif mutation == "hardlink":
        package = (
            ready.parent
            / "candidate/site/deep_gemm_glm52_w2_em8_bm16_stage11_v4"
        )
        target = package / "utils/jit_runtime.py"
        target.unlink()
        target.hardlink_to(package / "__init__.py")
    elif mutation == "source_replay":
        (ready.parent / "source_replay.json").unlink()
    elif mutation == "provenance":
        (
            sglang
            / "third_party/deepgemm_w2_em8_bm16_stage11_v4"
            / "build_provenance.json"
        ).write_text("{}\n")
    elif mutation == "source_input":
        (
            sglang
            / "third_party/deepgemm_w2_em8_bm16_stage11_v4"
            / "source.patch"
        ).write_text("corrupt source identity\n")
    else:
        (harness / "UNTRACKED").write_text("dirty\n")
    with pytest.raises(READY.ReadinessError):
        _verify(ready, sglang, harness)


def test_ready_tool_source_has_no_cuda_or_framework_import() -> None:
    source = TOOL_PATH.read_text()
    assert "import torch" not in source
    assert "import deep_gemm" not in source
    assert "nvidia-smi" not in source
    assert "CUDA_VISIBLE_DEVICES" not in source
