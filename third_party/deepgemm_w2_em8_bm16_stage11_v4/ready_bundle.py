#!/usr/bin/env python3
"""Publish and verify the Task26 v4 content-addressed READY bundle.

This tool is intentionally CPU-only.  It never imports DeepGEMM, torch, or a
CUDA binding and never queries a GPU.  A build produces the manifest,
packages, tracked provenance, and source-replay record first.  After the
generated provenance is committed, ``write-ready`` binds those artifacts to
clean exact SGLang and Kernel-Harness heads and atomically publishes ``READY``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SGLANG_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_KERNEL_HARNESS_ROOT = DEFAULT_SGLANG_ROOT.parent / "kernel-harness"
BUNDLE_ROOT_RELATIVE = Path("build/deepgemm-w2-em8-bm16-stage11-v4-ready-bundles")
TRACKED_PROVENANCE_RELATIVE = Path(
    "third_party/deepgemm_w2_em8_bm16_stage11_v4/build_provenance.json"
)
MANIFEST_NAME = "manifest.json"
SOURCE_REPLAY_NAME = "source_replay.json"
READY_NAME = "READY"

BASE_COMMIT = "edcf77b276965de8f03cdc47c23f01b08bf7c7ab"
CUTLASS_COMMIT = "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8"
FMT_COMMIT = "553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28"
VARIANT = {
    "name": "em8_bm16_stage11",
    "version": 4,
    "predeclared_fallback": "em8_bm16_stage10",
    "fallback_eligible": False,
}
BUILD_TOOL_SHA256 = "dc731d5442c0bdf0758b17380e02e67b580cf3aa579f4832a497d1b68e3a85c7"
SOURCE_PATCH_SHA256 = "9b227e5cf597c3f620245f82a66c7e22c7c483be91d54c711e68027947a005c8"
CORE_HASHES_SHA256 = "e2b904139f5891ca7eb11a66f221e902a35b2d3bce536478829dcd8c7b374bcb"
HEX64 = re.compile(r"[0-9a-f]{64}")
GIT_OID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")


class ReadinessError(RuntimeError):
    """A fail-closed READY contract violation."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReadinessError(f"{label} is unreadable: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReadinessError(f"{label} must be a JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReadinessError(message)


def _source_inputs(sglang_root: Path) -> tuple[Path, Path, Path]:
    source_dir = sglang_root / TRACKED_PROVENANCE_RELATIVE.parent
    return (
        source_dir / "source.patch",
        source_dir / "build_tool.patch",
        source_dir / "core_source_hashes.sha256",
    )


def expected_build_key(sglang_root: Path) -> str:
    source_patch, build_tool_patch, core_hashes = _source_inputs(sglang_root)
    source_sha = _sha256(source_patch)
    build_tool_sha = _sha256(build_tool_patch)
    core_hashes_sha = _sha256(core_hashes)
    _require(
        source_sha == SOURCE_PATCH_SHA256,
        "v4 source patch differs from its released identity",
    )
    _require(
        build_tool_sha == BUILD_TOOL_SHA256,
        "v4 build-tool patch differs from its released identity",
    )
    _require(
        core_hashes_sha == CORE_HASHES_SHA256,
        "v4 core-source hash list differs from its released identity",
    )
    return f"{BASE_COMMIT[:12]}-{source_sha[:12]}-{build_tool_sha[:12]}"


def _safe_bundle_member(bundle_dir: Path, relative: Any, label: str) -> Path:
    _require(
        isinstance(relative, str) and relative and not Path(relative).is_absolute(),
        f"{label} must be a non-empty relative path",
    )
    unresolved = bundle_dir / relative
    _require(not unresolved.is_symlink(), f"{label} must not be a symlink")
    path = unresolved.resolve()
    root = bundle_dir.resolve()
    _require(root in path.parents, f"{label} escapes the bundle: {relative}")
    return path


def _package_contract(
    bundle_dir: Path,
    manifest: dict[str, Any],
    role: str,
) -> dict[str, Any]:
    record = manifest.get(role)
    _require(isinstance(record, dict), f"manifest.{role} is missing")
    package = _safe_bundle_member(
        bundle_dir,
        record.get("package_relpath"),
        f"manifest.{role}.package_relpath",
    )
    files = {
        "init": package / "__init__.py",
        "version": package / "VERSION",
        "extension": package / "_C.so",
    }
    for label, path in files.items():
        _require(path.is_file(), f"{role} {label} artifact is missing: {path}")
        _require(not path.is_symlink(), f"{role} {label} must not be a symlink")
    observed = {
        "package_relpath": record.get("package_relpath"),
        "import_name": record.get("import_name"),
        "build_id": record.get("build_id"),
        "version_literal": (package / "VERSION").read_text().strip(),
        "version_sha256": _sha256(files["version"]),
        "init_sha256": _sha256(files["init"]),
        "extension_sha256": _sha256(files["extension"]),
        "extension_bytes": files["extension"].stat().st_size,
    }
    for field, value in observed.items():
        _require(
            record.get(field) == value,
            f"manifest.{role}.{field} does not match the package",
        )
    return observed


def _verify_source_replay(
    replay: dict[str, Any],
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    sglang_root: Path,
) -> None:
    source_patch, build_tool_patch, core_hashes = _source_inputs(sglang_root)
    verify_script = (
        sglang_root
        / TRACKED_PROVENANCE_RELATIVE.parent
        / "verify_source_reproducibility.sh"
    )
    expected = {
        "schema_version": 1,
        "status": "PASS",
        "variant": VARIANT,
        "build_key": manifest.get("build_key"),
        "base": {
            "commit": BASE_COMMIT,
            "submodules": {
                "third-party/cutlass": CUTLASS_COMMIT,
                "third-party/fmt": FMT_COMMIT,
            },
        },
        "patches": {
            "source_sha256": _sha256(source_patch),
            "build_tool_sha256": _sha256(build_tool_patch),
        },
        "core_source_hashes_sha256": _sha256(core_hashes),
        "replayed_manifest_sha256": _sha256(manifest_path),
        "verification_script_sha256": _sha256(verify_script),
    }
    _require(replay == expected, "source replay record is not the exact v4 record")


def write_source_replay(
    manifest_path: Path,
    output: Path,
    *,
    sglang_root: Path,
) -> None:
    """Write the deterministic record after the replay script has passed."""
    manifest_path = manifest_path.resolve()
    manifest = _json(manifest_path, "overlay manifest")
    source_patch, build_tool_patch, core_hashes = _source_inputs(sglang_root)
    verify_script = (
        sglang_root
        / TRACKED_PROVENANCE_RELATIVE.parent
        / "verify_source_reproducibility.sh"
    )
    record = {
        "schema_version": 1,
        "status": "PASS",
        "variant": VARIANT,
        "build_key": manifest.get("build_key"),
        "base": {
            "commit": BASE_COMMIT,
            "submodules": {
                "third-party/cutlass": CUTLASS_COMMIT,
                "third-party/fmt": FMT_COMMIT,
            },
        },
        "patches": {
            "source_sha256": _sha256(source_patch),
            "build_tool_sha256": _sha256(build_tool_patch),
        },
        "core_source_hashes_sha256": _sha256(core_hashes),
        "replayed_manifest_sha256": _sha256(manifest_path),
        "verification_script_sha256": _sha256(verify_script),
    }
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def _run_overlay_verifier(
    manifest_path: Path,
    *,
    sglang_root: Path,
    check_env: bool,
) -> None:
    command = [
        sys.executable,
        str(sglang_root / TRACKED_PROVENANCE_RELATIVE.parent / "overlay_manifest.py"),
        "verify",
        "--manifest",
        str(manifest_path),
        "--check-provenance",
    ]
    if check_env:
        command.append("--check-env")
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise ReadinessError(f"overlay manifest verification failed: {detail}")


def bundle_content_contract(
    manifest_path: Path,
    *,
    sglang_root: Path,
    check_env: bool = False,
    manifest_verifier: Callable[..., None] = _run_overlay_verifier,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    bundle_dir = manifest_path.parent
    _require(
        manifest_path.name == MANIFEST_NAME,
        f"manifest must be named {MANIFEST_NAME}",
    )
    provenance_unresolved = sglang_root / TRACKED_PROVENANCE_RELATIVE
    provenance_path = provenance_unresolved.resolve()
    replay_path = bundle_dir / SOURCE_REPLAY_NAME
    _require(not manifest_path.is_symlink(), "manifest must not be a symlink")
    _require(not replay_path.is_symlink(), "source replay must not be a symlink")
    _require(
        not provenance_unresolved.is_symlink(),
        "tracked provenance must not be a symlink",
    )
    _require(
        provenance_path.is_file(),
        f"tracked build provenance missing: {provenance_path}",
    )
    _require(replay_path.is_file(), f"source replay record missing: {replay_path}")

    manifest_verifier(
        manifest_path,
        sglang_root=sglang_root,
        check_env=check_env,
    )
    manifest = _json(manifest_path, "overlay manifest")
    provenance = _json(provenance_path, "tracked build provenance")
    replay = _json(replay_path, "source replay record")
    source_patch, build_tool_patch, core_hashes = _source_inputs(sglang_root)

    _require(manifest.get("schema_version") == 5, "manifest schema must be v5")
    _require(manifest.get("variant") == VARIANT, "manifest variant is not v4")
    _require(
        manifest.get("build_key") == expected_build_key(sglang_root),
        "manifest build key is not derived from the v4 source inputs",
    )
    _require(provenance.get("schema_version") == 4, "provenance schema must be v4")
    _require(provenance.get("variant") == VARIANT, "provenance variant is not v4")
    _require(
        provenance.get("build_key") == manifest.get("build_key"),
        "provenance build key differs from manifest",
    )
    _require(
        provenance.get("generated_manifest_sha256") == _sha256(manifest_path),
        "provenance does not bind the exact manifest",
    )
    _verify_source_replay(
        replay,
        manifest=manifest,
        manifest_path=manifest_path,
        sglang_root=sglang_root,
    )

    stock = _package_contract(bundle_dir, manifest, "stock")
    candidate = _package_contract(bundle_dir, manifest, "candidate")
    _require(
        provenance.get("stock") == manifest.get("stock"),
        "tracked provenance stock package differs from manifest",
    )
    _require(
        provenance.get("candidate") == manifest.get("candidate"),
        "tracked provenance candidate package differs from manifest",
    )
    _require(
        provenance.get("source_identity") == manifest.get("source_identity"),
        "tracked provenance source identity differs from manifest",
    )

    return {
        "schema_version": 1,
        "variant": VARIANT,
        "build_key": manifest["build_key"],
        "manifest_sha256": _sha256(manifest_path),
        "build_provenance_sha256": _sha256(provenance_path),
        "source_replay_sha256": _sha256(replay_path),
        "patches": {
            "source_sha256": _sha256(source_patch),
            "build_tool_sha256": _sha256(build_tool_patch),
        },
        "core_source_hashes_sha256": _sha256(core_hashes),
        "stock": stock,
        "candidate": candidate,
    }


def bundle_digest(
    manifest_path: Path,
    *,
    sglang_root: Path,
    check_env: bool = False,
    manifest_verifier: Callable[..., None] = _run_overlay_verifier,
) -> str:
    return _canonical_sha256(
        bundle_content_contract(
            manifest_path,
            sglang_root=sglang_root,
            check_env=check_env,
            manifest_verifier=manifest_verifier,
        )
    )


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise ReadinessError(
            f"git inspection failed for {repo}: {exc.output.strip()}"
        ) from exc


def _clean_repository(repo: Path, label: str) -> dict[str, str]:
    status = _git(repo, "status", "--porcelain=v1", "--untracked-files=normal")
    _require(not status, f"{label} repository must be clean before READY")
    head = _git(repo, "rev-parse", "HEAD")
    _require(bool(GIT_OID.fullmatch(head)), f"{label} HEAD is malformed: {head}")
    return {"path": str(repo.resolve()), "head": head}


def _require_tracked_provenance(sglang_root: Path) -> None:
    relative = TRACKED_PROVENANCE_RELATIVE.as_posix()
    observed = _git(sglang_root, "ls-files", "--error-unmatch", "--", relative)
    _require(observed == relative, "generated build provenance is not tracked")


def _expected_ready_contract(
    *,
    manifest_path: Path,
    sglang_root: Path,
    kernel_harness_root: Path,
    check_env: bool,
    manifest_verifier: Callable[..., None],
) -> dict[str, Any]:
    content = bundle_content_contract(
        manifest_path,
        sglang_root=sglang_root,
        check_env=check_env,
        manifest_verifier=manifest_verifier,
    )
    digest = _canonical_sha256(content)
    bundle_dir = manifest_path.resolve().parent
    _require(
        bundle_dir.name == digest,
        f"bundle directory is not its content digest: {bundle_dir.name} != {digest}",
    )
    _require_tracked_provenance(sglang_root)
    return {
        "schema_version": 1,
        "variant": VARIANT,
        "bundle_digest": digest,
        "bundle_content": content,
        "repositories": {
            "kernel_harness": _clean_repository(kernel_harness_root, "Kernel-Harness"),
            "sglang": _clean_repository(sglang_root, "SGLang"),
        },
        "tracked_provenance_relative": TRACKED_PROVENANCE_RELATIVE.as_posix(),
        "ready_tool_sha256": _sha256(Path(__file__).resolve()),
        "release_policy": {
            "build_under_gpu_lease": False,
            "gpu_driver_may_build": False,
            "ready_before_gpu_query": True,
            "ready_before_run_root": True,
            "ready_before_attempt_claim": True,
            "required_lanes": [
                "leaf_eager",
                "leaf_cuda_graph",
                "containing_region_eager",
                "containing_region_cuda_graph",
            ],
        },
    }


def write_ready(
    manifest_path: Path,
    *,
    sglang_root: Path,
    kernel_harness_root: Path,
    check_env: bool = False,
    manifest_verifier: Callable[..., None] = _run_overlay_verifier,
) -> Path:
    manifest_path = manifest_path.resolve()
    ready_path = manifest_path.parent / READY_NAME
    _require(
        not ready_path.exists(), f"READY already exists and is immutable: {ready_path}"
    )
    contract = _expected_ready_contract(
        manifest_path=manifest_path,
        sglang_root=sglang_root.resolve(),
        kernel_harness_root=kernel_harness_root.resolve(),
        check_env=check_env,
        manifest_verifier=manifest_verifier,
    )
    document = {
        "schema_version": 1,
        "status": "READY",
        "contract": contract,
        "contract_sha256": _canonical_sha256(contract),
        "published_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    ready_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=".READY.",
        dir=ready_path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(json.dumps(document, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, ready_path)
        directory_fd = os.open(ready_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    return ready_path


def verify_ready(
    ready_path: Path,
    *,
    sglang_root: Path,
    kernel_harness_root: Path,
    check_env: bool = False,
    manifest_verifier: Callable[..., None] = _run_overlay_verifier,
) -> dict[str, Any]:
    ready_path = ready_path.resolve()
    _require(ready_path.name == READY_NAME, f"ready record must be named {READY_NAME}")
    _require(not ready_path.is_symlink(), "READY must not be a symlink")
    document = _json(ready_path, "READY record")
    _require(
        set(document)
        == {
            "schema_version",
            "status",
            "contract",
            "contract_sha256",
            "published_utc",
        },
        "READY record has an unexpected field set",
    )
    _require(document.get("schema_version") == 1, "READY schema must be v1")
    _require(document.get("status") == "READY", "READY status is not exact")
    _require(
        isinstance(document.get("published_utc"), str)
        and re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:"
            r"[0-9]{2}:[0-9]{2}Z",
            document["published_utc"],
        )
        is not None,
        "READY publication timestamp is malformed",
    )
    contract = document.get("contract")
    _require(isinstance(contract, dict), "READY contract is missing")
    _require(
        document.get("contract_sha256") == _canonical_sha256(contract),
        "READY contract hash mismatch",
    )
    expected = _expected_ready_contract(
        manifest_path=ready_path.parent / MANIFEST_NAME,
        sglang_root=sglang_root.resolve(),
        kernel_harness_root=kernel_harness_root.resolve(),
        check_env=check_env,
        manifest_verifier=manifest_verifier,
    )
    _require(contract == expected, "READY contract differs from current exact inputs")
    return {
        "ready_path": str(ready_path),
        "ready_sha256": _sha256(ready_path),
        "contract_sha256": document["contract_sha256"],
        "bundle_digest": expected["bundle_digest"],
        "manifest_path": str((ready_path.parent / MANIFEST_NAME).resolve()),
        "manifest_sha256": expected["bundle_content"]["manifest_sha256"],
        "source_replay_path": str((ready_path.parent / SOURCE_REPLAY_NAME).resolve()),
        "source_replay_sha256": expected["bundle_content"]["source_replay_sha256"],
        "build_provenance_path": str(
            (sglang_root / TRACKED_PROVENANCE_RELATIVE).resolve()
        ),
        "build_provenance_sha256": expected["bundle_content"][
            "build_provenance_sha256"
        ],
        "stock_site": str((ready_path.parent / "stock/site").resolve()),
        "candidate_package": str(
            (
                ready_path.parent
                / expected["bundle_content"]["candidate"]["package_relpath"]
            ).resolve()
        ),
    }


def locate_bundle(
    *,
    sglang_root: Path,
    bundle_root: Path | None = None,
    ready_state: str = "ready",
) -> Path:
    _require(
        ready_state in {"ready", "unready", "any"},
        f"invalid ready state: {ready_state}",
    )
    root = (
        bundle_root.resolve()
        if bundle_root is not None
        else (sglang_root / BUNDLE_ROOT_RELATIVE).resolve()
    )
    build_key = expected_build_key(sglang_root)
    matches: list[Path] = []
    if root.is_dir():
        for bundle_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            if HEX64.fullmatch(bundle_dir.name) is None:
                continue
            ready_path = bundle_dir / READY_NAME
            if ready_state == "ready" and not ready_path.is_file():
                continue
            if ready_state == "unready" and ready_path.exists():
                continue
            manifest_path = bundle_dir / MANIFEST_NAME
            try:
                manifest = _json(manifest_path, "manifest candidate")
            except ReadinessError:
                continue
            if (
                manifest.get("variant") == VARIANT
                and manifest.get("build_key") == build_key
            ):
                matches.append(bundle_dir.resolve())
    _require(
        len(matches) == 1,
        f"expected exactly one v4 {ready_state} bundle for {build_key}, "
        f"found {len(matches)}",
    )
    return matches[0]


def locate_ready(
    *,
    sglang_root: Path,
    bundle_root: Path | None = None,
) -> Path:
    return (
        locate_bundle(
            sglang_root=sglang_root,
            bundle_root=bundle_root,
            ready_state="ready",
        )
        / READY_NAME
    )


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    compute = subparsers.add_parser("compute-digest")
    compute.add_argument("--manifest", type=Path, required=True)
    compute.add_argument("--sglang-root", type=Path, default=DEFAULT_SGLANG_ROOT)
    compute.add_argument("--check-env", action="store_true")

    replay = subparsers.add_parser("write-source-replay")
    replay.add_argument("--manifest", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--sglang-root", type=Path, default=DEFAULT_SGLANG_ROOT)

    publish = subparsers.add_parser("write-ready")
    publish.add_argument("--manifest", type=Path, required=True)
    publish.add_argument("--sglang-root", type=Path, default=DEFAULT_SGLANG_ROOT)
    publish.add_argument(
        "--kernel-harness-root",
        type=Path,
        default=DEFAULT_KERNEL_HARNESS_ROOT,
    )
    publish.add_argument("--check-env", action="store_true")

    verify = subparsers.add_parser("verify")
    verify.add_argument("--ready", type=Path, required=True)
    verify.add_argument("--sglang-root", type=Path, default=DEFAULT_SGLANG_ROOT)
    verify.add_argument(
        "--kernel-harness-root",
        type=Path,
        default=DEFAULT_KERNEL_HARNESS_ROOT,
    )
    verify.add_argument("--check-env", action="store_true")
    verify.add_argument("--json", action="store_true")

    locate = subparsers.add_parser("locate")
    locate.add_argument("--sglang-root", type=Path, default=DEFAULT_SGLANG_ROOT)
    locate.add_argument("--bundle-root", type=Path)
    locate.add_argument(
        "--print",
        choices=("ready", "manifest", "stock-site", "candidate-package"),
        default="ready",
    )

    locate_bundle_parser = subparsers.add_parser("locate-bundle")
    locate_bundle_parser.add_argument(
        "--sglang-root", type=Path, default=DEFAULT_SGLANG_ROOT
    )
    locate_bundle_parser.add_argument("--bundle-root", type=Path)
    locate_bundle_parser.add_argument(
        "--ready-state",
        choices=("ready", "unready", "any"),
        default="ready",
    )
    return parser.parse_args()


def main() -> int:
    args = _args()
    try:
        if args.command == "compute-digest":
            print(
                bundle_digest(
                    args.manifest,
                    sglang_root=args.sglang_root.resolve(),
                    check_env=args.check_env,
                )
            )
        elif args.command == "write-source-replay":
            write_source_replay(
                args.manifest,
                args.output,
                sglang_root=args.sglang_root.resolve(),
            )
            print(args.output.resolve())
        elif args.command == "write-ready":
            ready = write_ready(
                args.manifest,
                sglang_root=args.sglang_root.resolve(),
                kernel_harness_root=args.kernel_harness_root.resolve(),
                check_env=args.check_env,
            )
            print(ready)
        elif args.command == "verify":
            evidence = verify_ready(
                args.ready,
                sglang_root=args.sglang_root.resolve(),
                kernel_harness_root=args.kernel_harness_root.resolve(),
                check_env=args.check_env,
            )
            print(
                json.dumps(evidence, sort_keys=True)
                if args.json
                else f"PASS Task26 stage11-v4 READY: {evidence['ready_path']}"
            )
        elif args.command == "locate":
            ready = locate_ready(
                sglang_root=args.sglang_root.resolve(),
                bundle_root=args.bundle_root,
            )
            if args.print == "ready":
                print(ready)
            elif args.print == "manifest":
                print(ready.parent / MANIFEST_NAME)
            else:
                document = _json(ready, "READY record")
                content = document["contract"]["bundle_content"]
                relative = (
                    "stock/site"
                    if args.print == "stock-site"
                    else content["candidate"]["package_relpath"]
                )
                print((ready.parent / relative).resolve())
        else:
            print(
                locate_bundle(
                    sglang_root=args.sglang_root.resolve(),
                    bundle_root=args.bundle_root,
                    ready_state=args.ready_state,
                )
            )
    except (OSError, ReadinessError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
