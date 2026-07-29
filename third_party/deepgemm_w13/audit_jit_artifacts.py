#!/usr/bin/env python3
"""CPU-only generated-source/PTX/SASS/resource audit for exact W13 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


PREFIX = "infini_kernel_glm52_moe_w13_decode"
EXPECTED_M_VALUES = (4, 5, 8, 9)
CANDIDATE_INCLUDE_HASH = "d72839997b6ce1f022ac1c19647aee29"
CUOBJDUMP = Path("/usr/local/cuda-13.2/bin/cuobjdump")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count(text: str, pattern: str) -> int:
    return len(re.findall(pattern, text, flags=re.MULTILINE))


def run_tool(*args: str) -> str:
    process = subprocess.run(
        [str(CUOBJDUMP), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode:
        raise RuntimeError(
            f"{CUOBJDUMP.name} {' '.join(args)} failed: {process.stderr.strip()}"
        )
    return process.stdout


def parse_resource(text: str, symbol: str) -> dict[str, int]:
    match = re.search(
        rf"Function {re.escape(symbol)}:\s*\n"
        r"\s*REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)",
        text,
    )
    if match is None:
        raise AssertionError(f"resource record missing for {symbol}")
    return dict(
        zip(
            ("registers", "stack_bytes", "static_shared_bytes", "local_bytes"),
            map(int, match.groups()),
        )
    )


def assert_source_constants(
    source: str,
    *,
    symbol: str,
    block_m: int,
    stages: int,
    cluster_size: int,
) -> None:
    required = (
        '#include <deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh>',
        'extern "C" __global__ void __launch_bounds__(256, 1)',
        symbol,
        "cute::UMMA::Major::K, cute::UMMA::Major::K, 128,",
        "0, 4096, 6144,",
        f"{block_m}, 128, 128,",
        f"{stages},",
        f"{cluster_size}, {'true' if cluster_size == 2 else 'false'},",
        "148,",
        "GemmType::MGroupedMasked, false,",
        "cutlass::float_e4m3_t, cutlass::float_e4m3_t, cutlass::bfloat16_t,",
    )
    missing = [value for value in required if value not in source]
    if missing:
        raise AssertionError(f"{symbol}: generated source constants missing {missing}")


def _candidate_directory_identity(
    directory: Path,
) -> tuple[str, int, str]:
    match = re.fullmatch(
        rf"kernel\.({PREFIX}_em(4|5|8|9)_bm16_(1sm|2sm))\.[0-9a-f]+",
        directory.name,
    )
    if match is None:
        raise AssertionError(f"unexpected candidate directory: {directory.name}")
    symbol, expected_m_text, topology = match.groups()
    return symbol, int(expected_m_text), topology


def _context_duplicate_record(
    symbol: str,
    canonical: Path,
    alternatives: list[Path],
) -> dict[str, Any]:
    canonical_source = (canonical / "kernel.cu").read_text()
    canonical_body = canonical_source.splitlines()[1:]
    canonical_sass_sha = sha256(canonical / "kernel.sass")
    rendered = []
    for alternative in alternatives:
        source = (alternative / "kernel.cu").read_text()
        rendered.append(
            {
                "directory": str(alternative.resolve()),
                "include_hash_line": source.splitlines()[0],
                "source_body_identical_after_hash_comment": (
                    source.splitlines()[1:] == canonical_body
                ),
                "sass_sha256": sha256(alternative / "kernel.sass"),
                "sass_identical": (
                    sha256(alternative / "kernel.sass") == canonical_sass_sha
                ),
                "cubin_sha256": sha256(alternative / "kernel.cubin"),
                "ptx_sha256": sha256(alternative / "kernel.ptx"),
            }
        )
    return {
        "symbol": symbol,
        "reason": (
            "provider-only startup hashed the candidate include tree; current "
            "SGLang production startup loads stock DeepGEMM first and hashes "
            "that include tree before compiling the same named wrapper"
        ),
        "canonical_production_directory": str(canonical.resolve()),
        "canonical_include_hash": CANDIDATE_INCLUDE_HASH,
        "canonical_sass_sha256": canonical_sass_sha,
        "alternatives": rendered,
    }


def audit_candidate(
    cache: Path,
    *,
    allow_context_duplicates: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_directories = sorted((cache / "cache").glob(f"kernel.{PREFIX}_*"))
    groups: dict[tuple[int, str], list[Path]] = {}
    symbols: dict[tuple[int, str], str] = {}
    for directory in all_directories:
        symbol, expected_m, topology = _candidate_directory_identity(directory)
        key = (expected_m, topology)
        groups.setdefault(key, []).append(directory)
        symbols[key] = symbol
    expected_keys = {
        (expected_m, topology)
        for expected_m in EXPECTED_M_VALUES
        for topology in ("1sm", "2sm")
    }
    if set(groups) != expected_keys:
        raise AssertionError(
            f"candidate identity set mismatch: {set(groups)} != {expected_keys}"
        )
    directories: list[Path] = []
    context_duplicates: list[dict[str, Any]] = []
    for key in sorted(groups):
        choices = groups[key]
        production = [
            directory
            for directory in choices
            if (directory / "kernel.cu").read_text().splitlines()[0]
            == f"// Includes' hash value: {CANDIDATE_INCLUDE_HASH}"
        ]
        if len(production) != 1:
            raise AssertionError(
                f"{symbols[key]}: expected one production-context key, "
                f"found {len(production)}"
            )
        canonical = production[0]
        alternatives = [path for path in choices if path != canonical]
        if alternatives and not allow_context_duplicates:
            raise AssertionError(
                f"{symbols[key]}: non-production context cache keys remain"
            )
        if alternatives:
            context_duplicates.append(
                _context_duplicate_record(
                    symbols[key],
                    canonical,
                    alternatives,
                )
            )
        directories.append(canonical)
    if len(directories) != 8:
        raise AssertionError(
            f"expected 8 production candidate kernels, found {len(directories)}"
        )
    records: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for directory in directories:
        symbol, expected_m, topology = _candidate_directory_identity(directory)
        if (expected_m, topology) in seen:
            raise AssertionError(f"duplicate candidate identity {(expected_m, topology)}")
        seen.add((expected_m, topology))

        files = {suffix: directory / f"kernel.{suffix}" for suffix in ("cu", "cubin", "ptx", "sass")}
        if not all(path.is_file() for path in files.values()):
            raise AssertionError(f"{symbol}: incomplete generated artifact set")
        source = files["cu"].read_text()
        ptx = files["ptx"].read_text()
        sass = files["sass"].read_text()
        is_2sm = topology == "2sm"
        group = 2 if is_2sm else 1
        stages = 12 if is_2sm else 11
        assert_source_constants(
            source,
            symbol=symbol,
            block_m=16,
            stages=stages,
            cluster_size=group,
        )
        if count(ptx, rf"\.visible \.entry {re.escape(symbol)}\(") != 1:
            raise AssertionError(f"{symbol}: exact visible PTX entry missing")
        if count(ptx, r"\.visible \.entry ") != 1:
            raise AssertionError(f"{symbol}: unexpected extra PTX entry")
        ptx_counts = {
            "tcgen05_alloc_group": count(ptx, rf"tcgen05\.alloc\.cta_group::{group}\."),
            "tcgen05_cp_group": count(ptx, rf"tcgen05\.cp\.cta_group::{group}\."),
            "tcgen05_mma_group": count(ptx, rf"tcgen05\.mma\.cta_group::{group}\."),
            "tcgen05_dealloc_group": count(ptx, rf"tcgen05\.dealloc\.cta_group::{group}\."),
            "tcgen05_commit_group": count(ptx, rf"tcgen05\.commit\.cta_group::{group}\."),
            "tcgen05_other_group": count(ptx, rf"cta_group::{3 - group}"),
            "cluster_multicast": count(ptx, r"multicast::cluster"),
        }
        expected_ptx = {
            "tcgen05_alloc_group": 1,
            "tcgen05_cp_group": 2,
            "tcgen05_mma_group": 16,
            "tcgen05_dealloc_group": 1,
            "tcgen05_commit_group": 5,
            "tcgen05_other_group": 0,
            "cluster_multicast": 5 if is_2sm else 0,
        }
        if ptx_counts != expected_ptx:
            raise AssertionError(f"{symbol}: PTX topology {ptx_counts} != {expected_ptx}")

        sass_counts = {
            "utcqmma": count(sass, r"^\s*/\*[^*]+\*/\s+UTCQMMA(?:\.2CTA)?\b"),
            "utcqmma_2cta": count(sass, r"UTCQMMA\.2CTA\b"),
            "ucgabar_arrive": count(sass, r"\bUCGABAR_ARV\b"),
            "ucgabar_wait": count(sass, r"\bUCGABAR_WAIT\b"),
            "utmaldg_2d": count(sass, r"\bUTMALDG\.2D\b"),
            "ldtm": count(sass, r"\bLDTM\.16dp256bit\b"),
            "ldl": count(sass, r"\bLDL\b"),
            "stl": count(sass, r"\bSTL\b"),
        }
        expected_sass = {
            "utcqmma": 16,
            "utcqmma_2cta": 16 if is_2sm else 0,
            "ucgabar_arrive": 3 if is_2sm else 0,
            "ucgabar_wait": 3 if is_2sm else 0,
            "utmaldg_2d": 10,
            "ldtm": 4,
            "ldl": 0,
            "stl": 0,
        }
        if sass_counts != expected_sass:
            raise AssertionError(
                f"{symbol}: SASS topology {sass_counts} != {expected_sass}"
            )

        symbols = run_tool("--dump-elf-symbols", str(files["cubin"]))
        if count(
            symbols,
            rf"^STT_FUNC\s+STB_GLOBAL\s+STO_ENTRY\s+{re.escape(symbol)}$",
        ) != 1:
            raise AssertionError(f"{symbol}: unmangled global ELF entry missing")
        resources = parse_resource(
            run_tool("--dump-resource-usage", str(files["cubin"])), symbol
        )
        expected_resources = {
            "registers": 35 if is_2sm else 31,
            "stack_bytes": 0,
            "static_shared_bytes": 1024,
            "local_bytes": 0,
        }
        if resources != expected_resources:
            raise AssertionError(
                f"{symbol}: resources {resources} != {expected_resources}"
            )
        records.append(
            {
                "symbol": symbol,
                "expected_m": expected_m,
                "topology": topology,
                "config": [16, 128, 128, stages, group],
                "intended_launch": {
                    "grid_ctas": 148,
                    "block_threads": 256,
                    "cluster_size": group,
                    "dynamic_shared_bytes": 230188 if is_2sm else 223020,
                },
                "resources": resources,
                "ptx_counts": ptx_counts,
                "sass_counts": sass_counts,
                "artifacts": {
                    suffix: {
                        "path": str(path.resolve()),
                        "sha256": sha256(path),
                        "size_bytes": path.stat().st_size,
                    }
                    for suffix, path in files.items()
                },
            }
        )
    expected = {
        (expected_m, topology)
        for expected_m in EXPECTED_M_VALUES
        for topology in ("1sm", "2sm")
    }
    if seen != expected:
        raise AssertionError(f"candidate identity set mismatch: {seen} != {expected}")
    return records, context_duplicates


def audit_stock(cache: Path) -> list[dict[str, Any]]:
    directories = sorted((cache / "cache").glob("kernel.*"))
    if len(directories) != 1:
        raise AssertionError(f"expected one stock kernel, found {len(directories)}")
    directory = directories[0]
    files = {suffix: directory / f"kernel.{suffix}" for suffix in ("cu", "cubin", "ptx", "sass")}
    if not all(path.is_file() for path in files.values()):
        raise AssertionError("stock generated artifact set is incomplete")
    source = files["cu"].read_text()
    ptx = files["ptx"].read_text()
    sass = files["sass"].read_text()
    entries = re.findall(r"^\.entry ([^(]+)\(", ptx, flags=re.MULTILINE)
    if len(entries) != 1:
        raise AssertionError(f"stock PTX entry count is {len(entries)}")
    symbol = entries[0]
    required = (
        "cute::UMMA::Major::K, cute::UMMA::Major::K, 128,",
        "0, 4096, 6144,",
        "128, 128, 128,",
        "8,",
        "2, true,",
        "148,",
        "GemmType::MGroupedMasked, false,",
    )
    missing = [value for value in required if value not in source]
    if missing:
        raise AssertionError(f"stock generated constants missing {missing}")
    ptx_counts = {
        "tcgen05_mma_group2": count(ptx, r"tcgen05\.mma\.cta_group::2\."),
        "tcgen05_other_group": count(ptx, r"cta_group::1"),
        "cluster_multicast": count(ptx, r"multicast::cluster"),
    }
    if ptx_counts != {
        "tcgen05_mma_group2": 16,
        "tcgen05_other_group": 0,
        "cluster_multicast": 5,
    }:
        raise AssertionError(f"stock PTX topology mismatch: {ptx_counts}")
    sass_counts = {
        "utcqmma_2cta": count(sass, r"UTCQMMA\.2CTA\b"),
        "ucgabar_arrive": count(sass, r"\bUCGABAR_ARV\b"),
        "ucgabar_wait": count(sass, r"\bUCGABAR_WAIT\b"),
        "utmaldg_2d": count(sass, r"\bUTMALDG\.2D\b"),
        "ldtm": count(sass, r"\bLDTM\.16dp256bit\b"),
        "ldl": count(sass, r"\bLDL\b"),
        "stl": count(sass, r"\bSTL\b"),
    }
    expected_sass = {
        "utcqmma_2cta": 16,
        "ucgabar_arrive": 3,
        "ucgabar_wait": 3,
        "utmaldg_2d": 10,
        "ldtm": 32,
        "ldl": 0,
        "stl": 0,
    }
    if sass_counts != expected_sass:
        raise AssertionError(f"stock SASS topology {sass_counts} != {expected_sass}")
    resources = parse_resource(
        run_tool("--dump-resource-usage", str(files["cubin"])), symbol
    )
    if resources["stack_bytes"] or resources["local_bytes"]:
        raise AssertionError(f"stock uses stack/local memory: {resources}")
    return [
        {
            "symbol": symbol,
            "expected_m_values_sharing_identity": list(EXPECTED_M_VALUES),
            "config": [128, 128, 128, 8, 2],
            "intended_launch": {
                "grid_ctas": 148,
                "block_threads": 256,
                "cluster_size": 2,
            },
            "resources": resources,
            "ptx_counts": ptx_counts,
            "sass_counts": sass_counts,
            "artifacts": {
                suffix: {
                    "path": str(path.resolve()),
                    "sha256": sha256(path),
                    "size_bytes": path.stat().st_size,
                }
                for suffix, path in files.items()
            },
        }
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-context-duplicates", action="store_true")
    args = parser.parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    candidate, context_duplicates = audit_candidate(
        Path(manifest["variants"]["candidate"]["jit_cache"]).resolve(),
        allow_context_duplicates=args.allow_context_duplicates,
    )
    result = {
        "schema_version": 1,
        "audit": "cpu_only_generated_source_ptx_sass_elf_resources",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "candidate": candidate,
        "candidate_context_duplicates": context_duplicates,
        "stock": audit_stock(
            Path(manifest["variants"]["stock"]["jit_cache"]).resolve()
        ),
        "status": "PASS",
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
