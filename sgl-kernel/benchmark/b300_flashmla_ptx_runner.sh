#!/usr/bin/env bash

set -euo pipefail

MODE="${1:-stock}"
OFFLINE_DIR="${B300_OFFLINE_DIR:-/mnt/b300-shared/home/qinhaiyan/B300-OPT-GLM52/offline/glm-flashmla-ptx}"
STOCK_WHEEL="/mnt/b300-shared/home/qinhaiyan/wwxq/sglang_kernel-0.4.4-cp310-abi3-manylinux2014_x86_64.whl"
STOCK_WHEEL_SHA256="558bc7035e3c0a795e8c3eca3cc29e189c7d29f1db2471a0a71b9156a8ee7fa1"
STOCK_EXTENSION_SHA256="d8d97150bd86381c73406603cb7d6b682767535e0526053f04e3acefadb13316"
BENCHMARK="${B300_SOURCE_DIR}/sgl-kernel/benchmark/bench_flashmla_glm52_decode.py"
ANALYZER="${B300_SOURCE_DIR}/sgl-kernel/benchmark/analyze_flashmla_glm52.py"
EXISTING_TEST="${B300_SOURCE_DIR}/sgl-kernel/tests/test_flashmla.py"
CUOBJDUMP="/usr/local/cuda/bin/cuobjdump"
NVDISASM="/usr/local/cuda/bin/nvdisasm"
NCU="/usr/local/cuda/bin/ncu"
NSYS="/usr/local/cuda/bin/nsys"
IMAGE="${B300_FLASHMLA_IMAGE:-m.daocloud.io/docker.io/lmsysorg/sglang@sha256:ceaf8b16e02d165143633ac228bbb994a05fe77d7e0526cf035ae4bbf4eacc36}"
export B300_FLASHMLA_IMAGE="${IMAGE}"

: "${B300_AGENT:?}"
: "${B300_COMMIT:?}"
: "${B300_RUN_ID:?}"
: "${B300_SOURCE_DIR:?}"
: "${B300_BUILD_DIR:?}"
: "${B300_ARTIFACT_DIR:?}"

if [[ "${B300_FLASHMLA_IN_CONTAINER:-0}" != 1 ]]; then
    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || ! "${CUDA_VISIBLE_DEVICES}" =~ ^[0-7]$ ]]; then
        echo "Select exactly one physical campaign GPU" >&2
        exit 2
    fi
    docker image inspect "${IMAGE}" > "${B300_ARTIFACT_DIR}/container-image.json"
    exec docker run \
        --rm \
        --pull never \
        --network none \
        --ipc host \
        --shm-size 32g \
        --gpus "device=${CUDA_VISIBLE_DEVICES}" \
        --user "$(id -u):$(id -g)" \
        --mount type=bind,src=/mnt/b300-shared,dst=/mnt/b300-shared \
        --mount type=bind,src=/usr/local/cuda-13.1,dst=/usr/local/cuda,readonly \
        --workdir "${B300_SOURCE_DIR}" \
        --env B300_AGENT \
        --env B300_COMMIT \
        --env B300_RUN_ID \
        --env B300_SOURCE_DIR \
        --env B300_BUILD_DIR \
        --env B300_ARTIFACT_DIR \
        --env B300_OFFLINE_DIR \
        --env B300_BASELINE_ARTIFACT \
        --env B300_FLASHMLA_IMAGE \
        --env B300_FLASHMLA_IN_CONTAINER=1 \
        --env CUDA_HOME=/usr/local/cuda \
        --env CUDA_PATH=/usr/local/cuda \
        --env PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
        --env CUDA_VISIBLE_DEVICES=0 \
        "${IMAGE}" \
        bash "${B300_SOURCE_DIR}/sgl-kernel/benchmark/b300_flashmla_ptx_runner.sh" "${MODE}"
fi

export PYTHONDONTWRITEBYTECODE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CCACHE_DIR="${B300_BUILD_DIR}/ccache"
export CCACHE_TEMPDIR="${B300_BUILD_DIR}/ccache-tmp"
mkdir -p \
    "${CCACHE_DIR}" \
    "${CCACHE_TEMPDIR}" \
    "${B300_BUILD_DIR}" \
    "${B300_ARTIFACT_DIR}/benchmark" \
    "${B300_ARTIFACT_DIR}/build" \
    "${B300_ARTIFACT_DIR}/correctness" \
    "${B300_ARTIFACT_DIR}/ptx" \
    "${B300_ARTIFACT_DIR}/sass" \
    "${B300_ARTIFACT_DIR}/ncu" \
    "${B300_ARTIFACT_DIR}/nsys"

python3 - "${B300_ARTIFACT_DIR}/manifest.json" "${MODE}" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

path, mode = Path(sys.argv[1]), sys.argv[2]
manifest = json.loads(path.read_text())
source_root = Path(os.environ["B300_SOURCE_DIR"])

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

harness_paths = {
    "benchmark": source_root
    / "sgl-kernel/benchmark/bench_flashmla_glm52_decode.py",
    "analyzer": source_root
    / "sgl-kernel/benchmark/analyze_flashmla_glm52.py",
    "runner": source_root
    / "sgl-kernel/benchmark/b300_flashmla_ptx_runner.sh",
}
driver = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=driver_version",
        "--format=csv,noheader",
        "-i",
        "0",
    ],
    text=True,
).splitlines()[0]
manifest.update(
    {
        "base_sha": "2786889e7d5d35bb39f8b3785766ebab9fdab2fe",
        "mode": mode,
        "driver": driver,
        "cuda": torch.version.cuda,
        "pytorch": torch.__version__,
        "harness_sha": os.environ["B300_COMMIT"],
        "harness_source_sha256": sha256(harness_paths["benchmark"]),
        "harness_files": {
            name: {
                "path": str(path.relative_to(source_root)),
                "sha256": sha256(path),
            }
            for name, path in harness_paths.items()
        },
        "shape": {
            "batch_sizes": [16, 32],
            "s_q": 1,
            "h_q": 64,
            "d_qk": 576,
            "d_v": 512,
            "topk": 2048,
            "page_size": 64,
            "captured_decode_length_window": [512, 575],
            "endpoint_length": 576,
            "packed_kv_row_bytes": 656,
            "block_table_shape": ["B", 0],
        },
        "workload": {
            "stock": "stock-wheel production-shape reproduction",
            "build": "offline source build and binary inspection",
            "ab": "paired source-build correctness and microbenchmark",
            "profile": "source-build targeted ncu/nsys comparison",
            "all": "offline build plus paired correctness and microbenchmark",
        }.get(mode, mode),
        "baseline_artifact": os.environ.get("B300_BASELINE_ARTIFACT") or None,
        "container_image": {
            "reference": os.environ["B300_FLASHMLA_IMAGE"],
            "inspect_artifact": "container-image.json",
            "network": "none",
        },
        "protocol": {
            "warmup": 100,
            "iterations": 1000,
            "fresh_process_pairs": 3,
            "pair_order": ["baseline/candidate", "candidate/baseline", "baseline/candidate"],
            "authoritative_timing": "CUDA events only; profiler timings diagnostic",
        },
        "dependencies": {
            "cutlass_top": {
                "commit": "57e3cfb47a2d9e0d46eb6335c3dc411498efa198",
                "archive_sha256": "09237099a70f80bff1dc8bb80c843a674bb4fdcb46e43cc6993e711c5ca89bb5",
            },
            "fmt": {
                "commit": "553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28",
                "archive_sha256": "c314292789d28c3c3b420e75a7b2d1706f685f7fb63289128d46aeaea2c6be71",
            },
            "triton": {
                "tag": "v3.6.0",
                "archive_sha256": "be270ed11ca5a8fbd9d7941c5bbe9a23a9f6e2ffd372c8398346928bee464774",
            },
            "flashinfer": {
                "commit": "bc29697ba20b7e6bdb728ded98f04788e16ee021",
                "archive_sha256": "931dfd118f4b6de8c7d98702153c7c03840139170af21a07607693bd9749744d",
            },
            "flash_attention": {
                "commit": "f89bc2306632d1ec5f97b014dded4254f5b4a907",
                "archive_sha256": "418b5681584dc3efff496a1cab5ffd58d2728d89dcfe0ea16e6985d6ef35c68c",
            },
            "flashmla": {
                "commit": "05e26647fe840b8baedae486c2d86d5ce4efeb7c",
                "archive_sha256": "ce369489bbfc42cdfbba9aa949de0270e64469d530748dea9f4f60b3c69dea9b",
            },
            "flashmla_cutlass": {
                "commit": "147f5673d0c1c3dcf66f78d677fd647e4a020219",
                "archive_sha256": "9f6c53320a85b4a570975e557918cde65168cd311f081920446c238437347dc6",
            },
            "stock_wheel_sha256": "558bc7035e3c0a795e8c3eca3cc29e189c7d29f1db2471a0a71b9156a8ee7fa1",
        },
        "build_configuration": {
            "cuda_compiler": "/usr/local/cuda/bin/nvcc",
            "cuda_version_cmake": "13.1",
            "architectures": ["sm_103a"],
            "common_flags": [
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--use_fast_math",
                "-lineinfo",
                "-Xptxas=-v",
                "--keep",
            ],
            "baseline_specialization": False,
            "candidate_specialization": True,
            "parallel_compile_jobs": 1,
            "ccache_dir": os.environ["CCACHE_DIR"],
        },
    }
)
path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY

verify_sha256() {
    local path="$1"
    local expected="$2"
    local actual
    actual="$(sha256sum "${path}" | awk '{print $1}')"
    if [[ "${actual}" != "${expected}" ]]; then
        echo "SHA256 mismatch: ${path}: ${actual} != ${expected}" >&2
        return 1
    fi
}

verify_stock_wheel() {
    test -f "${STOCK_WHEEL}"
    verify_sha256 "${STOCK_WHEEL}" "${STOCK_WHEEL_SHA256}"
}

verify_offline_archives() {
    (
        cd "${OFFLINE_DIR}"
        sha256sum --check <<'EOF'
09237099a70f80bff1dc8bb80c843a674bb4fdcb46e43cc6993e711c5ca89bb5  cutlass-top.tar.gz
c314292789d28c3c3b420e75a7b2d1706f685f7fb63289128d46aeaea2c6be71  fmt.tar.gz
be270ed11ca5a8fbd9d7941c5bbe9a23a9f6e2ffd372c8398346928bee464774  triton.tar.gz
931dfd118f4b6de8c7d98702153c7c03840139170af21a07607693bd9749744d  flashinfer.tar.gz
418b5681584dc3efff496a1cab5ffd58d2728d89dcfe0ea16e6985d6ef35c68c  sgl-attn.tar.gz
ce369489bbfc42cdfbba9aa949de0270e64469d530748dea9f4f60b3c69dea9b  flashmla.tar.gz
9f6c53320a85b4a570975e557918cde65168cd311f081920446c238437347dc6  cutlass.tar.gz
EOF
    )
}

stage_archive() {
    local archive="$1"
    local expected="$2"
    local destination="$3"
    local marker="${destination}/.b300-extracted-sha256"
    verify_sha256 "${archive}" "${expected}"
    if [[ -f "${marker}" ]]; then
        test "$(tr -d '\n' < "${marker}")" = "${expected}"
        return
    fi
    if [[ -d "${destination}" ]] \
        && [[ -z "$(find "${destination}" -mindepth 1 -print -quit)" ]]; then
        rmdir "${destination}"
    elif [[ -e "${destination}" ]]; then
        echo "Refusing unverified dependency directory: ${destination}" >&2
        return 1
    fi
    local parent
    local staging
    parent="$(dirname "${destination}")"
    mkdir -p "${parent}"
    staging="$(mktemp -d "${parent}/.extract.XXXXXX")"
    tar -xzf "${archive}" -C "${staging}" --strip-components=1
    printf '%s\n' "${expected}" > "${staging}/.b300-extracted-sha256"
    mv "${staging}" "${destination}"
}

stage_dependencies() {
    local arm="$1"
    local dependency_root="${B300_BUILD_DIR}/${arm}/deps"
    stage_archive \
        "${OFFLINE_DIR}/cutlass-top.tar.gz" \
        09237099a70f80bff1dc8bb80c843a674bb4fdcb46e43cc6993e711c5ca89bb5 \
        "${dependency_root}/cutlass-top"
    stage_archive \
        "${OFFLINE_DIR}/fmt.tar.gz" \
        c314292789d28c3c3b420e75a7b2d1706f685f7fb63289128d46aeaea2c6be71 \
        "${dependency_root}/fmt"
    stage_archive \
        "${OFFLINE_DIR}/triton.tar.gz" \
        be270ed11ca5a8fbd9d7941c5bbe9a23a9f6e2ffd372c8398346928bee464774 \
        "${dependency_root}/triton"
    stage_archive \
        "${OFFLINE_DIR}/flashinfer.tar.gz" \
        931dfd118f4b6de8c7d98702153c7c03840139170af21a07607693bd9749744d \
        "${dependency_root}/flashinfer"
    stage_archive \
        "${OFFLINE_DIR}/sgl-attn.tar.gz" \
        418b5681584dc3efff496a1cab5ffd58d2728d89dcfe0ea16e6985d6ef35c68c \
        "${dependency_root}/sgl-attn"
    stage_archive \
        "${OFFLINE_DIR}/flashmla.tar.gz" \
        ce369489bbfc42cdfbba9aa949de0270e64469d530748dea9f4f60b3c69dea9b \
        "${dependency_root}/flashmla"
    stage_archive \
        "${OFFLINE_DIR}/cutlass.tar.gz" \
        9f6c53320a85b4a570975e557918cde65168cd311f081920446c238437347dc6 \
        "${dependency_root}/flashmla/csrc/cutlass"
}

preflight_site() {
    local site="$1"
    local expected="$2"
    local label="$3"
    PYTHONPATH="${site}" python3 - "${expected}" \
        "${B300_ARTIFACT_DIR}/build/${label}-import.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from sgl_kernel import flashmla_ops

expected, output_path = sys.argv[1:]
module_path = Path(flashmla_ops.__file__).resolve()
actual = hashlib.sha256(module_path.read_bytes()).hexdigest()
record = {
    "module": str(module_path),
    "expected_sha256": expected,
    "actual_sha256": actual,
    "match": actual == expected,
}
Path(output_path).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
if actual != expected:
    raise SystemExit(f"wrong FlashMLA extension imported: {actual} != {expected}")
PY
}

make_site() {
    local extension="$1"
    local site="$2"
    local arm="$3"
    local extension_sha
    verify_stock_wheel
    extension_sha="$(sha256sum "${extension}" | awk '{print $1}')"
    if [[ ! -d "${site}/sgl_kernel" ]]; then
        mkdir -p "${site}"
        python3 -m pip install \
            --no-index \
            --no-deps \
            --target "${site}" \
            "${STOCK_WHEEL}"
    fi
    cp -a "${B300_SOURCE_DIR}/sgl-kernel/python/sgl_kernel/." "${site}/sgl_kernel/"
    find "${site}/sgl_kernel" -maxdepth 1 -type f -name 'flashmla_ops*.so' -delete
    cp -a "${extension}" "${site}/sgl_kernel/$(basename "${extension}")"
    preflight_site "${site}" "${extension_sha}" "${arm}"
}

build_arm() {
    local arm="$1"
    local specialization="$2"
    local arm_root="${B300_BUILD_DIR}/${arm}"
    local dependency_root="${arm_root}/deps"
    local cmake_build="${arm_root}/cmake"
    local keep_dir="${arm_root}/keep"
    local site="${arm_root}/site"
    local arm_ccache="${arm_root}/ccache"
    local torch_prefix
    local extension
    local configure_log="${B300_ARTIFACT_DIR}/build/${arm}-configure.log"
    local build_log="${B300_ARTIFACT_DIR}/build/${arm}-build.log"

    stage_dependencies "${arm}"
    torch_prefix="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"
    mkdir -p "${cmake_build}" "${keep_dir}" "${arm_ccache}"
    (
        export CCACHE_DIR="${arm_ccache}"
        cmake \
            -S "${B300_SOURCE_DIR}/sgl-kernel" \
            -B "${cmake_build}" \
            -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
            -DCMAKE_CUDA_ARCHITECTURES=OFF \
            -DCMAKE_PREFIX_PATH="${torch_prefix}" \
            -DPython_EXECUTABLE="$(command -v python3)" \
            -DSKBUILD_SABI_COMPONENT=Development.SABIModule \
            -DSKBUILD_SABI_VERSION=3.10 \
            -DCUDA_VERSION=13.1 \
            -DENABLE_CCACHE=ON \
            -DENABLE_BELOW_SM90=OFF \
            -DSGL_KERNEL_ENABLE_FA3=OFF \
            -DSGL_KERNEL_COMPILE_THREADS=1 \
            -DSGL_FLASHMLA_SM103_ONLY=ON \
            -DSGL_FLASHMLA_GLM52_FLAT_TOKEN_INDEX="${specialization}" \
            -DSGL_FLASHMLA_KEEP_DIR="${keep_dir}" \
            -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
            "-DFETCHCONTENT_SOURCE_DIR_REPO-CUTLASS=${dependency_root}/cutlass-top" \
            "-DFETCHCONTENT_SOURCE_DIR_REPO-FMT=${dependency_root}/fmt" \
            "-DFETCHCONTENT_SOURCE_DIR_REPO-TRITON=${dependency_root}/triton" \
            "-DFETCHCONTENT_SOURCE_DIR_REPO-FLASHINFER=${dependency_root}/flashinfer" \
            "-DFETCHCONTENT_SOURCE_DIR_REPO-FLASH-ATTENTION=${dependency_root}/sgl-attn" \
            "-DFETCHCONTENT_SOURCE_DIR_REPO-FLASHMLA=${dependency_root}/flashmla" \
            "-DFETCHCONTENT_SOURCE_DIR_REPO-FLASHMLA-CUTLASS=${dependency_root}/flashmla/csrc/cutlass" \
            2>&1 | tee "${configure_log}"
        cmake --build "${cmake_build}" --target flashmla_ops --parallel 1 \
            2>&1 | tee "${build_log}"
    )
    extension="$(find "${cmake_build}" -type f -name 'flashmla_ops*.so' -print -quit)"
    test -n "${extension}"
    make_site "${extension}" "${site}" "${arm}"
    sha256sum "${extension}" > "${B300_ARTIFACT_DIR}/${arm}-binary.sha256"
    find "${keep_dir}" -type f -printf '%P\n' | sort \
        > "${B300_ARTIFACT_DIR}/build/${arm}-keep-inventory.txt"
    (
        cd "${keep_dir}"
        find . -type f -print0 | sort -z | xargs -0 -r sha256sum
    ) > "${B300_ARTIFACT_DIR}/build/${arm}-keep-sha256.txt"
}

install_stock_site() {
    local site="${B300_BUILD_DIR}/stock/site"
    verify_stock_wheel
    if [[ ! -d "${site}/sgl_kernel" ]]; then
        mkdir -p "${site}"
        python3 -m pip install \
            --no-index \
            --no-deps \
            --target "${site}" \
            "${STOCK_WHEEL}"
    fi
    local extension
    extension="$(find "${site}/sgl_kernel" -maxdepth 1 -type f -name 'flashmla_ops*.so' -print -quit)"
    test -n "${extension}"
    verify_sha256 "${extension}" "${STOCK_EXTENSION_SHA256}"
    preflight_site "${site}" "${STOCK_EXTENSION_SHA256}" stock
    sha256sum \
        "${STOCK_WHEEL}" \
        "${extension}" \
        > "${B300_ARTIFACT_DIR}/stock-binary.sha256"
}

run_benchmark_process() {
    local site="$1"
    local label="$2"
    local batch_size="$3"
    local seed="$4"
    local output="${B300_ARTIFACT_DIR}/benchmark/${label}.json"
    PYTHONPATH="${site}" python3 "${BENCHMARK}" \
        --output "${output}" \
        --label "${label}" \
        --batch-size "${batch_size}" \
        --active-tokens 576 \
        --length-pattern production \
        --topk 2048 \
        --page-size 64 \
        --warmup 100 \
        --iterations 1000 \
        --seed "${seed}"
}

run_non_target_process() {
    local site="$1"
    local label="$2"
    local case_name="$3"
    local seed="$4"
    local batch_size
    local heads_q
    local local_heads_q
    local active_tokens
    local length_pattern
    local topk
    case "${case_name}" in
        h128)
            batch_size=2
            heads_q=128
            local_heads_q=128
            active_tokens=576
            length_pattern=production
            topk=2048
            ;;
        topk128)
            batch_size=6
            heads_q=64
            local_heads_q=8
            active_tokens=65
            length_pattern=uniform
            topk=128
            ;;
        *)
            echo "Unknown non-target case: ${case_name}" >&2
            return 2
            ;;
    esac
    PYTHONPATH="${site}" python3 "${BENCHMARK}" \
        --output "${B300_ARTIFACT_DIR}/benchmark/${label}.json" \
        --label "${label}" \
        --batch-size "${batch_size}" \
        --heads-q "${heads_q}" \
        --local-heads-q "${local_heads_q}" \
        --active-tokens "${active_tokens}" \
        --length-pattern "${length_pattern}" \
        --topk "${topk}" \
        --page-size 64 \
        --warmup 100 \
        --iterations 1000 \
        --seed "${seed}"
}

run_correctness_case() {
    local site="$1"
    local label="$2"
    local batch_size="$3"
    local active_tokens="$4"
    local topk="$5"
    local length_pattern="${6:-uniform}"
    PYTHONPATH="${site}" python3 "${BENCHMARK}" \
        --output "${B300_ARTIFACT_DIR}/correctness/${label}.json" \
        --label "${label}" \
        --batch-size "${batch_size}" \
        --active-tokens "${active_tokens}" \
        --length-pattern "${length_pattern}" \
        --topk "${topk}" \
        --page-size 64 \
        --seed 20260729 \
        --correctness-only
}

run_existing_sparse_decode_cases() {
    local site="$1"
    local arm="$2"
    local output="${B300_ARTIFACT_DIR}/correctness/${arm}-existing-sparse-decode.json"
    PYTHONPATH="${site}" python3 - "${EXISTING_TEST}" "${output}" "${arm}" <<'PY'
import json
import random
import runpy
import sys
import traceback
from pathlib import Path

import torch

test_path, output_path, arm = sys.argv[1:]
namespace = runpy.run_path(test_path)
test = namespace["test_flash_mla_decode"]
cases = [
    {
        "name": "h128_sparse_topk2048",
        "b": 2,
        "s_q": 1,
        "s_k": 4096,
        "is_varlen": False,
        "causal_topk": (False, 2048),
        "dtype": torch.bfloat16,
    },
    {
        "name": "h128_sparse_topk128",
        "b": 2,
        "s_q": 1,
        "s_k": 140,
        "is_varlen": False,
        "causal_topk": (False, 128),
        "dtype": torch.bfloat16,
    },
    {
        "name": "h128_sparse_sq2_fallback",
        "b": 2,
        "s_q": 2,
        "s_k": 140,
        "is_varlen": True,
        "causal_topk": (False, 2048),
        "dtype": torch.bfloat16,
    },
]
records = []
verdict = "PASS"
for index, case in enumerate(cases):
    torch.manual_seed(20260729 + index)
    random.seed(20260729 + index)
    arguments = {key: value for key, value in case.items() if key != "name"}
    try:
        test(**arguments)
        records.append({"name": case["name"], "verdict": "PASS"})
    except Exception as error:
        verdict = "FAIL"
        records.append(
            {
                "name": case["name"],
                "verdict": "FAIL",
                "exception_type": type(error).__name__,
                "exception": str(error),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        torch.cuda.empty_cache()
result = {
    "schema_version": 1,
    "label": f"{arm}-existing-sparse-decode",
    "source_test": test_path,
    "cases": records,
    "correctness": {"verdict": verdict},
}
temporary = Path(output_path + ".tmp")
temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
temporary.replace(output_path)
if verdict != "PASS":
    raise SystemExit(1)
PY
}

run_stock_suite() {
    local site="${B300_BUILD_DIR}/stock/site"
    local round
    install_stock_site
    for round in 1 2 3; do
        run_benchmark_process "${site}" "stock-b16-r${round}" 16 "$((20260729 + round))"
        run_benchmark_process "${site}" "stock-b32-r${round}" 32 "$((20260729 + round))"
    done
    run_correctness_case "${site}" "stock-production-irregular" 8 576 2048 mixed
    run_correctness_case "${site}" "stock-fallback-topk128" 6 65 128
    run_correctness_case "${site}" "stock-empty" 4 0 2048
    run_correctness_case "${site}" "stock-endpoint576" 16 576 2048
}

run_ab_suite_for_batch() {
    local batch_size="$1"
    local baseline_site="${B300_BUILD_DIR}/baseline/site"
    local candidate_site="${B300_BUILD_DIR}/candidate/site"
    local pair
    for pair in 1 2 3; do
        if (( pair % 2 == 1 )); then
            run_benchmark_process \
                "${baseline_site}" "baseline-b${batch_size}-p${pair}" \
                "${batch_size}" "$((20260729 + pair))"
            run_benchmark_process \
                "${candidate_site}" "candidate-b${batch_size}-p${pair}" \
                "${batch_size}" "$((20260729 + pair))"
        else
            run_benchmark_process \
                "${candidate_site}" "candidate-b${batch_size}-p${pair}" \
                "${batch_size}" "$((20260729 + pair))"
            run_benchmark_process \
                "${baseline_site}" "baseline-b${batch_size}-p${pair}" \
                "${batch_size}" "$((20260729 + pair))"
        fi
    done
}

run_non_target_ab_suite() {
    local baseline_site="${B300_BUILD_DIR}/baseline/site"
    local candidate_site="${B300_BUILD_DIR}/candidate/site"
    local case_name
    local pair
    for case_name in h128 topk128; do
        for pair in 1 2 3; do
            if (( pair % 2 == 1 )); then
                run_non_target_process \
                    "${baseline_site}" "baseline-nontarget-${case_name}-p${pair}" \
                    "${case_name}" "$((20260729 + pair))"
                run_non_target_process \
                    "${candidate_site}" "candidate-nontarget-${case_name}-p${pair}" \
                    "${case_name}" "$((20260729 + pair))"
            else
                run_non_target_process \
                    "${candidate_site}" "candidate-nontarget-${case_name}-p${pair}" \
                    "${case_name}" "$((20260729 + pair))"
                run_non_target_process \
                    "${baseline_site}" "baseline-nontarget-${case_name}-p${pair}" \
                    "${case_name}" "$((20260729 + pair))"
            fi
        done
    done
}

run_built_correctness() {
    local arm
    for arm in baseline candidate; do
        run_correctness_case \
            "${B300_BUILD_DIR}/${arm}/site" "${arm}-production-irregular" 8 576 2048 mixed
        run_correctness_case \
            "${B300_BUILD_DIR}/${arm}/site" "${arm}-fallback-topk128" 6 65 128
        run_correctness_case \
            "${B300_BUILD_DIR}/${arm}/site" "${arm}-empty" 4 0 2048
        run_correctness_case \
            "${B300_BUILD_DIR}/${arm}/site" "${arm}-endpoint576" 16 576 2048
        run_existing_sparse_decode_cases "${B300_BUILD_DIR}/${arm}/site" "${arm}"
    done
}

extract_binary_evidence() {
    local arm="$1"
    local extension
    local keep_dir="${B300_BUILD_DIR}/${arm}/keep"
    local arm_sass="${B300_ARTIFACT_DIR}/sass/${arm}"
    local elf_dir="${arm_sass}/cubins"
    local ptx_dir="${B300_ARTIFACT_DIR}/ptx/${arm}"
    local target_hits=0
    extension="$(find "${B300_BUILD_DIR}/${arm}/site/sgl_kernel" -type f -name 'flashmla_ops*.so' -print -quit)"
    test -n "${extension}"
    extension="$(realpath "${extension}")"
    mkdir -p "${arm_sass}" "${elf_dir}" "${ptx_dir}"
    test -z "$(find "${elf_dir}" -mindepth 1 -print -quit)"
    "${CUOBJDUMP}" --dump-resource-usage "${extension}" > "${arm_sass}/resource.txt"
    "${CUOBJDUMP}" --dump-ptx "${extension}" > "${ptx_dir}/embedded.ptx"
    "${CUOBJDUMP}" --dump-sass "${extension}" > "${arm_sass}/extension.cuobjdump.sass"
    "${CUOBJDUMP}" --list-elf "${extension}" > "${arm_sass}/elf-list.txt"

    mapfile -d '' -t retained_ptx < <(
        find "${keep_dir}" -type f -name '*.ptx' -print0 | sort -z
    )
    : > "${ptx_dir}/retained-sha256.txt"
    local retained_target_hits=0
    local ptx
    local ptx_tag
    for ptx in "${retained_ptx[@]}"; do
        if grep -a -q 'flash_fwd_splitkv_mla_fp8_sparse_kernel' "${ptx}"; then
            printf -v ptx_tag 'target-%04d.ptx' "${retained_target_hits}"
            cp "${ptx}" "${ptx_dir}/${ptx_tag}"
            sha256sum "${ptx_dir}/${ptx_tag}" >> "${ptx_dir}/retained-sha256.txt"
            retained_target_hits=$((retained_target_hits + 1))
        fi
    done
    ((retained_target_hits > 0))

    (
        cd "${elf_dir}"
        "${CUOBJDUMP}" --extract-elf all "${extension}"
    )
    mapfile -d '' -t cubins < <(
        find "${elf_dir}" -maxdepth 1 -type f -print0 | sort -z
    )
    ((${#cubins[@]} > 0))
    : > "${arm_sass}/cubin-sha256.txt"
    : > "${arm_sass}/target-cubins.txt"
    local cubin
    local tag
    local index
    for index in "${!cubins[@]}"; do
        cubin="${cubins[$index]}"
        printf -v tag '%04d' "${index}"
        sha256sum "${cubin}" >> "${arm_sass}/cubin-sha256.txt"
        "${CUOBJDUMP}" --dump-elf-symbols "${cubin}" > "${arm_sass}/${tag}.symbols.txt"
        "${CUOBJDUMP}" --dump-resource-usage "${cubin}" > "${arm_sass}/${tag}.resource.txt"
        "${CUOBJDUMP}" --dump-sass "${cubin}" > "${arm_sass}/${tag}.cuobjdump.sass"
        "${NVDISASM}" -g -c "${cubin}" > "${arm_sass}/${tag}.nvdisasm.txt"
        if grep -a -q 'flash_fwd_splitkv_mla_fp8_sparse_kernel' \
            "${arm_sass}/${tag}.symbols.txt" "${arm_sass}/${tag}.cuobjdump.sass"; then
            printf '%s\n' "${cubin}" >> "${arm_sass}/target-cubins.txt"
            target_hits=$((target_hits + 1))
        fi
    done
    ((target_hits > 0))
}

analyze_ab_results() {
    python3 "${ANALYZER}" \
        "${B300_ARTIFACT_DIR}/benchmark" \
        --output "${B300_ARTIFACT_DIR}/analysis.json"
    cp "${B300_ARTIFACT_DIR}/analysis.json" "${B300_BUILD_DIR}/analysis.json"
}

run_ncu_capture() {
    local arm="$1"
    local site="${B300_BUILD_DIR}/${arm}/site"
    local prefix="${B300_ARTIFACT_DIR}/ncu/${arm}-main"
    PYTHONPATH="${site}" "${NCU}" \
        --target-processes all \
        --profile-from-start off \
        --kernel-name-base demangled \
        --kernel-name 'regex:.*flash_fwd_splitkv_mla_fp8_sparse_kernel.*' \
        --launch-count 1 \
        --set full \
        --replay-mode kernel \
        --force-overwrite \
        --export "${prefix}" \
        env PYTHONPATH="${site}" python3 "${BENCHMARK}" \
            --output "${B300_ARTIFACT_DIR}/ncu/${arm}-input.json" \
            --label "${arm}-ncu" \
            --batch-size 16 \
            --active-tokens 576 \
            --length-pattern production \
            --topk 2048 \
            --page-size 64 \
            --profile-only \
            --profile-region operator \
            --profile-iterations 1
    "${NCU}" --import "${prefix}.ncu-rep" --page details \
        > "${prefix}-details.txt"
    "${NCU}" --import "${prefix}.ncu-rep" --page raw --csv \
        > "${prefix}-raw.csv"
}

run_nsys_capture() {
    local label="$1"
    local site="$2"
    local topk="$3"
    local pattern="$4"
    local active_tokens="$5"
    local region="${6:-containing}"
    local prefix="${B300_ARTIFACT_DIR}/nsys/${label}-region"
    PYTHONPATH="${site}" "${NSYS}" profile \
        -t cuda,nvtx \
        -s none \
        --cpuctxsw=none \
        --cuda-graph-trace=node \
        -c cudaProfilerApi \
        --capture-range-end=stop-shutdown \
        --kill=none \
        --force-overwrite=true \
        -o "${prefix}" \
        env PYTHONPATH="${site}" python3 "${BENCHMARK}" \
            --output "${B300_ARTIFACT_DIR}/nsys/${label}-input.json" \
            --label "${label}-nsys" \
            --batch-size 16 \
            --active-tokens "${active_tokens}" \
            --length-pattern "${pattern}" \
            --topk "${topk}" \
            --page-size 64 \
            --profile-only \
            --profile-region "${region}" \
            --profile-iterations 5
    "${NSYS}" stats --report cuda_gpu_kern_sum --format csv "${prefix}.nsys-rep" \
        > "${prefix}-cuda_gpu_kern_sum.csv"
    "${NSYS}" stats --report nvtx_kern_sum --format csv "${prefix}.nsys-rep" \
        > "${prefix}-nvtx_kern_sum.csv"
}

run_profile_suite() {
    local arm
    local extension
    local extension_sha
    for arm in baseline candidate; do
        extension="$(find "${B300_BUILD_DIR}/${arm}/site/sgl_kernel" \
            -maxdepth 1 -type f -name 'flashmla_ops*.so' -print -quit)"
        test -n "${extension}"
        extension_sha="$(sha256sum "${extension}" | awk '{print $1}')"
        preflight_site \
            "${B300_BUILD_DIR}/${arm}/site" \
            "${extension_sha}" \
            "${arm}-profile-preflight"
        run_ncu_capture "${arm}"
        run_nsys_capture \
            "${arm}-target" "${B300_BUILD_DIR}/${arm}/site" 2048 production 576 containing
        run_nsys_capture \
            "${arm}-graph" "${B300_BUILD_DIR}/${arm}/site" 2048 production 576 graph
    done
    run_nsys_capture \
        candidate-fallback "${B300_BUILD_DIR}/candidate/site" 128 uniform 65 containing
    {
        grep -h 'flash_fwd_splitkv_mla_fp8_sparse_kernel' \
            "${B300_ARTIFACT_DIR}"/nsys/*cuda_gpu_kern_sum.csv || true
    } > "${B300_ARTIFACT_DIR}/nsys/dispatch-symbols.txt"
    test -s "${B300_ARTIFACT_DIR}/nsys/dispatch-symbols.txt"
}

finalize_results() {
    local payload_rc="$1"
    python3 - "${B300_ARTIFACT_DIR}" "${MODE}" "${payload_rc}" <<'PY'
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

artifact = Path(sys.argv[1])
mode = sys.argv[2]
payload_rc = int(sys.argv[3])
result_files = sorted(
    list((artifact / "benchmark").glob("*.json"))
    + list((artifact / "correctness").glob("*.json"))
    + list((artifact / "ncu").glob("*-input.json"))
    + list((artifact / "nsys").glob("*-input.json"))
)
records = []
all_passed = True
for path in result_files:
    try:
        record = json.loads(path.read_text())
    except Exception as error:
        records.append(
            {
                "file": str(path.relative_to(artifact)),
                "parse_error": str(error),
            }
        )
        all_passed = False
        continue
    correctness = record.get("correctness", {})
    verdict = correctness.get("verdict")
    records.append(
        {
            "file": str(path.relative_to(artifact)),
            "label": record.get("label"),
            "verdict": verdict,
        }
    )
    all_passed &= verdict == "PASS"

if mode == "build":
    correctness_summary = {
        "status": "not_applicable",
        "verdict": "PASS" if payload_rc == 0 else "FAIL",
        "reason": "build and binary-inspection run",
        "records": records,
    }
else:
    correctness_summary = {
        "status": "complete" if result_files else "missing",
        "verdict": "PASS" if all_passed and result_files else "FAIL",
        "records": records,
    }
(artifact / "correctness.json").write_text(
    json.dumps(correctness_summary, indent=2, sort_keys=True) + "\n"
)

benchmark_records = [
    record for record in records if record.get("file", "").startswith("benchmark/")
]
benchmark_summary = {"records": benchmark_records}
analysis_path = artifact / "analysis.json"
if analysis_path.exists():
    benchmark_summary["analysis"] = json.loads(analysis_path.read_text())
(artifact / "benchmark.json").write_text(
    json.dumps(benchmark_summary, indent=2, sort_keys=True) + "\n"
)

inventory_patterns = (
    "analysis.json",
    "container-image.json",
    "*-binary.sha256",
    "stock-binary.sha256",
    "build/*",
    "ptx/**/*",
    "sass/**/*",
    "ncu/**/*",
    "nsys/**/*",
)
inventory = []
seen = set()
for pattern in inventory_patterns:
    for path_text in glob.glob(str(artifact / pattern), recursive=True):
        path = Path(path_text)
        if not path.is_file() or path in seen:
            continue
        seen.add(path)
        hasher = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        inventory.append(
            {
                "path": str(path.relative_to(artifact)),
                "sha256": digest,
                "bytes": path.stat().st_size,
            }
        )
inventory.sort(key=lambda item: item["path"])
(artifact / "evidence-sha256.json").write_text(
    json.dumps(inventory, indent=2, sort_keys=True) + "\n"
)

manifest_path = artifact / "manifest.json"
manifest = json.loads(manifest_path.read_text())
manifest["evidence"] = inventory
build_hashes = {}
for arm in ("baseline", "candidate", "stock"):
    paths = list(artifact.glob(f"{arm}-binary.sha256"))
    if paths:
        lines = paths[0].read_text().splitlines()
        selected = lines[-1] if arm == "stock" else lines[0]
        build_hashes[arm] = selected.split()[0]
if build_hashes:
    manifest["build_sha256"] = build_hashes
if payload_rc != 0 or correctness_summary["verdict"] != "PASS":
    manifest["verdict"] = "INCORRECT"
elif "analysis" in benchmark_summary:
    manifest["verdict"] = benchmark_summary["analysis"]["verdict"]
else:
    manifest["verdict"] = "FLAT"
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY
}

on_exit() {
    local rc=$?
    trap - EXIT
    if ! finalize_results "${rc}"; then
        echo "Artifact finalization failed" >&2
        if ((rc == 0)); then
            rc=1
        fi
    fi
    exit "${rc}"
}
trap on_exit EXIT

case "${MODE}" in
    stock)
        run_stock_suite
        ;;
    build)
        verify_offline_archives
        build_arm baseline OFF
        build_arm candidate ON
        extract_binary_evidence baseline
        extract_binary_evidence candidate
        ;;
    ab)
        run_ab_suite_for_batch 16
        run_ab_suite_for_batch 32
        run_non_target_ab_suite
        run_built_correctness
        analyze_ab_results
        ;;
    profile)
        extract_binary_evidence baseline
        extract_binary_evidence candidate
        run_profile_suite
        ;;
    all)
        verify_offline_archives
        build_arm baseline OFF
        build_arm candidate ON
        extract_binary_evidence baseline
        extract_binary_evidence candidate
        run_ab_suite_for_batch 16
        run_ab_suite_for_batch 32
        run_non_target_ab_suite
        run_built_correctness
        analyze_ab_results
        ;;
    *)
        echo "Unknown mode: ${MODE}" >&2
        exit 2
        ;;
esac
