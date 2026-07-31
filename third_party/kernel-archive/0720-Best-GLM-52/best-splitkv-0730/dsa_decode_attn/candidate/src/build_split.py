"""JIT-build the split-KV MLA extension.

Toolchain copied verbatim from the campaign's own build recipe
(mla_ptx_work/ext/build_ext.py, itself copied from
flashmla_combine_decode_provider._jit_load) so the compiled result is comparable
with everything already measured in this tree -- same nvcc flags, same
-gencode, same CUTLASS include set.
"""
import os
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

SRC = Path("/mnt/b300-shared/home/qinhaiyan/wwxq/FlashMLA").resolve()
HERE = Path(__file__).parent.resolve()

sources = [HERE / "api_split.cpp", HERE / "mla_split_phase1.cu",
           HERE / "mla_split_combine.cu", HERE / "inst_identity.cu"]
for p in sources:
    assert p.is_file(), p

inc = [SRC / "csrc", SRC / "csrc/kerutils/include", SRC / "csrc/sm90",
       SRC / "csrc/cutlass/include", SRC / "csrc/cutlass/tools/util/include",
       HERE, Path("/usr/local/cuda/targets/x86_64-linux/include/cccl")]


# The pybind init symbol is PyInit_<name>, so a vendored .so must be BUILT under
# the name it will be imported as -- renaming the file afterwards does not work.
EXT_NAME = os.environ.get("MLA_EXT_NAME", "mla_split_ext")


def build(verbose=True):
    return load(
        name=EXT_NAME,
        sources=[str(p) for p in sources],
        extra_cflags=["-O3", "-std=c++20", "-DNDEBUG", "-Wno-deprecated-declarations"],
        extra_cuda_cflags=[
            "-O3", "-std=c++20", "-DNDEBUG", "-D_USE_MATH_DEFINES",
            "-Wno-deprecated-declarations",
            "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "--expt-relaxed-constexpr", "--expt-extended-lambda", "--use_fast_math",
            "-gencode", "arch=compute_100f,code=sm_100f",
            "--threads", os.environ.get("NVCC_THREADS", "16"),
        ],
        extra_include_paths=[str(p) for p in inc],
        # cuTensorMapReplaceAddress is a driver API; CUTLASS's dlopen wrapper does
        # not cover it, so link libcuda directly.
        extra_ldflags=["-lcuda"],
        with_cuda=True, verbose=verbose,
    )


if __name__ == "__main__":
    ext = build()
    print("BUILD OK", ext)
