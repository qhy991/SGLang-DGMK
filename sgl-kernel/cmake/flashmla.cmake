# flash_mla
# sm90 dense decode HEAD_DIM_K=512 support (sgl-project/FlashMLA#9, merged).
FetchContent_Declare(
    repo-flashmla
    URL      https://${GITHUB_ARTIFACTORY}/sgl-project/FlashMLA/archive/05e26647fe840b8baedae486c2d86d5ce4efeb7c.tar.gz
    URL_HASH SHA256=ce369489bbfc42cdfbba9aa949de0270e64469d530748dea9f4f60b3c69dea9b
)
FetchContent_Populate(repo-flashmla)

option(
    SGL_FLASHMLA_GLM52_FLAT_TOKEN_INDEX
    "Specialize contiguous V3.2 sparse-decode token addressing"
    OFF
)
if(SGL_FLASHMLA_GLM52_FLAT_TOKEN_INDEX)
    set(
        FLASHMLA_GLM52_PATCH
        "${CMAKE_CURRENT_LIST_DIR}/patches/flashmla/glm52_v32_flat_token_index.patch"
    )
    set(
        FLASHMLA_GLM52_KERNEL
        "${repo-flashmla_SOURCE_DIR}/csrc/sm100/decode/head64/kernel.cuh"
    )
    set(
        FLASHMLA_GLM52_CONFIG
        "${repo-flashmla_SOURCE_DIR}/csrc/sm100/decode/head64/config.h"
    )
    file(READ "${FLASHMLA_GLM52_KERNEL}" FLASHMLA_GLM52_KERNEL_CONTENT)
    file(READ "${FLASHMLA_GLM52_CONFIG}" FLASHMLA_GLM52_CONFIG_CONTENT)
    string(
        FIND
        "${FLASHMLA_GLM52_KERNEL_CONTENT}"
        "use_flat_page64_v32"
        FLASHMLA_GLM52_KERNEL_PATCHED
    )
    string(
        FIND
        "${FLASHMLA_GLM52_KERNEL_CONTENT}"
        "KernelTemplate<MODEL_TYPE, true>::run(params)"
        FLASHMLA_GLM52_DISPATCH_PATCHED
    )
    string(
        FIND
        "${FLASHMLA_GLM52_CONFIG_CONTENT}"
        "template<ModelType MODEL_TYPE, bool FLAT_PAGE64_V32 = false>"
        FLASHMLA_GLM52_CONFIG_PATCHED
    )
    if(
        FLASHMLA_GLM52_KERNEL_PATCHED EQUAL -1
        AND FLASHMLA_GLM52_DISPATCH_PATCHED EQUAL -1
        AND FLASHMLA_GLM52_CONFIG_PATCHED EQUAL -1
    )
        execute_process(
            COMMAND patch -p1 --forward --input=${FLASHMLA_GLM52_PATCH}
            WORKING_DIRECTORY "${repo-flashmla_SOURCE_DIR}"
            RESULT_VARIABLE FLASHMLA_GLM52_PATCH_RESULT
            OUTPUT_VARIABLE FLASHMLA_GLM52_PATCH_STDOUT
            ERROR_VARIABLE FLASHMLA_GLM52_PATCH_STDERR
        )
        if(NOT FLASHMLA_GLM52_PATCH_RESULT EQUAL 0)
            message(
                FATAL_ERROR
                "Failed to apply ${FLASHMLA_GLM52_PATCH}:\n"
                "${FLASHMLA_GLM52_PATCH_STDOUT}\n${FLASHMLA_GLM52_PATCH_STDERR}"
            )
        endif()
        file(READ "${FLASHMLA_GLM52_KERNEL}" FLASHMLA_GLM52_KERNEL_CONTENT)
        file(READ "${FLASHMLA_GLM52_CONFIG}" FLASHMLA_GLM52_CONFIG_CONTENT)
        if(
            NOT FLASHMLA_GLM52_KERNEL_CONTENT MATCHES "use_flat_page64_v32"
            OR NOT FLASHMLA_GLM52_KERNEL_CONTENT MATCHES
                "KernelTemplate<MODEL_TYPE, true>::run\\(params\\)"
            OR NOT FLASHMLA_GLM52_CONFIG_CONTENT MATCHES
                "bool FLAT_PAGE64_V32 = false"
        )
            message(FATAL_ERROR "GLM-5.2 FlashMLA patch verification failed")
        endif()
        message(STATUS "Applied GLM-5.2 V3.2 flat-token-index specialization")
    elseif(
        NOT FLASHMLA_GLM52_KERNEL_PATCHED EQUAL -1
        AND NOT FLASHMLA_GLM52_DISPATCH_PATCHED EQUAL -1
        AND NOT FLASHMLA_GLM52_CONFIG_PATCHED EQUAL -1
    )
        message(STATUS "GLM-5.2 V3.2 flat-token-index specialization already applied")
    else()
        message(FATAL_ERROR "Partially patched GLM-5.2 FlashMLA dependency")
    endif()
endif()

# flashmla submodule pin: NVIDIA/cutlass @ 147f5673d0c1c3dcf66f78d677fd647e4a020219
FetchContent_Declare(
    repo-flashmla-cutlass
    URL      https://${GITHUB_ARTIFACTORY}/NVIDIA/cutlass/archive/147f5673d0c1c3dcf66f78d677fd647e4a020219.tar.gz
    URL_HASH SHA256=9f6c53320a85b4a570975e557918cde65168cd311f081920446c238437347dc6
    SOURCE_DIR ${repo-flashmla_SOURCE_DIR}/csrc/cutlass
)
FetchContent_Populate(repo-flashmla-cutlass)

set(FLASHMLA_CUDA_FLAGS
    "--expt-relaxed-constexpr"
    "--expt-extended-lambda"
    "--use_fast_math"

    "-Xcudafe=--diag_suppress=177"   # variable was declared but never referenced
)

set(
    SGL_FLASHMLA_KEEP_DIR
    ""
    CACHE PATH
    "Optional directory for retained FlashMLA CUDA intermediate files"
)
if(SGL_FLASHMLA_KEEP_DIR)
    file(MAKE_DIRECTORY "${SGL_FLASHMLA_KEEP_DIR}")
endif()

set(FLASHMLA_ENABLE_SM100 OFF)
option(
    SGL_FLASHMLA_SM103_ONLY
    "Build FlashMLA device code only for SM103a (campaign diagnostics)"
    OFF
)
if(
    SGL_FLASHMLA_SM103_ONLY
    AND (
        CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "13.0"
        OR CUDA_VERSION VERSION_LESS "13.0"
    )
)
    message(
        FATAL_ERROR
        "SGL_FLASHMLA_SM103_ONLY requires CUDA 13 or newer compiler and CUDA_VERSION"
    )
endif()

# The FlashMLA kernels only work on hopper and require CUDA 12.4 or later.
# Only build FlashMLA kernels if we are building for something compatible with
# sm90a
if(NOT SGL_FLASHMLA_SM103_ONLY AND ${CUDA_VERSION} VERSION_GREATER 12.4)
    list(APPEND FLASHMLA_CUDA_FLAGS
        "-gencode=arch=compute_90a,code=sm_90a"
    )
endif()
if(${CUDA_VERSION} VERSION_GREATER 12.8)
    if(NOT SGL_FLASHMLA_SM103_ONLY)
        list(APPEND FLASHMLA_CUDA_FLAGS
            "-gencode=arch=compute_100a,code=sm_100a"
        )
    endif()
    set(FLASHMLA_ENABLE_SM100 ON)
endif()
if(${CUDA_VERSION} VERSION_GREATER_EQUAL "13.0")
    # Patch FlashMLA sources for SM103a support.
    # These patches are only needed (and only valid) with CUDA 13+.

    # Patch utils.h: widen IS_SM100 to cover the full SM100 family.
    # Newer FlashMLA versions use csrc/utils.h.
    set(FLASHMLA_UTILS_FILE "${repo-flashmla_SOURCE_DIR}/csrc/utils.h")
    file(READ "${FLASHMLA_UTILS_FILE}" FLASHMLA_UTILS_CONTENT)
    string(REPLACE
        "#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
#define IS_SM100 1"
        "#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && (__CUDA_ARCH__ < 1100)
#define IS_SM100 1"
        FLASHMLA_UTILS_CONTENT "${FLASHMLA_UTILS_CONTENT}")
    file(WRITE "${FLASHMLA_UTILS_FILE}" "${FLASHMLA_UTILS_CONTENT}")
    message(STATUS "Patched utils.h for SM103a support")

    # Patch cutlass/arch/config.h: add SM103 architecture defines.
    # The new block is inserted right before the existing "// SM101 and SM101a"
    # anchor in the upstream header.
    set(CUTLASS_CONFIG_FILE "${repo-flashmla_SOURCE_DIR}/csrc/cutlass/include/cutlass/arch/config.h")
    file(READ "${CUTLASS_CONFIG_FILE}" CUTLASS_CONFIG_CONTENT)
    string(FIND "${CUTLASS_CONFIG_CONTENT}" "SM103" SM103_FOUND)
    if(SM103_FOUND EQUAL -1)
        string(REPLACE
"// SM101 and SM101a"
"// SM103 and SM103a
#if !CUTLASS_CLANG_CUDA && (__CUDACC_VER_MAJOR__ >= 13)
  #define CUTLASS_ARCH_MMA_SM103_SUPPORTED 1
  #if (!defined(CUTLASS_ARCH_MMA_SM103_ENABLED) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1030)
    #define CUTLASS_ARCH_MMA_SM103_ENABLED 1
    #if !defined(CUTLASS_ARCH_MMA_SM100A_ENABLED)
      #define CUTLASS_ARCH_MMA_SM100A_ENABLED 1
    #endif
    #if !defined(CUTLASS_ARCH_MMA_SM100F_ENABLED)
      #define CUTLASS_ARCH_MMA_SM100F_ENABLED 1
    #endif
  #endif
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

// SM101 and SM101a"
            CUTLASS_CONFIG_CONTENT "${CUTLASS_CONFIG_CONTENT}")
        file(WRITE "${CUTLASS_CONFIG_FILE}" "${CUTLASS_CONFIG_CONTENT}")
        message(STATUS "Patched cutlass/arch/config.h for SM103a support")
    else()
        message(STATUS "cutlass/arch/config.h already patched for SM103a")
    endif()

    list(APPEND FLASHMLA_CUDA_FLAGS
        "-gencode=arch=compute_103a,code=sm_103a"
    )
endif()


set(FlashMLA_SOURCES
    "csrc/flashmla_extension.cc"

    # Compatibility shim for sgl-kernel torch.ops API.
    ${repo-flashmla_SOURCE_DIR}/csrc/python_api.cpp

    # Decode metadata/combine kernels.
    ${repo-flashmla_SOURCE_DIR}/csrc/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu
    ${repo-flashmla_SOURCE_DIR}/csrc/smxx/decode/combine/combine.cu

    # sm90 dense decode.
    ${repo-flashmla_SOURCE_DIR}/csrc/sm90/decode/dense/instantiations/fp16.cu
    ${repo-flashmla_SOURCE_DIR}/csrc/sm90/decode/dense/instantiations/bf16.cu

    # sm90 sparse decode.
    ${repo-flashmla_SOURCE_DIR}/csrc/sm90/decode/sparse_fp8/instantiations/model1_persistent_h64.cu
    ${repo-flashmla_SOURCE_DIR}/csrc/sm90/decode/sparse_fp8/instantiations/model1_persistent_h128.cu
    ${repo-flashmla_SOURCE_DIR}/csrc/sm90/decode/sparse_fp8/instantiations/v32_persistent_h64.cu
    ${repo-flashmla_SOURCE_DIR}/csrc/sm90/decode/sparse_fp8/instantiations/v32_persistent_h128.cu

    # sm90 sparse prefill.
    ${repo-flashmla_SOURCE_DIR}/csrc/sm90/prefill/sparse/fwd.cu
    ${repo-flashmla_SOURCE_DIR}/csrc/sm90/prefill/sparse/instantiations/phase1_k512.cu
    ${repo-flashmla_SOURCE_DIR}/csrc/sm90/prefill/sparse/instantiations/phase1_k512_topklen.cu
    ${repo-flashmla_SOURCE_DIR}/csrc/sm90/prefill/sparse/instantiations/phase1_k576.cu
    ${repo-flashmla_SOURCE_DIR}/csrc/sm90/prefill/sparse/instantiations/phase1_k576_topklen.cu

    ${repo-flashmla_SOURCE_DIR}/csrc/extension/sm90/dense_fp8/dense_fp8_python_api.cpp
    ${repo-flashmla_SOURCE_DIR}/csrc/extension/sm90/dense_fp8/flash_fwd_mla_fp8_sm90.cu
    ${repo-flashmla_SOURCE_DIR}/csrc/extension/sm90/dense_fp8/flash_fwd_mla_metadata.cu
)

set(
    FLASHMLA_SM100_V32_SOURCE
    "${repo-flashmla_SOURCE_DIR}/csrc/sm100/decode/head64/instantiations/v32.cu"
)
if(FLASHMLA_ENABLE_SM100)
    list(APPEND FlashMLA_SOURCES
        # sm100 dense prefill/bwd.
        ${repo-flashmla_SOURCE_DIR}/csrc/sm100/prefill/dense/fmha_cutlass_fwd_sm100.cu
        ${repo-flashmla_SOURCE_DIR}/csrc/sm100/prefill/dense/fmha_cutlass_bwd_sm100.cu

        # sm100 sparse prefill.
        ${repo-flashmla_SOURCE_DIR}/csrc/sm100/prefill/sparse/fwd/head64/instantiations/phase1_k512.cu
        ${repo-flashmla_SOURCE_DIR}/csrc/sm100/prefill/sparse/fwd/head64/instantiations/phase1_k576.cu
        ${repo-flashmla_SOURCE_DIR}/csrc/sm100/prefill/sparse/fwd/head128/instantiations/phase1_k512.cu
        ${repo-flashmla_SOURCE_DIR}/csrc/sm100/prefill/sparse/fwd/head128/instantiations/phase1_k576.cu
        ${repo-flashmla_SOURCE_DIR}/csrc/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_prefill_k512.cu

        # sm100 sparse decode.
        ${FLASHMLA_SM100_V32_SOURCE}
        ${repo-flashmla_SOURCE_DIR}/csrc/sm100/decode/head64/instantiations/model1.cu
        ${repo-flashmla_SOURCE_DIR}/csrc/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_decode_k512.cu
    )
endif()

Python_add_library(flashmla_ops MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI ${FlashMLA_SOURCES})
target_compile_options(flashmla_ops PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++20>
    $<$<COMPILE_LANGUAGE:CUDA>:-std=c++20>
    $<$<COMPILE_LANGUAGE:CUDA>:${FLASHMLA_CUDA_FLAGS}>
)
if(SGL_FLASHMLA_KEEP_DIR)
    # Retain only the target V3.2 translation unit. nvcc derives retained
    # filenames from source basenames, and the full source list contains
    # duplicate basenames that would otherwise collide in a shared directory.
    set_property(
        SOURCE "${FLASHMLA_SM100_V32_SOURCE}"
        APPEND
        PROPERTY COMPILE_OPTIONS
        "-lineinfo"
        "-Xptxas=-v"
        "--keep"
        "--keep-dir=${SGL_FLASHMLA_KEEP_DIR}"
    )
endif()
if(SGL_FLASHMLA_SM103_ONLY)
    # Manual -gencode flags above are the complete architecture contract.
    set_property(TARGET flashmla_ops PROPERTY CUDA_ARCHITECTURES OFF)
endif()
if(FLASHMLA_ENABLE_SM100)
    target_compile_definitions(flashmla_ops PRIVATE FLASHMLA_ENABLE_SM100)
endif()

# CUDA 13 moved cuda/std/* under cccl/cuda/std/*. The vendored cutlass routes
# <cuda/std/...> to <cccl/cuda/std/...> when __CUDACC_VER_MAJOR__ >= 13, so the
# host C++ TU (compiled by g++, where that macro is unset for the legacy path)
# needs the cccl include root on the search path.
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "13.0")
    find_path(FLASHMLA_CCCL_INCLUDE NAMES cuda/std/utility
        HINTS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
              ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/cccl)
endif()
target_include_directories(flashmla_ops PRIVATE
    ${repo-flashmla_SOURCE_DIR}/csrc
    ${repo-flashmla_SOURCE_DIR}/csrc/kerutils/include
    ${repo-flashmla_SOURCE_DIR}/csrc/sm90
    ${repo-flashmla_SOURCE_DIR}/csrc/extension/sm90/dense_fp8/
    ${repo-flashmla_SOURCE_DIR}/csrc/cutlass/include
    ${repo-flashmla_SOURCE_DIR}/csrc/cutlass/tools/util/include
    ${FLASHMLA_CCCL_INCLUDE}
)

target_link_libraries(flashmla_ops PRIVATE ${TORCH_LIBRARIES} c10 cuda)

install(TARGETS flashmla_ops LIBRARY DESTINATION "sgl_kernel")

target_compile_definitions(flashmla_ops PRIVATE)
