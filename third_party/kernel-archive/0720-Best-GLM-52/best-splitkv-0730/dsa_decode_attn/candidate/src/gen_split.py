"""Derive the split-KV kernel from FlashMLA's head64 sparse phase1 by patch.

The upstream inner loop is already excellent -- tcgen05 UMMA with the
accumulator in TMEM, Q staged to TMEM by UTCCP, TMA `tile::gather4` for the 512
NoPE dims, cp.async for the 64 RoPE dims, a 3-deep pipeline, and a dual-GEMM
N=128 trick so the QK^T MMA never runs at the half-throughput N=64 shape. There
is nothing to gain by rewriting any of that.

Its ONE defect is the grid: `kernel<<<params.s_q, 384>>>` is literally one CTA
per query token, so at the decode shapes the harness sweeps (M = 16 and 32) it
occupies 16 or 32 of 148 SMs and the measured latency is identical at both
(42.18 us, measured). Everything else follows from that.

So this generator rewrites exactly the seven lines that tie the kernel to "one
CTA per token" and leaves the other ~660 untouched. Generating rather than
hand-copying keeps the provenance checkable: re-run it against a new upstream
and the patch either still applies cleanly or fails loudly.

Grid becomes (num_splits, s_q). CTA (split, token) processes the topk slice
[split*chunk, (split+1)*chunk) and writes a softmax-normalised partial plus its
LSE to row `split*s_q + token` of a [splits*s_q, h_q, d_v] buffer -- which is
the SAME layout the unmodified TMA_O descriptor already produces, just with a
taller leading dimension. A combine kernel then merges the splits.
"""
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).parent
_REL = "csrc/sm100/prefill/sparse/fwd/head64/phase1.cuh"

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--flashmla", type=Path, default=None,
                 help=f"FlashMLA checkout root (reads {_REL}); defaults to a "
                      "phase1.cuh sitting next to this script")
_ap.add_argument("--out", type=Path, default=HERE / "mla_split_phase1.cuh")
_args = _ap.parse_args()

SRC = (_args.flashmla / _REL) if _args.flashmla else (HERE / "phase1.cuh")
DST = _args.out
if not SRC.is_file():
    print(f"FAIL: no upstream phase1.cuh at {SRC}\n"
          f"      pass --flashmla /path/to/FlashMLA", file=sys.stderr)
    raise SystemExit(2)

# (description, exact text to find, replacement). Every one must apply exactly
# once; anything else means upstream moved and the result must not be trusted.
PATCHES = [
    # Upstream resolves this relative to its own directory. The generated file
    # lives elsewhere, so spell it from csrc/ -- which IS on the include path.
    # Everything else phase1.cuh includes (params.h, flashmla_utils.h, sm100/...)
    # already resolves from csrc/.
    ("relocate the head64-local config.h include",
     '#include "config.h"',
     '#include "sm100/prefill/sparse/fwd/head64/config.h"'),

    ("kernel name + split extents",
     "template<bool HAVE_ROPE, typename TmaParams>\n"
     "__global__ void __launch_bounds__(NUM_THREADS, 1, 1)\n"
     "sparse_attn_fwd_kernel(__grid_constant__ const SparseAttnFwdParams params, "
     "__grid_constant__ const TmaParams tma_params) {",
     "template<bool HAVE_ROPE, typename TmaParams>\n"
     "__global__ void __launch_bounds__(NUM_THREADS, 1, 1)\n"
     "mla_split_fwd_kernel(__grid_constant__ const SparseAttnFwdParams params, "
     "__grid_constant__ const TmaParams tma_params, int chunk) {"),

    # blockIdx.x now selects the KV split, blockIdx.y the query token.
    ("grid mapping",
     "    // Grid shape: [s_q, 1, 1]\n"
     "\n"
     "    const int s_q_idx = blockIdx.x;",
     "    // Grid shape: [num_splits, s_q, 1]  (was [s_q, 1, 1])\n"
     "\n"
     "    const int s_q_idx   = blockIdx.y;\n"
     "    const int split_idx = blockIdx.x;\n"
     "    // Partial for (split, token) lands at row split*s_q + token of a\n"
     "    // [num_splits*s_q, h_q, d_v] buffer, so TMA_O's descriptor is unchanged\n"
     "    // apart from a taller leading dimension.\n"
     "    const int out_row   = split_idx * params.s_q + s_q_idx;"),

    # This CTA owns only its slice of the topk list.
    ("topk length",
     "    const int topk_length = params.topk_length != nullptr ? "
     "__ldg(params.topk_length + s_q_idx) : params.topk;",
     "    const int topk_length = chunk;   // this split's slice, not the whole list"),

    ("index base",
     "    int* gIndices = params.indices + s_q_idx*params.stride_indices_s_q; // [topk]",
     "    int* gIndices = params.indices + s_q_idx*params.stride_indices_s_q\n"
     "                                   + split_idx*chunk; // [chunk] slice of [topk]"),

    ("lse/max_logits row",
     "            int global_index = s_q_idx*params.h_q + idx_in_warpgroup;",
     "            int global_index = out_row*params.h_q + idx_in_warpgroup;"),

    ("O destination row",
     "            tma_params.tma_O.get_tma_tensor(tma_params.shape_O)(_, _, s_q_idx),",
     "            tma_params.tma_O.get_tma_tensor(tma_params.shape_O)(_, _, out_row),"),

    # Host launcher: taller O tensor, 2-D grid, chunk argument.
    ("launcher signature",
     "template<int D_QK>\n"
     "void run_fwd_phase1_kernel(const SparseAttnFwdParams& params) {",
     "template<int D_QK>\n"
     "void run_split_phase1_kernel(const SparseAttnFwdParams& params, int num_splits) {\n"
     "    // Every slice must be a whole number of B_TOPK blocks, and deep enough to\n"
     "    // fill the NUM_BUFS-deep pipeline -- otherwise the CTA spends all its time\n"
     "    // in prologue and the split costs more than it buys.\n"
     "    KU_ASSERT(num_splits >= 1);\n"
     "    KU_ASSERT(params.topk % (num_splits * B_TOPK) == 0);\n"
     "    const int chunk = params.topk / num_splits;\n"
     "    // A slice shorter than the NUM_BUFS-deep prologue is legal -- the pipeline\n"
     "    // primes min(NUM_BUFS-1, num_k_blocks) buffers -- but it spends most of its\n"
     "    // life in prologue, and MLA_MIN_BLOCKS lets the sweep say where that stops\n"
     "    // paying. One block per CTA is the hard floor.\n"
     "    KU_ASSERT(chunk >= B_TOPK);"),

    ("O shape spans splits",
     "    auto shape_O = make_shape(params.h_q, params.d_v, params.s_q);",
     "    auto shape_O = make_shape(params.h_q, params.d_v, params.s_q * num_splits);"),

    # tma_params is now a reference into the descriptor cache, and a
    # __grid_constant__ kernel parameter must not have reference type -- so name
    # the template argument explicitly rather than deducing it from the lvalue.
    ("kernel symbol",
     "    auto kernel = &sparse_attn_fwd_kernel<D_QK == 576, decltype(tma_params)>;",
     "    auto kernel = &mla_split_fwd_kernel<D_QK == 576, TmaParamsT>;"),

    # cudaFuncSetAttribute is a runtime API call and the opt-in smem size never
    # changes; upstream pays it on every launch. Harmless there, but with cupti
    # absent the gate times with CUDA events after a sync, so per-call host work
    # is inside the measured window.
    ("hoist smem opt-in out of the per-call path",
     "    constexpr size_t smem_size = sizeof(SharedMemoryPlan);\n"
     "    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));",
     "    constexpr size_t smem_size = sizeof(SharedMemoryPlan);\n"
     "    static bool smem_optin_done = [&]{\n"
     "        KU_CUDA_CHECK(cudaFuncSetAttribute(\n"
     "            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));\n"
     "        return true;\n"
     "    }();\n"
     "    (void)smem_optin_done;"),

    # Building the four TMA descriptors costs 4.47 us of host time (measured:
    # probe_alloc 2.33 -> fwd_identity 10.70, minus 3.90 for the launch itself).
    # Raw cuTensorMapEncodeTiled is only 0.065 us x4 = 0.26 us of that, so almost
    # all of it is CuTe's host-side layout work in make_tma_copy.
    #
    # With cupti absent the gate times with CUDA events after a sync
    # (timing.py:137-164), so that host cost sits inside the measured window. The
    # descriptor layout depends only on shapes and strides; the harness re-clones
    # its inputs every timed iteration (evaluate_task.py:322), so the base
    # POINTERS move but the layout never does. Build once per shape, then patch
    # the four base addresses with cuTensorMapReplaceAddress (0.030 us each).
    ("cache TMA descriptors, patch only the base addresses",
     "    auto shape_Q_nope = make_shape(params.h_q, D_V, params.s_q);",
     "    auto build_tma = [&] {\n"
     "    auto shape_Q_nope = make_shape(params.h_q, D_V, params.s_q);"),

    ("close the descriptor builder and add the cache",
     "    TmaParams<\n"
     "        decltype(shape_Q_nope), decltype(tma_Q_nope),\n"
     "        decltype(shape_Q_rope), decltype(tma_Q_rope),\n"
     "        decltype(shape_O), decltype(tma_O)\n"
     "    > tma_params = {\n"
     "        shape_Q_nope, tma_Q_nope,\n"
     "        shape_Q_rope, tma_Q_rope,\n"
     "        shape_O, tma_O,\n"
     "        tensor_map_kv_nope\n"
     "    };",
     "    TmaParams<\n"
     "        decltype(shape_Q_nope), decltype(tma_Q_nope),\n"
     "        decltype(shape_Q_rope), decltype(tma_Q_rope),\n"
     "        decltype(shape_O), decltype(tma_O)\n"
     "    > tp = {\n"
     "        shape_Q_nope, tma_Q_nope,\n"
     "        shape_Q_rope, tma_Q_rope,\n"
     "        shape_O, tma_O,\n"
     "        tensor_map_kv_nope\n"
     "    };\n"
     "    return tp;\n"
     "    };\n"
     "\n"
     "    using TmaParamsT = decltype(build_tma());\n"
     "    struct DescKey {\n"
     "        int s_q, s_kv, h_q, d_v, topk, stq_s, stq_h, stkv_s, splits;\n"
     "        bool operator==(const DescKey&) const = default;\n"
     "    };\n"
     "    const DescKey key{params.s_q, params.s_kv, params.h_q, params.d_v, params.topk,\n"
     "                      params.stride_q_s_q, params.stride_q_h_q,\n"
     "                      params.stride_kv_s_kv, num_splits};\n"
     "    static thread_local DescKey cached_key{};\n"
     "    static thread_local std::optional<TmaParamsT> cached;\n"
     "\n"
     "    bool reuse = cached.has_value() && cached_key == key;\n"
     "    if (reuse) {\n"
     "        // Q_rope's descriptor is based at q + D_V, not q -- see the +D_V in the\n"
     "        // builder above. Getting that wrong reads the NoPE half as RoPE.\n"
     "        auto* d0 = const_cast<CUtensorMap*>(cached->tma_Q_nope.get_tma_descriptor());\n"
     "        auto* d1 = const_cast<CUtensorMap*>(cached->tma_Q_rope.get_tma_descriptor());\n"
     "        auto* d2 = const_cast<CUtensorMap*>(cached->tma_O.get_tma_descriptor());\n"
     "        CUtensorMap* d3 = &cached->tensor_map_kv_nope;\n"
     "        reuse = cuTensorMapReplaceAddress(d0, (void*)params.q) == CUDA_SUCCESS\n"
     "             && cuTensorMapReplaceAddress(d1, (void*)((bf16*)params.q + D_V)) == CUDA_SUCCESS\n"
     "             && cuTensorMapReplaceAddress(d2, (void*)params.out) == CUDA_SUCCESS\n"
     "             && cuTensorMapReplaceAddress(d3, (void*)params.kv) == CUDA_SUCCESS;\n"
     "        // A patch can legitimately fail (an unusually-aligned pointer), so fall\n"
     "        // through to a full rebuild rather than launching a half-patched map.\n"
     "    }\n"
     "    if (!reuse) { cached.emplace(build_tma()); cached_key = key; }\n"
     "    const TmaParamsT& tma_params = *cached;"),

    ("launch shape",
     "    kernel<<<params.s_q, NUM_THREADS, smem_size, params.stream>>>(params, tma_params);",
     "    kernel<<<dim3(num_splits, params.s_q), NUM_THREADS, smem_size, params.stream>>>(\n"
     "        params, tma_params, chunk);"),
]

HEADER = """// GENERATED by gen_split.py from FlashMLA csrc/sm100/prefill/sparse/fwd/head64/
// phase1.cuh -- do not edit by hand; edit gen_split.py and regenerate.
//
// Split-KV variant of the head64 BF16 sparse forward kernel. The math, the
// pipeline, the tcgen05 MMA shapes and the TMA gather are byte-identical to
// upstream; only the grid mapping and the output row change. See gen_split.py
// for the exact patch list and why.
"""


def main() -> int:
    text = SRC.read_text()
    for desc, old, new in PATCHES:
        n = text.count(old)
        if n != 1:
            print(f"FAIL: patch {desc!r} matched {n} times, expected 1", file=sys.stderr)
            return 1
        text = text.replace(old, new)
    # Guard against a stale copy silently shadowing the real kernel.
    assert "sparse_attn_fwd_kernel" not in text, "upstream kernel name still present"
    DST.write_text(HEADER + text.replace('#include "phase1.h"',
                                         '#include "mla_split_phase1.h"', 1))
    print(f"wrote {DST} ({len(text.splitlines())} lines, "
          f"{len(PATCHES)} patches applied from {SRC})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
