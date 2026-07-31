// pybind surface for the split-KV MLA forward.
//
// Two entry points, deliberately:
//   fwd_identity  -- the UNMODIFIED upstream kernel through this same build.
//                    Without it, a correctness failure is ambiguous between
//                    "the split is wrong" and "this build's toolchain differs
//                    from the installed sgl_kernel". With it, that is one test.
//   fwd_split     -- the split-KV kernel plus the combine.
//
// Only `out` is returned. The harness compares element [0] of the reference
// tuple and ignores max_logits/lse (glm52_ops.py:441-449), and sgl_kernel's own
// wrapper spends two extra kernel launches rescaling them by log2(e).

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cutlass/bfloat16.h>

#include "params.h"
#include "sm100/prefill/sparse/fwd/head64/phase1.h"
#include "mla_split_phase1.h"

using bf16 = cutlass::bfloat16_t;

namespace {

constexpr float kLog2e = 1.44269504f;

void check_inputs(const at::Tensor &q, const at::Tensor &kv,
                  const at::Tensor &indices, int64_t d_v) {
    TORCH_CHECK(q.dim() == 3 && kv.dim() == 3 && indices.dim() == 3);
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16, "q must be bf16");
    TORCH_CHECK(kv.scalar_type() == torch::kBFloat16, "kv must be bf16");
    TORCH_CHECK(indices.scalar_type() == torch::kInt32, "indices must be int32");
    TORCH_CHECK(q.size(2) == 576, "only d_qk=576 is built here");
    TORCH_CHECK(d_v == 512, "only d_v=512");
    TORCH_CHECK(q.size(1) == 64, "only h_q=64 (head64 kernel)");
    TORCH_CHECK(q.stride(2) == 1 && kv.stride(2) == 1 && indices.stride(2) == 1);
}

SparseAttnFwdParams make_params(const at::Tensor &q, const at::Tensor &kv,
                                const at::Tensor &indices, float sm_scale, int d_v,
                                bf16 *out, float *max_logits, float *lse) {
    SparseAttnFwdParams p = {};
    p.s_q = (int)q.size(0);
    p.s_kv = (int)kv.size(0);
    p.h_q = (int)q.size(1);
    p.h_kv = (int)kv.size(1);
    p.d_qk = (int)q.size(2);
    p.d_v = d_v;
    p.topk = (int)indices.size(2);
    p.sm_scale = sm_scale;
    p.sm_scale_div_log2 = sm_scale * kLog2e;
    p.q = (bf16 *)q.data_ptr();
    p.kv = (bf16 *)kv.data_ptr();
    p.indices = (int *)indices.data_ptr();
    p.attn_sink = nullptr;
    p.topk_length = nullptr;
    p.stride_q_s_q = (int)q.stride(0);
    p.stride_q_h_q = (int)q.stride(1);
    p.stride_kv_s_kv = (int)kv.stride(0);
    p.stride_kv_h_kv = (int)kv.stride(1);
    p.stride_indices_s_q = (int)indices.stride(0);
    p.stride_indices_h_kv = (int)indices.stride(1);
    p.out = out;
    p.max_logits = max_logits;
    p.lse = lse;
    p.num_sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    p.stream = at::cuda::getCurrentCUDAStream().stream();
    return p;
}

at::Tensor fwd_identity(const at::Tensor &q, const at::Tensor &kv,
                        const at::Tensor &indices, double sm_scale, int64_t d_v) {
    check_inputs(q, kv, indices, d_v);
    at::cuda::CUDAGuard guard{(char)q.get_device()};
    auto opts = q.options();
    at::Tensor out = torch::empty({q.size(0), q.size(1), d_v}, opts);
    at::Tensor lse = torch::empty({q.size(0), q.size(1)}, opts.dtype(torch::kFloat));
    at::Tensor ml = torch::empty({q.size(0), q.size(1)}, opts.dtype(torch::kFloat));
    auto p = make_params(q, kv, indices, (float)sm_scale, (int)d_v,
                         (bf16 *)out.data_ptr(), (float *)ml.data_ptr(),
                         (float *)lse.data_ptr());
    sm100::fwd::head64::run_fwd_phase1_kernel<576>(p);
    return out;
}

// o_acc: [splits, s_q, h_q, d_v] bf16   l_acc: [splits, s_q, h_q] fp32
//
// Both are caller-owned workspace, cached across calls exactly like the
// o_accum / lse_accum / tile_scheduler_metadata buffers SGLang already
// preallocates for FlashMLA's own split-KV decode path. `out` is allocated per
// call so the comparison against the reference -- which allocates its outputs
// inside its own interface every call -- stays apples-to-apples.
at::Tensor fwd_split(const at::Tensor &q, const at::Tensor &kv,
                     const at::Tensor &indices, double sm_scale, int64_t d_v,
                     at::Tensor &o_acc, at::Tensor &l_acc, int64_t splits) {
    check_inputs(q, kv, indices, d_v);
    TORCH_CHECK(o_acc.scalar_type() == torch::kBFloat16, "o_acc must be bf16");
    TORCH_CHECK(l_acc.scalar_type() == torch::kFloat, "l_acc must be fp32");
    TORCH_CHECK(o_acc.is_contiguous() && l_acc.is_contiguous());
    TORCH_CHECK(o_acc.size(0) >= splits && l_acc.size(0) >= splits,
                "workspace must have at least `splits` planes");
    at::cuda::CUDAGuard guard{(char)q.get_device()};
    at::Tensor out = torch::empty({q.size(0), q.size(1), d_v}, q.options());

    // max_logits is written by the kernel but nothing downstream reads it, so it
    // shares l_acc's shape in a throwaway buffer rather than costing an alloc.
    static thread_local at::Tensor ml_scratch;
    if (!ml_scratch.defined() || ml_scratch.numel() < l_acc.numel() ||
        ml_scratch.device() != l_acc.device()) {
        ml_scratch = torch::empty_like(l_acc);
    }

    auto p = make_params(q, kv, indices, (float)sm_scale, (int)d_v,
                         (bf16 *)o_acc.data_ptr(), (float *)ml_scratch.data_ptr(),
                         (float *)l_acc.data_ptr());
    sm100::fwd::head64::run_split_phase1_kernel<576>(p, (int)splits);
    sm100::fwd::head64::run_combine((const bf16 *)o_acc.data_ptr(),
                                    (const float *)l_acc.data_ptr(),
                                    (bf16 *)out.data_ptr(), p.s_q, p.h_q, p.d_v,
                                    (int)splits, p.stream);
    return out;
}

// Host-cost attribution. With cupti absent the gate times with CUDA events
// after a sync, so everything the CPU does before the first kernel reaches the
// GPU is inside the measured window -- 10.3 us of a 28.6 us call at M=16. These
// stubs peel that apart layer by layer, so the optimisation target is chosen on
// a measurement rather than a guess. Each does strictly more than the previous.
at::Tensor probe_pybind(const at::Tensor &q, const at::Tensor &kv,
                        const at::Tensor &indices, double, int64_t) {
    return q;                                    // arg marshalling only
}

at::Tensor probe_checks(const at::Tensor &q, const at::Tensor &kv,
                        const at::Tensor &indices, double, int64_t d_v) {
    check_inputs(q, kv, indices, d_v);
    at::cuda::CUDAGuard guard{(char)q.get_device()};
    return q;                                    // + TORCH_CHECKs + device guard
}

at::Tensor probe_alloc(const at::Tensor &q, const at::Tensor &kv,
                       const at::Tensor &indices, double, int64_t d_v) {
    check_inputs(q, kv, indices, d_v);
    at::cuda::CUDAGuard guard{(char)q.get_device()};
    return torch::empty({q.size(0), q.size(1), d_v}, q.options());   // + the output alloc
}

// probe_alloc -> fwd_identity is a 7.5 us jump covering two things: building
// four TMA descriptors and getting one kernel launched. Raw
// cuTensorMapEncodeTiled measures 0.065 us, so if the descriptors were the cost
// the jump would be ~0.3 us, not 7.5. Timing a bare launch settles which it is.
at::Tensor probe_launch(const at::Tensor &q, const at::Tensor &kv,
                        const at::Tensor &indices, double, int64_t d_v) {
    check_inputs(q, kv, indices, d_v);
    at::cuda::CUDAGuard guard{(char)q.get_device()};
    at::Tensor out = torch::empty({q.size(0), q.size(1), d_v}, q.options());
    sm100::fwd::head64::launch_empty(at::cuda::getCurrentCUDAStream().stream());
    return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd_identity", &fwd_identity, "unmodified upstream head64 sparse fwd");
    m.def("fwd_split", &fwd_split, "split-KV head64 sparse fwd + combine");
    m.def("probe_pybind", &probe_pybind, "host-cost probe: arg marshalling only");
    m.def("probe_checks", &probe_checks, "host-cost probe: + checks + CUDAGuard");
    m.def("probe_alloc", &probe_alloc, "host-cost probe: + output allocation");
    m.def("probe_launch", &probe_launch, "host-cost probe: + one trivial kernel launch");
}
