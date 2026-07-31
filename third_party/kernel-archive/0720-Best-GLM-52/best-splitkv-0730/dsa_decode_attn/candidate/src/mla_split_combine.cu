// Flash-decoding combine for the split-KV MLA forward.
//
// Shapes are tiny and completely fixed: rows = s_q*h_q (1024 at M=16), d_v=512,
// num_splits <= 32. The whole job is 1024 rows x 512 bf16 = 1 MB of output, so
// the only thing that matters is that every load is a full 16 B and that the
// splits for one row are read back-to-back while they are still in L2 -- they
// were written microseconds earlier by the split kernel.
//
// One warp owns one (token, head) row: 512 dims / 32 lanes = 16 bf16 = exactly
// two uint4 per lane per split. No shared memory, no cross-lane reduction on
// the data path; the only broadcast is the per-row softmax weight, which every
// lane recomputes from the same <= 32 LSE values rather than paying a shuffle.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cutlass/bfloat16.h>

#include <kerutils/kerutils.cuh>   // KU_ASSERT / KU_CHECK_KERNEL_LAUNCH

#include "mla_split_phase1.h"

namespace sm100::fwd::head64 {

namespace {

using bf16_t = cutlass::bfloat16_t;

constexpr int MAX_SPLITS  = 32;
constexpr int WARPS_PER_CTA = 8;
constexpr int DV          = 512;
constexpr int ELEMS_PER_LANE = DV / 32;             // 16 bf16 = 32 B = 2 x uint4

__global__ __launch_bounds__(WARPS_PER_CTA * 32)
void combine_kernel(const __nv_bfloat16* __restrict__ o_acc,
                    const float* __restrict__ lse_acc,
                    __nv_bfloat16* __restrict__ out,
                    int rows, int num_splits, long stride_split) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row  = blockIdx.x * WARPS_PER_CTA + warp;
    if (row >= rows) return;

    // ---- weights: softmax over the splits' LSEs -----------------------------
    // +inf marks "this split saw no valid index" (phase1.cuh:264), which must be
    // weightless. Mapping it to -inf before the max does that and also keeps a
    // fully-empty row from producing NaN.
    float lse[MAX_SPLITS];
    float m = -INFINITY;
    for (int s = 0; s < num_splits; ++s) {
        float v = lse_acc[(long)s * rows + row];
        v = (v == INFINITY) ? -INFINITY : v;
        lse[s] = v;
        m = fmaxf(m, v);
    }
    if (m == -INFINITY) m = 0.f;                    // every split empty
    float denom = 0.f;
    for (int s = 0; s < num_splits; ++s) {
        lse[s] = __expf(lse[s] - m);                // -inf -> 0, so it drops out
        denom += lse[s];
    }
    const float inv = (denom > 0.f) ? __frcp_rn(denom) : 0.f;

    // ---- weighted sum over the splits ---------------------------------------
    float acc[ELEMS_PER_LANE];
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_LANE; ++i) acc[i] = 0.f;

    const long off = (long)row * DV + lane * ELEMS_PER_LANE;
    for (int s = 0; s < num_splits; ++s) {
        const float w = lse[s] * inv;
        if (w == 0.f) continue;                     // empty split: skip the loads
        const __nv_bfloat16* src = o_acc + (long)s * stride_split + off;
        uint4 raw[2];
        raw[0] = *reinterpret_cast<const uint4*>(src);
        raw[1] = *reinterpret_cast<const uint4*>(src + 8);
        const __nv_bfloat16* v = reinterpret_cast<const __nv_bfloat16*>(raw);
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_LANE; ++i) acc[i] = fmaf(__bfloat162float(v[i]), w, acc[i]);
    }

    __nv_bfloat16 res[ELEMS_PER_LANE];
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_LANE; ++i) res[i] = __float2bfloat16(acc[i]);
    uint4* dst = reinterpret_cast<uint4*>(out + off);
    dst[0] = reinterpret_cast<const uint4*>(res)[0];
    dst[1] = reinterpret_cast<const uint4*>(res)[1];
}

}  // namespace

void run_combine(const bf16_t* o_acc, const float* lse_acc, bf16_t* out,
                 int s_q, int h_q, int d_v, int num_splits, cudaStream_t stream) {
    KU_ASSERT(d_v == DV);
    KU_ASSERT(num_splits >= 1 && num_splits <= MAX_SPLITS);
    const int rows = s_q * h_q;
    const int ctas = (rows + WARPS_PER_CTA - 1) / WARPS_PER_CTA;
    combine_kernel<<<ctas, WARPS_PER_CTA * 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(o_acc), lse_acc,
        reinterpret_cast<__nv_bfloat16*>(out), rows, num_splits, (long)rows * d_v);
    KU_CHECK_KERNEL_LAUNCH();
}

__global__ void empty_kernel() {}

void launch_empty(cudaStream_t stream) { empty_kernel<<<1, 32, 0, stream>>>(); }

}  // namespace sm100::fwd::head64
