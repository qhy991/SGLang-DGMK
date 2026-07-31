#pragma once

// Host-visible declarations only. This header is included by api_split.cpp,
// which is compiled by the host compiler, so it must NOT pull in kerutils /
// CuTe device headers (they use __cvta_generic_to_shared and friends). That is
// the same discipline upstream's phase1.h follows.

#include <cuda.h>
#include <cuda_runtime.h>

#include <optional>
#include <cutlass/bfloat16.h>

#include "params.h"

namespace sm100::fwd::head64 {

// Split-KV forward. Grid (num_splits, s_q); CTA (split, token) covers the topk
// slice [split*chunk, (split+1)*chunk) and writes its softmax-normalised partial
// to row split*s_q + token of params.out, with the matching LSE at the same row
// of params.lse. Requires topk % (num_splits*B_TOPK) == 0 and a slice of at
// least NUM_BUFS blocks.
template<int D_QK>
void run_split_phase1_kernel(const SparseAttnFwdParams& params, int num_splits);

// Merge the partials: out[t,h,:] = sum_s softmax(lse[:,t,h])_s * o_acc[s,t,h,:].
//
// Each partial is already divided by its own l_s, so exp(lse_s) * o_s is that
// split's unnormalised numerator and a softmax over lse is the exact weight.
// A split whose slice held no valid index reports lse = +inf (phase1.cuh:264);
// it must be treated as weightless, not as the maximum.
void run_combine(const cutlass::bfloat16_t* o_acc, const float* lse_acc,
                 cutlass::bfloat16_t* out, int s_q, int h_q, int d_v,
                 int num_splits, cudaStream_t stream);

// Host-cost attribution helper (see api_split.cpp). Not part of the kernel path.
void launch_empty(cudaStream_t stream);

}  // namespace sm100::fwd::head64
