#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cfloat>
#include <cstdint>

namespace {

constexpr int kGlm52RouterExperts = 256;
constexpr int kGlm52RouterTopK = 8;
constexpr int kGlm52RouterWarpSize = 32;
constexpr uint32_t kFullWarpMask = 0xffffffffu;

__device__ __forceinline__ float glm52_add_f32(float lhs, float rhs) {
  float result;
  asm("add.f32 %0, %1, %2;" : "=f"(result) : "f"(lhs), "f"(rhs));
  return result;
}

__device__ __forceinline__ float glm52_max_f32(float lhs, float rhs) {
  float result;
  asm("max.f32 %0, %1, %2;" : "=f"(result) : "f"(lhs), "f"(rhs));
  return result;
}

__device__ __forceinline__ float glm52_warp_max_f32(float value) {
  // Triton's sm_100a PTX can use redux.sync.max.f32, but the TVM-FFI NVCC
  // path deliberately builds portable sm_100. NaNs have already been mapped
  // away, so a butterfly of the same PTX max.f32 operation is equivalent.
#pragma unroll
  for (int32_t delta = 16; delta > 0; delta >>= 1) {
    value = glm52_max_f32(
        value, __shfl_xor_sync(kFullWarpMask, value, delta));
  }
  return value;
}

__device__ __forceinline__ int32_t glm52_warp_min_s32(int32_t value) {
  int32_t result;
  asm("redux.sync.min.s32 %0, %1, 0xffffffff;" : "=r"(result) : "r"(value));
  return result;
}

__device__ __forceinline__ float glm52_div_full_f32(float numerator, float denominator) {
  float result;
  asm("div.full.f32 %0, %1, %2;" : "=f"(result) : "f"(numerator), "f"(denominator));
  return result;
}

// Match Triton's tl.sigmoid lowering for the stock SM100 router exactly:
//   sub.f32 -> mul.f32(log2(e)) -> ex2.approx.f32 -> add.f32 -> div.full.f32.
__device__ __forceinline__ float glm52_stock_sigmoid(float value) {
  float negated;
  float exponent;
  float denominator;
  float result;
  asm("sub.f32 %0, 0f00000000, %1;" : "=f"(negated) : "f"(value));
  asm("mul.f32 %0, %1, 0f3FB8AA3B;" : "=f"(exponent) : "f"(negated));
  asm("ex2.approx.f32 %0, %1;" : "=f"(exponent) : "f"(exponent));
  asm("add.f32 %0, %1, 0f3F800000;" : "=f"(denominator) : "f"(exponent));
  asm("div.full.f32 %0, 0f3F800000, %1;" : "=f"(result) : "f"(denominator));
  return result;
}

template <int kWarpsPerCTA, bool kUsePDL>
__launch_bounds__(kWarpsPerCTA * kGlm52RouterWarpSize) __global__
    void glm52_router_sigmoid_topk_kernel(
        const float* __restrict__ scores,
        const float* __restrict__ bias,
        float* __restrict__ output,
        int32_t* __restrict__ indices,
        int32_t num_rows) {
  static_assert(kWarpsPerCTA == 4 || kWarpsPerCTA == 8);

  const int32_t lane = static_cast<int32_t>(threadIdx.x);
  const int32_t warp = static_cast<int32_t>(threadIdx.y);
  const int32_t row = static_cast<int32_t>(blockIdx.x) * kWarpsPerCTA + warp;

  // The exact Task 32 shapes are divisible by both candidate CTA row tiles.
  if (row >= num_rows) return;

  const int32_t first_expert = lane * 4;
  const auto bias_lo = *reinterpret_cast<const float4*>(bias + first_expert);
  const auto bias_hi = *reinterpret_cast<const float4*>(bias + 128 + first_expert);

  device::PDLWaitPrimary<kUsePDL>();

  const float* row_scores = scores + row * kGlm52RouterExperts;
  const auto score_lo = *reinterpret_cast<const float4*>(row_scores + first_expert);
  const auto score_hi = *reinterpret_cast<const float4*>(row_scores + 128 + first_expert);

  float activated[kGlm52RouterTopK];
  float ranked[kGlm52RouterTopK];
  const float score_values[kGlm52RouterTopK] = {
      score_lo.x, score_lo.y, score_lo.z, score_lo.w, score_hi.x, score_hi.y, score_hi.z, score_hi.w};
  const float bias_values[kGlm52RouterTopK] = {
      bias_lo.x, bias_lo.y, bias_lo.z, bias_lo.w, bias_hi.x, bias_hi.y, bias_hi.z, bias_hi.w};

#pragma unroll
  for (int32_t local = 0; local < kGlm52RouterTopK; ++local) {
    const float sigmoid = glm52_stock_sigmoid(score_values[local]);
    const float biased = glm52_add_f32(sigmoid, bias_values[local]);
    activated[local] = sigmoid;
    // Stock maps a NaN ranking value to the exact FP32 representation of -1e30.
    ranked[local] = (biased == biased) ? biased : __int_as_float(0xf149f2ca);
  }

  float selected_weights[kGlm52RouterTopK];
  int32_t selected_indices[kGlm52RouterTopK];

#pragma unroll
  for (int32_t k = 0; k < kGlm52RouterTopK; ++k) {
    float local_max = ranked[0];
#pragma unroll
    for (int32_t local = 1; local < kGlm52RouterTopK; ++local) {
      local_max = glm52_max_f32(local_max, ranked[local]);
    }
    const float row_max = glm52_warp_max_f32(local_max);

    int32_t local_winner = kGlm52RouterExperts + 1;
#pragma unroll
    for (int32_t local = 0; local < kGlm52RouterTopK; ++local) {
      const int32_t expert = first_expert + (local < 4 ? local : 128 + local - 4);
      if (ranked[local] == row_max && expert < local_winner) {
        local_winner = expert;
      }
    }
    const int32_t winner = glm52_warp_min_s32(local_winner);
    selected_indices[k] = winner;

    const int32_t owner_lane = (winner & 127) >> 2;
    const int32_t owner_local = (winner & 3) + ((winner >= 128) ? 4 : 0);
    float winner_weight = lane == owner_lane ? activated[owner_local] : 0.0f;
    winner_weight = __shfl_sync(kFullWarpMask, winner_weight, owner_lane);
    selected_weights[k] = winner_weight;

    if (lane == owner_lane) {
      ranked[owner_local] = __int_as_float(0xff800000);
    }
  }

  // Match the stock BLOCK_K=8 Triton reduction layout: four serial adds in
  // each logical half, followed by the final half-to-half add.
  float routed_lo = glm52_add_f32(selected_weights[0], selected_weights[1]);
  routed_lo = glm52_add_f32(routed_lo, selected_weights[2]);
  routed_lo = glm52_add_f32(routed_lo, selected_weights[3]);
  float routed_hi = glm52_add_f32(selected_weights[4], selected_weights[5]);
  routed_hi = glm52_add_f32(routed_hi, selected_weights[6]);
  routed_hi = glm52_add_f32(routed_hi, selected_weights[7]);
  const float routed_sum = glm52_add_f32(routed_lo, routed_hi);

  device::PDLTriggerSecondary<kUsePDL>();

  const float norm = routed_sum > 0.0f ? routed_sum : 1.0f;
  if (lane < kGlm52RouterTopK) {
    const int32_t offset = row * kGlm52RouterTopK + lane;
    output[offset] = glm52_div_full_f32(selected_weights[lane], norm);
    indices[offset] = selected_indices[lane];
  }
}

template <int kWarpsPerCTA, bool kUsePDL>
struct Glm52RouterSigmoidTopKKernel {
  static void run(
      const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView bias,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView indices) {
    using namespace host;

    auto M = SymbolicSize{"num_rows"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({M, kGlm52RouterExperts}).with_dtype<float>().with_device(device).verify(scores);
    TensorMatcher({kGlm52RouterExperts}).with_dtype<float>().with_device(device).verify(bias);
    TensorMatcher({M, kGlm52RouterTopK}).with_dtype<float>().with_device(device).verify(output);
    TensorMatcher({M, kGlm52RouterTopK}).with_dtype<int32_t>().with_device(device).verify(indices);

    const int64_t num_rows = M.unwrap();
    RuntimeCheck(num_rows == 16 || num_rows == 32, "GLM-5.2 router CUDA candidate requires M=16 or M=32");
    RuntimeCheck(getSMVersion(device.unwrap().device_id) == 100, "GLM-5.2 router CUDA candidate requires SM100");
    RuntimeCheck(
        reinterpret_cast<uintptr_t>(scores.data_ptr()) % alignof(float4) == 0,
        "GLM-5.2 router scores must be 16-byte aligned");
    RuntimeCheck(
        reinterpret_cast<uintptr_t>(bias.data_ptr()) % alignof(float4) == 0,
        "GLM-5.2 router bias must be 16-byte aligned");

    const dim3 block(kGlm52RouterWarpSize, kWarpsPerCTA);
    const dim3 grid(div_ceil(num_rows, static_cast<int64_t>(kWarpsPerCTA)));
    LaunchKernel(grid, block, device.unwrap())
        .enable_pdl(kUsePDL)(
            glm52_router_sigmoid_topk_kernel<kWarpsPerCTA, kUsePDL>,
            static_cast<const float*>(scores.data_ptr()),
            static_cast<const float*>(bias.data_ptr()),
            static_cast<float*>(output.data_ptr()),
            static_cast<int32_t*>(indices.data_ptr()),
            static_cast<int32_t>(num_rows));
  }
};

}  // namespace
