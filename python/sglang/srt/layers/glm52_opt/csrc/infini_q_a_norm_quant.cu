#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace {

constexpr int kHidden = 2048;
constexpr int kRowsPerBlock = 4;
constexpr int kThreadsPerRow = 32;
constexpr int kThreads = kRowsPerBlock * kThreadsPerRow;
constexpr int kVec = 8;
constexpr int kVecBlocks = 8;
constexpr int kColStride = kThreadsPerRow * kVec;
constexpr int kGroup = 128;
constexpr int kLanesPerGroup = kGroup / kVec;
constexpr float kLocalAbsmax = 1e-10f;
constexpr float kFp8Max = 448.0f;

__device__ __forceinline__ float bfly(float value, int mask) {
  float peer;
  asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;"
               : "=f"(peer)
               : "f"(value), "r"(mask));
  return peer;
}

__device__ __forceinline__ float rsqrt_approx_ftz(float value) {
  float result;
  asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
  return result;
}

__device__ __forceinline__ float fast_pow2(int exponent) {
  return __uint_as_float(static_cast<uint32_t>((exponent + 127) << 23));
}

__device__ __forceinline__ int fast_log2_ceil(float value) {
  const uint32_t bits = __float_as_uint(value);
  const int exponent = static_cast<int>((bits >> 23) & 0xff);
  const uint32_t mantissa = bits & ((1u << 23) - 1u);
  return exponent - 127 + (mantissa != 0);
}

__device__ __forceinline__ float2 fmul2(float2 left, float2 right) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  return __fmul2_rn(left, right);
#else
  return make_float2(left.x * right.x, left.y * right.y);
#endif
}

__device__ __forceinline__ void cp_async_16(
    void* shared_destination, const void* global_source) {
  const uint32_t shared_address =
      static_cast<uint32_t>(__cvta_generic_to_shared(shared_destination));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                   "r"(shared_address), "l"(global_source));
}

__global__ __launch_bounds__(kThreads) void q_a_rmsnorm_quant_kernel(
    const __nv_bfloat16* __restrict__ input,
    const int64_t input_row_stride,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ normed,
    __nv_fp8_e4m3* __restrict__ quantized,
    uint32_t* __restrict__ packed_scales,
    const int scale_column_stride,
    const int rows,
    const float epsilon) {
  extern __shared__ __align__(16) char shared_raw[];
  __nv_bfloat16* shared_input =
      reinterpret_cast<__nv_bfloat16*>(shared_raw);

  const int thread = threadIdx.x;
  const int lane_in_row = thread % kThreadsPerRow;
  const int row_in_block = thread / kThreadsPerRow;
  const int lane = thread & 31;
  const int row = blockIdx.x * kRowsPerBlock + row_in_block;
  const bool live = row < rows;
  const int64_t input_base =
      static_cast<int64_t>(row) * input_row_stride + lane_in_row * kVec;
  const int output_base = row * kHidden + lane_in_row * kVec;
  const int shared_base = row_in_block * kHidden + lane_in_row * kVec;

  if (live) {
#pragma unroll
    for (int block = 0; block < kVecBlocks; ++block) {
      cp_async_16(
          shared_input + shared_base + block * kColStride,
          input + input_base + block * kColStride);
    }
  }
  asm volatile("cp.async.commit_group;\n" ::);
  asm volatile("cp.async.wait_group 0;\n" ::);

  float values[kVec * kVecBlocks];
  float sum_square = 0.0f;
  if (live) {
#pragma unroll
    for (int block = 0; block < kVecBlocks; ++block) {
      const int4 loaded = *reinterpret_cast<const int4*>(
          shared_input + shared_base + block * kColStride);
      const __nv_bfloat16* packed =
          reinterpret_cast<const __nv_bfloat16*>(&loaded);
#pragma unroll
      for (int item = 0; item < kVec; ++item) {
        const float value = __bfloat162float(packed[item]);
        values[block * kVec + item] = value;
        sum_square = __fmaf_rn(value, value, sum_square);
      }
    }
  }

#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    sum_square = sum_square + bfly(sum_square, offset);
  }
  if (!live) {
    return;
  }

  const float inverse_rms = rsqrt_approx_ftz(
      sum_square / static_cast<float>(kHidden) + epsilon);
  const int group_in_chunk = lane_in_row / kLanesPerGroup;
  const bool group_leader =
      (lane_in_row % kLanesPerGroup) == 0;

#pragma unroll
  for (int block = 0; block < kVecBlocks; ++block) {
    const int4 weight_vector = *reinterpret_cast<const int4*>(
        weight + lane_in_row * kVec + block * kColStride);
    const __nv_bfloat16* weight_values =
        reinterpret_cast<const __nv_bfloat16*>(&weight_vector);
    __nv_bfloat16 normalized[kVec];
    float absolute_max = kLocalAbsmax;
#pragma unroll
    for (int item = 0; item < kVec; ++item) {
      const __nv_bfloat16 output = __float2bfloat16(
          values[block * kVec + item] * inverse_rms *
          __bfloat162float(weight_values[item]));
      normalized[item] = output;
      absolute_max = fmaxf(
          absolute_max, fabsf(__bfloat162float(output)));
    }
    *reinterpret_cast<int4*>(
        normed + output_base + block * kColStride) =
        *reinterpret_cast<const int4*>(normalized);

#pragma unroll
    for (int offset = 1; offset < kLanesPerGroup; offset <<= 1) {
      absolute_max = fmaxf(absolute_max, bfly(absolute_max, offset));
    }

    const int exponent = fast_log2_ceil(absolute_max * (1.0f / kFp8Max));
    const int logical_group = block * 2 + group_in_chunk;
    if (group_leader) {
      uint8_t* scale_byte =
          reinterpret_cast<uint8_t*>(packed_scales) +
          (static_cast<int64_t>(logical_group / 4) *
               scale_column_stride * 4 +
           static_cast<int64_t>(row) * 4 + logical_group % 4);
      *scale_byte = static_cast<uint8_t>(
          __float_as_uint(fast_pow2(exponent)) >> 23);
    }

    const float output_scale = fast_pow2(-exponent);
    const float2 scales = make_float2(output_scale, output_scale);
    uint2 packed_output;
    __nv_fp8x2_storage_t* pairs =
        reinterpret_cast<__nv_fp8x2_storage_t*>(&packed_output);
#pragma unroll
    for (int item = 0; item < kVec; item += 2) {
      float2 pair = make_float2(
          __bfloat162float(normalized[item]),
          __bfloat162float(normalized[item + 1]));
      pair = fmul2(pair, scales);
      pair.x = fminf(fmaxf(pair.x, -kFp8Max), kFp8Max);
      pair.y = fminf(fmaxf(pair.y, -kFp8Max), kFp8Max);
      pairs[item >> 1] = __nv_cvt_float2_to_fp8x2(
          pair, __NV_SATFINITE, __NV_E4M3);
    }
    *reinterpret_cast<uint2*>(
        quantized + output_base + block * kColStride) = packed_output;
  }
}

}  // namespace

void fused_q_a_rmsnorm_quant_ue8m0(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor normed,
    at::Tensor quantized,
    at::Tensor packed_scales,
    double epsilon) {
  TORCH_CHECK(input.is_cuda() && input.dim() == 2, "input must be 2-D CUDA");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16, "input must be BF16");
  TORCH_CHECK(
      input.size(1) == kHidden && input.stride(1) == 1,
      "input must have logical shape [M,2048] and contiguous columns");
  TORCH_CHECK(
      weight.is_cuda() && weight.is_contiguous() &&
          weight.scalar_type() == at::kBFloat16 && weight.numel() == kHidden,
      "weight must be contiguous CUDA BF16 [2048]");
  TORCH_CHECK(
      normed.is_cuda() && normed.is_contiguous() &&
          normed.scalar_type() == at::kBFloat16 &&
          normed.sizes() == input.sizes(),
      "normed must be contiguous CUDA BF16 [M,2048]");
  TORCH_CHECK(
      quantized.is_cuda() && quantized.is_contiguous() &&
          quantized.scalar_type() == at::kFloat8_e4m3fn &&
          quantized.sizes() == input.sizes(),
      "quantized must be contiguous CUDA FP8 [M,2048]");
  TORCH_CHECK(
      packed_scales.is_cuda() && packed_scales.scalar_type() == at::kInt &&
          packed_scales.size(0) == input.size(0) &&
          packed_scales.size(1) == kHidden / kGroup / 4 &&
          packed_scales.stride(0) == 1,
      "packed scales must use the MN-major int32 UE8M0 ABI");

  const int rows = static_cast<int>(input.size(0));
  const int blocks = (rows + kRowsPerBlock - 1) / kRowsPerBlock;
  constexpr int shared_bytes =
      kRowsPerBlock * kHidden * sizeof(__nv_bfloat16);
  auto stream = at::cuda::getCurrentCUDAStream();
  q_a_rmsnorm_quant_kernel<<<blocks, kThreads, shared_bytes, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
      input.stride(0),
      reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(normed.data_ptr()),
      reinterpret_cast<__nv_fp8_e4m3*>(quantized.data_ptr()),
      reinterpret_cast<uint32_t*>(packed_scales.data_ptr()),
      static_cast<int>(packed_scales.stride(1)),
      rows,
      static_cast<float>(epsilon));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "fused_q_a_rmsnorm_quant_ue8m0",
      &fused_q_a_rmsnorm_quant_ue8m0,
      "fused strided q_a RMSNorm + BF16 passthrough + packed UE8M0 FP8 quant");
}
