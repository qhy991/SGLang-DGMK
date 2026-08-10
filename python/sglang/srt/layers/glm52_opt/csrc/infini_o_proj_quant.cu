#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace {

constexpr int kHidden = 16384;
constexpr int kGroup = 128;
constexpr int kGroupsPerRow = kHidden / kGroup;
constexpr int kThreadsPerGroup = 8;
constexpr int kValuesPerThread = kGroup / kThreadsPerGroup;
constexpr int kGroupsPerBlock = 32;
constexpr float kLocalAbsmax = 1e-10f;
constexpr float kFp8Max = 448.0f;

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

__global__ __launch_bounds__(kGroupsPerBlock * kThreadsPerGroup)
void o_proj_quant_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_fp8_e4m3* __restrict__ quantized,
    uint32_t* __restrict__ packed_scales,
    const int scale_column_stride,
    const int rows) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaGridDependencySynchronize();
#endif

  const int thread = threadIdx.x;
  const int group_in_block = thread / kThreadsPerGroup;
  const int lane_in_group = thread % kThreadsPerGroup;
  const int group = blockIdx.x * kGroupsPerBlock + group_in_block;
  const int row = group / kGroupsPerRow;
  const int group_in_row = group % kGroupsPerRow;
  const int offset = group * kGroup + lane_in_group * kValuesPerThread;

  int4 raw[2];
  if (row < rows) {
    raw[0] = *reinterpret_cast<const int4*>(input + offset);
    raw[1] = *reinterpret_cast<const int4*>(input + offset + 8);
  } else {
    raw[0] = make_int4(0, 0, 0, 0);
    raw[1] = make_int4(0, 0, 0, 0);
  }
  const __nv_bfloat16* values = reinterpret_cast<const __nv_bfloat16*>(raw);

  float absolute_max = kLocalAbsmax;
#pragma unroll
  for (int item = 0; item < kValuesPerThread; ++item) {
    absolute_max = fmaxf(
        absolute_max, fabsf(__bfloat162float(values[item])));
  }
#pragma unroll
  for (int delta = kThreadsPerGroup / 2; delta > 0; delta >>= 1) {
    absolute_max = fmaxf(
        absolute_max,
        __shfl_xor_sync(0xffffffffu, absolute_max, delta, kThreadsPerGroup));
  }

  const int exponent = fast_log2_ceil(absolute_max * (1.0f / kFp8Max));
  const uint32_t exponent_byte =
      __float_as_uint(fast_pow2(exponent)) >> 23;

  // A warp owns four adjacent group-128 reductions.  All lanes participate in
  // the shuffles; lane zero combines their UE8M0 bytes into one aligned int32
  // store matching DeepGEMM's MN-major/TMA-aligned activation-scale ABI.
  const int lane_in_warp = thread & 31;
  const uint32_t e0 = __shfl_sync(0xffffffffu, exponent_byte, 0);
  const uint32_t e1 = __shfl_sync(0xffffffffu, exponent_byte, 8);
  const uint32_t e2 = __shfl_sync(0xffffffffu, exponent_byte, 16);
  const uint32_t e3 = __shfl_sync(0xffffffffu, exponent_byte, 24);
  if (lane_in_warp == 0 && row < rows) {
    const uint32_t packed = e0 | (e1 << 8) | (e2 << 16) | (e3 << 24);
    const int pack = group_in_row / 4;
    packed_scales[pack * scale_column_stride + row] = packed;
  }

  const float output_scale = fast_pow2(-exponent);
  const float2 scales = make_float2(output_scale, output_scale);
  uint4 packed_output;
  auto* pairs = reinterpret_cast<__nv_fp8x2_storage_t*>(&packed_output);
#pragma unroll
  for (int item = 0; item < kValuesPerThread; item += 2) {
    float2 pair = make_float2(
        __bfloat162float(values[item]),
        __bfloat162float(values[item + 1]));
    pair = fmul2(pair, scales);
    pair.x = fminf(fmaxf(pair.x, -kFp8Max), kFp8Max);
    pair.y = fminf(fmaxf(pair.y, -kFp8Max), kFp8Max);
    pairs[item / 2] = __nv_cvt_float2_to_fp8x2(
        pair, __NV_SATFINITE, __NV_E4M3);
  }
  if (row < rows) {
    *reinterpret_cast<uint4*>(quantized + offset) = packed_output;
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

}  // namespace

void o_proj_quant_ue8m0(
    at::Tensor input,
    at::Tensor quantized,
    at::Tensor packed_scales) {
  TORCH_CHECK(
      input.is_cuda() && input.is_contiguous() && input.dim() == 2 &&
          input.scalar_type() == at::kBFloat16 && input.size(1) == kHidden,
      "input must be contiguous CUDA BF16 [M,16384]");
  TORCH_CHECK(
      input.size(0) > 0 && input.size(0) <= 16,
      "o_proj quant supports 1 <= M <= 16");
  TORCH_CHECK(
      quantized.is_cuda() && quantized.is_contiguous() &&
          quantized.scalar_type() == at::kFloat8_e4m3fn &&
          quantized.sizes() == input.sizes(),
      "quantized must be contiguous CUDA FP8 [M,16384]");
  TORCH_CHECK(
      packed_scales.is_cuda() && packed_scales.scalar_type() == at::kInt &&
          packed_scales.size(0) == input.size(0) &&
          packed_scales.size(1) == kGroupsPerRow / 4 &&
          packed_scales.stride(0) == 1,
      "packed scales must use the MN-major int32 UE8M0 ABI");
  TORCH_CHECK(
      input.device() == quantized.device() &&
          input.device() == packed_scales.device(),
      "all tensors must be on one CUDA device");

  auto stream = at::cuda::getCurrentCUDAStream();
  const int rows = static_cast<int>(input.size(0));
  const int total_groups = rows * kGroupsPerRow;
  const int blocks = (total_groups + kGroupsPerBlock - 1) / kGroupsPerBlock;
  const int scale_column_stride = static_cast<int>(packed_scales.stride(1));
  const auto* input_ptr =
      reinterpret_cast<const __nv_bfloat16*>(input.data_ptr());
  auto* quantized_ptr =
      reinterpret_cast<__nv_fp8_e4m3*>(quantized.data_ptr());
  auto* scale_ptr = reinterpret_cast<uint32_t*>(packed_scales.data_ptr());

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(blocks);
  config.blockDim = dim3(kGroupsPerBlock * kThreadsPerGroup);
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute.val.programmaticStreamSerializationAllowed = 1;
  config.attrs = &attribute;
  config.numAttrs = 1;
  cudaLaunchKernelEx(
      &config,
      o_proj_quant_kernel,
      input_ptr,
      quantized_ptr,
      scale_ptr,
      scale_column_stride,
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "o_proj_quant_ue8m0",
      &o_proj_quant_ue8m0,
      "GLM-5.2 o_proj BF16 -> FP8 + packed UE8M0 quant");
}
