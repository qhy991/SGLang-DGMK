#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace {

constexpr int kExperts = 32;
constexpr int kRows = 35200;
constexpr int kGateUp = 4096;
constexpr int kHidden = 2048;
constexpr int kGroup = 128;
constexpr int kGroupsPerRow = kHidden / kGroup;
constexpr int kSubwarp = 8;
constexpr int kValuesPerLane = kGroup / kSubwarp;
constexpr int kThreads = kGroupsPerRow * kSubwarp;
constexpr int kSubwarp16 = 16;
constexpr int kValuesPerLane16 = kGroup / kSubwarp16;
constexpr int kThreads256 = kGroupsPerRow * kSubwarp16;
constexpr float kAbsmaxFloor = 1.0e-10f;
constexpr float kFp8Max = 448.0f;

static_assert(kGroupsPerRow == 16);
static_assert(kValuesPerLane == 16);
static_assert(kThreads == 128);
static_assert(kValuesPerLane16 == 8);
static_assert(kThreads256 == 256);

__device__ __forceinline__ int4 load_global_nc(const int4* ptr) {
  int4 value;
  asm volatile(
      "ld.global.nc.v4.s32 {%0, %1, %2, %3}, [%4];"
      : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
      : "l"(ptr));
  return value;
}

__device__ __forceinline__ void store_global(int4* ptr, const int4& value) {
  asm volatile(
      "st.global.v4.s32 [%0], {%1, %2, %3, %4};"
      :
      : "l"(ptr), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
}

__device__ __forceinline__ void store_global(uint2* ptr, const uint2& value) {
  asm volatile(
      "st.global.v2.s32 [%0], {%1, %2};"
      :
      : "l"(ptr), "r"(value.x), "r"(value.y));
}

__device__ __forceinline__ float subwarp_reduce_max(float value) {
  value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 4));
  value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 2));
  value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 1));
  return value;
}

__device__ __forceinline__ float subwarp16_reduce_max(float value) {
  value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 8));
  value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 4));
  value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 2));
  value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 1));
  return value;
}

__device__ __forceinline__ float fast_pow2(int exponent) {
  const uint32_t bits = static_cast<uint32_t>(exponent + 127) << 23;
  return __uint_as_float(bits);
}

__device__ __forceinline__ int fast_log2_ceil(float value) {
  const uint32_t bits = __float_as_uint(value);
  const int exponent = static_cast<int>((bits >> 23) & 0xffu);
  const uint32_t mantissa = bits & 0x7fffffu;
  return exponent - 127 + (mantissa != 0);
}

__device__ __forceinline__ float2 multiply_rn(float2 lhs, float2 rhs) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  return __fmul2_rn(lhs, rhs);
#else
  return make_float2(lhs.x * rhs.x, lhs.y * rhs.y);
#endif
}

__global__ void swiglu_quant_prefill_s8_v16_b128_kernel(
    const __nv_bfloat16* __restrict__ gateup,
    uint8_t* __restrict__ output,
    int32_t* __restrict__ scale_storage,
    const int32_t* __restrict__ m_indices,
    const int32_t* __restrict__ endpoint) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaGridDependencySynchronize();
#endif

  const int row = static_cast<int>(blockIdx.x);
  const int warp_lane = static_cast<int>(threadIdx.x) & 31;
  int live = 0;
  if (warp_lane == 0) {
    const int expert = m_indices[row];
    live = row < endpoint[expert];
  }
  live = __shfl_sync(0xffffffffu, live, 0);
  if (!live) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaTriggerProgrammaticLaunchCompletion();
#endif
    return;
  }

  const int group = static_cast<int>(threadIdx.x) / kSubwarp;
  const int subwarp_lane = static_cast<int>(threadIdx.x) & (kSubwarp - 1);
  const int64_t row_base = static_cast<int64_t>(row) * kGateUp;
  const int64_t group_base =
      row_base + static_cast<int64_t>(group) * kGroup +
      subwarp_lane * kValuesPerLane;

  int4 gate_values[2];
  int4 up_values[2];
  const int4* gate_ptr =
      reinterpret_cast<const int4*>(gateup + group_base);
  const int4* up_ptr =
      reinterpret_cast<const int4*>(gateup + group_base + kHidden);
  gate_values[0] = load_global_nc(gate_ptr);
  gate_values[1] = load_global_nc(gate_ptr + 1);
  up_values[0] = load_global_nc(up_ptr);
  up_values[1] = load_global_nc(up_ptr + 1);

  auto* activated = reinterpret_cast<__nv_bfloat16*>(gate_values);
  const auto* gate_bf16 =
      reinterpret_cast<const __nv_bfloat16*>(gate_values);
  const auto* up_bf16 =
      reinterpret_cast<const __nv_bfloat16*>(up_values);
  float local_absmax = kAbsmaxFloor;

#pragma unroll
  for (int index = 0; index < kValuesPerLane; ++index) {
    const float gate = __bfloat162float(gate_bf16[index]);
    const float up = __bfloat162float(up_bf16[index]);
    const float silu = gate / (1.0f + expf(-gate));
    const __nv_bfloat16 rounded =
        __float2bfloat16_rn(silu * up);
    activated[index] = rounded;
    local_absmax =
        fmaxf(local_absmax, fabsf(__bfloat162float(rounded)));
  }

  const float absmax = subwarp_reduce_max(local_absmax);
  constexpr float kFp8MaxInv = 1.0f / kFp8Max;
  const int scale_exponent = fast_log2_ceil(absmax * kFp8MaxInv);
  const float quant_scale = fast_pow2(-scale_exponent);
  const float inverse_scale = fast_pow2(scale_exponent);
  const uint32_t stored_exponent = __float_as_uint(inverse_scale) >> 23;

  // Each warp owns four adjacent 128-value groups and therefore one complete
  // little-endian packed scale word. Only the four subwarp leaders
  // participate, so there are no byte-store races or post-pack work.
  if (subwarp_lane == 0) {
    constexpr uint32_t kLeaderMask = 0x01010101u;
    const uint32_t exponent_0 =
        __shfl_sync(kLeaderMask, stored_exponent, 0);
    const uint32_t exponent_1 =
        __shfl_sync(kLeaderMask, stored_exponent, 8);
    const uint32_t exponent_2 =
        __shfl_sync(kLeaderMask, stored_exponent, 16);
    const uint32_t exponent_3 =
        __shfl_sync(kLeaderMask, stored_exponent, 24);
    if (warp_lane == 0) {
      const int word = static_cast<int>(threadIdx.x) / 32;
      scale_storage[static_cast<int64_t>(word) * kRows + row] =
          static_cast<int32_t>(
              exponent_0 | (exponent_1 << 8) | (exponent_2 << 16) |
              (exponent_3 << 24));
    }
  }

  int4 quantized;
  auto* quantized_pairs =
      reinterpret_cast<__nv_fp8x2_storage_t*>(&quantized);
  const float2 repeated_scale = make_float2(quant_scale, quant_scale);
#pragma unroll
  for (int index = 0; index < kValuesPerLane; index += 2) {
    float2 values = make_float2(
        __bfloat162float(activated[index]),
        __bfloat162float(activated[index + 1]));
    values = multiply_rn(values, repeated_scale);
    values.x = fminf(fmaxf(values.x, -kFp8Max), kFp8Max);
    values.y = fminf(fmaxf(values.y, -kFp8Max), kFp8Max);
    quantized_pairs[index / 2] =
        __nv_cvt_float2_to_fp8x2(values, __NV_SATFINITE, __NV_E4M3);
  }

  const int64_t output_offset =
      static_cast<int64_t>(row) * kHidden +
      static_cast<int64_t>(group) * kGroup +
      subwarp_lane * kValuesPerLane;
  store_global(
      reinterpret_cast<int4*>(output + output_offset), quantized);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// NCU showed that the s8/v16 mapping is the latency frontier but that its two
// BF16 halves use 43% excessive L2 sectors.  Keep the same four-warp,
// warp-local reduction/packing structure while distributing each 64-value
// half contiguously across the eight lanes.  The matching two 64-bit output
// stores preserve the exact logical order without shared memory or a helper
// launch.
__global__ void swiglu_quant_prefill_s8_v16_b128_cgld_kernel(
    const __nv_bfloat16* __restrict__ gateup,
    uint8_t* __restrict__ output,
    int32_t* __restrict__ scale_storage,
    const int32_t* __restrict__ m_indices,
    const int32_t* __restrict__ endpoint) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaGridDependencySynchronize();
#endif

  const int row = static_cast<int>(blockIdx.x);
  const int warp_lane = static_cast<int>(threadIdx.x) & 31;
  int live = 0;
  if (warp_lane == 0) {
    const int expert = m_indices[row];
    live = row < endpoint[expert];
  }
  live = __shfl_sync(0xffffffffu, live, 0);
  if (!live) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaTriggerProgrammaticLaunchCompletion();
#endif
    return;
  }

  const int group = static_cast<int>(threadIdx.x) / kSubwarp;
  const int subwarp_lane = static_cast<int>(threadIdx.x) & (kSubwarp - 1);
  const int64_t row_base = static_cast<int64_t>(row) * kGateUp;
  const int64_t group_base =
      row_base + static_cast<int64_t>(group) * kGroup;
  constexpr int kValuesPerVector = 8;
  constexpr int kHalfGroup = kGroup / 2;
  const int64_t first_offset =
      group_base + subwarp_lane * kValuesPerVector;
  const int64_t second_offset = first_offset + kHalfGroup;

  int4 gate_values[2];
  int4 up_values[2];
  gate_values[0] =
      load_global_nc(reinterpret_cast<const int4*>(gateup + first_offset));
  gate_values[1] =
      load_global_nc(reinterpret_cast<const int4*>(gateup + second_offset));
  up_values[0] = load_global_nc(
      reinterpret_cast<const int4*>(gateup + first_offset + kHidden));
  up_values[1] = load_global_nc(
      reinterpret_cast<const int4*>(gateup + second_offset + kHidden));

  auto* activated = reinterpret_cast<__nv_bfloat16*>(gate_values);
  const auto* gate_bf16 =
      reinterpret_cast<const __nv_bfloat16*>(gate_values);
  const auto* up_bf16 =
      reinterpret_cast<const __nv_bfloat16*>(up_values);
  float local_absmax = kAbsmaxFloor;

#pragma unroll
  for (int index = 0; index < kValuesPerLane; ++index) {
    const float gate = __bfloat162float(gate_bf16[index]);
    const float up = __bfloat162float(up_bf16[index]);
    const float silu = gate / (1.0f + expf(-gate));
    const __nv_bfloat16 rounded = __float2bfloat16_rn(silu * up);
    activated[index] = rounded;
    local_absmax =
        fmaxf(local_absmax, fabsf(__bfloat162float(rounded)));
  }

  const float absmax = subwarp_reduce_max(local_absmax);
  constexpr float kFp8MaxInv = 1.0f / kFp8Max;
  const int scale_exponent = fast_log2_ceil(absmax * kFp8MaxInv);
  const float quant_scale = fast_pow2(-scale_exponent);
  const float inverse_scale = fast_pow2(scale_exponent);
  const uint32_t stored_exponent = __float_as_uint(inverse_scale) >> 23;

  if (subwarp_lane == 0) {
    constexpr uint32_t kLeaderMask = 0x01010101u;
    const uint32_t exponent_0 =
        __shfl_sync(kLeaderMask, stored_exponent, 0);
    const uint32_t exponent_1 =
        __shfl_sync(kLeaderMask, stored_exponent, 8);
    const uint32_t exponent_2 =
        __shfl_sync(kLeaderMask, stored_exponent, 16);
    const uint32_t exponent_3 =
        __shfl_sync(kLeaderMask, stored_exponent, 24);
    if (warp_lane == 0) {
      const int word = static_cast<int>(threadIdx.x) / 32;
      scale_storage[static_cast<int64_t>(word) * kRows + row] =
          static_cast<int32_t>(
              exponent_0 | (exponent_1 << 8) | (exponent_2 << 16) |
              (exponent_3 << 24));
    }
  }

  int4 quantized;
  auto* quantized_pairs =
      reinterpret_cast<__nv_fp8x2_storage_t*>(&quantized);
  const float2 repeated_scale = make_float2(quant_scale, quant_scale);
#pragma unroll
  for (int index = 0; index < kValuesPerLane; index += 2) {
    float2 values = make_float2(
        __bfloat162float(activated[index]),
        __bfloat162float(activated[index + 1]));
    values = multiply_rn(values, repeated_scale);
    values.x = fminf(fmaxf(values.x, -kFp8Max), kFp8Max);
    values.y = fminf(fmaxf(values.y, -kFp8Max), kFp8Max);
    quantized_pairs[index / 2] =
        __nv_cvt_float2_to_fp8x2(values, __NV_SATFINITE, __NV_E4M3);
  }

  const uint2* quantized_halves =
      reinterpret_cast<const uint2*>(&quantized);
  const int64_t output_base =
      static_cast<int64_t>(row) * kHidden +
      static_cast<int64_t>(group) * kGroup;
  const int64_t first_output =
      output_base + subwarp_lane * kValuesPerVector;
  const int64_t second_output = first_output + kHalfGroup;
  store_global(
      reinterpret_cast<uint2*>(output + first_output),
      quantized_halves[0]);
  store_global(
      reinterpret_cast<uint2*>(output + second_output),
      quantized_halves[1]);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// A bounded occupancy/vector-width alternative to the four-warp mapping above.
// Each 16-lane subwarp owns one 128-value quant group, halving the number of
// precise expf evaluations serialized by a lane.  The 16 scale exponents are
// packed through 64 bytes of shared memory because a packed word now spans two
// warps; there are still no helper kernels or adapter launches.
__global__ void swiglu_quant_prefill_s16_v8_b256_kernel(
    const __nv_bfloat16* __restrict__ gateup,
    uint8_t* __restrict__ output,
    int32_t* __restrict__ scale_storage,
    const int32_t* __restrict__ m_indices,
    const int32_t* __restrict__ endpoint) {
  __shared__ uint32_t scale_exponents[kGroupsPerRow];

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaGridDependencySynchronize();
#endif

  const int row = static_cast<int>(blockIdx.x);
  const int warp_lane = static_cast<int>(threadIdx.x) & 31;
  int live = 0;
  if (warp_lane == 0) {
    const int expert = m_indices[row];
    live = row < endpoint[expert];
  }
  live = __shfl_sync(0xffffffffu, live, 0);
  if (!live) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaTriggerProgrammaticLaunchCompletion();
#endif
    return;
  }

  const int group = static_cast<int>(threadIdx.x) / kSubwarp16;
  const int subwarp_lane =
      static_cast<int>(threadIdx.x) & (kSubwarp16 - 1);
  const int64_t row_base = static_cast<int64_t>(row) * kGateUp;
  const int64_t group_base =
      row_base + static_cast<int64_t>(group) * kGroup +
      subwarp_lane * kValuesPerLane16;

  int4 gate_values;
  int4 up_values;
  const int4* gate_ptr =
      reinterpret_cast<const int4*>(gateup + group_base);
  const int4* up_ptr =
      reinterpret_cast<const int4*>(gateup + group_base + kHidden);
  gate_values = load_global_nc(gate_ptr);
  up_values = load_global_nc(up_ptr);

  auto* activated = reinterpret_cast<__nv_bfloat16*>(&gate_values);
  const auto* gate_bf16 =
      reinterpret_cast<const __nv_bfloat16*>(&gate_values);
  const auto* up_bf16 =
      reinterpret_cast<const __nv_bfloat16*>(&up_values);
  float local_absmax = kAbsmaxFloor;

#pragma unroll
  for (int index = 0; index < kValuesPerLane16; ++index) {
    const float gate = __bfloat162float(gate_bf16[index]);
    const float up = __bfloat162float(up_bf16[index]);
    const float silu = gate / (1.0f + expf(-gate));
    const __nv_bfloat16 rounded =
        __float2bfloat16_rn(silu * up);
    activated[index] = rounded;
    local_absmax =
        fmaxf(local_absmax, fabsf(__bfloat162float(rounded)));
  }

  const float absmax = subwarp16_reduce_max(local_absmax);
  constexpr float kFp8MaxInv = 1.0f / kFp8Max;
  const int scale_exponent = fast_log2_ceil(absmax * kFp8MaxInv);
  const float quant_scale = fast_pow2(-scale_exponent);
  const float inverse_scale = fast_pow2(scale_exponent);
  const uint32_t stored_exponent = __float_as_uint(inverse_scale) >> 23;
  if (subwarp_lane == 0) {
    scale_exponents[group] = stored_exponent;
  }

  uint2 quantized;
  auto* quantized_pairs =
      reinterpret_cast<__nv_fp8x2_storage_t*>(&quantized);
  const float2 repeated_scale = make_float2(quant_scale, quant_scale);
#pragma unroll
  for (int index = 0; index < kValuesPerLane16; index += 2) {
    float2 values = make_float2(
        __bfloat162float(activated[index]),
        __bfloat162float(activated[index + 1]));
    values = multiply_rn(values, repeated_scale);
    values.x = fminf(fmaxf(values.x, -kFp8Max), kFp8Max);
    values.y = fminf(fmaxf(values.y, -kFp8Max), kFp8Max);
    quantized_pairs[index / 2] =
        __nv_cvt_float2_to_fp8x2(values, __NV_SATFINITE, __NV_E4M3);
  }

  const int64_t output_offset =
      static_cast<int64_t>(row) * kHidden +
      static_cast<int64_t>(group) * kGroup +
      subwarp_lane * kValuesPerLane16;
  store_global(
      reinterpret_cast<uint2*>(output + output_offset), quantized);

  __syncthreads();
  if (threadIdx.x < 4) {
    const int first_group = static_cast<int>(threadIdx.x) * 4;
    const uint32_t exponent_0 = scale_exponents[first_group];
    const uint32_t exponent_1 = scale_exponents[first_group + 1];
    const uint32_t exponent_2 = scale_exponents[first_group + 2];
    const uint32_t exponent_3 = scale_exponents[first_group + 3];
    scale_storage[static_cast<int64_t>(threadIdx.x) * kRows + row] =
        static_cast<int32_t>(
            exponent_0 | (exponent_1 << 8) | (exponent_2 << 16) |
            (exponent_3 << 24));
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

void run_cuda(
    torch::Tensor gateup,
    torch::Tensor output,
    torch::Tensor scale_storage,
    torch::Tensor m_indices,
    torch::Tensor endpoint,
    int64_t variant) {
  TORCH_CHECK(gateup.is_cuda() && output.is_cuda() &&
                  scale_storage.is_cuda() && m_indices.is_cuda() &&
                  endpoint.is_cuda(),
              "Task-29 CUDA tensors must be on CUDA");
  TORCH_CHECK(gateup.scalar_type() == at::ScalarType::BFloat16,
              "gateup must be BF16");
  TORCH_CHECK(output.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "output must be Float8_e4m3fn");
  TORCH_CHECK(scale_storage.scalar_type() == at::ScalarType::Int &&
                  m_indices.scalar_type() == at::ScalarType::Int &&
                  endpoint.scalar_type() == at::ScalarType::Int,
              "scale and metadata tensors must be int32");
  TORCH_CHECK(gateup.is_contiguous() && output.is_contiguous() &&
                  scale_storage.is_contiguous() &&
                  m_indices.is_contiguous() && endpoint.is_contiguous(),
              "Task-29 CUDA tensors must use exact contiguous storage");
  TORCH_CHECK(gateup.numel() == static_cast<int64_t>(kRows) * kGateUp &&
                  output.numel() ==
                      static_cast<int64_t>(kRows) * kHidden &&
                  scale_storage.numel() ==
                      static_cast<int64_t>(kRows) * 4 &&
                  m_indices.numel() == kRows &&
                  endpoint.numel() == kExperts,
              "Task-29 CUDA tensor size mismatch");
  TORCH_CHECK(gateup.get_device() == output.get_device() &&
                  gateup.get_device() == scale_storage.get_device() &&
                  gateup.get_device() == m_indices.get_device() &&
                  gateup.get_device() == endpoint.get_device(),
              "Task-29 CUDA tensors must share one device");
  TORCH_CHECK(
      variant == 0 || variant == 1 || variant == 2,
      "Task-29 CUDA variant must be 0 (s8/v16/b128), 1 "
      "(s16/v8/b256), or 2 (s8/v16/b128/coalesced-load)");

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(kRows);
  config.blockDim = dim3(variant == 1 ? kThreads256 : kThreads);
  config.dynamicSmemBytes = 0;
  config.stream = at::cuda::getCurrentCUDAStream(gateup.get_device());
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute.val.programmaticStreamSerializationAllowed = 1;
  config.attrs = &attribute;
  config.numAttrs = 1;

  if (variant == 0) {
    C10_CUDA_CHECK(cudaLaunchKernelEx(
        &config,
        swiglu_quant_prefill_s8_v16_b128_kernel,
        reinterpret_cast<const __nv_bfloat16*>(gateup.data_ptr()),
        static_cast<uint8_t*>(output.data_ptr()),
        scale_storage.data_ptr<int32_t>(),
        m_indices.data_ptr<int32_t>(),
        endpoint.data_ptr<int32_t>()));
  } else if (variant == 1) {
    C10_CUDA_CHECK(cudaLaunchKernelEx(
        &config,
        swiglu_quant_prefill_s16_v8_b256_kernel,
        reinterpret_cast<const __nv_bfloat16*>(gateup.data_ptr()),
        static_cast<uint8_t*>(output.data_ptr()),
        scale_storage.data_ptr<int32_t>(),
        m_indices.data_ptr<int32_t>(),
        endpoint.data_ptr<int32_t>()));
  } else {
    C10_CUDA_CHECK(cudaLaunchKernelEx(
        &config,
        swiglu_quant_prefill_s8_v16_b128_cgld_kernel,
        reinterpret_cast<const __nv_bfloat16*>(gateup.data_ptr()),
        static_cast<uint8_t*>(output.data_ptr()),
        scale_storage.data_ptr<int32_t>(),
        m_indices.data_ptr<int32_t>(),
        endpoint.data_ptr<int32_t>()));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "run_cuda",
      &run_cuda,
      "GLM-5.2 exact prefill SwiGLU plus packed UE8M0 (CUDA)");
}
