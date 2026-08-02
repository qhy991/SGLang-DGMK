// gate_v1_fusednq — fused (residual-add + RMSNorm + per-token-group UE8M0 FP8 quant)
// for the GLM-5.2 norm_quant_gate PREFILL region.
//
// Replaces TWO stock kernels with ONE:
//   flashinfer fused_add_rmsnorm  (CuTe-DSL FusedAddRMSNormKernel)
//   sglang per_token_group_quant_8bit_v2_kernel<NaiveScheduler,128,8,bf16,e4m3,...>
// and hands the same (x_fp8, packed-UE8M0 x_scale) pair to the same
// deep_gemm.fp8_gemm_nt the reference calls.  No FP8 GEMM is hand-rolled.
//
// WHY IT PAYS
// -----------
// The stock pair spills the normalized [M,K] bf16 activation to HBM and reads it
// straight back, but nothing outside the region consumes it — the task contract
// gates only (out, residual).  Fusing removes 4*M*K bytes of HBM traffic and one
// graph node.  Measured on B200 under CUDA-graph replay, the stock norm+quant pair
// is 37% / 45% / 50% of the whole region at M = 1024 / 2048 / 4096.
//
// WHY IT IS BIT-EXACT, AND WHY THAT IS MANDATORY
// ----------------------------------------------
// The harness gates elementwise on (abs_err < abs_tol OR rel_err < rel_tol).  One
// flipped fp8 code perturbs all 2048 outputs of its row, and the near-zero ones
// among those fail both tolerances: an earlier build of this kernel differed in
// FOUR fp8 bytes out of 6291456 and the harness rejected it with 147 failing
// elements at M=1024.  "Within tolerance" is therefore not available here; the
// quantized activation has to match byte for byte, which means `sum_sq` has to
// match bit for bit, which means reproducing the reference's reduction TREE.
//
// The reference reduction is NOT flashinfer's C++ norm.cuh kernel.  sgl_kernel
// forwards to flashinfer.norm.fused_add_rmsnorm, and with FLASHINFER_USE_CUDA_NORM
// unset (the default) that lands on the CuTe-DSL path in
// flashinfer/norm/kernels/fused_add_rmsnorm.py.  Its shape for H=6144 / bf16 /
// sm100, derived from that source and CONFIRMED by tests/solve_rms.py and
// tests/search_tree.py, which solve the reference's own rstd out of its bf16 output
// and then score 26 candidate reduction spellings on the 915 rows where they
// disagree — this one is the only 915/915:
//
//   cluster_n=1, threads_per_row=64, num_threads=128, rows_per_block=2,
//   vec_size=8, num_vec_blocks=12, warps_per_row=2, grid=(ceil(M/2),1,1)
//   TV layout ((64,2),(8,12)):((16,1),(2,1024)) over a (2,6144) tile
//     -> thread tid owns row tid/64 and columns (tid%64)*8 + j + 512*b
//   sum_sq  : accumulated SEQUENTIALLY over the thread's 96 values in flat
//             fragment order v = j + 8*b, CONTRACTED to one FMA per value;
//             then warp_reduce butterfly with ASCENDING offsets 1,2,4,8,16
//             (flashinfer's C++ kernel goes 16..1 — a different sum);
//             then lane0-per-warp -> smem[row][warp%2] -> a second 32-wide
//             butterfly over [a,b,0,...], which yields exactly a + b.
//   mean    : sum_sq / 6144.0f  (IEEE div.rn)
//   rstd    : rsqrt.approx.ftz.f32(mean + eps)
//   y       : (h * rstd) * (weight_bias + w), rounded to bf16
//
// The quant epilogue mirrors sglang's v2 kernel
// (gemm/per_token_group_quant_8bit_v2.cuh); amax is order-independent so the
// differing subwarp width (16 lanes x 8 values here vs 8 lanes x 16 values there)
// is not a divergence:
//   amax  = max(1e-10f, max_j |float(bf16 y_j)|)
//   e     = fast_log2_ceil(amax * (1.0f/448.0f))
//   scale = fast_pow2(-e); stored byte = bits(fast_pow2(e)) >> 23
//   q     = cvt.rn.satfinite.e4m3x2.f32( clamp(__fmul2_rn(v, scale), -448, 448) )
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace {

constexpr int THREADS_PER_ROW = 64;
constexpr int VEC = 8;                                          // bf16 per copy (16 B)
constexpr int NVB = 12;                                         // num_vec_blocks
constexpr int VALS = VEC * NVB;                                 // 96 per thread
constexpr int COL_STRIDE = VEC * THREADS_PER_ROW;               // 512
constexpr int HSIZE = COL_STRIDE * NVB;                         // 6144
constexpr int GROUP = 128;
constexpr int LANES_PER_GROUP = GROUP / VEC;                    // 16
constexpr float LOCAL_ABSMAX_ABS = 1e-10f;                      // sglang v2 init
constexpr float FP8_MAX = 448.0f;

__device__ __forceinline__ float bfly(float x, int mask) {
  float y;
  asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;"
               : "=f"(y)
               : "f"(x), "r"(mask));
  return y;
}

__device__ __forceinline__ float rsqrt_approx_ftz(float x) {
  float y;
  asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// DeepEP-derived helpers, as copied by sglang's v2 quant kernel.
__device__ __forceinline__ float fast_pow2(int x) {
  return __uint_as_float(static_cast<uint32_t>((x + 127) << 23));
}
__device__ __forceinline__ int fast_log2_ceil(float x) {
  const uint32_t b = __float_as_uint(x);
  const int e = static_cast<int>((b >> 23) & 0xff);
  const uint32_t man = b & ((1u << 23) - 1u);
  return e - 127 + (man != 0);
}

__device__ __forceinline__ float2 fmul2(float2 a, float2 b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  return __fmul2_rn(a, b);
#else
  return make_float2(a.x * b.x, a.y * b.y);
#endif
}

__device__ __forceinline__ void cp_async_16(void* smem_dst, const void* gmem_src) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem_src));
}

// ROWS rows per block, 64 threads per row (the reduction shape is pinned by
// bit-exactness; ROWS is free and only trades block size against occupancy).
// SMEM_STAGE=true : cp.async hidden+residual into smem, recompute h in pass 2.
//                   Lowest registers (32) and best at M=1024, where the whole grid is
//                   one resident wave.
// SMEM_STAGE=false: plain LDG, carry h as 96 fp32 registers across the barrier.
//                   142 registers, but no smem cap on blocks/SM and no second smem
//                   read; measurably better once the grid exceeds one wave.
template <int ROWS, bool SMEM_STAGE, bool HAS_PRE_ADD = false, bool WRITE_BF16 = false>
__global__ __launch_bounds__(ROWS * THREADS_PER_ROW) void infini_glm52_fused_add_rmsnorm_quant_ue8m0_kernel(
    const __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ pre_add,
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ y_bf16,
    __nv_fp8_e4m3* __restrict__ xq,
    uint32_t* __restrict__ xs,
    const int M,
    const int scale_hidden_stride,   // xs.stride(1), in int32 elements
    const float eps,
    const float weight_bias) {
  static_assert(!(SMEM_STAGE && HAS_PRE_ADD),
                "the three-input path deliberately uses the register-staged kernel");
  extern __shared__ __align__(16) char smem_raw[];
  __nv_bfloat16* sX = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  __nv_bfloat16* sR = sX + (SMEM_STAGE ? ROWS * HSIZE : 0);
  float* red = reinterpret_cast<float*>(sR + (SMEM_STAGE ? ROWS * HSIZE : 0));

  const int tid = threadIdx.x;
  const int t0 = tid % THREADS_PER_ROW;
  const int rib = tid / THREADS_PER_ROW;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int row = blockIdx.x * ROWS + rib;
  const bool live = (row < M);

  const int64_t g_base = static_cast<int64_t>(row) * HSIZE + t0 * VEC;
  const int s_base = rib * HSIZE + t0 * VEC;

  // ── stage hidden + residual into smem; the copies fly without holding registers,
  //    which is what lets this kernel keep 24 loads in flight per thread ──
  if (SMEM_STAGE && live) {
#pragma unroll
    for (int b = 0; b < NVB; ++b) {
      cp_async_16(sX + s_base + b * COL_STRIDE, hidden + g_base + b * COL_STRIDE);
      cp_async_16(sR + s_base + b * COL_STRIDE, residual + g_base + b * COL_STRIDE);
    }
  }
  if constexpr (SMEM_STAGE) {
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
  }

  // ── pass 1: h = hidden + residual -> residual, and sum_sq ──
  float acc = 0.0f;
  float hreg[SMEM_STAGE ? 1 : VALS];
  if (live) {
#pragma unroll
    for (int b = 0; b < NVB; ++b) {
      const int4 hv = SMEM_STAGE
          ? *reinterpret_cast<const int4*>(sX + s_base + b * COL_STRIDE)
          : *reinterpret_cast<const int4*>(hidden + g_base + b * COL_STRIDE);
      const int4 rv = SMEM_STAGE
          ? *reinterpret_cast<const int4*>(sR + s_base + b * COL_STRIDE)
          : *reinterpret_cast<const int4*>(residual + g_base + b * COL_STRIDE);
      int4 pv{};
      if constexpr (HAS_PRE_ADD) {
        pv = *reinterpret_cast<const int4*>(pre_add + g_base + b * COL_STRIDE);
      }
      const __nv_bfloat16* hb = reinterpret_cast<const __nv_bfloat16*>(&hv);
      const __nv_bfloat16* rb = reinterpret_cast<const __nv_bfloat16*>(&rv);
      const __nv_bfloat16* pb = reinterpret_cast<const __nv_bfloat16*>(&pv);
      __nv_bfloat16 ob[VEC];
#pragma unroll
      for (int j = 0; j < VEC; ++j) {
        float hidden_value = __bfloat162float(hb[j]);
        if constexpr (HAS_PRE_ADD) {
          // Production materializes shared.add_(routed) in BF16 before the
          // next residual add. Preserve that observable store/reload round;
          // reassociating three FP32 values changes residual, norm and FP8 bytes.
          hidden_value = __bfloat162float(
              __float2bfloat16(hidden_value + __bfloat162float(pb[j])));
        }
        const float v = hidden_value + __bfloat162float(rb[j]);
        ob[j] = __float2bfloat16(v);
        if constexpr (!SMEM_STAGE) hreg[b * VEC + j] = v;
        // Sequential over the flat fragment order v = j + 8*b, CONTRACTED to one FMA
        // per value.  Both the order and the contraction are load-bearing:
        // tests/search_tree.py scored 26 candidate spellings against the reference's
        // own output over 915 informative rows and only this one is 915/915 — the
        // same order without the contraction is 912/915, and that 3-row gap left one
        // wrong fp8 byte at M=4096 in an earlier build.
        acc = __fmaf_rn(v, v, acc);
      }
      *reinterpret_cast<int4*>(residual + g_base + b * COL_STRIDE) =
          *reinterpret_cast<const int4*>(ob);
    }
  }

#pragma unroll
  for (int off = 1; off < 32; off <<= 1) acc = acc + bfly(acc, off);
  if (lane == 0) red[(warp >> 1) * 2 + (warp & 1)] = acc;
  __syncthreads();
  float bv = (lane < 2) ? red[rib * 2 + lane] : 0.0f;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) bv = bv + bfly(bv, off);
  if (!live) return;

  const float rstd = rsqrt_approx_ftz(bv / static_cast<float>(HSIZE) + eps);

  // ── pass 2: normalize, per-128-group UE8M0 quant, store fp8 + scale ──
  const int gsub = t0 / LANES_PER_GROUP;         // which 128-group inside this 512 block
  const bool group_leader = (t0 % LANES_PER_GROUP) == 0;
#pragma unroll
  for (int b = 0; b < NVB; ++b) {
    const int soff = s_base + b * COL_STRIDE;
    int4 hv, rv;
    if constexpr (SMEM_STAGE) {
      hv = *reinterpret_cast<const int4*>(sX + soff);
      rv = *reinterpret_cast<const int4*>(sR + soff);
    }
    const int4 wv = *reinterpret_cast<const int4*>(weight + t0 * VEC + b * COL_STRIDE);
    const __nv_bfloat16* hb = reinterpret_cast<const __nv_bfloat16*>(&hv);
    const __nv_bfloat16* rb = reinterpret_cast<const __nv_bfloat16*>(&rv);
    const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wv);

    __nv_bfloat16 y[VEC];
    float amax = LOCAL_ABSMAX_ABS;
#pragma unroll
    for (int j = 0; j < VEC; ++j) {
      // h recomputed from smem rather than carried in registers: a deterministic
      // fp32 add, so bit-identical, and it frees the 96 registers that were
      // capping memory-level parallelism.
      const float hh = SMEM_STAGE
          ? __bfloat162float(hb[j]) + __bfloat162float(rb[j])
          : hreg[b * VEC + j];
      const float wf = __bfloat162float(wb[j]);
      const __nv_bfloat16 yb = __float2bfloat16(hh * rstd * (weight_bias + wf));
      y[j] = yb;
      amax = fmaxf(amax, fabsf(__bfloat162float(yb)));
    }
    if constexpr (WRITE_BF16) {
      *reinterpret_cast<int4*>(y_bf16 + g_base + b * COL_STRIDE) =
          *reinterpret_cast<const int4*>(y);
    }
#pragma unroll
    for (int s = 1; s < LANES_PER_GROUP; s <<= 1) amax = fmaxf(amax, bfly(amax, s));

    const int e = fast_log2_ceil(amax * (1.0f / FP8_MAX));
    if (group_leader) {
      // column-major TMA-aligned packed UE8M0: int32[g/4][row], byte g%4,
      // with g = 4*b + gsub, so g/4 == b and g%4 == gsub.
      uint8_t* so = reinterpret_cast<uint8_t*>(xs) +
                    (static_cast<int64_t>(b) * scale_hidden_stride * 4 +
                     static_cast<int64_t>(row) * 4 + gsub);
      *so = static_cast<uint8_t>(__float_as_uint(fast_pow2(e)) >> 23);
    }

    const float y_scale = fast_pow2(-e);
    const float2 ys = make_float2(y_scale, y_scale);
    uint2 ob;
    __nv_fp8x2_storage_t* op = reinterpret_cast<__nv_fp8x2_storage_t*>(&ob);
#pragma unroll
    for (int j = 0; j < VEC; j += 2) {
      float2 in2 = make_float2(__bfloat162float(y[j]), __bfloat162float(y[j + 1]));
      float2 o2 = fmul2(in2, ys);
      o2.x = fminf(fmaxf(o2.x, -FP8_MAX), FP8_MAX);
      o2.y = fminf(fmaxf(o2.y, -FP8_MAX), FP8_MAX);
      op[j >> 1] = __nv_cvt_float2_to_fp8x2(o2, __NV_SATFINITE, __NV_E4M3);
    }
    *reinterpret_cast<uint2*>(xq + g_base + b * COL_STRIDE) = ob;
  }
}

}  // namespace

void fused_add_rmsnorm_quant_ue8m0(at::Tensor hidden, at::Tensor residual,
                                   at::Tensor weight, at::Tensor xq, at::Tensor xs,
                                   double eps, int64_t rows_per_block, int64_t smem_stage) {
  TORCH_CHECK(hidden.is_cuda() && hidden.is_contiguous(), "hidden must be contiguous cuda");
  TORCH_CHECK(residual.is_cuda() && residual.is_contiguous(), "residual must be contiguous cuda");
  TORCH_CHECK(hidden.scalar_type() == at::kBFloat16, "bf16 only");
  TORCH_CHECK(residual.sizes() == hidden.sizes(), "shape mismatch");
  TORCH_CHECK(xq.scalar_type() == at::kFloat8_e4m3fn && xq.is_contiguous(), "xq fp8 contiguous");
  TORCH_CHECK(xs.scalar_type() == at::kInt, "xs must be int32 (packed ue8m0)");
  TORCH_CHECK(xs.stride(0) == 1, "xs must be mn-major (stride(0)==1)");

  const int M = static_cast<int>(hidden.size(0));
  TORCH_CHECK(hidden.size(1) == HSIZE,
              "this build is specialised for K=6144 (GLM-5.2 hidden size)");
  TORCH_CHECK(weight.numel() == HSIZE, "weight numel");

  auto stream = at::cuda::getCurrentCUDAStream();
#define LAUNCH_ROWS(R, S)                                                                \
  do {                                                                                \
    constexpr int kThreads = (R) * THREADS_PER_ROW;                                   \
    const int smem = ((S) ? 2 * (R) * HSIZE * static_cast<int>(sizeof(__nv_bfloat16))  \
                          : 0)                                                          \
                     + (R) * 2 * static_cast<int>(sizeof(float));                      \
    auto kfn = infini_glm52_fused_add_rmsnorm_quant_ue8m0_kernel<R, S>;                              \
    static bool once = false;                                                          \
    if (!once) {                                                                       \
      C10_CUDA_CHECK(cudaFuncSetAttribute(                                              \
          kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));                     \
      once = true;                                                                      \
    }                                                                                   \
    const int grid = (M + (R) - 1) / (R);                                              \
    kfn<<<grid, kThreads, smem, stream>>>(                                             \
        reinterpret_cast<const __nv_bfloat16*>(hidden.data_ptr()),                     \
        nullptr,                                                                        \
        reinterpret_cast<__nv_bfloat16*>(residual.data_ptr()),                         \
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),                     \
        nullptr,                                                                        \
        reinterpret_cast<__nv_fp8_e4m3*>(xq.data_ptr()),                               \
        reinterpret_cast<uint32_t*>(xs.data_ptr()),                                    \
        M, static_cast<int>(xs.stride(1)), static_cast<float>(eps), 0.0f);             \
  } while (0)
  const bool st = (smem_stage != 0);
  switch (rows_per_block * 2 + (st ? 1 : 0)) {
    case 2: LAUNCH_ROWS(1, false); break;
    case 3: LAUNCH_ROWS(1, true); break;
    case 4: LAUNCH_ROWS(2, false); break;
    case 5: LAUNCH_ROWS(2, true); break;
    case 8: LAUNCH_ROWS(4, false); break;
    case 9: LAUNCH_ROWS(4, true); break;
    default: TORCH_CHECK(false, "rows_per_block must be 1, 2 or 4");
  }
#undef LAUNCH_ROWS
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fused_shared_add_rmsnorm_quant_ue8m0(
    at::Tensor shared, at::Tensor routed, at::Tensor residual,
    at::Tensor weight, at::Tensor y_bf16, at::Tensor xq, at::Tensor xs,
    double eps, int64_t rows_per_block) {
  TORCH_CHECK(shared.is_cuda() && shared.is_contiguous(),
              "shared must be contiguous cuda");
  TORCH_CHECK(routed.is_cuda() && routed.is_contiguous(),
              "routed must be contiguous cuda");
  TORCH_CHECK(residual.is_cuda() && residual.is_contiguous(),
              "residual must be contiguous cuda");
  TORCH_CHECK(shared.scalar_type() == at::kBFloat16 &&
              routed.scalar_type() == at::kBFloat16 &&
              residual.scalar_type() == at::kBFloat16 &&
              weight.scalar_type() == at::kBFloat16 &&
              y_bf16.scalar_type() == at::kBFloat16,
              "shared/routed/residual/weight/y must be bf16");
  TORCH_CHECK(routed.sizes() == shared.sizes() &&
              residual.sizes() == shared.sizes() &&
              y_bf16.sizes() == shared.sizes(),
              "shared/routed/residual/y shape mismatch");
  TORCH_CHECK(xq.scalar_type() == at::kFloat8_e4m3fn && xq.is_contiguous(),
              "xq must be contiguous float8_e4m3fn");
  TORCH_CHECK(xq.sizes() == shared.sizes(), "xq shape mismatch");
  TORCH_CHECK(xs.scalar_type() == at::kInt && xs.stride(0) == 1,
              "xs must be mn-major packed int32 UE8M0");

  const int M = static_cast<int>(shared.size(0));
  TORCH_CHECK(shared.dim() == 2 && shared.size(1) == HSIZE,
              "this build is specialised for [M,6144]");
  TORCH_CHECK(weight.numel() == HSIZE, "weight numel");

  auto stream = at::cuda::getCurrentCUDAStream();
#define LAUNCH_SHARED_ROWS(R)                                                         \
  do {                                                                                \
    constexpr int kThreads = (R) * THREADS_PER_ROW;                                   \
    constexpr int smem = (R) * 2 * static_cast<int>(sizeof(float));                   \
    auto kfn = infini_glm52_fused_add_rmsnorm_quant_ue8m0_kernel<                     \
        R, false, true, true>;                                                         \
    const int grid = (M + (R) - 1) / (R);                                             \
    kfn<<<grid, kThreads, smem, stream>>>(                                             \
        reinterpret_cast<const __nv_bfloat16*>(shared.data_ptr()),                    \
        reinterpret_cast<const __nv_bfloat16*>(routed.data_ptr()),                    \
        reinterpret_cast<__nv_bfloat16*>(residual.data_ptr()),                        \
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),                    \
        reinterpret_cast<__nv_bfloat16*>(y_bf16.data_ptr()),                          \
        reinterpret_cast<__nv_fp8_e4m3*>(xq.data_ptr()),                              \
        reinterpret_cast<uint32_t*>(xs.data_ptr()),                                   \
        M, static_cast<int>(xs.stride(1)), static_cast<float>(eps), 0.0f);             \
  } while (0)
  switch (rows_per_block) {
    case 1: LAUNCH_SHARED_ROWS(1); break;
    case 2: LAUNCH_SHARED_ROWS(2); break;
    case 4: LAUNCH_SHARED_ROWS(4); break;
    default: TORCH_CHECK(false, "rows_per_block must be 1, 2 or 4");
  }
#undef LAUNCH_SHARED_ROWS
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("infini_fused_add_rmsnorm_quant_ue8m0", &fused_add_rmsnorm_quant_ue8m0,
        "fused residual-add + RMSNorm + per-token-group UE8M0 fp8 quant");
  m.def("infini_fused_shared_add_rmsnorm_quant_ue8m0",
        &fused_shared_add_rmsnorm_quant_ue8m0,
        "fused BF16 shared+routed round + residual-add + RMSNorm + UE8M0 quant");
}
