#include "silu_and_mul_masked_post_quant.cuh"

// The stock wrapper launches from the physical expert slab.  The first Task-25
// candidate replaced that with source_rank_M * topk blocks, but that quantity is
// only the *mean* number of assignments received by an EP rank.  Routing skew can
// make sum(masked_m) larger on a destination rank, so the old candidate silently
// left valid rows unwritten.
//
// Keep a graph-static, host-known CTA pool, but let every CTA consume a
// grid-stride sequence of device-owned work IDs.  The loop terminates from the
// prefix sum of masked_m, so it covers the complete destination-rank route table
// without a host synchronization or a dynamic CUDA-graph launch.
namespace {

struct Task25RouteWork {
  CTAWork work;
  uint32_t total_work;
};

SGL_DEVICE Task25RouteWork task25_get_work(
    const SiluMulQuantVarlenParams& params, uint32_t work_index) {
  // Preconditions are identical to stock get_work():
  // 1. blockDim.x >= params.num_experts
  // 2. params.num_experts <= kMaxExperts
  using namespace device;
  static_assert(kWarpThreads == 32);

  static __shared__ uint32_t s_warp_sum[32];
  static __shared__ CTAWork result;

  result.valid = false;

  const uint32_t tx = threadIdx.x;
  const uint32_t lane_id = tx % kWarpThreads;
  const uint32_t warp_id = tx / kWarpThreads;
  const uint32_t val = tx < params.num_experts ? params.masked_m[tx] : 0u;

  const uint32_t warp_inclusive = warp_inclusive_sum(lane_id, val);
  const uint32_t warp_exclusive = warp_inclusive - val;

  if (lane_id == kWarpThreads - 1) s_warp_sum[warp_id] = warp_inclusive;
  __syncthreads();
  const auto tmp_val = lane_id < warp_id ? s_warp_sum[lane_id] : 0u;
  const auto prefix_exclusive = warp::reduce_sum(tmp_val) + warp_exclusive;
  const auto total_work = warp::reduce_sum(
      lane_id < (blockDim.x / kWarpThreads) ? s_warp_sum[lane_id] : 0u);
  if (prefix_exclusive <= work_index && work_index < prefix_exclusive + val) {
    result = {tx, work_index - prefix_exclusive, true};
  }
  __syncthreads();
  return {result, total_work};
}

template <bool kUsePDL>
__global__ __launch_bounds__(1024, 2) void task25_silu_mul_quant_grid_stride_kernel(
    const SiluMulQuantVarlenParams __grid_constant__ params) {
  using namespace device;

  constexpr uint32_t kGroupSize = 128u;
  constexpr uint32_t kWorkThreads = 16u;
  using InputVec = AlignedVector<bf16x2_t, 4>;
  using OutputVec = AlignedVector<fp8x2_e4m3_t, 4>;
  static_assert(8 * kWorkThreads == kGroupSize, "Invalid tiling");

  bool pdl_waited = false;
  bool pdl_triggered = false;

  for (uint32_t route_work_id = blockIdx.x;; route_work_id += gridDim.x) {
    const auto route_work = task25_get_work(params, route_work_id);
    const auto [expert_id, token_id, valid] = route_work.work;
    if (!valid) break;

    const auto group_id = threadIdx.x / kWorkThreads;
    const auto offset = expert_id * params.num_tokens + token_id;
    const auto input = params.input + offset * params.hidden_dim * 2;
    const auto output = params.output + offset * params.hidden_dim;
    const auto num_groups = params.hidden_dim / kGroupSize;
    const auto output_scale = [&] {
      const auto base = reinterpret_cast<uint8_t*>(params.output_scale);
      return base + expert_id * num_groups * params.num_tokens +
             (group_id / 4u) * (params.num_tokens * 4u) + token_id * 4u + (group_id % 4u);
    }();

    if (!pdl_waited) {
      PDLWaitPrimary<kUsePDL>();
      pdl_waited = true;
    }

    InputVec gate_vec, up_vec;
    gate_vec.load(input, threadIdx.x);
    up_vec.load(input, threadIdx.x + blockDim.x);

    float local_max = 0.0f;
    float results[8];

#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      const auto [x, y] = silu_and_mul<false>(gate_vec[i], up_vec[i], params.swiglu_limit);
      results[2 * i + 0] = x;
      results[2 * i + 1] = y;
      local_max = fmaxf(local_max, fmaxf(fabsf(x), fabsf(y)));
    }

    local_max = warp::reduce_max<kWorkThreads>(local_max);
    const float absmax = fmaxf(local_max, 1e-10f);
    const float raw_scale = absmax / math::FP8_E4M3_MAX;
    const uint32_t ue8m0_exp = cast_to_ue8m0(raw_scale);
    const float inv_scale = 1.0f / __uint_as_float(ue8m0_exp << 23);

    OutputVec out_vec;
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      out_vec[i] = pack_fp8(results[2 * i + 0] * inv_scale, results[2 * i + 1] * inv_scale);
    }

    // A grid-stride CTA must not release its dependent W2 grid after its first
    // item: later routed rows would still be unproduced. Trigger immediately
    // before this CTA's *last* store, matching stock's producer/store ordering
    // while keeping every earlier loop iteration hidden from the consumer.
    if (!pdl_triggered && route_work_id + gridDim.x >= route_work.total_work) {
      PDLTriggerSecondary<kUsePDL>();
      pdl_triggered = true;
    }

    out_vec.store(output, threadIdx.x);
    *output_scale = ue8m0_exp;
  }
}

}  // namespace

template <bool kUsePDL>
struct Task25SiluAndMulMaskedPostQuantGridStrideKernel {
  static constexpr auto kernel = task25_silu_mul_quant_grid_stride_kernel<kUsePDL>;

  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView output_scale,
      const tvm::ffi::TensorView masked_m,
      const uint32_t topk,
      const uint32_t num_real_tokens) {
    using namespace host;

    auto device = SymbolicDevice{};
    auto E = SymbolicSize{"num_experts"};
    auto T = SymbolicSize{"num_tokens_padded"};
    auto D = SymbolicSize{"hidden_dim x 2"};
    auto N = SymbolicSize{"hidden_dim"};
    auto G4 = SymbolicSize{"packed groups"};
    device.set_options<kDLCUDA>();

    TensorMatcher({E, T, D}).with_dtype<bf16_t>().with_device(device).verify(input);
    TensorMatcher({E, T, N}).with_dtype<fp8_e4m3_t>().with_device(device).verify(output);
    TensorMatcher({E, G4, T}).with_dtype<int32_t>().with_device(device).verify(output_scale);
    TensorMatcher({E}).with_dtype<int32_t>().with_device(device).verify(masked_m);

    RuntimeCheck(E.unwrap() == 32, "Task-25 candidate requires E=32");
    RuntimeCheck(
        T.unwrap() == 1024 || T.unwrap() == 8192,
        "candidate requires an audited 1024- or 8192-row expert slab");
    RuntimeCheck(D.unwrap() == 4096, "Task-25 candidate requires gate/up width 4096");
    RuntimeCheck(N.unwrap() == 2048, "Task-25 candidate requires hidden width 2048");
    RuntimeCheck(G4.unwrap() == 4, "Task-25 candidate requires four packed scale words");
    RuntimeCheck(topk == 8, "Task-25 candidate requires topk=8");
    RuntimeCheck(
        num_real_tokens == 1 || num_real_tokens == 2 || num_real_tokens == 4 ||
            num_real_tokens == 8 || num_real_tokens == 12 || num_real_tokens == 16 ||
            num_real_tokens == 32,
        "Task-25 candidate requires an audited M1/M2/M4/M8/M12/M16/M32 bucket");

    const auto params = SiluMulQuantVarlenParams{
        .input = static_cast<const bf16_t*>(input.data_ptr()),
        .output = static_cast<fp8_e4m3_t*>(output.data_ptr()),
        .output_scale = static_cast<float*>(output_scale.data_ptr()),
        .masked_m = static_cast<const int32_t*>(masked_m.data_ptr()),
        .swiglu_limit = 0.0f,
        .hidden_dim = 2048,
        .num_tokens = static_cast<uint32_t>(T.unwrap()),
        .num_experts = 32,
    };

    // Pool size is a throughput choice, not a correctness bound.  The kernel
    // loops over the complete device-side route table even when this EP rank
    // receives more than source_rank_M * topk assignments.
    LaunchKernel(num_real_tokens * topk, 256, device.unwrap())
        .enable_pdl(kUsePDL)(kernel, params);
  }
};
