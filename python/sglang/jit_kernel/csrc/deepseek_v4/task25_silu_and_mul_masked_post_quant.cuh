#include "silu_and_mul_masked_post_quant.cuh"

// Keep the audited stock kernel body and packed-UE8M0 store exactly unchanged.
// Production sizes the grid from the padded expert slab (1024 on the original
// B200 contract, 8192 on the profiled B300 contract), so almost every block
// scans masked_m and returns during decode. This wrapper launches one block per
// routed assignment instead.
template <bool kUsePDL>
struct Task25SiluAndMulMaskedPostQuantKernel {
  static constexpr auto kernel = silu_mul_quant_varlen_kernel<true, true, false, kUsePDL, false>;

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

    TensorMatcher({E, T, D})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(input);
    TensorMatcher({E, T, N})
        .with_dtype<fp8_e4m3_t>()
        .with_device(device)
        .verify(output);
    TensorMatcher({E, G4, T})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(output_scale);
    TensorMatcher({E})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(masked_m);

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

    LaunchKernel(num_real_tokens * topk, 256, device.unwrap())
        .enable_pdl(kUsePDL)(kernel, params);
  }
};
