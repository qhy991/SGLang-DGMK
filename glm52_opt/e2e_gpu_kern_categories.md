# Default-path e2e GPU kernel breakdown (OPT0 nsys)

Source: `bench_results/nsys_op_single/opt0.nsys-rep` / `opt0.sqlite`.
Serving under nsys, **OPT=0**, **deepep=low_latency**.

This is **full GPU kernel time including communication** (NCCL + DeepEP),
unlike the earlier llm_flops single-op layer shares which intentionally omit comm.

Caveat: capture is from `nsys_op_single` (short-prompt, decode-heavy).
Prefill@64k mix will differ, but **comm remains first-class** on the default EP path.

| category | GPU time % | total ms | instances | #variants |
|----------|-----------:|---------:|----------:|----------:|
| comm_nccl | 31.3 | 13751.6 | 174746 | 1 |
| comm_deepep | 21.2 | 9290.8 | 312000 | 2 |
| gemm_deepgemm | 20.3 | 8884.4 | 761232 | 14 |
| elementwise_misc | 7.5 | 3309.7 | 2365326 | 35 |
| moe_act_quant | 7.2 | 3141.3 | 158028 | 2 |
| dsa_attn_mla | 5.1 | 2230.5 | 241110 | 4 |
| norm_rope | 2.9 | 1291.4 | 443901 | 9 |
| quant | 2.3 | 1001.5 | 501357 | 2 |
| router_topk | 1.3 | 579.2 | 252147 | 8 |
| other | 0.4 | 182.2 | 85439 | 17 |
| index_score_mqa | 0.3 | 110.7 | 22198 | 2 |
| gemm_cublas | 0.2 | 96.4 | 23847 | 2 |

## Top raw kernels

| GPU % | name (truncated) |
|------:|------------------|
| 31.3 | `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` |
| 14.9 | `void deep_ep::internode_ll::dispatch<(bool)1, (bool)1, (int)6144>(void *, void *, int *, long *, int *, int...` |
| 10.5 | `void deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, (unsigned int)128,...` |
| 6.7 | `void <unnamed>::silu_mul_quant_varlen_kernel<(bool)1, (bool)1, (bool)0, (bool)1, (bool)0>(<unnamed>::SiluMu...` |
| 6.3 | `void deep_ep::internode_ll::combine<(bool)0, (int)6144, (int)11, (int)4>(void *, void *, int *, void *, con...` |
| 2.8 | `void deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, (unsigned int)128,...` |
| 2.6 | `void smxx::decode::flash_fwd_mla_combine_kernel<cutlass::bfloat16_t, (int)512, (int)8, (int)160, (int)256>(...` |
| 2.2 | `void sm100::decode::head64::flash_fwd_splitkv_mla_fp8_sparse_kernel<sm100::decode::head64::KernelTemplate<(...` |
| 2.2 | `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::d...` |
| 2.1 | `void deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, (unsigned int)128,...` |
| 2.1 | `void <unnamed>::per_token_group_quant_8bit_v2_kernel<<unnamed>::NaiveScheduler, (int)128, (int)8, __nv_bflo...` |
| 1.9 | `void deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, (unsigned int)128,...` |
| 1.6 | `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<long, at::native::func_wrapper_t<long...` |
| 1.5 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16g...` |
| 0.9 | `nvjet_sm103_tst_128x8_64x12_2x1_v_bz_TNT` |
| 0.9 | `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<long, at::native::func_wrapper_t<long...` |
| 0.8 | `nvjet_sm103_tst_256x8_64x6_2x1_v_bz_TNT` |
| 0.7 | `void at_cuda_detail::cub::detail::select::DeviceSelectSweepKernel<at_cuda_detail::cub::detail::select::poli...` |
| 0.6 | `nvjet_sm103_tst_192x8_64x8_2x1_v_bz_splitK_TNT` |
| 0.5 | `void <unnamed>::fused_rope_kernel<(bool)0, (long)64, (bool)1, __nv_bfloat16, long, (unsigned int)8>(<unname...` |

## Rollup

- **Communication (NCCL + DeepEP): 52.5%** of summed GPU kernel time
- Non-comm kernels: 47.5%
- Largest singles: NCCL AllGather ~31%; DeepEP LL dispatch ~15%; DeepEP combine ~6%

## Why earlier docs omitted this

Previous shares came from `llm_flops_style` / accept_layer: **single-GPU op microbench**
with no DeepEP/NCCL. That answers 'which compute ops dominate a layer', not
'what fraction of real multi-GPU serving GPU time is communication'.
