# nsys S=64k incremental prefill (default path OPT0)

- Machine: B300 / `sglang_0512`
- Config: `deepep=low_latency`, OPT=0, DSA `flashmla_kv`, TP8/DP8/EP8
- Method: serve under `nsys profile -c cudaProfilerApi`; warm S=64k cache **outside** capture;
  arm trigger; run incremental prefill M=1024+2048 **inside** 45s window
- Trace: `prefill64k_opt0.nsys-rep` (~437 MB) / `.sqlite`

## Capture one_batch

| run | last_ttft | cache_hit |
|-----|----------:|----------:|
| warmup_M1024 | 2.1108 | 0.9846 |
| capture_M1024 | 2.2994 | 0.9846 |
| capture_M2048 | 4.2368 | 0.9697 |

## Interpretation notes

1. This is **GPU kernel-time share** (sum of kernel durations), not wall-clock exclusive time.
2. DeepEP-LL `dispatch` often **busy-waits** (avg >> median): GPU% can look larger than useful work.
3. On this capture: **DeepEP alone ~78%** of summed kernel time; NCCL AllGather ~1.4%;
   compute GEMM ~12%; DSA MLA ~2.8%; index_score/MQA ~0.2%.
4. Contrast vs earlier short-decode nsys (NCCL-heavy ~31%): prefill@64k incremental
   shifts dominance to **DeepEP dispatch**.

Source CSV: `/home/ubuntu/wwxq/bench_results/nsys_prefill_64k/prefill64k_opt0_gpu_kern_sum.csv`

Full GPU kernel time **including communication** (NCCL + DeepEP).

| category | GPU time % | total ms | instances | #variants |
|----------|-----------:|---------:|----------:|----------:|
| comm_deepep | 77.9 | 190648.0 | 417863 | 2 |
| gemm_deepgemm | 12.2 | 29979.7 | 1116355 | 18 |
| dsa_attn_mla | 2.8 | 6937.1 | 325768 | 4 |
| moe_act_quant | 2.1 | 5087.2 | 212589 | 2 |
| comm_nccl | 1.4 | 3541.3 | 40945 | 1 |
| quant | 1.0 | 2542.8 | 673845 | 2 |
| norm_rope | 0.9 | 2156.3 | 596232 | 9 |
| elementwise_misc | 0.7 | 1652.8 | 871856 | 34 |
| router_topk | 0.4 | 900.5 | 247410 | 6 |
| index_score_mqa | 0.2 | 517.9 | 25213 | 3 |
| other | 0.2 | 480.7 | 146727 | 14 |
| gemm_cublas | 0.2 | 375.2 | 125456 | 2 |

## Top raw kernels

| GPU % | name (truncated) |
|------:|------------------|
| 71.9 | `void deep_ep::internode_ll::dispatch<(bool)1, (bool)1, (int)6144>(void *, void *, int *, long *, int *, int...` |
| 6.0 | `void deep_ep::internode_ll::combine<(bool)0, (int)6144, (int)11, (int)4>(void *, void *, int *, void *, con...` |
| 5.0 | `void deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, (unsigned int)128,...` |
| 2.7 | `void deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, (unsigned int)128,...` |
| 2.6 | `void sm100::decode::head64::flash_fwd_splitkv_mla_fp8_sparse_kernel<sm100::decode::head64::KernelTemplate<(...` |
| 1.8 | `void deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, (unsigned int)128,...` |
| 1.7 | `void <unnamed>::silu_mul_quant_varlen_kernel<(bool)1, (bool)1, (bool)0, (bool)1, (bool)0>(<unnamed>::SiluMu...` |
| 1.4 | `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)` |
| 1.0 | `void <unnamed>::per_token_group_quant_8bit_v2_kernel<<unnamed>::NaiveScheduler, (int)128, (int)8, __nv_bflo...` |
| 0.5 | `void deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, (unsigned int)128,...` |
| 0.5 | `void deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, (unsigned int)128,...` |
| 0.4 | `void deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, (unsigned int)128,...` |
| 0.4 | `void <unnamed>::act_and_mul_kernel<__nv_bfloat16, (<unnamed>::ActivationKind)0, (bool)1, (bool)0>(<unnamed>...` |
| 0.4 | `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16g...` |
| 0.3 | `nvjet_sm103_tst_256x128_64x5_2x1_2cta_v_bz_TNT` |
| 0.3 | `nvjet_sm103_tst_128x128_64x8_2x1_2cta_v_bz_TNT` |
| 0.2 | `void <unnamed>::fused_rope_kernel<(bool)0, (long)64, (bool)1, __nv_bfloat16, long, (unsigned int)8>(<unname...` |
| 0.2 | `void deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, (unsigned int)128,...` |
| 0.2 | `nvjet_sm103_tss_64x32_64x16_2x4_2cta_h_bz_splitK_TNT` |
| 0.2 | `void deep_gemm::sm100_mqa_logits<(bool)0, (unsigned int)32, (unsigned int)128, (bool)0, (unsigned int)4, (u...` |
| 0.2 | `_router_triton_kernel` |
| 0.2 | `nvjet_sm103_tst_192x8_64x8_2x1_v_bz_splitK_TNT` |
| 0.1 | `<unnamed>::concat_mla_absorb_q_kernel(__nv_bfloat16 *, __nv_bfloat16 *, __nv_bfloat16 *, int, int, long, in...` |
| 0.1 | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &):...` |
| 0.1 | `void smxx::decode::flash_fwd_mla_combine_kernel<cutlass::bfloat16_t, (int)512, (int)8, (int)160, (int)256>(...` |

## Rollup

- **Communication (NCCL + DeepEP): 79.3%** of summed GPU kernel time
- Non-comm kernels: 20.7%
