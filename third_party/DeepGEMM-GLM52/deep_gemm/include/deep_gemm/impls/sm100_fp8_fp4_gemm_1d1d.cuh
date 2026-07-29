#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>

#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/epilogue/transform.cuh>
#include <deep_gemm/epilogue/sm100_store_cd.cuh>
#include <deep_gemm/epilogue/sm100_store_cd_swap_ab.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/scheduler/gemm.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

template <cute::UMMA::Major kMajorA, cute::UMMA::Major kMajorB,
          uint32_t kGranKA, uint32_t kGranKB, uint32_t kKAlignment,
          uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumGroups,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleCDMode,
          uint32_t kNumStages,
          uint32_t kNumNonEpilogueThreads, uint32_t kNumEpilogueThreads,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          uint32_t kNumSMs,
          bool kSwapAB, bool kEnsureZeroPadding,
          GemmType kGemmType, bool kWithAccumulation, bool kFuseScalePack, bool kProfile,
          bool kTask06GatedDual, bool kTask07OneSm,
          typename a_dtype_t, typename b_dtype_t, typename cd_dtype_t,
          typename epilogue_type_t>
CUTLASS_GLOBAL void __launch_bounds__(kNumNonEpilogueThreads + kNumEpilogueThreads, 1)
sm100_fp8_fp4_gemm_1d1d_impl(int* grouped_layout,
                             uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                             const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                             const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                             const __grid_constant__ cute::TmaDescriptor tensor_map_sfa,
                             const __grid_constant__ cute::TmaDescriptor tensor_map_sfb,
                             const __grid_constant__ cute::TmaDescriptor tensor_map_cd,
                             // GLM-5.2 fused UE8M0 scale pack: raw f32 SF operands + strides
                             // (elements). Only read when kFuseScalePack; nullptr otherwise.
                             const float* __restrict__ sfa_raw = nullptr,
                             const float* __restrict__ sfb_raw = nullptr,
                             int sfa_s0 = 0, int sfa_s1 = 0,
                             int sfb_s0 = 0, int sfb_s1 = 0,
                             // Opt-in span phase-decomposition probe (nullptr in production;
                             // 8 u64 %globaltimer slots per CTA: 0 kernel-start,
                             // 1 first-TMA-issue, 2 last-TMA-issue, 3 first-MMA-issue,
                             // 4 last-MMA-complete, 5 epilogue-start, 6 epilogue-end,
                             // 7 CTA-end). Only read when kProfile.
                             unsigned long long* prof = nullptr) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::conditional_t<kNumMulticast == 1, cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm>;

    // C/D type: BF16 and FP32 are supported, with or without accumulation
    DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, float> or cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>, "Invalid C/D data dtype");

    // MMA Configs
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t UMMA_M = LAYOUT_AD_M * kNumMulticast;
    constexpr uint32_t UMMA_N = kSwapAB ? BLOCK_M : BLOCK_N;
    constexpr uint32_t UMMA_K = 32;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / (kIsMulticastOnA ? kNumMulticast: 1);
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N / (kIsMulticastOnA ? 1 : kNumMulticast);
    constexpr uint32_t kNumNSubtiles =
        kTask07OneSm ? BLOCK_N / LAYOUT_AD_M : 1;
    DG_STATIC_ASSERT(BLOCK_K == 128, "Invalid block K");
    DG_STATIC_ASSERT(BLOCK_K % UMMA_K == 0, "Block K must be divisible by UMMA K");
    DG_STATIC_ASSERT(kNumMulticast == 1 or kNumMulticast == 2, "Only support 1/2 multicast");
    DG_STATIC_ASSERT((kSwapAB and
                      (BLOCK_N == LAYOUT_AD_M or
                       (kTask07OneSm and BLOCK_N == 2 * LAYOUT_AD_M))) or
                     (not kSwapAB and (BLOCK_M == 32 or BLOCK_M == 64 or BLOCK_M == LAYOUT_AD_M)), "Invalid block size");
    DG_STATIC_ASSERT(not kTask06GatedDual or
                     (kSwapAB and kNumMulticast == 1 and
                      (BLOCK_M == 16 or BLOCK_M == 32) and BLOCK_N == 128 and BLOCK_K == 128 and
                      SHAPE_N == 2048 and SHAPE_K == 6144),
                     "Task 06 gated-dual is restricted to the exact GLM-5.2 shared-expert decode tiles");
    DG_STATIC_ASSERT(not kTask07OneSm or
                     (kSwapAB and kNumMulticast == 1 and
                      (BLOCK_M == 16 or BLOCK_M == 32) and
                      (BLOCK_N == 128 or BLOCK_N == 256) and
                      BLOCK_K == 128 and SHAPE_N == 6144 and SHAPE_K == 2048),
                     "Task 07 one-SM route is restricted to the exact GLM-5.2 shared-down decode tiles");

    // SF configs
    constexpr uint32_t kNumUTCCPAlignedElems = 128;
    constexpr uint32_t SF_BLOCK_M = math::constexpr_align(BLOCK_M, kNumUTCCPAlignedElems);
    constexpr uint32_t SF_BLOCK_N = math::constexpr_align(BLOCK_N, kNumUTCCPAlignedElems);
    constexpr uint32_t kNumSFAStagesPerLoad = kGranKA == 32 ? 1 : 4;
    constexpr uint32_t kNumSFBStagesPerLoad = kGranKB == 32 ? 1 : 4;
    DG_STATIC_ASSERT(kGranKA == 32 or kGranKA == 128, "Invalid granularity K for A");
    DG_STATIC_ASSERT(kGranKB == 32 or kGranKB == 128, "Invalid granularity K for B");
    DG_STATIC_ASSERT(not is_k_grouped_contiguous(kGemmType) or kGranKA == kGranKB, "K-grouped SF requires kGranKA == kGranKB");
    DG_STATIC_ASSERT(not is_k_grouped_contiguous(kGemmType) or kKAlignment % UMMA_K == 0, "K alignment must be divisible by UMMA K");

    // Epilogue configs
    // Always enable pipeline for better performance
    constexpr uint32_t kNumEpilogueStages = 2;
    constexpr uint32_t kNumTMAStoreStages = 2;
    // NOTES: To maximize epilogue threads utilization, process an entire BLOCK_N
    //        per store stage for swap-AB cases, and an entire BLOCK_M for non-swap cases
    constexpr uint32_t STORE_BLOCK_M =        kSwapAB ? 16      : cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t STORE_BLOCK_N =        kSwapAB ? (kTask07OneSm ? LAYOUT_AD_M : BLOCK_N) : kSwizzleCDMode / sizeof(cd_dtype_t);
    constexpr uint32_t kNumUMMAStoreThreads = kSwapAB ? kNumEpilogueThreads: STORE_BLOCK_M;
    DG_STATIC_ASSERT(kNumUMMAStoreThreads % 32 == 0, "Invalid store block M");

    // Share memory sizes
    constexpr uint32_t SMEM_CD_SIZE_PER_STAGE = STORE_BLOCK_M * STORE_BLOCK_N * sizeof(cd_dtype_t);
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(a_dtype_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(b_dtype_t);
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = SF_BLOCK_M * sizeof(uint32_t);
    constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = SF_BLOCK_N * sizeof(uint32_t);
    constexpr uint32_t kNumWeightStreams = kTask06GatedDual ? 2 : 1;
    DG_STATIC_ASSERT(SMEM_CD_SIZE % 1024 == 0 and SMEM_A_SIZE_PER_STAGE % 1024 == 0 and SMEM_B_SIZE_PER_STAGE % 1024 == 0, 
                     "Shared memory of A/B must be aligned to 1024 bytes");
    // NOTES: Make sure we have enough shared memory for UMMA padding
    constexpr uint32_t UMMA_A_SIZE_PER_STAGE = math::constexpr_align(LOAD_BLOCK_M, LAYOUT_AD_M) * BLOCK_K * sizeof(a_dtype_t);
    DG_STATIC_ASSERT(UMMA_A_SIZE_PER_STAGE <= SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE * kNumStages, "Memory Out of bound for UMMA");

    // Tensor memory size and offsets
    // The gated-dual route owns two disjoint accumulator regions.  Gate stages
    // occupy [0, 2*UMMA_N), and up stages occupy
    // [2*UMMA_N, 4*UMMA_N).  No BF16 [M,4096] intermediate exists.
    constexpr uint32_t kNumAccumTmemCols =
        UMMA_N * kNumEpilogueStages * kNumWeightStreams * kNumNSubtiles;
    constexpr uint32_t kNumSFATmemCols = SF_BLOCK_M / 32;
    constexpr uint32_t kNumSFBTmemColsPerStream = SF_BLOCK_N / 32;
    constexpr uint32_t kNumSFBTmemColsPerNSubtile = LAYOUT_AD_M / 32;
    constexpr uint32_t kNumSFBTmemCols = kNumSFBTmemColsPerStream * kNumWeightStreams;
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFATmemCols + kNumSFBTmemCols>();
    constexpr uint32_t kTmemStartColOfSFA = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFB = kNumAccumTmemCols + kNumSFATmemCols;
    constexpr uint32_t kTmemStartColOfSFBUp = kTmemStartColOfSFB + kNumSFBTmemColsPerStream;
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    // Synchronize the cluster before 2-CTA TMEM allocation
    kNumMulticast > 1 ? comm::cluster_sync_with_relaxed_arrive() : void();

    // Utils
    const bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto lane_idx = ptx::get_lane_idx();

    // Opt-in span phase probe: a single lane stamps %globaltimer (ns) into a per-CTA
    // slot. nullptr in production (uniform branch, no cost). Each slot is written by
    // exactly one warp to avoid cross-warp races; values across CTAs are aggregated
    // host-side into TMA / MMA / epilogue / tail phase spans.
    auto prof_mark = [&](uint32_t slot) {
        if constexpr (kProfile) {
            if (prof != nullptr and lane_idx == 0) {
                unsigned long long t;
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory");
                prof[static_cast<unsigned long long>(blockIdx.x) * 8ull + slot] = t;
            }
        }
    };
    if (warp_idx == 0) prof_mark(0);   // kernel-start

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == 0) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        if constexpr (not kFuseScalePack) {
            cute::prefetch_tma_descriptor(&tensor_map_sfa);
            cute::prefetch_tma_descriptor(&tensor_map_sfb);
        }
        cute::prefetch_tma_descriptor(&tensor_map_cd);
    }

    // Overwrite shape constants if the compiler gives
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;
    const auto shape_sfa_k = math::ceil_div(shape_k, kGranKA * 4);
    const auto shape_sfb_k = math::ceil_div(shape_k, kGranKB * 4);

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    // D/A/B shared memory
    auto smem_cd = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cd_dtype_t*>(smem_buffer + i * SMEM_CD_SIZE_PER_STAGE); 
    });
    auto smem_a  = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<a_dtype_t*>(smem_buffer + SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b  = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<b_dtype_t*>(smem_buffer + SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });
    auto smem_b_up = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<b_dtype_t*>(smem_buffer + SMEM_CD_SIZE +
            kNumStages * SMEM_A_SIZE_PER_STAGE +
            (kNumStages + i) * SMEM_B_SIZE_PER_STAGE);
    });

    // SFA/SFB shared memory
    auto sf_start_ptr = smem_buffer + SMEM_CD_SIZE +
                        kNumStages * SMEM_A_SIZE_PER_STAGE +
                        kNumWeightStreams * kNumStages * SMEM_B_SIZE_PER_STAGE;
    auto smem_sfa = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE);
    });
    auto smem_sfb = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE + i * SMEM_SFB_SIZE_PER_STAGE);
    });
    auto smem_sfb_up = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(sf_start_ptr +
            kNumStages * SMEM_SFA_SIZE_PER_STAGE +
            (kNumStages + i) * SMEM_SFB_SIZE_PER_STAGE);
    });

    // Barriers and tensor memory pointer
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(
        sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE +
        kNumWeightStreams * kNumStages * SMEM_SFB_SIZE_PER_STAGE);
    auto full_barriers          = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (i); });
    auto empty_barriers         = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages + i); });
    auto with_sf_full_barriers  = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 2 + i); });
    auto tmem_full_barriers     = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 3 + i); });
    auto tmem_empty_barriers    = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 3 + kNumEpilogueStages + i); });
    auto tmem_ptr_in_smem  = reinterpret_cast<uint32_t*>(barrier_start_ptr + kNumStages * 3 + kNumEpilogueStages * 2);

    // Initialize barriers
    if (warp_idx == 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            // Arrive at all CTAs
            full_barriers[i]->init(1);
            empty_barriers[i]->init(1);
            // Arrive only at the leader CTA
            with_sf_full_barriers[i]->init(kNumMulticast * 32);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
            // Arrive at all CTAs
            tmem_full_barriers[i]->init(1);
            // Arrive only at the leader CTA
            tmem_empty_barriers[i]->init(kNumMulticast * kNumUMMAStoreThreads);
        }

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        // Allocate tensor memory
        Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    kNumMulticast > 1 ? comm::cluster_sync_with_relaxed_arrive() : __syncthreads();

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = sched::Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumMulticast, kIsMulticastOnA, kNumSMs, kEnsureZeroPadding, kKAlignment, kGranKA * 4>(
        shape_m, shape_n, shape_k, grouped_layout);

    // Pipeline and TMA phases
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;

        // Flip phases only if reach the next first stage
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    // Dispatch warps into different roles
    if (warp_idx == 0 and cute::elect_one_sync()) {
        // TMA load warp
        // Persistently schedule over blocks
        bool first_tma = true;
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            // Use dynamic load block M, when swap-AB is enabled
            const auto load_block_m = kSwapAB ? scheduler.get_aligned_effective_m_in_block(m_block_idx) / kNumMulticast : LOAD_BLOCK_M;

            // For k-grouped layout, the number of block K is variable
            const auto num_total_k_blocks = math::ceil_div(scheduler.current_shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                // Wait consumer release
                empty_barriers[stage_idx]->wait(phase ^ 1);
                if (first_tma) { prof_mark(1); first_tma = false; }   // first-TMA-issue

                // Compute offsets
                // NOTES: the group is always concatenated with the outer dimension
                uint32_t m_idx = scheduler.template get_global_idx<(kGemmType == GemmType::MGroupedMasked), sched::IndexType::MN> (
                    shape_m, BLOCK_M, m_block_idx);
                uint32_t n_idx = scheduler.template get_global_idx<(kMajorB == cute::UMMA::Major::K), sched::IndexType::MN> (
                    shape_n, BLOCK_N, n_block_idx, m_block_idx);

                // NOTES: `k_idx` is actually the k index default for K-major, while `k_b_idx` may be MN-major
                // And for all m-grouped GEMMs, A must be K-majored
                DG_STATIC_ASSERT(kGemmType == GemmType::Normal or is_k_grouped_contiguous(kGemmType) or kGemmType == GemmType::Batched or
                                 kMajorA == cute::UMMA::Major::K, "Invalid major");
                uint32_t k_idx = k_block_idx * BLOCK_K;
                uint32_t k_a_idx = scheduler.template get_global_idx<(kMajorA == cute::UMMA::Major::MN), sched::IndexType::K> (
                    shape_k, BLOCK_K, k_block_idx, m_block_idx);
                uint32_t k_b_idx = scheduler.template get_global_idx<(kMajorB == cute::UMMA::Major::MN), sched::IndexType::K> (
                    shape_k, BLOCK_K, k_block_idx, m_block_idx);

                // Add 2 CTA offsets
                if constexpr (kNumMulticast > 1) {
                    m_idx += kIsMulticastOnA ? (cute::block_rank_in_cluster() * load_block_m) : 0;
                    n_idx += kIsMulticastOnA ? 0 : (cute::block_rank_in_cluster() * LOAD_BLOCK_N);
                }

                // Issue TMAs
                constexpr bool kIsBatchedMM = (kGemmType == GemmType::Batched);
                const uint32_t batch_idx = (kIsBatchedMM ? scheduler.current_group_idx : 0);
                if constexpr (kMajorA == cute::UMMA::Major::K)
                    tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t, kIsBatchedMM>(
                        &tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], k_a_idx, m_idx, 1, batch_idx);
                if constexpr (kMajorA == cute::UMMA::Major::MN)
                    tma::copy<LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode, a_dtype_t, kIsBatchedMM>(
                        &tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], m_idx, k_a_idx, 1, batch_idx);
                if constexpr (kMajorB == cute::UMMA::Major::K) {
                    if constexpr (kTask07OneSm and BLOCK_N == 256) {
                        // Reset the 128B-swizzle origin for each physical-M128
                        // UMMA subtile. Treating a single 256-row TMA tile as
                        // two independent UMMA descriptors rotates the second
                        // half by one row because its descriptor needs a fresh
                        // swizzle base.
                        #pragma unroll
                        for (uint32_t n_subtile = 0;
                             n_subtile < kNumNSubtiles;
                             ++n_subtile) {
                            tma::copy<
                                BLOCK_K, LAYOUT_AD_M,
                                kSwizzleBMode, b_dtype_t, kIsBatchedMM>(
                                &tensor_map_b, full_barriers[stage_idx],
                                smem_b[stage_idx] +
                                    n_subtile * LAYOUT_AD_M * BLOCK_K,
                                k_b_idx,
                                n_idx + n_subtile * LAYOUT_AD_M,
                                1, batch_idx);
                        }
                    } else {
                        tma::copy<
                            BLOCK_K, LOAD_BLOCK_N,
                            kSwizzleBMode, b_dtype_t, kIsBatchedMM>(
                            &tensor_map_b, full_barriers[stage_idx],
                            smem_b[stage_idx], k_b_idx, n_idx,
                            1, batch_idx);
                    }
                }
                if constexpr (kMajorB == cute::UMMA::Major::MN)
                    tma::copy<LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode, b_dtype_t, kIsBatchedMM>(
                        &tensor_map_b, full_barriers[stage_idx], smem_b[stage_idx], n_idx, k_b_idx, 1, batch_idx);
                if constexpr (kTask06GatedDual) {
                    // The merged production order is [gate(2048), up(2048)].
                    // Reuse the same activation tile and fetch the matching up
                    // rows into a second pipe owned by this CTA.
                    if constexpr (kMajorB == cute::UMMA::Major::K)
                        tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t, kIsBatchedMM>(
                            &tensor_map_b, full_barriers[stage_idx], smem_b_up[stage_idx],
                            k_b_idx, n_idx + shape_n, 1, batch_idx);
                    if constexpr (kMajorB == cute::UMMA::Major::MN)
                        tma::copy<LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode, b_dtype_t, kIsBatchedMM>(
                            &tensor_map_b, full_barriers[stage_idx], smem_b_up[stage_idx],
                            n_idx + shape_n, k_b_idx, 1, batch_idx);
                }
                auto num_arrival_bytes = SMEM_A_SIZE_PER_STAGE / (std::is_same_v<a_dtype_t, cutlass::float_e4m3_t> ? 1 : 2) +
                                         kNumWeightStreams * SMEM_B_SIZE_PER_STAGE /
                                             (std::is_same_v<b_dtype_t, cutlass::float_e4m3_t> ? 1 : 2);

                // Issue SFA and SFB TMAs at certain stages
                // No swizzling, so one TMA for one SF is enough
                if constexpr (kFuseScalePack) {
                    // GLM-5.2 fused UE8M0 pack: the SF smem is produced by warp 2 (the
                    // UTCCP transposer) directly from the raw f32 scale operands, using
                    // all 32 lanes and off THIS weight-stream producer's critical path.
                    // The producer only streams A/B: no SF TMA, no SF arrival bytes.
                } else {
                if (k_block_idx % kNumSFAStagesPerLoad == 0) {
                    uint32_t sfa_m_idx = m_block_idx * BLOCK_M;
                    uint32_t sfa_k_idx = scheduler.template get_global_idx<(not is_m_grouped_contiguous(kGemmType)), sched::IndexType::SF_K>(
                        shape_sfa_k, 1, math::ceil_div(k_idx, BLOCK_K * kNumSFAStagesPerLoad));
                    tma::copy<BLOCK_M, 1, 0>(&tensor_map_sfa, full_barriers[stage_idx], smem_sfa[stage_idx], sfa_m_idx, sfa_k_idx);
                    num_arrival_bytes += BLOCK_M * sizeof(uint32_t);
                }
                if (k_block_idx % kNumSFBStagesPerLoad == 0) {
                    uint32_t sfb_n_idx = n_block_idx * BLOCK_N;
                    uint32_t sfb_k_idx = scheduler.template get_global_idx<true, sched::IndexType::SF_K>(
                        shape_sfb_k, 1, math::ceil_div(k_idx, BLOCK_K * kNumSFBStagesPerLoad), m_block_idx);
                    tma::copy<BLOCK_N, 1, 0>(&tensor_map_sfb, full_barriers[stage_idx], smem_sfb[stage_idx], sfb_n_idx, sfb_k_idx);
                    num_arrival_bytes += BLOCK_N * sizeof(uint32_t);
                    if constexpr (kTask06GatedDual) {
                        tma::copy<BLOCK_N, 1, 0>(&tensor_map_sfb, full_barriers[stage_idx],
                                                smem_sfb_up[stage_idx],
                                                sfb_n_idx + shape_n, sfb_k_idx);
                        num_arrival_bytes += BLOCK_N * sizeof(uint32_t);
                    }
                }
                }

                // Arrive at full barriers
                full_barriers[stage_idx]->arrive_and_expect_tx(num_arrival_bytes);
            }
        }
        prof_mark(2);   // producer-end (all A/B TMAs issued)
    } else if (warp_idx == 1 and is_leader_cta) {
        // MMA issue warp
        // NOTES: only the leader CTA will do this
        // Make instruction descriptor
        auto instr_desc = kSwapAB ? cute::UMMA::make_instr_desc_block_scaled<b_dtype_t, a_dtype_t, float, cutlass::float_ue8m0_t,
                                                                             UMMA_M, UMMA_N, kMajorB, kMajorA>()
                                  : cute::UMMA::make_instr_desc_block_scaled<a_dtype_t, b_dtype_t, float, cutlass::float_ue8m0_t,
                                                                             UMMA_M, UMMA_N, kMajorA, kMajorB>();
        auto sf_desc = mma::sm100::make_sf_desc(nullptr);

        DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
        auto a_desc = mma::sm100::make_umma_desc<kMajorA, LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode>(smem_a[0], 0, 0);
        auto b_desc = mma::sm100::make_umma_desc<kMajorB, LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode>(smem_b[0], 0, 0);
        auto b_up_desc = mma::sm100::make_umma_desc<kMajorB, LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode>(smem_b_up[0], 0, 0);
        uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * SMEM_A_SIZE_PER_STAGE / 16 : 0u;
        uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16 : 0u;
        uint32_t b_up_desc_lo = lane_idx < kNumStages ? b_up_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16 : 0u;

        // Checks for MMA instructions
        // NOTES: CUTLASS does not have such checks except the MMA traits, but we are not using these traits
        DG_STATIC_ASSERT((UMMA_M == 64  and UMMA_N %  8 == 0 and  8 <= UMMA_N and UMMA_N <= 256) or
                         (UMMA_M == 128 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256) or
                         (UMMA_M == 256 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256),
                         "Invalid MMA instruction shape");

        // Persistently schedule over blocks
        bool first_mma = true;
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            // Wait tensor memory empty barrier arrival
            auto accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;
            auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;
            tmem_empty_barriers[accum_stage_idx]->wait(accum_phase_idx ^ 1);
            ptx::tcgen05_after_thread_sync();

            // Empty barrier arrival
            auto empty_barrier_arrive = [&](const bool& do_tmem_full_arrive) {
                auto umma_arrive = [](const uint64_t* barrier) {
                    if constexpr (kNumMulticast == 1) {
                        cutlass::arch::umma_arrive(barrier);
                    } else {
                        constexpr uint16_t kCTAMask = (1 << kNumMulticast) - 1;
                        cutlass::arch::umma_arrive_multicast_2x1SM(barrier, kCTAMask);
                    }
                };
                umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers[stage_idx]));

                // NOTES: the tensor memory accumulator pipeline has nothing to do with multicasting
                if (do_tmem_full_arrive)
                    umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barriers[accum_stage_idx]));
                __syncwarp();
            };

            // Dynamic update of UMMA N based on effective M, when swap-AB is enabled
            if constexpr (kSwapAB) {
                uint32_t umma_n = scheduler.get_aligned_effective_m_in_block(m_block_idx);
                mma::sm100::update_instr_desc_with_umma_n(instr_desc, umma_n);
            }

            // Launch MMAs
            if (first_mma) { prof_mark(3); first_mma = false; }   // first-MMA-issue
            const auto num_total_k_blocks = math::ceil_div(scheduler.current_shape_k, BLOCK_K);
            constexpr bool kMayHaveTailKBlock = is_k_grouped_contiguous(kGemmType) ? (kKAlignment % BLOCK_K != 0) : (SHAPE_K == 0 or SHAPE_K % BLOCK_K != 0);
            #pragma unroll 4
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                // Wait TMA and SF-transpose arrival
                with_sf_full_barriers[stage_idx]->wait(phase);
                ptx::tcgen05_after_thread_sync();

                const auto a_desc_base_lo = ptx::exchange(a_desc_lo, stage_idx);
                const auto b_desc_base_lo = ptx::exchange(b_desc_lo, stage_idx);
                const auto b_up_desc_base_lo = ptx::exchange(b_up_desc_lo, stage_idx);
                if (cute::elect_one_sync()) {
                    // Do SF copy at certain stages
                    // TODO: process shared memory descriptor by addition
                    using cute_utccp_t = cute::conditional_t<kNumMulticast == 1,
                        cute::SM100_UTCCP_4x32dp128bit_1cta, cute::SM100_UTCCP_4x32dp128bit_2cta>;
                    const uint32_t sfa_stage_in_group_idx = k_block_idx % kNumSFAStagesPerLoad;
                    if (sfa_stage_in_group_idx == 0) {
                        #pragma unroll
                        for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i) {
                            auto smem_ptr = smem_sfa[stage_idx] + i * kNumUTCCPAlignedElems;
                            mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                            cute_utccp_t::copy(sf_desc, kTmemStartColOfSFA + i * 4);
                        }
                    }
                    const uint32_t sfb_stage_in_group_idx = k_block_idx % kNumSFBStagesPerLoad;
                    if (sfb_stage_in_group_idx == 0) {
                        #pragma unroll
                        for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++ i) {
                            auto smem_ptr = smem_sfb[stage_idx] + i * kNumUTCCPAlignedElems;
                            mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                            cute_utccp_t::copy(sf_desc, kTmemStartColOfSFB + i * 4);
                            if constexpr (kTask06GatedDual) {
                                auto smem_up_ptr = smem_sfb_up[stage_idx] + i * kNumUTCCPAlignedElems;
                                mma::sm100::replace_smem_desc_addr(sf_desc, smem_up_ptr);
                                cute_utccp_t::copy(sf_desc, kTmemStartColOfSFBUp + i * 4);
                            }
                        }
                    }

                    // Issue UMMA
                    using mma_t = cute::conditional_t<
                        kNumMulticast == 1, ptx::SM100_MMA_MXF8F6F4_SS, ptx::SM100_MMA_MXF8F6F4_2x1SM_SS>;
                    auto issue_umma = [&]<uint32_t kUMMAKIdx>() {
                        constexpr uint32_t kOffset = kUMMAKIdx * UMMA_K;
                        const uint32_t sfa_id = (kGranKA == 32 ? kUMMAKIdx : sfa_stage_in_group_idx);
                        const uint32_t sfb_id = (kGranKB == 32 ? kUMMAKIdx : sfb_stage_in_group_idx);
                        const auto runtime_instr_desc = kSwapAB ?
                            mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, sfb_id, sfa_id):
                            mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, sfa_id, sfb_id);

                        a_desc.lo = mma::sm100::advance_umma_desc_lo<kMajorA, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(a_desc_base_lo, 0, kOffset);
                        b_up_desc.lo = mma::sm100::advance_umma_desc_lo<kMajorB, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(b_up_desc_base_lo, 0, kOffset);
                        if constexpr (kSwapAB) {
                            #pragma unroll
                            for (uint32_t n_subtile = 0;
                                 n_subtile < kNumNSubtiles;
                                 ++n_subtile) {
                                auto b_subtile_desc =
                                    mma::sm100::make_umma_desc<
                                        kMajorB, LAYOUT_AD_M, BLOCK_K,
                                        kSwizzleBMode>(
                                        smem_b[stage_idx] +
                                            n_subtile *
                                                LAYOUT_AD_M * BLOCK_K,
                                        0,
                                        kOffset);
                                const uint32_t accum_offset =
                                    accum_stage_idx * UMMA_N +
                                    n_subtile *
                                        kNumEpilogueStages * UMMA_N;
                                const uint32_t sfb_offset =
                                    kTmemStartColOfSFB +
                                    n_subtile *
                                        kNumSFBTmemColsPerNSubtile;
                                mma_t::fma(
                                    b_subtile_desc, a_desc, accum_offset,
                                    kUMMAKIdx > 0 or k_block_idx > 0,
                                    runtime_instr_desc,
                                    sfb_offset, kTmemStartColOfSFA);
                            }
                            if constexpr (kTask06GatedDual) {
                                constexpr uint32_t kUpAccumOffset =
                                    kNumEpilogueStages * UMMA_N *
                                    kNumNSubtiles;
                                mma_t::fma(
                                           b_up_desc, a_desc,
                                           kUpAccumOffset +
                                               accum_stage_idx * UMMA_N,
                                           kUMMAKIdx > 0 or k_block_idx > 0, runtime_instr_desc,
                                           kTmemStartColOfSFBUp, kTmemStartColOfSFA);
                            }
                        } else {
                            b_desc.lo = mma::sm100::advance_umma_desc_lo<kMajorB, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(b_desc_base_lo, 0, kOffset);
                            mma_t::fma(a_desc, b_desc, accum_stage_idx * UMMA_N,
                                       kUMMAKIdx > 0 or k_block_idx > 0, runtime_instr_desc,
                                       kTmemStartColOfSFA, kTmemStartColOfSFB);
                        }
                    };
                    auto issue_full_k_block = [&]() {
                        utils::for_each_static_until<BLOCK_K / UMMA_K>(std::make_integer_sequence<uint32_t, BLOCK_K / UMMA_K>(), issue_umma);
                    };

                    if constexpr (kMayHaveTailKBlock) {
                        auto issue_tail_k_block = [&](const uint32_t& remaining_k) {
                            const auto num_valid_umma_k = math::ceil_div(remaining_k, UMMA_K);
                            // Prefix expansion uses switch only for small cases to avoid long SASS.
                            utils::for_each_static_prefix(std::make_integer_sequence<uint32_t, BLOCK_K / UMMA_K>(), num_valid_umma_k, issue_umma);
                        };
                        const auto is_last_k_block = k_block_idx == num_total_k_blocks - 1;
                        if (is_last_k_block) {
                            const auto remaining_k = scheduler.current_shape_k - k_block_idx * BLOCK_K;
                            if (remaining_k < BLOCK_K)
                                issue_tail_k_block(remaining_k);
                            else
                                issue_full_k_block();
                        } else {
                            issue_full_k_block();
                        }
                    } else {
                        issue_full_k_block();
                    }
                }
                __syncwarp();

                // Commit to the mbarrier object
                // No explicit `tcgen05.fence::before_thread_sync` is needed, as this is implicitly performed by `tcgen05.commit`
                empty_barrier_arrive(k_block_idx == num_total_k_blocks - 1);
            }
        }

        // To safely deconstruct barriers, we need another round of waits
        const auto iter_idx = scheduler.current_iter - 1;
        if (kNumMulticast > 1 and iter_idx >= 0) {
            const auto accum_phase_idx = (iter_idx / kNumEpilogueStages) & 1;
            tmem_empty_barriers[iter_idx % kNumEpilogueStages]->wait(accum_phase_idx);
        }
        prof_mark(4);   // mma-end
    } else if (warp_idx == 2) {
        // UTCCP transposer
        auto utccp_required_smem_warp_transpose = [&](const uint32_t* smem_ptr) {
            DG_STATIC_ASSERT(kNumUTCCPAlignedElems == 128, "Invalid aligned elements");
            uint32_t values[4];
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++ i)
                values[i] = ptx::ld_shared(smem_ptr + i * 32 + lane_idx);
            __syncwarp();
            ptx::st_shared(smem_ptr + lane_idx * 4, values[0], values[1], values[2], values[3]);
        };

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            const auto num_total_k_blocks = math::ceil_div(scheduler.current_shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                // Wait TMA arrival
                full_barriers[stage_idx]->wait(phase);

                // GLM-5.2 fused UE8M0 pack: build the pre-transpose SF smem directly
                // from the raw f32 scale operands (byte-identical to what the SF TMA
                // would have produced), parallelised across this warp's 32 lanes. Each
                // N-tile's SFB is CTA-local, so no grid sync is needed; the existing
                // transpose + UTCCP consume the result unchanged. Doing this here (not
                // in the weight-stream producer) keeps the pack off warp 0's critical path.
                if constexpr (kFuseScalePack) {
                    const uint32_t base_m = m_block_idx * BLOCK_M;
                    // Activation SF is per-token (1:1 with rows). Frozen tiny-M path
                    // (BLOCK_M<=32): one row per lane, branchless (clamp the row index so a
                    // partial last M-block reads a valid, MMA-unused row instead of OOB;
                    // for M==BLOCK_M this is identical to the proven kernel's direct read).
                    // BLOCK_M>32: loop the extra rows the same way.
                    if (k_block_idx % kNumSFAStagesPerLoad == 0) {
                        const uint32_t kgrp = k_block_idx / kNumSFAStagesPerLoad;
                        auto pack_sfa_row = [&](uint32_t row) {
                            const uint32_t ar = min(base_m + row, shape_m - 1u);
                            const float* p = sfa_raw + static_cast<long>(ar) * sfa_s0
                                                     + static_cast<long>(4 * kgrp) * sfa_s1;
                            uint32_t acc = 0;
                            #pragma unroll
                            for (uint32_t t = 0; t < 4; ++ t)
                                acc |= ((__float_as_int(p[t * sfa_s1]) >> 23) & 0xFFu) << (8 * t);
                            smem_sfa[stage_idx][row] = acc;
                        };
                        if constexpr (BLOCK_M <= 32) {
                            if (lane_idx < BLOCK_M) pack_sfa_row(lane_idx);
                        } else {
                            for (uint32_t row = lane_idx; row < BLOCK_M; row += 32) pack_sfa_row(row);
                        }
                    }
                    // Weight SF is per-128-block. Frozen path is exactly BLOCK_N==128 (tile ==
                    // one weight scale block): pack once from `n_block_idx` and broadcast,
                    // identical to the proven kernel. Otherwise map each tile row to its
                    // per-128 block (wr>>7), correct for BLOCK_N<128 and >128; bound by shape_n.
                    if (k_block_idx % kNumSFBStagesPerLoad == 0) {
                        const uint32_t kgrp = k_block_idx / kNumSFBStagesPerLoad;
                        if constexpr (BLOCK_N == 128) {
                            const float* p = sfb_raw + static_cast<long>(n_block_idx) * sfb_s0
                                                     + static_cast<long>(4 * kgrp) * sfb_s1;
                            uint32_t acc = 0;
                            #pragma unroll
                            for (uint32_t t = 0; t < 4; ++ t)
                                acc |= ((__float_as_int(p[t * sfb_s1]) >> 23) & 0xFFu) << (8 * t);
                            #pragma unroll
                            for (uint32_t j = lane_idx; j < BLOCK_N; j += 32)
                                smem_sfb[stage_idx][j] = acc;
                        } else {
                            #pragma unroll
                            for (uint32_t j = lane_idx; j < BLOCK_N; j += 32) {
                                const uint32_t wr = n_block_idx * BLOCK_N + j;
                                uint32_t acc = 0;
                                if (wr < shape_n) {
                                    const float* p = sfb_raw + static_cast<long>(wr >> 7) * sfb_s0
                                                             + static_cast<long>(4 * kgrp) * sfb_s1;
                                    #pragma unroll
                                    for (uint32_t t = 0; t < 4; ++ t)
                                        acc |= ((__float_as_int(p[t * sfb_s1]) >> 23) & 0xFFu) << (8 * t);
                                }
                                smem_sfb[stage_idx][j] = acc;
                            }
                        }
                    }
                    __syncwarp();
                }

                // Transpose for UTCCP at certain stages
                if (k_block_idx % kNumSFAStagesPerLoad == 0) {
                    #pragma unroll
                    for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i)
                        utccp_required_smem_warp_transpose(smem_sfa[stage_idx] + i * kNumUTCCPAlignedElems);
                    // TODO: figure out whether the proxy fence is valid for 2-CTA cases
                    cutlass::arch::fence_view_async_shared();
                }
                if (k_block_idx % kNumSFBStagesPerLoad == 0) {
                    #pragma unroll
                    for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++ i) {
                        utccp_required_smem_warp_transpose(smem_sfb[stage_idx] + i * kNumUTCCPAlignedElems);
                        if constexpr (kTask06GatedDual)
                            utccp_required_smem_warp_transpose(smem_sfb_up[stage_idx] + i * kNumUTCCPAlignedElems);
                    }
                    // TODO: figure out whether the proxy fence is valid for 2-CTA cases
                    cutlass::arch::fence_view_async_shared();
                }

                // Arrive
                with_sf_full_barriers[stage_idx]->arrive(0u);
            }
        }
    } else if (warp_idx >= kNumNonEpilogueThreads / 32 and warp_idx < (kNumNonEpilogueThreads + kNumUMMAStoreThreads) / 32) {
        // Epilogue warp groups
        const auto epilogue_warp_idx = warp_idx - (kNumNonEpilogueThreads / 32);

        // NOTES: tensor memory addresses are simplified, as the hardware will ignore the warp index bits,
        // i.e., no need for `tmem_ptr |= (epilogue_warp_idx * 32) << 16`.
        // NOTES: we also forbid two CTAs to share the same SM and its tensor memory
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(tmem_ptr_in_smem) == 0);

        // Share store pipeline between blocks
        uint32_t tma_stage_idx = 0;

        // Persistently schedule over blocks
        bool first_store = true;
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            if (epilogue_warp_idx == 0 and first_store) { prof_mark(5); first_store = false; }   // epilogue-start (first store)
            auto accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;
            auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;

            // Wait UMMA arrival
            tmem_full_barriers[accum_stage_idx]->wait(accum_phase_idx);
            ptx::tcgen05_after_thread_sync();

            const auto tmem_base_addr = accum_stage_idx * UMMA_N;
            const auto base_m_idx = scheduler.template get_global_idx<(not is_m_grouped_contiguous(kGemmType)), sched::IndexType::MN>(shape_m, BLOCK_M, m_block_idx);
            const auto base_n_idx = n_block_idx * BLOCK_N;

            if constexpr (kSwapAB) {
                const auto effective_m = scheduler.get_aligned_effective_m_in_block(m_block_idx);
                if constexpr (kTask06GatedDual) {
                    constexpr uint32_t kUpAccumOffset =
                        kNumEpilogueStages * UMMA_N * kNumNSubtiles;
                    epilogue::sm100_store_swiglu_swap_ab<
                        BLOCK_M, BLOCK_N, STORE_BLOCK_M, STORE_BLOCK_N,
                        kSwizzleCDMode, kNumTMAStoreStages, kNumUMMAStoreThreads,
                        cd_dtype_t>
                    (smem_cd, tma_stage_idx, tmem_base_addr,
                     kUpAccumOffset + tmem_base_addr,
                     base_m_idx, base_n_idx, effective_m,
                     epilogue_warp_idx, lane_idx,
                     tmem_empty_barriers[accum_stage_idx],
                     tensor_map_cd);
                } else if constexpr (kTask07OneSm) {
                    #pragma unroll
                    for (uint32_t n_subtile = 0;
                         n_subtile < kNumNSubtiles;
                         ++n_subtile) {
                        const uint32_t subtile_tmem_base =
                            tmem_base_addr +
                            n_subtile *
                                kNumEpilogueStages * UMMA_N;
                        epilogue::sm100_store_cd_swap_ab<
                            BLOCK_M, BLOCK_N,
                            STORE_BLOCK_M, STORE_BLOCK_N,
                            kSwizzleCDMode, kNumTMAStoreStages,
                            kNumUMMAStoreThreads,
                            kGemmType, kWithAccumulation,
                            cd_dtype_t, epilogue_type_t>
                        (smem_cd, tma_stage_idx, subtile_tmem_base,
                         base_m_idx,
                         base_n_idx + n_subtile * LAYOUT_AD_M,
                         scheduler.current_group_idx,
                         effective_m,
                         epilogue_warp_idx, lane_idx,
                         tmem_empty_barriers[accum_stage_idx],
                         tensor_map_cd,
                         /*release_tmem=*/
                         n_subtile + 1 == kNumNSubtiles);
                    }
                } else {
                    epilogue::sm100_store_cd_swap_ab<
                        BLOCK_M, BLOCK_N, STORE_BLOCK_M, STORE_BLOCK_N,
                        kSwizzleCDMode, kNumTMAStoreStages, kNumUMMAStoreThreads,
                        kGemmType, kWithAccumulation,
                        cd_dtype_t, epilogue_type_t>
                    (smem_cd, tma_stage_idx, tmem_base_addr,
                     base_m_idx, base_n_idx, scheduler.current_group_idx,
                     effective_m,
                     epilogue_warp_idx, lane_idx,
                     tmem_empty_barriers[accum_stage_idx],
                     tensor_map_cd);
                }
            } else {
                epilogue::sm100_store_cd<
                    BLOCK_M, BLOCK_N, STORE_BLOCK_M, STORE_BLOCK_N,
                    kSwizzleCDMode, kNumTMAStoreStages, kNumUMMAStoreThreads,
                    kGemmType, kWithAccumulation,
                    cd_dtype_t, epilogue_type_t>
                (smem_cd, tma_stage_idx, tmem_base_addr,
                 base_m_idx, base_n_idx, scheduler.current_group_idx,
                 epilogue_warp_idx, lane_idx,
                 tmem_empty_barriers[accum_stage_idx],
                 tensor_map_cd);
            }
        }
        if (epilogue_warp_idx == 0) prof_mark(6);   // epilogue-end (all stores issued)
    }

    // TODO: Remove redundant synchronization
    kNumMulticast > 1 ? comm::cluster_sync_with_relaxed_arrive() : __syncthreads();

    // Deallocate tensor memory
    if (warp_idx == 0)
        Allocator().free(0, kNumTmemCols);

    if (warp_idx == 0) prof_mark(7);   // CTA-end

#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");
#endif
}

};  // namespace deep_gemm

#pragma clang diagnostic pop
