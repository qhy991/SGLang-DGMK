#pragma once

#include "../utils/compatibility.hpp"

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
#include "../jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp"
#include "../jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp"
#include "../jit_kernels/impls/sm90_bf16_gemm.hpp"
#include "../jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp"
#include "../jit_kernels/impls/sm100_bf16_gemm.hpp"
#endif 

#include "../jit_kernels/impls/smxx_cublaslt.hpp"

#include "layout.hpp"

namespace deep_gemm::gemm {

static bool early_return(const int& m, const int &n, const int& k,
                         const torch::Tensor& d, const std::optional<torch::Tensor>& c) {
    // Do nothing if the problem is empty
    if (m == 0 or n == 0)
        return true;

    // Checks
    const bool is_cd_same = c.has_value() and c->data_ptr() == d.data_ptr();
    if (is_cd_same)
        DG_HOST_ASSERT(c->sizes() == d.sizes() and c->strides() == d.strides());
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat);
    if (c.has_value()) {
        check_major_type_cd(c.value());
        DG_HOST_ASSERT(d.scalar_type() == c.value().scalar_type());
    }

    // No accumulation
    if (k == 0) {
        if (not is_cd_same)
            c.has_value() ? d.copy_(c.value()) : d.zero_();
        return true;
    }

    // With accumulation, do copy before GEMM (assuming the GEMM kernel does not support different C/D)
    if (c.has_value() and not is_cd_same)
        d.copy_(c.value());
    return false;
}

static int check_k_grouped_args(const std::optional<std::vector<int>>& ks_cpu,
                                const torch::Tensor& grouped_layout,
                                const int& num_groups,
                                const bool& use_psum_layout,
                                const int& k_alignment,
                                const int& sum_k_if_ks_cpu_missing = 0) {
    DG_HOST_ASSERT(grouped_layout.is_contiguous());
    DG_HOST_ASSERT(grouped_layout.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(static_cast<int>(grouped_layout.numel()) == num_groups);

    if (ks_cpu.has_value() and not ks_cpu.value().empty()) {
        DG_HOST_ASSERT(static_cast<int>(ks_cpu.value().size()) == num_groups);
        int sum_k = 0;
        for (const auto k: ks_cpu.value()) {
            DG_HOST_ASSERT(k % k_alignment == 0);
            sum_k += k;
        }
        return sum_k;
    }
    DG_HOST_ASSERT(use_psum_layout);
    return sum_k_if_ks_cpu_missing;
}

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE

static void fp8_fp4_gemm_nt(const std::pair<torch::Tensor, torch::Tensor>& a,
                            const std::pair<torch::Tensor, torch::Tensor>& b,
                            const torch::Tensor& d,
                            const std::optional<torch::Tensor>& c,
                            std::optional<std::tuple<int, int, int>> recipe,
                            std::optional<std::tuple<int, int>> recipe_a,
                            std::optional<std::tuple<int, int>> recipe_b,
                            const std::string& compiled_dims,
                            const bool& disable_ue8m0_cast,
                            const bool& task06_one_sm = false) {
    // Shape must be `[M, K] @ [N, K].T`
    const auto major_a = get_major_type_ab(a.first);
    const auto major_b = get_major_type_ab(b.first);
    if (fp8_requires_k_major()) {
        DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);
        DG_HOST_ASSERT(major_b == cute::UMMA::Major::K);
    }

    // C/D must be N-major
    check_major_type_cd(d);

    // Type and shape checks
    const auto arch_major = device_runtime->get_arch_major();
    const auto [m , k ] = check_ab_fp8_fp4(a.first, major_a, arch_major);
    const auto [n , k_] = check_ab_fp8_fp4(b.first, major_b, arch_major);
    const auto [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat);

    // Early return for trivial cases
    if (early_return(m, n, k, d, c))
        return;

    // Transform SFA and SFB into compute-required layout
    const auto [sfa, sfb, gran_k_a, gran_k_b] = layout::transform_sf_pair_into_required_layout(
        a.second, b.second, m, n, k, recipe, recipe_a, recipe_b, std::nullopt, std::nullopt, disable_ue8m0_cast);

    // Dispatch into different implements
    if (arch_major == 9 and sfa.scalar_type() == torch::kFloat) {
        const int gran_n = recipe.has_value() ? std::get<1>(recipe.value()) : std::get<0>(recipe_b.value());
        if (gran_n == 1) {
            sm90_fp8_gemm_1d1d(a.first, sfa, b.first, sfb, c, d, m, n, k, major_a, major_b, compiled_dims);
        } else {
            const auto major_sfb = get_major_type_ab(sfb);
            sm90_fp8_gemm_1d2d(a.first, sfa, b.first, sfb, c, d, m, n, k, major_a, major_b, major_sfb, compiled_dims);
        }
    } else if (arch_major == 10 and sfa.scalar_type() == torch::kInt) {
        sm100_fp8_fp4_gemm_1d1d(a.first, sfa, b.first, sfb, c, d, m, n, k, gran_k_a, gran_k_b,
                                major_a, major_b, compiled_dims, std::nullopt,
                                /*fuse_scale_pack=*/false, /*prof=*/nullptr,
                                task06_one_sm);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture or scaling factor types");
    }
}

// Goal 06 route B1: an explicit, fail-closed one-SM feasibility entry.
// It consumes the already-packed production UE8M0 tensors and writes the
// caller-owned BF16 [M,4096] buffer.  It is intentionally not aliased to the
// normal API and cannot affect stock dispatch.
static void fp8_fp4_gemm_nt_task06_one_sm(
        const std::pair<torch::Tensor, torch::Tensor>& a,
        const std::pair<torch::Tensor, torch::Tensor>& b,
        const torch::Tensor& d) {
    DG_HOST_ASSERT(a.first.dim() == 2 and b.first.dim() == 2 and
                   a.second.dim() == 2 and b.second.dim() == 2 and
                   d.dim() == 2);
    const auto m = a.first.size(0);
    const auto k = a.first.size(1);
    const auto n = b.first.size(0);
    DG_HOST_ASSERT((m == 16 or m == 32) and n == 4096 and k == 6144);
    DG_HOST_ASSERT(b.first.size(1) == k and
                   d.size(0) == m and d.size(1) == n);
    DG_HOST_ASSERT(a.first.scalar_type() == torch::kFloat8_e4m3fn and
                   b.first.scalar_type() == torch::kFloat8_e4m3fn and
                   d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(a.second.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(b.second.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(a.first.is_contiguous() and b.first.is_contiguous() and
                   d.is_contiguous() and d.storage_offset() == 0);
    DG_HOST_ASSERT(a.second.size(0) == m and a.second.size(1) == k / 128 / 4 and
                   b.second.size(0) == n and b.second.size(1) == k / 128 / 4);
    DG_HOST_ASSERT(a.second.stride(0) == 1 and a.second.stride(1) == m and
                   b.second.stride(0) == 1 and b.second.stride(1) == n);
    fp8_fp4_gemm_nt(
        a, b, d, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt,
        /*compiled_dims=*/"mnk",
        /*disable_ue8m0_cast=*/false,
        /*task06_one_sm=*/true);
}

// Goal 06 route B2: a true gated dual GEMM over the production merged
// [gate(2048), up(2048)] weight. The dedicated SM100 kernel reuses each
// activation tile, keeps gate/up in disjoint TMEM accumulator regions, applies
// exact FP32 SwiGLU in its epilogue, and writes BF16 [M,2048]
// directly. This entry is deliberately not reachable from the stock API.
static void fp8_fp4_gemm_nt_task06_gated_dual(
        const std::pair<torch::Tensor, torch::Tensor>& a,
        const std::pair<torch::Tensor, torch::Tensor>& b,
        const torch::Tensor& d) {
    const auto major_a = get_major_type_ab(a.first);
    const auto major_b = get_major_type_ab(b.first);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and
                   major_b == cute::UMMA::Major::K);
    check_major_type_cd(d);

    const auto arch_major = device_runtime->get_arch_major();
    const auto [m, k] = check_ab_fp8_fp4(a.first, major_a, arch_major);
    const auto [merged_n, k_] = check_ab_fp8_fp4(b.first, major_b, arch_major);
    const auto [m_out, n_out] = get_shape<2>(d);
    DG_HOST_ASSERT(arch_major == 10);
    DG_HOST_ASSERT((m == 16 or m == 32) and k == 6144);
    DG_HOST_ASSERT(merged_n == 4096 and k_ == k);
    DG_HOST_ASSERT(m_out == m and n_out == 2048);
    DG_HOST_ASSERT(a.first.scalar_type() == torch::kFloat8_e4m3fn and
                   b.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(a.second.scalar_type() == torch::kInt and
                   b.second.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(a.first.is_contiguous() and b.first.is_contiguous() and
                   d.is_contiguous() and d.storage_offset() == 0);
    DG_HOST_ASSERT(a.second.size(0) == m and a.second.size(1) == k / 128 / 4 and
                   b.second.size(0) == merged_n and
                   b.second.size(1) == k / 128 / 4);
    DG_HOST_ASSERT(a.second.stride(0) == 1 and a.second.stride(1) == m and
                   b.second.stride(0) == 1 and
                   b.second.stride(1) == merged_n);

    // Preserve the production packed-int32 UE8M0 layout conversion/validation.
    // `merged_n`, not output N, owns the weight scale tensor.
    std::optional<std::tuple<int, int, int>> recipe = std::nullopt;
    const std::optional<std::tuple<int, int>> recipe_a = std::nullopt;
    const std::optional<std::tuple<int, int>> recipe_b = std::nullopt;
    const auto [sfa, sfb, gran_k_a, gran_k_b] =
        layout::transform_sf_pair_into_required_layout(
            a.second, b.second, m, merged_n, k,
            recipe, recipe_a, recipe_b,
            std::nullopt, std::nullopt,
            /*disable_ue8m0_cast=*/false);
    DG_HOST_ASSERT(sfa.scalar_type() == torch::kInt and
                   sfb.scalar_type() == torch::kInt);

    sm100_fp8_fp4_gemm_1d1d(
        a.first, sfa, b.first, sfb, std::nullopt, d,
        m, n_out, k, gran_k_a, gran_k_b,
        major_a, major_b,
        /*compiled_dims=*/"mnk",
        std::nullopt,
        /*fuse_scale_pack=*/false,
        /*prof=*/nullptr,
        /*task06_one_sm=*/false,
        /*task06_gated_dual=*/true);
}

// Goal 08 route B2: exact M4096 two-SM gated dual GEMM. The dedicated
// SM100 kernel reads the production packed scales and original
// [gate(2048), up(2048)] weight directly, reuses each activation tile for
// both MMA streams, retains the two FP32 accumulators in disjoint two-SM
// TMEM regions, and writes only the final BF16 [4096,2048] SwiGLU result.
// This entry is deliberately unreachable from the stock API.
static void fp8_fp4_gemm_nt_task08_gated_dual(
        const std::pair<torch::Tensor, torch::Tensor>& a,
        const std::pair<torch::Tensor, torch::Tensor>& b,
        const torch::Tensor& d) {
    const auto major_a = get_major_type_ab(a.first);
    const auto major_b = get_major_type_ab(b.first);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and
                   major_b == cute::UMMA::Major::K);
    check_major_type_cd(d);

    const auto arch_major = device_runtime->get_arch_major();
    const auto [m, k] = check_ab_fp8_fp4(a.first, major_a, arch_major);
    const auto [merged_n, k_] =
        check_ab_fp8_fp4(b.first, major_b, arch_major);
    const auto [m_out, n_out] = get_shape<2>(d);
    DG_HOST_ASSERT(arch_major == 10);
    DG_HOST_ASSERT(m == 4096 and k == 6144);
    DG_HOST_ASSERT(merged_n == 4096 and k_ == k);
    DG_HOST_ASSERT(m_out == m and n_out == 2048);
    DG_HOST_ASSERT(a.first.scalar_type() == torch::kFloat8_e4m3fn and
                   b.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(a.second.scalar_type() == torch::kInt and
                   b.second.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(a.first.is_contiguous() and b.first.is_contiguous() and
                   d.is_contiguous() and d.storage_offset() == 0);
    DG_HOST_ASSERT(
        a.second.size(0) == m and
        a.second.size(1) == k / 128 / 4 and
        b.second.size(0) == merged_n and
        b.second.size(1) == k / 128 / 4);
    DG_HOST_ASSERT(
        a.second.stride(0) == 1 and a.second.stride(1) == m and
        b.second.stride(0) == 1 and
        b.second.stride(1) == merged_n);

    // The exact packed int32 UE8M0 operands are already in the required
    // column-major scale layout. Pass them through without a transform,
    // expansion, repack, transpose, workspace, or helper launch.
    sm100_fp8_fp4_gemm_1d1d(
        a.first, a.second, b.first, b.second, std::nullopt, d,
        m, n_out, k,
        /*gran_k_a=*/128, /*gran_k_b=*/128,
        major_a, major_b,
        /*compiled_dims=*/"mnk",
        std::nullopt,
        /*fuse_scale_pack=*/false,
        /*prof=*/nullptr,
        /*task06_one_sm=*/false,
        /*task06_gated_dual=*/false,
        /*task08_gated_dual=*/true);
}

// GLM-5.2 fused UE8M0 scale pack: same math as `fp8_fp4_gemm_nt`, but the raw
// per-128-block f32 scale operands are packed to UE8M0 *inside* the GEMM kernel
// (device-side, per-CTA local) instead of by a separate pre-pass kernel. This
// removes the standalone scale-pack launch from the timed span for decode-shaped
// skinny FP8 GEMMs, whose span is otherwise dominated by that launch overhead.
// `a.second` / `b.second` are the raw float32 SF tensors (NOT pre-packed int32).
static void fp8_fp4_gemm_nt_fused(const std::pair<torch::Tensor, torch::Tensor>& a,
                                  const std::pair<torch::Tensor, torch::Tensor>& b,
                                  const torch::Tensor& d,
                                  const std::optional<torch::Tensor>& c,
                                  const std::string& compiled_dims,
                                  const std::optional<torch::Tensor>& prof = std::nullopt) {
    const auto major_a = get_major_type_ab(a.first);
    const auto major_b = get_major_type_ab(b.first);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);
    check_major_type_cd(d);

    const auto arch_major = device_runtime->get_arch_major();
    const auto [m, k]  = check_ab_fp8_fp4(a.first, major_a, arch_major);
    const auto [n, k_] = check_ab_fp8_fp4(b.first, major_b, arch_major);
    const auto [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(arch_major == 10 and "Fused scale pack is SM100-only");

    // Raw f32 per-128-block scales (1D1D, gran_k = 128), consumed directly by the kernel.
    // The in-kernel pack extracts UE8M0 exponents from fp8-e4m3 per-128-block scales, so
    // bound the fused entry to that envelope (fp4 / non-128 granularity is not supported
    // here). The kernel packs every SFA row and per-128 SFB block with shape bounds, so it
    // is correct for any block_m/block_n the SM100 dense heuristic selects.
    DG_HOST_ASSERT(a.first.scalar_type() == torch::kFloat8_e4m3fn and
                   b.first.scalar_type() == torch::kFloat8_e4m3fn and
                   "Fused scale pack supports fp8 e4m3 operands only");
    DG_HOST_ASSERT(a.second.scalar_type() == torch::kFloat and b.second.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(static_cast<int>(a.second.size(-1)) == k / 128 and
                   static_cast<int>(b.second.size(-1)) == k / 128 and
                   "Fused scale pack expects per-128-block (k/128) f32 scales");

    if (early_return(m, n, k, d, c))
        return;

    sm100_fp8_fp4_gemm_1d1d(a.first, a.second, b.first, b.second, c, d, m, n, k,
                            /*gran_k_a=*/128, /*gran_k_b=*/128,
                            major_a, major_b, compiled_dims, std::nullopt, /*fuse_scale_pack=*/true,
                            prof.has_value() ? reinterpret_cast<unsigned long long*>(prof->data_ptr()) : nullptr);
}

// GLM-5.2 diagnostic-only: profile the SM100 1D1D GEMM span phases for BOTH the
// baseline/pre-pack path (`fuse_scale_pack=false`, packed int32 SF via TMA) and the
// fused path (`fuse_scale_pack=true`, raw f32 SF packed in-kernel), threading the
// per-CTA %globaltimer probe. Not a production entry (always takes `prof`).
static void fp8_fp4_gemm_nt_prof(const std::pair<torch::Tensor, torch::Tensor>& a,
                                 const std::pair<torch::Tensor, torch::Tensor>& b,
                                 const torch::Tensor& d,
                                 const std::optional<torch::Tensor>& c,
                                 const std::string& compiled_dims,
                                 const torch::Tensor& prof,
                                 const bool& fuse_scale_pack) {
    const auto major_a = get_major_type_ab(a.first);
    const auto major_b = get_major_type_ab(b.first);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);
    check_major_type_cd(d);
    const auto arch_major = device_runtime->get_arch_major();
    const auto [m, k]  = check_ab_fp8_fp4(a.first, major_a, arch_major);
    const auto [n, k_] = check_ab_fp8_fp4(b.first, major_b, arch_major);
    const auto [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(arch_major == 10 and "Span phase probe is SM100-only");
    if (early_return(m, n, k, d, c))
        return;
    auto* prof_ptr = reinterpret_cast<unsigned long long*>(prof.data_ptr());
    // fused: raw f32 per-128 scales; baseline: pre-packed int32 SF (as the candidate
    // reference produces). Both drive the same SM100 kernel; only the SF source differs.
    sm100_fp8_fp4_gemm_1d1d(a.first, a.second, b.first, b.second, c, d, m, n, k,
                            /*gran_k_a=*/128, /*gran_k_b=*/128,
                            major_a, major_b, compiled_dims, std::nullopt,
                            fuse_scale_pack, prof_ptr);
}

static void fp8_fp4_gemm_nn(const std::pair<torch::Tensor, torch::Tensor>& a,
                            const std::pair<torch::Tensor, torch::Tensor>& b,
                            const torch::Tensor& d,
                            const std::optional<torch::Tensor>& c,
                            const std::optional<std::tuple<int, int, int>>& recipe,
                            const std::optional<std::tuple<int, int>>& recipe_a,
                            const std::optional<std::tuple<int, int>>& recipe_b,
                            const std::string& compiled_dims,
                            const bool& disable_ue8m0_cast) {
    fp8_fp4_gemm_nt(a, {b.first.transpose(0, 1), b.second.transpose(0, 1)},
                    d, c, recipe, recipe_a, recipe_b, compiled_dims, disable_ue8m0_cast);
}

static void fp8_fp4_gemm_tn(const std::pair<torch::Tensor, torch::Tensor>& a,
                            const std::pair<torch::Tensor, torch::Tensor>& b,
                            const torch::Tensor& d,
                            const std::optional<torch::Tensor>& c,
                            const std::optional<std::tuple<int, int, int>>& recipe,
                            const std::optional<std::tuple<int, int>>& recipe_a,
                            const std::optional<std::tuple<int, int>>& recipe_b,
                            const std::string& compiled_dims,
                            const bool& disable_ue8m0_cast) {
    fp8_fp4_gemm_nt({a.first.transpose(0, 1), a.second.transpose(0, 1)},
                    {b.first.transpose(0, 1), b.second.transpose(0, 1)},
                    d, c, recipe, recipe_a, recipe_b, compiled_dims, disable_ue8m0_cast);
}

static void fp8_fp4_gemm_tt(const std::pair<torch::Tensor, torch::Tensor>& a,
                            const std::pair<torch::Tensor, torch::Tensor>& b,
                            const torch::Tensor& d,
                            const std::optional<torch::Tensor>& c,
                            const std::optional<std::tuple<int, int, int>>& recipe,
                            const std::optional<std::tuple<int, int>>& recipe_a,
                            const std::optional<std::tuple<int, int>>& recipe_b,
                            const std::string& compiled_dims,
                            const bool& disable_ue8m0_cast) {
    fp8_fp4_gemm_nt({a.first.transpose(0, 1), a.second.transpose(0, 1)}, b,
                    d, c, recipe, recipe_a, recipe_b, compiled_dims, disable_ue8m0_cast);
}

static void m_grouped_fp8_fp4_gemm_nt_contiguous(const std::pair<torch::Tensor, torch::Tensor>& a,
                                                 const std::pair<torch::Tensor, torch::Tensor>& b,
                                                 const torch::Tensor& d,
                                                 const torch::Tensor& grouped_layout,
                                                 std::optional<std::tuple<int, int, int>> recipe,
                                                 std::optional<std::tuple<int, int>> recipe_a,
                                                 std::optional<std::tuple<int, int>> recipe_b,
                                                 const std::string& compiled_dims,
                                                 const bool& disable_ue8m0_cast,
                                                 const bool& use_psum_layout,
                                                 const bool& ensure_zero_padding,
                                                 const std::optional<int>& expected_m_for_psum_layout) {
    // Shape must be `[M, K] @ [G, N, K].mT`
    const auto major_a = get_major_type_ab(a.first);
    const auto major_b = get_major_type_ab(b.first);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);
    if (fp8_requires_k_major())
        DG_HOST_ASSERT(major_b == cute::UMMA::Major::K);
    DG_HOST_ASSERT(grouped_layout.is_contiguous());

    // Type and shape checks
    const auto arch_major = device_runtime->get_arch_major();
    const auto [m , k ] = check_ab_fp8_fp4(a.first, major_a, arch_major);
    const auto [num_groups, n, k_] = check_grouped_ab_fp8_fp4(b.first, major_b, arch_major);
    const auto [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(grouped_layout.scalar_type() == torch::kInt);

    // Layout checks
    if (use_psum_layout) {
        const auto [num_groups_] = get_shape<1>(grouped_layout);
        DG_HOST_ASSERT(num_groups == num_groups_);
    } else {
        const auto [m__] = get_shape<1>(grouped_layout);
        DG_HOST_ASSERT(m == m__);
        DG_HOST_ASSERT(not expected_m_for_psum_layout.has_value());
    }

    // D must be N-major
    check_major_type_cd(d);

    // Do nothing if empty
    if (m == 0)
        return;

    // Pass PSUM layout so SFA packing skips gap rows
    const std::optional<torch::Tensor> psum_sfa_layout = use_psum_layout ? std::make_optional(grouped_layout) : std::nullopt;
    const auto [sfa, sfb, gran_k_a, gran_k_b] = layout::transform_sf_pair_into_required_layout(
        a.second, b.second, m, n, k, recipe, recipe_a, recipe_b, std::nullopt, num_groups, disable_ue8m0_cast,
        psum_sfa_layout);

    // Dispatch implementation
    if (arch_major == 9 and sfa.scalar_type() == torch::kFloat) {
        const auto major_sfb = get_major_type_ab(sfb);
        sm90_m_grouped_fp8_gemm_contiguous_1d2d(a.first, sfa, b.first, sfb, d, grouped_layout,
                                                num_groups, m, n, k, major_a, major_b, major_sfb,
                                                compiled_dims, use_psum_layout, expected_m_for_psum_layout);
    } else if (arch_major == 10 and sfa.scalar_type() == torch::kInt) {
        sm100_m_grouped_fp8_fp4_gemm_contiguous_1d1d(a.first, sfa, b.first, sfb, d, grouped_layout,
                                                     num_groups, m, n, k, gran_k_a, gran_k_b, major_a, major_b,
                                                     compiled_dims, use_psum_layout, ensure_zero_padding, expected_m_for_psum_layout);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture or scaling factor types");
    }
}

static void m_grouped_fp8_fp4_gemm_nn_contiguous(const std::pair<torch::Tensor, torch::Tensor>& a,
                                                 const std::pair<torch::Tensor, torch::Tensor>& b,
                                                 const torch::Tensor& d,
                                                 const torch::Tensor& grouped_layout,
                                                 const std::optional<std::tuple<int, int, int>>& recipe,
                                                 const std::optional<std::tuple<int, int>>& recipe_a,
                                                 const std::optional<std::tuple<int, int>>& recipe_b,
                                                 const std::string& compiled_dims,
                                                 const bool& disable_ue8m0_cast,
                                                 const bool& use_psum_layout,
                                                 const bool& ensure_zero_padding) {
    m_grouped_fp8_fp4_gemm_nt_contiguous(a, {b.first.transpose(1, 2), b.second.transpose(1, 2)},
                                         d, grouped_layout, recipe, recipe_a, recipe_b, compiled_dims, disable_ue8m0_cast,
                                         use_psum_layout, ensure_zero_padding, std::nullopt);
}

static void m_grouped_fp8_fp4_gemm_nt_masked(const std::pair<torch::Tensor, torch::Tensor>& a,
                                             const std::pair<torch::Tensor, torch::Tensor>& b,
                                             const torch::Tensor& d,
                                             const torch::Tensor& masked_m,
                                             const int& expected_m,
                                             std::optional<std::tuple<int, int, int>> recipe,
                                             std::optional<std::tuple<int, int>> recipe_a,
                                             std::optional<std::tuple<int, int>> recipe_b,
                                             const std::string& compiled_dims,
                                             const bool& disable_ue8m0_cast) {
    // Shape must be `[G, M, K] @ [G, N, K].mT`
    const auto major_a = get_major_type_ab(a.first);
    const auto major_b = get_major_type_ab(b.first);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);
    DG_HOST_ASSERT(masked_m.is_contiguous());

    // Type and shape checks
    const auto arch_major = device_runtime->get_arch_major();
    const auto [num_groups  , m , k ] = check_grouped_ab_fp8_fp4(a.first, major_a, arch_major);
    const auto [num_groups_ , n , k_] = check_grouped_ab_fp8_fp4(b.first, major_b, arch_major);
    const auto [num_groups__, m_, n_] = get_shape<3>(d);
    const auto num_groups___ = static_cast<int>(masked_m.numel());
    DG_HOST_ASSERT(num_groups == num_groups_ and num_groups == num_groups__ and num_groups == num_groups___);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(expected_m > 0 and m > 0 and n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(masked_m.scalar_type() == torch::kInt);

    // D must be N-major
    check_major_type_cd(d);

    // Transform scaling factors
    const auto [sfa, sfb, gran_k_a, gran_k_b] = layout::transform_sf_pair_into_required_layout(
        a.second, b.second, m, n, k, recipe, recipe_a, recipe_b, num_groups, num_groups, disable_ue8m0_cast);

    // Dispatch implementation
    if (arch_major == 9 and sfa.scalar_type() == torch::kFloat) {
        const auto major_sfb = get_major_type_ab(sfb);
        sm90_m_grouped_fp8_gemm_masked_1d2d(a.first, sfa, b.first, sfb, d, masked_m,
                                            num_groups, m, n, k, expected_m, major_a, major_b, major_sfb, compiled_dims);
    } else if (arch_major == 10 and sfa.scalar_type() == torch::kInt) {
        sm100_m_grouped_fp8_fp4_gemm_masked_1d1d(a.first, sfa, b.first, sfb, d, masked_m,
                                                 num_groups, m, n, k, expected_m, gran_k_a, gran_k_b,
                                                 major_a, major_b, compiled_dims);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture or scaling factor types");
    }
}

static void k_grouped_fp8_gemm_tn_contiguous(const std::pair<torch::Tensor, torch::Tensor>& a,
                                             const std::pair<torch::Tensor, torch::Tensor>& b,
                                             const torch::Tensor& d,
                                             const std::optional<std::vector<int>>& ks_cpu,
                                             const torch::Tensor& grouped_layout,
                                             const std::optional<torch::Tensor>& c,
                                             const std::tuple<int, int, int>& recipe,
                                             const std::string& compiled_dims,
                                             const bool& use_psum_layout) {
    // Must be 1D1D kernel
    DG_HOST_ASSERT(std::get<0>(recipe) == 1 and std::get<1>(recipe) == 1);

    const int gran_k = std::get<2>(recipe);
    DG_HOST_ASSERT(gran_k == 32 or gran_k == 128);
    const int k_alignment = heuristics_runtime->get_mk_alignment_for_contiguous_layout();
    DG_HOST_ASSERT(k_alignment % 32 == 0);

    // Shape checks
    const auto [num_groups, m, n] = get_shape<3>(d);
    const auto [sum_k_ , m_] = get_shape<2>(a.first);
    const auto [sum_k__, n_] = get_shape<2>(b.first);

    const int sum_k = check_k_grouped_args(ks_cpu, grouped_layout, num_groups,
                                           use_psum_layout, k_alignment, static_cast<int>(a.first.size(0)));
    DG_HOST_ASSERT(m == m_ and n == n_ and sum_k == sum_k_ and sum_k == sum_k__);
    // Contiguity checks
    DG_HOST_ASSERT(a.first.is_contiguous());
    DG_HOST_ASSERT(b.first.is_contiguous());
    DG_HOST_ASSERT(d.is_contiguous());
    DG_HOST_ASSERT(c.has_value() and c.value().is_contiguous());

    // Early return for trivial cases
    if (early_return(m, n, sum_k, d, c))
        return;

    // Transform SF with padding
    const auto sfa = layout::transform_k_grouped_sf_into_required_layout(a.second, ks_cpu, grouped_layout, recipe, k_alignment, use_psum_layout);
    const auto sfb = layout::transform_k_grouped_sf_into_required_layout(b.second, ks_cpu, grouped_layout, recipe, k_alignment, use_psum_layout);

    // Dispatch implementation
    const auto arch_major = device_runtime->get_arch_major();
    if (arch_major == 10) {
        sm100_k_grouped_fp8_gemm_1d1d(a.first, sfa, b.first, sfb, c, d, m, n, grouped_layout, gran_k, k_alignment,
                                       cute::UMMA::Major::MN, cute::UMMA::Major::MN, compiled_dims, use_psum_layout);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

static void k_grouped_fp8_gemm_nt_contiguous(const std::pair<torch::Tensor, torch::Tensor>& a,
                                             const std::pair<torch::Tensor, torch::Tensor>& b,
                                             const torch::Tensor& d,
                                             const std::optional<std::vector<int>>& ks_cpu,
                                             const torch::Tensor& grouped_layout,
                                             const std::optional<torch::Tensor>& c,
                                             const std::tuple<int, int, int>& recipe,
                                             const std::string& compiled_dims,
                                             const bool& use_psum_layout) {
    // Must be 1D1D kernel
    DG_HOST_ASSERT(recipe == std::make_tuple(1, 1, 128));

    // No psum on FP8 NT
    DG_HOST_ASSERT(not use_psum_layout and ks_cpu.has_value() and not ks_cpu.value().empty());

    // Shape checks
    const auto [num_groups, m, n] = get_shape<3>(d);
    const auto sum_mk = a.first.numel();
    const auto sum_nk = b.first.numel();
    const int sum_k = check_k_grouped_args(ks_cpu, grouped_layout, num_groups,
                                           use_psum_layout, 128);
    DG_HOST_ASSERT(sum_mk == static_cast<int64_t>(sum_k) * m);
    DG_HOST_ASSERT(sum_nk == static_cast<int64_t>(sum_k) * n);

    // Contiguity checks
    DG_HOST_ASSERT(a.first.is_contiguous());
    DG_HOST_ASSERT(b.first.is_contiguous());
    DG_HOST_ASSERT(d.is_contiguous());
    DG_HOST_ASSERT(c.has_value() and c.value().is_contiguous());

    // Early return for trivial cases
    if (early_return(m, n, sum_k, d, c))
        return;

    // Transform SF with padding
    const auto sfa = layout::transform_k_grouped_sf_into_required_layout(a.second, ks_cpu, grouped_layout, recipe, 128, false);
    const auto sfb = layout::transform_k_grouped_sf_into_required_layout(b.second, ks_cpu, grouped_layout, recipe, 128, false);

    // Allocate tensormap buffer
    // `4` means the double buffering for both A and B operands (2 * 2)
    const auto num_sms = device_runtime->get_num_sms();
    const auto tensor_map_buffer = torch::empty({num_sms * 4 * static_cast<int>(sizeof(CUtensorMap))},
                                                a.first.options().dtype(torch::kByte));

    // Dispatch implementation
    const auto arch_major = device_runtime->get_arch_major();
    if (arch_major == 9) {
        sm90_k_grouped_fp8_gemm_1d1d(a.first, sfa, b.first, sfb, c, d, m, n, ks_cpu.value(), grouped_layout, tensor_map_buffer,
                                     cute::UMMA::Major::K, cute::UMMA::Major::K, compiled_dims);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}
#endif

#if DG_TENSORMAP_COMPATIBLE
static void bf16_gemm_nt(const torch::Tensor& a,
                         const torch::Tensor& b,
                         const torch::Tensor& d,
                         const std::optional<torch::Tensor>& c,
                         const std::string& compiled_dims) {
    // Shape must be `[M, K] @ [N, K].T`
    const auto major_a = get_major_type_ab(a);
    const auto major_b = get_major_type_ab(b);

    // C/D must be N-major
    check_major_type_cd(d);

    // Type and shape checks
    const auto [m , k ] = get_shape<2>(a);
    const auto [n , k_] = get_shape<2>(b);
    const auto [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(b.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat);

    // Early return for trivial cases
    if (early_return(m, n, k, d, c))
        return;

    // Dispatch into different implements
    const auto arch_major = device_runtime->get_arch_major();
    if (arch_major == 9) {
        sm90_bf16_gemm(a, b, c, d, m, n, k, major_a, major_b, compiled_dims);
    } else if (arch_major == 10) {
        sm100_bf16_gemm(a, b, c, d, m, n, k, major_a, major_b, compiled_dims);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

static void bf16_gemm_nn(const torch::Tensor& a,
                         const torch::Tensor& b,
                         const torch::Tensor& d,
                         const std::optional<torch::Tensor>& c,
                         const std::string& compiled_dims) {
    bf16_gemm_nt(a, b.transpose(0, 1), d, c, compiled_dims);
}

static void bf16_gemm_tn(const torch::Tensor& a,
                         const torch::Tensor& b,
                         const torch::Tensor& d,
                         const std::optional<torch::Tensor>& c,
                         const std::string& compiled_dims) {
    bf16_gemm_nt(a.transpose(0, 1), b.transpose(0, 1), d, c, compiled_dims);
}

static void bf16_gemm_tt(const torch::Tensor& a,
                         const torch::Tensor& b,
                         const torch::Tensor& d,
                         const std::optional<torch::Tensor>& c,
                         const std::string& compiled_dims) {
    bf16_gemm_nt(a.transpose(0, 1), b, d, c, compiled_dims);
}

static void m_grouped_bf16_gemm_nt_contiguous(const torch::Tensor& a, const torch::Tensor& b,
                                              const torch::Tensor& d, const torch::Tensor& grouped_layout,
                                              const std::string& compiled_dims,
                                              const bool& use_psum_layout,
                                              const bool& ensure_zero_padding,
                                              const std::optional<int>& expected_m_for_psum_layout) {
    // Shape must be `[M, K] @ [G, N, K].mT`
    const auto major_a = get_major_type_ab(a);
    const auto major_b = get_major_type_ab(b);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);
    DG_HOST_ASSERT(grouped_layout.is_contiguous());

    // Type and shape checks
    const auto [m, k] = get_shape<2>(a);
    const auto [num_groups, n, k_] = get_shape<3>(b);
    const auto [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(b.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(grouped_layout.scalar_type() == torch::kInt);

    // Layout checks
    if (use_psum_layout) {
        const auto [num_groups_] = get_shape<1>(grouped_layout);
        DG_HOST_ASSERT(num_groups == num_groups_);
    } else {
        const auto [m__] = get_shape<1>(grouped_layout);
        DG_HOST_ASSERT(m == m__);
        DG_HOST_ASSERT(not expected_m_for_psum_layout.has_value());
    }

    // D must be N-major
    check_major_type_cd(d);

    // Do nothing if empty
    if (m == 0)
        return;

    // Dispatch implementation
    const auto arch_major = device_runtime->get_arch_major();
    if (arch_major == 9) {
        sm90_m_grouped_bf16_gemm_contiguous(a, b, d, grouped_layout,
                                            num_groups, m, n, k, major_a, major_b, compiled_dims,
                                            use_psum_layout, expected_m_for_psum_layout);
    } else if (arch_major == 10) {
        sm100_m_grouped_bf16_gemm_contiguous(a, b, d, grouped_layout,
                                             num_groups, m, n, k, major_a, major_b, compiled_dims,
                                             use_psum_layout, ensure_zero_padding, expected_m_for_psum_layout);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

static void m_grouped_bf16_gemm_nn_contiguous(const torch::Tensor& a, const torch::Tensor& b,
                                              const torch::Tensor& d, const torch::Tensor& grouped_layout,
                                              const std::string& compiled_dims,
                                              const bool& use_psum_layout,
                                              const bool& ensure_zero_padding) {
    m_grouped_bf16_gemm_nt_contiguous(a, b.transpose(1, 2),
                                      d, grouped_layout, compiled_dims, use_psum_layout, ensure_zero_padding, std::nullopt);
}

static void m_grouped_bf16_gemm_nt_masked(const torch::Tensor& a, const torch::Tensor& b,
                                          const torch::Tensor& d, const torch::Tensor& masked_m,
                                          const int& expected_m, const std::string& compiled_dims) {
    // Shape must be `[G, M, K] @ [G, N, K].mT`
    const auto major_a = get_major_type_ab(a);
    const auto major_b = get_major_type_ab(b);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);
    DG_HOST_ASSERT(masked_m.is_contiguous());

    // Type and shape checks
    const auto [num_groups, m, k] = get_shape<3>(a);
    const auto [num_groups_, n, k_] = get_shape<3>(b);
    const auto [num_groups__, m_, n_] = get_shape<3>(d);
    const auto num_groups___ = static_cast<int>(masked_m.numel());
    DG_HOST_ASSERT(num_groups == num_groups_ and num_groups == num_groups__ and num_groups == num_groups___);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(expected_m > 0 and m > 0 and n > 0 and k > 0 and num_groups > 0);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(b.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(masked_m.scalar_type() == torch::kInt);

    // D must be N-major
    check_major_type_cd(d);

    // Dispatch implementation
    const auto arch_major = device_runtime->get_arch_major();
    if (arch_major == 9) {
        sm90_bf16_m_grouped_gemm_masked(a, b, d, masked_m,
                                        num_groups, m, n, k, expected_m, major_a, major_b, compiled_dims);
    } else if (arch_major == 10) {
        sm100_m_grouped_bf16_gemm_masked(a, b, d, masked_m,
                                         num_groups, m, n, k, expected_m, major_a, major_b, compiled_dims);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

static void k_grouped_bf16_gemm_tn_contiguous(const torch::Tensor& a,
                                              const torch::Tensor& b,
                                              const torch::Tensor& d,
                                              const std::optional<std::vector<int>>& ks_cpu,
                                              const torch::Tensor& grouped_layout,
                                              const std::optional<torch::Tensor>& c,
                                              const std::string& compiled_dims,
                                              const bool& use_psum_layout) {
    // Shape checks
    const auto [num_groups, m, n] = get_shape<3>(d);
    const auto [sum_k_ , m_] = get_shape<2>(a);
    const auto [sum_k__, n_] = get_shape<2>(b);

    const auto k_alignment = heuristics_runtime->get_mk_alignment_for_contiguous_layout();
    DG_HOST_ASSERT(k_alignment % 32 == 0);
    const int sum_k = check_k_grouped_args(ks_cpu, grouped_layout, num_groups,
                                           use_psum_layout, k_alignment, static_cast<int>(a.size(0)));
    DG_HOST_ASSERT(m == m_ and n == n_ and sum_k == sum_k_ and sum_k == sum_k__);

    // Contiguity checks
    DG_HOST_ASSERT(a.is_contiguous());
    DG_HOST_ASSERT(b.is_contiguous());
    DG_HOST_ASSERT(d.is_contiguous());
    DG_HOST_ASSERT(c.has_value() and c.value().is_contiguous());

    // Early return for trivial cases
    if (early_return(m, n, sum_k, d, c))
        return;

    // Dispatch implementation
    const auto arch_major = device_runtime->get_arch_major();
    if (arch_major == 9) {
        // No psum on SM90
        DG_HOST_ASSERT(not use_psum_layout and ks_cpu.has_value() and not ks_cpu.value().empty());
        sm90_bf16_k_grouped_gemm(a, b, c, d, m, n, ks_cpu.value(), grouped_layout,
                                 cute::UMMA::Major::MN, cute::UMMA::Major::MN, compiled_dims);
    } else if (arch_major == 10) {
        sm100_bf16_k_grouped_gemm(a, b, c, d, m, n, grouped_layout,
                                  cute::UMMA::Major::MN, cute::UMMA::Major::MN, compiled_dims, use_psum_layout);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}
#endif

static void cublaslt_gemm_nt(const torch::Tensor& a, const torch::Tensor& b,
                             const torch::Tensor& d, const std::optional<torch::Tensor>& c) {
    // Shape must be `[M, K] @ [N, K].T`
    const auto major_a = get_major_type_ab(a);
    const auto major_b = get_major_type_ab(b);

    // Type and shape checks
    const auto [m , k ] = get_shape<2>(a);
    const auto [n , k_] = get_shape<2>(b);
    const auto [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);

    // Early return for trivial cases
    if (early_return(m, n, k, d, c))
        return;

    cublaslt_gemm(a, b, d, m, n, k, major_a, major_b, c.has_value());
}

static void cublaslt_gemm_nn(const torch::Tensor& a, const torch::Tensor& b,
                             const torch::Tensor& d, const std::optional<torch::Tensor>& c) {
    cublaslt_gemm_nt(a, b.transpose(0, 1), d, c);
}

static void cublaslt_gemm_tn(const torch::Tensor& a, const torch::Tensor& b,
                             const torch::Tensor& d, const std::optional<torch::Tensor>& c) {
    cublaslt_gemm_nt(a.transpose(0, 1), b.transpose(0, 1), d, c);
}

static void cublaslt_gemm_tt(const torch::Tensor& a, const torch::Tensor& b,
                             const torch::Tensor& d, const std::optional<torch::Tensor>& c) {
    cublaslt_gemm_nt(a.transpose(0, 1), b, d, c);
}

#if 0

static void register_apis(pybind11::module_& m) {

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
    // FP8 FP4 GEMMs
    m.def("fp8_fp4_gemm_nt", &fp8_fp4_gemm_nt,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt, py::arg("recipe") = std::nullopt,
          py::arg("recipe_a") = std::nullopt, py::arg("recipe_b") = std::nullopt,
          py::arg("compiled_dims") = "nk",
          py::arg("disable_ue8m0_cast") = false);
    m.def("fp8_fp4_gemm_nt_fused", &fp8_fp4_gemm_nt_fused,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt,
          py::arg("compiled_dims") = "nk");
    m.def("fp8_fp4_gemm_nn", &fp8_fp4_gemm_nn,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt, py::arg("recipe") = std::nullopt,
          py::arg("recipe_a") = std::nullopt, py::arg("recipe_b") = std::nullopt,
          py::arg("compiled_dims") = "nk",
          py::arg("disable_ue8m0_cast") = false);
    m.def("fp8_fp4_gemm_tn", &fp8_fp4_gemm_tn,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt, py::arg("recipe") = std::nullopt,
          py::arg("recipe_a") = std::nullopt, py::arg("recipe_b") = std::nullopt,
          py::arg("compiled_dims") = "mn",
          py::arg("disable_ue8m0_cast") = false);
    m.def("fp8_fp4_gemm_tt", &fp8_fp4_gemm_tt,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt, py::arg("recipe") = std::nullopt,
          py::arg("recipe_a") = std::nullopt, py::arg("recipe_b") = std::nullopt,
          py::arg("compiled_dims") = "mn",
          py::arg("disable_ue8m0_cast") = false);
    m.def("m_grouped_fp8_fp4_gemm_nt_contiguous", &m_grouped_fp8_fp4_gemm_nt_contiguous,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("grouped_layout"),
          py::arg("recipe") = std::nullopt,
          py::arg("recipe_a") = std::nullopt, py::arg("recipe_b") = std::nullopt,
          py::arg("compiled_dims") = "nk",
          py::arg("disable_ue8m0_cast") = false,
          py::arg("use_psum_layout") = false,
          py::arg("ensure_zero_padding") = true,
          py::arg("expected_m_for_psum_layout") = std::nullopt);
    m.def("m_grouped_fp8_fp4_gemm_nn_contiguous", &m_grouped_fp8_fp4_gemm_nn_contiguous,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("grouped_layout"),
          py::arg("recipe") = std::nullopt,
          py::arg("recipe_a") = std::nullopt, py::arg("recipe_b") = std::nullopt,
          py::arg("compiled_dims") = "nk",
          py::arg("disable_ue8m0_cast") = false,
          py::arg("use_psum_layout") = false,
          py::arg("ensure_zero_padding") = true);
    m.def("m_grouped_fp8_fp4_gemm_nt_masked", &m_grouped_fp8_fp4_gemm_nt_masked,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("masked_m"),
          py::arg("expected_m"), py::arg("recipe") = std::nullopt,
          py::arg("recipe_a") = std::nullopt, py::arg("recipe_b") = std::nullopt,
          py::arg("compiled_dims") = "nk", py::arg("disable_ue8m0_cast") = false);
    m.def("k_grouped_fp8_gemm_tn_contiguous", &k_grouped_fp8_gemm_tn_contiguous,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("ks_cpu"), py::arg("grouped_layout"),
          py::arg("c") = std::nullopt,
          py::arg("recipe") = std::make_tuple(1, 1, 128),
          py::arg("compiled_dims") = "mn",
          py::arg("use_psum_layout") = false);
    m.def("k_grouped_fp8_gemm_nt_contiguous", &k_grouped_fp8_gemm_nt_contiguous,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("ks_cpu"), py::arg("grouped_layout"),
          py::arg("c") = std::nullopt,
          py::arg("recipe") = std::make_tuple(1, 1, 128),
          py::arg("compiled_dims") = "mn",
          py::arg("use_psum_layout") = false);

    // FP8 GEMM alias names
    m.attr("fp8_gemm_nt") = m.attr("fp8_fp4_gemm_nt");
    m.attr("fp8_gemm_nt_fused") = m.attr("fp8_fp4_gemm_nt_fused");
    m.attr("fp8_gemm_nn") = m.attr("fp8_fp4_gemm_nn");
    m.attr("fp8_gemm_tn") = m.attr("fp8_fp4_gemm_tn");
    m.attr("fp8_gemm_tt") = m.attr("fp8_fp4_gemm_tt");
    m.attr("m_grouped_fp8_gemm_nt_contiguous") = m.attr("m_grouped_fp8_fp4_gemm_nt_contiguous");
    m.attr("m_grouped_fp8_gemm_nn_contiguous") = m.attr("m_grouped_fp8_fp4_gemm_nn_contiguous");
    m.attr("m_grouped_fp8_gemm_nt_masked") = m.attr("m_grouped_fp8_fp4_gemm_nt_masked");
#endif

#if DG_TENSORMAP_COMPATIBLE
    // BF16 GEMMs
    m.def("bf16_gemm_nt", &bf16_gemm_nt,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt,
          py::arg("compiled_dims") = "nk");
    m.def("bf16_gemm_nn", &bf16_gemm_nn,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt,
          py::arg("compiled_dims") = "nk");
    m.def("bf16_gemm_tn", &bf16_gemm_tn,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt,
          py::arg("compiled_dims") = "mn");
    m.def("bf16_gemm_tt", &bf16_gemm_tt,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt,
          py::arg("compiled_dims") = "mn");
    m.def("m_grouped_bf16_gemm_nt_contiguous", &m_grouped_bf16_gemm_nt_contiguous,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("grouped_layout"),
          py::arg("compiled_dims") = "nk",
          py::arg("use_psum_layout") = false,
          py::arg("ensure_zero_padding") = true,
          py::arg("expected_m_for_psum_layout") = std::nullopt);
    m.def("m_grouped_bf16_gemm_nn_contiguous", &m_grouped_bf16_gemm_nn_contiguous,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("grouped_layout"),
          py::arg("compiled_dims") = "nk",
          py::arg("use_psum_layout") = false,
          py::arg("ensure_zero_padding") = true);
    m.def("m_grouped_bf16_gemm_nt_masked", &m_grouped_bf16_gemm_nt_masked,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("masked_m"),
          py::arg("expected_m"), py::arg("compiled_dims") = "nk");
    m.def("k_grouped_bf16_gemm_tn_contiguous", &k_grouped_bf16_gemm_tn_contiguous,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("ks_cpu"), py::arg("grouped_layout"),
          py::arg("c") = std::nullopt,
          py::arg("compiled_dims") = "mn",
          py::arg("use_psum_layout") = false);
#endif

    // cuBLASLt GEMMs
    m.def("cublaslt_gemm_nt", &cublaslt_gemm_nt,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("c") = std::nullopt);
    m.def("cublaslt_gemm_nn", &cublaslt_gemm_nn,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("c") = std::nullopt);
    m.def("cublaslt_gemm_tn", &cublaslt_gemm_tn,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("c") = std::nullopt);
    m.def("cublaslt_gemm_tt", &cublaslt_gemm_tt,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("c") = std::nullopt);
}

#endif

} // namespace deep_gemm::gemm
