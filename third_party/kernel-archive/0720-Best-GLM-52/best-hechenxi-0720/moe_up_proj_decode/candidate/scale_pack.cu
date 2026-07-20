// Fused per-call lossless repack of the 3D masked-grouped float32 UE8M0 scales into
// DeepGEMM's packed-int32 MN-major layout. BOTH operands, ALL E experts, in ONE
// kernel launch and ONE allocation — the CUPTI device span counts every kernel run()
// launches start-to-end, so a second launch or a host gap would widen the measured
// window (BL-fused-repack-single-launch / BL-single-kernel-span-floor). Stateless: the
// output buffer is freshly allocated each call and fully overwritten.
//
// The masked grouped GEMM with disable_ue8m0_cast=True wants the WEIGHT scale expanded
// to per-N-row (kernel assert sf.size(-2)==ceil_div(N,1)); the activation scale stays
// per-token (M rows). This kernel does the N-row expansion inline (row>>7 picks the
// block-scale row) so no (E,N,K) f32 tensor is ever materialized.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <vector>

// grid = (nxb + nwb, KP, E). blockIdx.z = expert e. The first nxb x-blocks pack the
// activation scale (M rows, 1:1); the rest pack the weight scale with N-row expansion.
// Each output int32 packs 4 consecutive K-block UE8M0 exponents little-endian. Output
// buffers are contiguous [E,KP,MN]; the caller views them transposed to (E,MN,KP),
// i.e. MN-contiguous (mn-major), exactly what DeepGEMM's packed path consumes.
__global__ void fused_pack_grouped_kernel(
        const float* __restrict__ xs, const float* __restrict__ ws,
        int* __restrict__ xbuf, int* __restrict__ wbuf,
        int M, int N, int KP, int nxb,
        long xs_se, long xs_sr, long xs_sk,
        long ws_se, long ws_sr, long ws_sk) {
    int e  = blockIdx.z;
    int kp = blockIdx.y;
    int b  = blockIdx.x;
    if (b < nxb) {
        int row = b * blockDim.x + threadIdx.x;
        if (row >= M) return;
        int acc = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float f = xs[(long)e * xs_se + (long)row * xs_sr + (long)(4 * kp + j) * xs_sk];
            acc |= ((__float_as_int(f) >> 23) & 0xFF) << (8 * j);
        }
        xbuf[(long)e * KP * M + (long)kp * M + row] = acc;
    } else {
        int row = (b - nxb) * blockDim.x + threadIdx.x;
        if (row >= N) return;
        int src = row >> 7;  // row / 128 -> block-scale row this weight row belongs to
        int acc = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float f = ws[(long)e * ws_se + (long)src * ws_sr + (long)(4 * kp + j) * ws_sk];
            acc |= ((__float_as_int(f) >> 23) & 0xFF) << (8 * j);
        }
        wbuf[(long)e * KP * N + (long)kp * N + row] = acc;
    }
}

// x_scale (E, M, K48) f32, w_scale (E, Nblk, K48) f32.
// Returns (xp [E,M,KP] mn-major int32, wp [E,N,KP] mn-major int32) with N = Nblk*128.
std::vector<torch::Tensor> pack_scales(torch::Tensor x_scale, torch::Tensor w_scale) {
    int E = x_scale.size(0);
    int M = x_scale.size(1);
    int Nblk = w_scale.size(1);
    int N = Nblk * 128;
    int K48 = x_scale.size(2);
    int KP = K48 / 4;  // 4 K-blocks packed per int32
    auto opt = torch::TensorOptions().dtype(torch::kInt32).device(x_scale.device());
    auto buf = torch::empty({(long)E * KP * (M + N)}, opt);
    int* xbuf = buf.data_ptr<int>();
    int* wbuf = xbuf + (long)E * KP * M;
    auto stream = at::cuda::getCurrentCUDAStream();
    const int BLK = 128;
    int nxb = (M + BLK - 1) / BLK;
    int nwb = (N + BLK - 1) / BLK;
    dim3 grid(nxb + nwb, KP, E);
    fused_pack_grouped_kernel<<<grid, BLK, 0, stream>>>(
        x_scale.data_ptr<float>(), w_scale.data_ptr<float>(), xbuf, wbuf,
        M, N, KP, nxb,
        x_scale.stride(0), x_scale.stride(1), x_scale.stride(2),
        w_scale.stride(0), w_scale.stride(1), w_scale.stride(2));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto xp = buf.narrow(0, 0, (long)E * KP * M).view({E, KP, M}).transpose(1, 2);
    auto wp = buf.narrow(0, (long)E * KP * M, (long)E * KP * N).view({E, KP, N}).transpose(1, 2);
    return {xp, wp};
}


__global__ void pack_x_kernel(const float* __restrict__ xs, int* __restrict__ xbuf, int M, int KP, long xs_se, long xs_sr, long xs_sk){ int e=blockIdx.z; int kp=blockIdx.y; int row=blockIdx.x*blockDim.x+threadIdx.x; if(row>=M) return; int acc=0; for(int j=0;j<4;++j){ float f=xs[(long)e*xs_se+(long)row*xs_sr+(long)(4*kp+j)*xs_sk]; acc|=((__float_as_int(f)>>23)&0xFF)<<(8*j); } xbuf[(long)e*KP*M+(long)kp*M+row]=acc; } torch::Tensor pack_x_scale(torch::Tensor x_scale){ int E=x_scale.size(0); int M=x_scale.size(1); int K48=x_scale.size(2); int KP=K48/4; auto opt=torch::TensorOptions().dtype(torch::kInt32).device(x_scale.device()); auto buf=torch::empty({(long)E*KP*M}, opt); int* xbuf=buf.data_ptr<int>(); auto stream=at::cuda::getCurrentCUDAStream(); const int BLK=128; int nxb=(M+BLK-1)/BLK; dim3 grid(nxb,KP,E); pack_x_kernel<<<grid,BLK,0,stream>>>(x_scale.data_ptr<float>(),xbuf,M,KP,x_scale.stride(0),x_scale.stride(1),x_scale.stride(2)); C10_CUDA_KERNEL_LAUNCH_CHECK(); return buf.view({E,KP,M}).transpose(1,2); }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("pack_x_scale", &pack_x_scale, "pack x scale");
    m.def("pack_scales", &pack_scales,
          "Fused pack of 3D masked-grouped f32 UE8M0 scales to packed int32 (x,w)");
}
