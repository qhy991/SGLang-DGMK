// Weight-only prepack: two entry points. Frozen weight scale packed ONCE (cached
// by the Python side); tiny per-token activation scale repacked every call into its
// own small buffer. Both emit DeepGEMM packed-int32 UE8M0 mn-major layout; each int32
// packs 4 consecutive K-block ue8m0 exponents little-endian.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

__global__ void pack_act_kernel(const float* __restrict__ xs, int* __restrict__ xbuf,
                                int M, long xs_sr, long xs_sk) {
    int kp = blockIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    int acc = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        float f = xs[(long)row * xs_sr + (long)(4 * kp + j) * xs_sk];
        acc |= ((__float_as_int(f) >> 23) & 0xFF) << (8 * j);
    }
    xbuf[(long)kp * M + row] = acc;
}

__global__ void pack_weight_kernel(const float* __restrict__ ws, int* __restrict__ wbuf,
                                   int N, long ws_sr, long ws_sk) {
    int kp = blockIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;
    int src = row >> 7;
    int acc = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        float f = ws[(long)src * ws_sr + (long)(4 * kp + j) * ws_sk];
        acc |= ((__float_as_int(f) >> 23) & 0xFF) << (8 * j);
    }
    wbuf[(long)kp * N + row] = acc;
}

torch::Tensor pack_weight(torch::Tensor w_scale, int64_t N_true) {
    int N = (int)N_true;
    int KP = w_scale.size(1) / 4;
    auto opt = torch::TensorOptions().dtype(torch::kInt32).device(w_scale.device());
    auto buf = torch::empty({(long)KP * N}, opt);
    auto stream = at::cuda::getCurrentCUDAStream();
    const int BLK = 128;
    int nwb = (N + BLK - 1) / BLK;
    dim3 grid(nwb, KP);
    pack_weight_kernel<<<grid, BLK, 0, stream>>>(
        w_scale.data_ptr<float>(), buf.data_ptr<int>(), N,
        w_scale.stride(0), w_scale.stride(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return buf.view({KP, N}).t();
}

torch::Tensor pack_act(torch::Tensor x_scale) {
    int M = x_scale.size(0);
    int KP = x_scale.size(1) / 4;
    auto opt = torch::TensorOptions().dtype(torch::kInt32).device(x_scale.device());
    auto buf = torch::empty({(long)KP * M}, opt);
    auto stream = at::cuda::getCurrentCUDAStream();
    const int BLK = 128;
    int nxb = (M + BLK - 1) / BLK;
    dim3 grid(nxb, KP);
    pack_act_kernel<<<grid, BLK, 0, stream>>>(
        x_scale.data_ptr<float>(), buf.data_ptr<int>(), M,
        x_scale.stride(0), x_scale.stride(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return buf.view({KP, M}).t();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack_weight", &pack_weight, "Pack frozen f32 ue8m0 weight scale to packed int32 (cached)");
    m.def("pack_act", &pack_act, "Pack per-call f32 ue8m0 activation scale to packed int32");
}
