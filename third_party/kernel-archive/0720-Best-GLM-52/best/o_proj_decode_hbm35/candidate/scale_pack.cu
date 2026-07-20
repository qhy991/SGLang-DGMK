// Per-call lossless repack of float32 ue8m0 scales into DeepGEMM's packed-int32
// UE8M0 layout, both operands in ONE fused kernel launch and ONE allocation, to
// minimize host-launch gaps (which the CUPTI device-span timer counts). Stateless:
// the output buffer is freshly allocated each call and fully overwritten.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <vector>

// One launch packs both operands. grid = (nxb + nwb, KP); the first nxb row-blocks
// pack the activation (rows_per_blk=1 -> xbuf[KP,M]); the rest pack the weight
// (rows_per_blk=128 -> wbuf[KP,N], the N-expanded block scale). Each output int32
// packs 4 consecutive K-block ue8m0 exponents little-endian.
__global__ void fused_pack_kernel(const float* __restrict__ xs, const float* __restrict__ ws,
                                  int* __restrict__ xbuf, int* __restrict__ wbuf,
                                  int M, int N, int nxb,
                                  long xs_sr, long xs_sk, long ws_sr, long ws_sk) {
    int kp = blockIdx.y;
    int b = blockIdx.x;
    if (b < nxb) {
        int row = b * blockDim.x + threadIdx.x;
        if (row >= M) return;
        int acc = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float f = xs[(long)row * xs_sr + (long)(4 * kp + j) * xs_sk];
            acc |= ((__float_as_int(f) >> 23) & 0xFF) << (8 * j);
        }
        xbuf[(long)kp * M + row] = acc;
    } else {
        int row = (b - nxb) * blockDim.x + threadIdx.x;
        if (row >= N) return;
        int src = row >> 7;  // row / 128 (block-scale row this weight row belongs to)
        int acc = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float f = ws[(long)src * ws_sr + (long)(4 * kp + j) * ws_sk];
            acc |= ((__float_as_int(f) >> 23) & 0xFF) << (8 * j);
        }
        wbuf[(long)kp * N + row] = acc;
    }
}

// Returns (xp [M,KP] mn-major, wp [N,KP] mn-major).
std::vector<torch::Tensor> pack_scales(torch::Tensor x_scale, torch::Tensor w_scale) {
    int M = x_scale.size(0);
    int Nblk = w_scale.size(0);
    int N = Nblk * 128;
    int KP = x_scale.size(1) / 4;
    auto opt = torch::TensorOptions().dtype(torch::kInt32).device(x_scale.device());
    auto buf = torch::empty({(long)KP * (M + N)}, opt);
    int* xbuf = buf.data_ptr<int>();
    int* wbuf = xbuf + (long)KP * M;
    auto stream = at::cuda::getCurrentCUDAStream();
    const int BLK = 128;
    int nxb = (M + BLK - 1) / BLK;
    int nwb = (N + BLK - 1) / BLK;
    dim3 grid(nxb + nwb, KP);
    fused_pack_kernel<<<grid, BLK, 0, stream>>>(
        x_scale.data_ptr<float>(), w_scale.data_ptr<float>(), xbuf, wbuf,
        M, N, nxb, x_scale.stride(0), x_scale.stride(1), w_scale.stride(0), w_scale.stride(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto xp = buf.narrow(0, 0, (long)KP * M).view({KP, M}).t();
    auto wp = buf.narrow(0, (long)KP * M, (long)KP * N).view({KP, N}).t();
    return {xp, wp};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack_scales", &pack_scales, "Fused pack f32 ue8m0 scales to packed int32 (x,w)");
}
