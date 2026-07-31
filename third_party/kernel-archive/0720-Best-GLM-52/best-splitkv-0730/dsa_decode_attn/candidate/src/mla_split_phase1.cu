// Instantiate the split-KV head64 sparse forward for d_qk = 576 (GLM-5.2 MLA).
#include "mla_split_phase1.cuh"

namespace sm100::fwd::head64 {
template void run_split_phase1_kernel<576>(const SparseAttnFwdParams&, int);
}
