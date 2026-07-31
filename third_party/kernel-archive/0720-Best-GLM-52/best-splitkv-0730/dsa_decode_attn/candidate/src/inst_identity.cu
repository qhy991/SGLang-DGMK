// Unmodified upstream instantiation, so the JIT build path can be validated
// against installed sgl_kernel before any kernel change is made.
#include "sm100/prefill/sparse/fwd/head64/phase1.h"
#include "sm100/prefill/sparse/fwd/head64/phase1.cuh"

namespace sm100::fwd::head64 {
template void run_fwd_phase1_kernel<576>(const SparseAttnFwdParams& params);
}
