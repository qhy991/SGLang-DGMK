# GLM-5.2 `infini_kernel` fixed-N/K E2E candidates

## Scope

This branch registers three exact, default-off DeepGEMM candidates through the
existing SGLang `glm52_opt` dispatcher. Candidate and stock runs share the
same `Fp8LinearMethod` callsite, dynamic activation quantizer, FP8 tensors,
packed int32 UE8M0 scales, BF16 output, CUDA Graph path, and serving scheduler.
The candidate-side GEMM change is only:

```python
deep_gemm.fp8_gemm_nt(..., compiled_dims="nk")
```

| Registry op | Exact phase and local shape | Profiler identity |
|---|---|---|
| `index_q_upproj` | DECODE, M=16/32, N=4096, K=2048 | `infini_kernel_glm52_index_q_upproj_decode_nk` |
| `o_proj` | DECODE, M=16/32, N=6144, K=16384 | `infini_kernel_glm52_attn_o_decode_nk` |
| `fused_qkv_a_proj` | PREFILL, M=4096, N=2624, K=6144 | `infini_kernel_glm52_fused_qkv_a_prefill_nk` |

The indexer and QKV-A entries are explicit-only: an empty `OPT_OPS` does not
enable them. The existing `o_proj` E2E default remains M16/M32 only. The
registry also prevents the prefill MoE PSUM names from selecting their
archived decode kernels.

## Fail-closed selection

Before direct fixed-N/K dispatch, SGLang checks:

- the exact registry op and built-in M bucket;
- exact `ForwardMode.DECODE` for decode or `ForwardMode.EXTEND` for prefill;
  speculative, mixed, split-prefill, and target-verify modes do not select;
- exact local N/K and 128x128 block size;
- contiguous CUDA FP8 E4M3 activation and weight tensors;
- CUDA int32 packed UE8M0 activation and weight scales;
- exact packed scale shapes and column-major strides;
- one common device, BF16 output, and no bias.

Any mismatch occurs before selection and returns to the stock SGLang path.
After selection, DeepGEMM launch errors propagate; this implementation never
relabels an archive or stock fallback as a successful fixed-N/K candidate.

## Run one operator per process

Indexer Q up-projection:

```bash
export SGLANG_GLM52_OPT=1
export SGLANG_GLM52_OPT_PROFILE=e2e_candidates
export SGLANG_GLM52_OPT_OPS=index_q_upproj
export SGLANG_GLM52_OPT_M_BUCKETS='index_q_upproj:16|32'
export SGLANG_GLM52_ALLOW_ABI_ADAPTER=0
export SGLANG_GLM52_INFINI_KERNEL_NVTX=0
python -m sglang.launch_server <unchanged GLM-5.2 arguments>
```

Attention O projection:

```bash
export SGLANG_GLM52_OPT_OPS=o_proj
export SGLANG_GLM52_OPT_M_BUCKETS='o_proj:16|32'
```

Fused QKV-A prefill:

```bash
export SGLANG_GLM52_OPT_OPS=fused_qkv_a_proj
export SGLANG_GLM52_OPT_M_BUCKETS='fused_qkv_a_proj:4096'
```

Restart every server worker after changing the environment. Keep all remaining
checkpoint, topology, GPU, clock, request, and server arguments identical.

## Nsys naming

For profiler collection only, set:

```bash
export SGLANG_GLM52_INFINI_KERNEL_NVTX=1
nsys profile \
  --trace=cuda,nvtx \
  --cuda-graph-trace=node \
  --output=glm52-infini-kernel \
  python -m sglang.launch_server <unchanged arguments>
```

The selected launch is enclosed by an exact range such as:

```text
infini_kernel_glm52_index_q_upproj_decode_nk[M=16,N=4096,K=2048]
```

Summarize it with:

```bash
nsys stats \
  --report nvtx_kern_sum \
  --report cuda_gpu_kern_gb_sum:nvtx-name \
  glm52-infini-kernel.nsys-rep
```

The actual device symbol remains the generated
`deep_gemm::sm100_fp8_fp4_gemm...` name. Nsys renders the useful association as
`infini_kernel.../void deep_gemm::...`. This avoids changing the installed
DeepGEMM generator or adding a marker kernel whose extra launch would distort
5--25 microsecond decode kernels. Set the NVTX switch back to `0` for
authoritative A/B timing.

This path was revalidated on the public branch with B200 and Nsys 2025.6.3.
Both `nvtx_kern_sum` and `cuda_gpu_kern_gb_sum:nvtx-name` reported all five
exact M/N/K ranges and associated each one with its generated DeepGEMM device
kernel. Nsys 2025.6.3 does not expose the newer CUDA Graph NVTX projection
modifier, so on that version use capture-time association plus the first-hit
log rather than treating an unlabeled replay node as stock.

## Why these remain diagnostics

- Indexer direct fixed-N/K was bit-exact against stock in the local production
  ABI launch test. A same-process B200 full-apply sanity campaign measured
  0.952--0.971x eager but 1.247--1.250x CUDA Graph replay at M16/M32 under
  both 148-SM and 80-SM budgets. The older archived indexer fast path that had
  a numerical failure is a different implementation and remains disabled.
- Attention O fixed-N/K has positive direct and CUDA Graph evidence, while the
  integrated eager dispatcher has been neutral or regressive.
- Fused QKV-A measured 1.075--1.109x in eager full apply, but only 1.004x when
  the same quantize-plus-GEMM sequence was replayed in a CUDA Graph.

These results rank experiments; they do not establish a serving improvement.
The required promotion chain is:

```text
leaf eager
  -> leaf CUDA Graph replay
  -> containing attention/indexer region
  -> checkpoint-backed serving E2E
```

For each stage, alternate fresh-process A/B and B/A order, warm both JIT paths,
exclude compilation and graph capture, validate outputs first, retain the
first-hit log and profiler trace, and report median plus dispersion. The
indexer acceptance additionally compares downstream top-k indices. Promotion
requires a stable gain above 3% in the relevant containing-region/E2E bucket
without correctness, other-bucket, or SLA regression.

## Local validation

CPU-safe registry and dispatch-contract tests:

```bash
PYTHONPATH=python python - <<'PY'
import inspect
import runpy

ns = runpy.run_path("test/registered/kernels/test_glm52_opt_registry.py")
tests = [
    value
    for name, value in ns.items()
    if name.startswith("test_") and inspect.isfunction(value)
]
for test in sorted(tests, key=lambda fn: fn.__name__):
    test()
print(f"PASS total={len(tests)}")
PY
```

Opt-in B200 production-interface and graph test:

```bash
CUDA_VISIBLE_DEVICES=0 \
SGLANG_RUN_GLM52_INFINI_FIXED_NK_GPU_TEST=1 \
PYTHONPATH=python \
python test/registered/kernels/test_glm52_infini_fixed_nk.py -q
```

The local result was 14/14 CPU-safe checks plus all five GPU buckets bit-exact
against stock in eager execution and after CUDA Graph capture/replay.
