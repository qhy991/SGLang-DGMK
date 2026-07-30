# GLM-5.2 W13 BM16 hotspot providers

This directory contains two default-off API-v1 `infini_kernel` providers for
the exact fused GLM-5.2 W13 decode ABI:

- `provider_bm16_2sm.py`: `(BM,BN,BK,stages,cluster-N)=(16,128,128,12,2)`
- `provider_bm16_1sm.py`: `(16,128,128,11,1)`

`build_variants.py` reconstructs stock from DeepGEMM
`731e7c7a97d269e4b9f482ea18d0e709a948f293` and candidate from the dedicated
task branch. It builds both extensions with the same clean command and stages
them under the task-local cache. Both DSOs use hidden C++ visibility plus
`-Wl,-Bsymbolic` so their JIT include-parser/compiler statics remain local
when stock and candidate are loaded side by side. It never installs or
overwrites a package.

CPU-only materialization audit:

```bash
CUDA_VISIBLE_DEVICES='' \
python3 third_party/deepgemm_w13/build_variants.py --audit-materialization
```

Clean host build:

```bash
CUDA_VISIBLE_DEVICES='' \
CUDA_HOME=/usr/local/cuda-13.2 \
MAX_JOBS=4 \
/home/qinhaiyan/miniconda3/envs/sglang/bin/python \
third_party/deepgemm_w13/build_variants.py --force
```

At startup the selected provider validates the manifest and DSO, binds its
private NVCC JIT cache, fixes PDL/SM/tensor-core state, and compiles all four
expected-M keys. The callback itself performs one current-stream launch,
mutates caller-owned output, and returns `None`.

Activation remains the stock API-v1 path:

```text
SGLANG_GLM52_OPT=1
SGLANG_GLM52_OPT_PROFILE=hotspot_candidates
SGLANG_GLM52_OPT_OPS=moe_w13
SGLANG_GLM52_OPT_M_BUCKETS=moe_gate_proj:16|32
SGLANG_GLM52_HOTSPOT_MODULE=<absolute provider_bm16_{2sm,1sm}.py>
```
