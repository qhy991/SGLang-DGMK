# GLM-5.2 kernel optimization runbook (dual-repo)

Maintain **sglang `glm52-opt`** + **DeepGEMM-GLM52** fork in parallel.

## Repositories

| Repo | Branch / variant | Role |
|------|------------------|------|
| `sglang` | `glm52-opt` | Phase-aware dispatch, SGLang hooks |
| `DeepGEMM-GLM52` | `41c6235` (`fp8_gemm_nt_fused`) | Decode q_b / fused_qkv_a GEMM |
| `Kernel-Harness` | `main` | Oracle + archive winners |

## Weekly rebase (sglang)

```bash
cd "${SGLANG_ROOT:?set SGLANG_ROOT to the repository checkout}"
git fetch origin
git checkout glm52-opt
git rebase origin/main
# resolve conflicts in fp8_utils.py, dsa_backend.py, model_runner.py if any
./scripts/glm52_opt_smoke.sh
```

## After a new harness WIN

1. Promote candidate to `Kernel-Harness/archive/0720-Best-GLM-52/best/`
2. Update `glm52_opt/manifest.json` `kernel_harness_commit`
3. Add or update entry in `python/sglang/srt/layers/glm52_opt/registry.py`
4. Run harness `run.sh` + `llm_flops_style/bench_*.py`
5. Commit sglang with manifest SHA in message

## DeepGEMM variant bump

```bash
cd "${DEEPGEMM_GLM52_ROOT:?set DEEPGEMM_GLM52_ROOT}"
git checkout <new-commit>
cd "${SGLANG_ROOT:?set SGLANG_ROOT}/third_party/deepgemm_glm52"
./build_overlay.sh
# update glm52_opt/manifest.json deepgemm_commit
export SGLANG_GLM52_DEEPGEMM_VARIANT=<short-sha>
./scripts/glm52_opt_smoke.sh
```

## Rollback

```bash
unset SGLANG_GLM52_OPT SGLANG_GLM52_DEEPGEMM_VARIANT
# server falls back to stock SGLang kernels immediately
```

## CI smoke (local)

```bash
cd "${SGLANG_ROOT:?set SGLANG_ROOT to the repository checkout}"
./scripts/glm52_opt_smoke.sh
cd third_party/kernel-archive/0720-Best-GLM-52/llm_flops_style
CUDA_VISIBLE_DEVICES=0 "${HARNESS_PYTHON:?set HARNESS_PYTHON}" bench_decode.py
```

Expected decode layer speedup ~1.44–1.51× vs stock (see `COMPARISON_TABLE.md`).
