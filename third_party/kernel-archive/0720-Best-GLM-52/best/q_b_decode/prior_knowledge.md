# Prior knowledge — q_b_decode DeepGEMM fork campaign

## Shared fork

- Source: `sgl-project/DeepGEMM` tag `v0.1.4` / `731e7c7`
- Local: `/home/qinhaiyan/DeepGEMM-GLM52` branch `glm52-experiments`
- Load alias: `deep_gemm_experimental` via `llm/scripts/deepgemm_glm52/loader.py`
- Stock reference remains `sgl-deep-gemm==0.1.4` in Kernel-Harness `.venv`

## Isolation

- Overlay builds never `pip install` into the harness venv.
- Per-commit JIT cache under `DeepGEMM-GLM52/overlays/<commit>/jit_cache`
- Dual smoke must pass before accepting a fork commit

## Related campaign

`glm52_harness_q_b_decode_hbm25` continues independently; this campaign must
not mutate its worktree.
