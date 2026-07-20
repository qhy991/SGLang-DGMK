# DeepGEMM-GLM52 tooling

Shared experimental fork for GLM-5.2 Kernel-Harness campaigns.

## Layout

- Source: `/home/qinhaiyan/DeepGEMM-GLM52` (`glm52-experiments`)
- Overlays: `/home/qinhaiyan/DeepGEMM-GLM52/overlays/<commit>/`
- Stock: `/home/qinhaiyan/Kernel-Harness/.venv/.../site-packages/deep_gemm` (never overwritten)

## Commands

```bash
# Build commit-partitioned overlay (uses Harness venv python)
./build_overlay.sh

# Dual isolation smoke
CUDA_VISIBLE_DEVICES=0 /home/qinhaiyan/Kernel-Harness/.venv/bin/python ./smoke_dual.py

# In candidate code
from loader import load_deep_gemm_experimental
dg = load_deep_gemm_experimental()
dg.fp8_gemm_nt(...)
```
