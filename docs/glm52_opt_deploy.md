# GLM-5.2 optimized kernel deployment (sglang `glm52-opt` branch)

Single-process SGLang with **phase-aware** kernel dispatch: decode and prefill
use different optimized kernels in the same server process.

路径约定（相对仓库根，运行时由 `glm52_opt/config.py` 解析为绝对路径）：

| 用途 | 路径 |
|------|------|
| Manifest | `glm52_opt/manifest.json` |
| Kernel archive | `third_party/kernel-archive/0720-Best-GLM-52` |
| DeepGEMM fork | `third_party/DeepGEMM-GLM52` |
| Overlay tooling | `third_party/deepgemm_glm52` |

更多总览见仓库根目录 [`README_DGMK.md`](../README_DGMK.md)。

## Prerequisites

- B200 GPU node with SGLang venv matching your GLM-5.2 checkout
- Vendored kernel archive at `third_party/kernel-archive/0720-Best-GLM-52`
  (pinned in `glm52_opt/manifest.json`)
- DeepGEMM-GLM52 overlay built:

```bash
cd third_party/deepgemm_glm52
# optional: export HARNESS_PYTHON=/path/to/venv/bin/python
./build_overlay.sh   # uses third_party/DeepGEMM-GLM52 @ 41c6235
```

## Enable optimized kernels

```bash
export SGLANG_GLM52_OPT=1
export SGLANG_GLM52_OPT_PROFILE=decode_max   # or full (adds prefill winners)
export SGLANG_GLM52_DEEPGEMM_VARIANT=41c6235

python -m sglang.launch_server --model-path <glm-5.2-checkpoint> ...
```

### Profiles

| Profile | Decode | Prefill |
|---------|--------|---------|
| `serving_safe` (default) | Stock unless explicitly allowlisted | Stock |
| `decode_max` | All 8 legacy decode swaps | Stock |
| `full` | Same | + fused_qkv_a, q_b, index_k/q/weights |
| `e2e_candidates` | Default-off/allowlisted fixed-N/K diagnostics | Allowlisted fixed-N/K and MoE PSUM diagnostics |
| `hotspot_candidates` | External FlashMLA hook or built-in source-scoped W13/W2, exactly one op per process | Stock |

Prefill `moe_gate` and `dsa_prefill_attn` stay on stock (CUDA Graph regressions).
For the three exact packed-UE8M0 fixed-N/K candidates, single-op commands,
fair A/B rules, and `infini_kernel` Nsys labels, see
[`glm52_opt/infini_kernel_fixed_nk_e2e.md`](../glm52_opt/infini_kernel_fixed_nk_e2e.md).
For default-off PTX/SASS, CUDA/CuTe, CUTLASS, or Triton experiments at the
production FlashMLA sparse-decode call site, plus the built-in same-source
DeepGEMM W13/W2 end-to-end registrations, see
[`glm52_opt/infini_kernel_hotspot_e2e.md`](../glm52_opt/infini_kernel_hotspot_e2e.md).

## Smoke test

```bash
./scripts/glm52_opt_smoke.sh
```

For the source-scoped W13/W2 registrations, first build the matching artifact,
set the single-op `hotspot_candidates` environment documented in
[`glm52_opt/infini_kernel_hotspot_e2e.md`](../glm52_opt/infini_kernel_hotspot_e2e.md),
then run the matching stock launcher around:

```bash
python scripts/glm52_moe_registration_smoke.py --op w13  # or --op w2
```

This additionally verifies one real CUDA Graph capture/replay and reports the
bound stock/candidate identities.

## Validation checklist

1. Per-op harness gate (if Kernel-Harness is available):
   `Kernel-Harness/testbench/tasks/glm52/<op>_<phase>/run.sh --candidate <archive>`
2. Layer CUDA Graph bench (vendored archive or external Harness):
   `third_party/kernel-archive/0720-Best-GLM-52/llm_flops_style/bench_{decode,prefill}.py`
3. SGLang unit tests (when GPU available):
   `pytest test/registered/kernels/test_block_fp8_deep_gemm_blackwell.py -q`
4. Serving TTFT/TPOT vs `SGLANG_GLM52_OPT=0` baseline

## Architecture

- `python/sglang/srt/layers/glm52_opt/` — registry, phase detection, dispatch
- Hooks: `fp8_utils.py`, `deep_gemm_wrapper/entrypoint.py`, `dsa_backend.py`,
  `moe_runner/deep_gemm.py`, `model_runner._forward_raw`
- DeepGEMM fork loaded as `deep_gemm_experimental` (stock `deep_gemm` untouched)

### Backend routing (after wiring fix)

| Op | Decode | Prefill (`PROFILE=full`) |
|----|--------|--------------------------|
| fused_qkv_a | **archive** hechenxi Triton | **archive** `fused_qkv_a_prefill.py` |
| q_b | DeepGEMM fork `fp8_gemm_nt_fused` | **archive** `q_b_prefill.py` |
| o_proj | native packed UE8M0 | native packed UE8M0 |
| index_k / index_q | **archive** Triton | **archive** (k=decode winner; q=PR#3) |
| index_weights | fusion path / CUDA Graph mm | **CUDA Graph** `torch.mm` |
| moe w13/w2 | pack+PDL (gate tag) | up/down pack+PDL；**gate 保持 stock** |
| dsa_decode_attn | **archive** hechenxi trtllm-gen (non-trtllm backends) | n/a |

Verify: `./scripts/glm52_opt_route_check.sh`

## Maintenance

See `docs/glm52_opt_runbook.md` for rebase and upgrade workflow.
