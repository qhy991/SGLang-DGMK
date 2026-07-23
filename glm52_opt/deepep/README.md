# GLM-5.2 DeepEP Config frontier (goal-25)

**Disposition: EP4 diagnostic only. Not a production default.**

This directory records the goal-25 joint normal-mode Config that cleared the
EP4 full-region ≥3% paired gate (~**1.36×** median) on stock DeepEP binaries.
Source geometry candidates were rejected. SGLang runtime defaults remain
unchanged (`SGLANG_GLM52_OPT=0`, no `--deepep-config`).

## Artifact

| File | Role |
|---|---|
| `ep4_joint_config_frontier.json` | Exact EP4 joint Config (dispatch SMS24/send32 + combine SMS24/send16) |

Canonical SHA-256 (sorted keys, compact JSON):

`b0dc47051e10c17a76f6b68b00c76a8690241e154c10f04e7b7cd42ecd3bfe0a`

Evidence (kernel-harness worktree):

- `serving_native/evidence/25_deepep_dispatch_combine/summaries/glm52_goal25_joint_config_summary_20260722_d.md`
- `serving_native/evidence/25_deepep_dispatch_combine/ep8_config_external_acceptance.md`

## How to A/B on EP4 (diagnostic)

```bash
# stock
# (omit --deepep-config)

# candidate Config only — stock DeepEP extension
--deepep-config /path/to/sglang/glm52_opt/deepep/ep4_joint_config_frontier.json
```

Do **not** relabel EP4 results as EP8. EP4 used 64 local experts / prefill M8192;
production EP8 uses 32 local experts / prefill M4096.

## EP8 adaptation handoff

An agent adapting this for EP8 should:

1. Treat this JSON as a **seed / frontier reference**, not a drop-in EP8 enable.
2. Re-search on an exclusive 8×B200 allocation with production ABI (local M4096).
3. Keep dispatch/combine `num_sms` equal and even (`<= 32` on B200 half-SM policy).
4. Gate independently on dispatch, combine, and full-region paired p50 ≥ 1.03×,
   plus CUDA graph / overlap / checkpoint-backed server acceptance.
5. Prefer the harness scripts under
   `glm52-goal-runs/25-deepep_dispatch_combine/kernel-harness/serving_native/tools/`:
   - `run_deepep_ep8_config_search.sh`
   - `summarize_deepep_ep8_config_acceptance.py`
   - `run_deepep_ep8_config_acceptance.sh`
6. Materialize a new `ep8_*.json` here only after EP8 acceptance; do not overwrite
   this EP4 frontier file in place.

## Related leaf wins (goals 07 / 08)

| Goal | Leaf | In this branch |
|---|---|---|
| 07 | DeepGEMM BM16 (~1.06–1.09×) | overlap-contract unit test only; **no** process-global alignment enable |
| 08 | DeepGEMM PSUM (~1.064×) | opt-in `grouped_gemm_nt_f8f8bf16_contig(..., use_psum_layout=True)` kwargs + unit test; production callers still stock `{}` |
