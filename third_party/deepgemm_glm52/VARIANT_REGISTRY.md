# Shared GLM-5.2 DeepGEMM variant registry

Only fork commits that pass the owning campaign's correctness + authoritative
gates are listed. Other campaigns must opt in explicitly by commit SHA.

## Registered variants

| Variant ID | Commit | Campaign | Notes |
|---|---|---|---|
| `glm52-qb-sm100-sms-clamp` | `0b39e972a2b5e3bc196df2a435eb6a4f5310ea75` | `glm52_harness_q_b_decode_deepgemm_fork` | Single-wave SM clamp; 3-gate >25% HBM |
| `glm52-qb-sm100-fused-ue8m0-pack` | `41c62355360c6ce07af1ca54eef612087a48f803` | `glm52_harness_q_b_decode_deepgemm_fork` | Opt-in `fp8_gemm_nt_fused`: device-side UE8M0 SF pack in warp 2, off weight-stream path. 3-gate >35% HBM (M16 36.1%, M32 36.4%); bit-exact for M16/32/64/128 (correct for all block_m/block_n). Additive/opt-in (includes A4 clamp). Supersedes round-0 `4c2c22f` (same mechanism, unsafe for block_m>32). |

## Opt-in

Point `llm/scripts/deepgemm_glm52/manifest.json` at the desired overlay
(rebuild with `build_overlay.sh` after `git checkout <commit>`), then load via
`deep_gemm_experimental`. Never silently swap stock `import deep_gemm`.

## Rollback

`glm52-qb-sm100-fused-ue8m0-pack` (`41c6235`) — revert to the prior 29% variant
(`0b39e97`) or to unmodified fork:

```bash
cd /home/qinhaiyan/DeepGEMM-GLM52
git checkout glm52-experiments
git revert --no-edit 41c62355360c6ce07af1ca54eef612087a48f803   # keeps A4 clamp (0b39e97)
# or hard reset past the fused commit:  git reset --hard 0b39e972a2b5e3bc196df2a435eb6a4f5310ea75
/home/qinhaiyan/KDA-Pilot-Exp/llm/scripts/deepgemm_glm52/build_overlay.sh   # rebuild overlay + refresh manifest
```

The opt-in fused path is additive (`kFuseScalePack` defaults false), so reverting
only removes `fp8_gemm_nt_fused`; all stock entrypoints are unaffected and the
q_b candidate auto-falls back to the pre-pack `fp8_gemm_nt` path. Stock
`import deep_gemm` in the Harness `.venv` is never touched by any variant.
