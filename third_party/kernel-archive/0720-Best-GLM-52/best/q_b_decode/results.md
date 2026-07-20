# Results — q_b_decode DeepGEMM-GLM52 fork

## Verdict

**TARGET MET** — strictly **>35%** B200 HBM on both frozen decode shapes, via ONE
new DeepGEMM source mechanism (device-side fused UE8M0 scale pack). The mechanism
is correct for the full instantiable envelope (any `block_m`/`block_n`), not just
the frozen shapes.

Three idle-GPU CUPTI cold-L2 device-kernel-span gates (NVIDIA B200), medians
(fork commit `41c62355`; the shipped kernel is unchanged from `3b883898` — the
delta is only the `kProfile`-gated span phase probe + a diagnostic profiling entry):

| M | median cand µs | ceiling µs (35%) | median HBM | pass |
|---:|---:|---:|---:|:---:|
| 16 | 11.75 | `<12.185966` | 36.30% | yes |
| 32 | 11.87 | `<12.385280` | 36.52% | yes |

Per-gate (all three pass on both shapes):

| gate | M16 µs | M16 HBM | M32 µs | M32 HBM |
|---:|---:|---:|---:|---:|
| 1 | 11.77 | 36.24% | 11.87 | 36.52% |
| 2 | 11.72 | 36.39% | 11.95 | 36.27% |
| 3 | 11.75 | 36.30% | 11.86 | 36.55% |

All gates: correctness PASS, `calc_diff = 0` (bit-exact vs stock oracle),
post-timing recheck passed, performance gate MET, geomean speedup vs stock 2.87×.

**Envelope correctness.** `fp8_gemm_nt_fused` is bit-exact (`calc_diff = 0` vs
stock and vs the pre-pack path) for M ∈ {16, 32, 64, 128} — i.e. `BLOCK_M` ∈ {16,
32, 64, 128}, exercising `BLOCK_M > 32` where the round-0 kernel packed only SFA
rows 0–31. Host asserts bound the fused entry to the fp8-e4m3 / per-128-block /
dense envelope its in-kernel pack implements.

## Shared fork registration

| Field | Value |
|---|---|
| Fork root | `/home/qinhaiyan/DeepGEMM-GLM52` |
| Branch | `glm52-experiments` |
| Registered commit | `41c62355360c6ce07af1ca54eef612087a48f803` |
| Upstream base | `v0.1.4` / `731e7c7a97d269e4b9f482ea18d0e709a948f293` |
| Import alias | `deep_gemm_experimental` |
| Overlay | `overlays/41c62355.../site/deep_gemm_experimental` |
| JIT cache | `overlays/41c62355.../jit_cache` |
| Stock | untouched `sgl-deep-gemm==0.1.4` in Kernel-Harness `.venv` (mtime Jul 14, pre-campaign) |

## Mechanism (single new fork source change)

**Device-fused UE8M0 scale pack** — opt-in `fp8_gemm_nt_fused` entry.

*Why it was needed.* The op is a 32 MiB weight stream (`out[M,16384] =
x[M,2048] @ w[16384,2048].T`, FP8). The GEMM alone already streams at ~35.9%
HBM (M16 11.87 µs), but the mandatory per-call UE8M0 scale repack ran as a
**separate kernel** whose ~1.3–2 µs launch-overhead floor cannot be hidden
(pipeline depth caps pre-sync overlap), dragging the timed device-kernel span to
~29%. The pack budget to clear 35% is only ~0.3 µs — below any separate launch.
So >35% *requires* removing the separate pack, i.e. fusing it into the GEMM.

*What changed.* `fp8_gemm_nt_fused` takes the raw f32 per-128-block scales and
packs them to the UE8M0 SF layout **inside** the kernel. A first naive attempt
that packed in the single-lane weight-stream producer was *slower* (the cold f32
reads + smem writes landed on warp 0's critical path; M32 regressed to 23%). The
working design packs in **warp 2 (the UTCCP transposer)** — 32 lanes,
coalesced f32 reads, in the SF warp's existing slack and off the weight-stream
producer's path — writing the byte-identical pre-transpose SF smem the TMA would
have. Consumers (transpose + UTCCP + MMA) are unchanged, so `calc_diff = 0`.
Each N-tile's SFB is CTA-local, so no grid sync is needed. Existing entrypoints
are untouched (`kFuseScalePack` defaults false; SF-TMA path unchanged).

Files: `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh`,
`csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp`, `csrc/apis/gemm.hpp`,
`csrc/tvm_ffi_api.cpp`, `sgl_deep_gemm/__init__.py`.

## Diagnostics that drove the mechanism

- Span decomposition (CUPTI cold-L2): `full = pack(2.6µs) + gemm(11.9µs)`;
  `gemm_only` = 35.9% (M16) / 35.7% (M32) — the pack was the entire deficit.
- NCU (cold-L2, base clock): GEMM is DRAM-bound but only ~25% of DRAM peak —
  latency/overhead-bound, not BW-saturated (headroom exists; the wall is not
  physical here, unlike the extreme-HBM span-floor no-go cases).
- Separate-kernel launch floor (~1.3–2 µs) proves a separate pack can never fit
  the ~0.3 µs budget → fusion is mandatory.

## Artifacts

- Gates: `results/gates_fused/gate{1,2,3}.log`, `results/gates_fused/summary.json`
- Diagnostics: `diag/{diag_decompose,gemm_only,test_fused_correctness,time_fused}.py`
- Prior (29.3%) baseline gates: `results/gates/`
- Tooling: `llm/scripts/deepgemm_glm52/{build_overlay.sh,loader.py,smoke_dual.py}`

## Evidence closure (phase decomposition, per-gate CV, idle GPU, NCU)

- **Span phase decomposition (AC-3).** Opt-in kernel `%globaltimer` probe
  (`kProfile` template, compiled out of production so `prof=nullptr` calls are
  byte-identical to the shipped kernel) via `fp8_gemm_nt_prof`, which drives the
  SAME SM100 kernel for BOTH the baseline/pre-pack path (`kFuseScalePack=false`,
  SF via TMA) and the fused path (`kFuseScalePack=true`, SF packed in-kernel).
  Per-CTA timestamps aggregated over the launched CTAs (`diag/phase_decomp.py` →
  `diag/phase_decomp.json`). Timeline fields are ns relative to kernel-start;
  MMA fields are leader-CTA (2-CTA UMMA); `%globaltimer` step ~256 ns (read as
  ratios). Reference floors (µs, `diag/decompose.json`): pack-only 2.59/2.43,
  gemm-only 11.87/12.14, full 14.62/14.78 (M16/M32).

  | field (ns) | M16 baseline | M16 fused | M32 baseline | M32 fused |
  |---|---:|---:|---:|---:|
  | first_tma_ns | 768 | 512 | 768 | 640 |
  | last_tma_ns | 4608 | 5888 | 5120 | 6656 |
  | first_mma_ns | 768 | 512 | 512 | 512 |
  | last_mma_ns | 8448 | 8704 | 8448 | 9472 |
  | mma_window_ns | 7680 | 8192 | 7936 | 8960 |
  | epilogue_ns | 7680 | 8192 | 7936 | 8960 |
  | cta_tail_ns | 1280 | 768 | 768 | 256 |
  | total_span_ns | 9472 | 9728 | 9216 | 9984 |

  Reading it: both variants are ONE GEMM kernel differing only in the SF source,
  and their phase structure is nearly identical — the producer (TMA), MMA, and
  epilogue windows overlap for ~90% of the span (`first_mma≈first_tma≈0.5–0.8 µs`,
  `last_mma≈last_epilogue≈8.4–9.5 µs`); the only fixed serial slices are the
  prologue (kernel-start→first-TMA, ~0.5–0.8 µs) and the CTA tail (~0.25–1.3 µs),
  both small. The fused GEMM's `last_tma`/`last_mma` run ~0.3–1.0 µs later than the
  baseline GEMM (the extra warp-2 in-kernel pack), **but the baseline path also
  needs the SEPARATE pack kernel** (pack-only floor 2.4–2.6 µs) ahead of it.
  So **the pack-launch floor IS removed by fusion**: fused full = one kernel
  (≈gemm-only, gate 11.8–11.9 µs) vs baseline full = pack + gemm (the `full` floor,
  14.6–14.8 µs) — a ~2.5 µs / +6–7 HBM-point win, which is the whole >35% story.
  Next optimization target: none of the first/last TMA/MMA/epilogue/tail components
  is a large removable fixed cost — the span is the memory-bound producer/MMA/
  epilogue pipeline (mainloop ~7.7–9 µs of the ~9.5–10 µs). Further gains would need
  raising the weight-stream bandwidth itself (gemm-only ≈36%, NCU latency-bound),
  which also bounds the 40% stretch as unreachable.

## Evidence closure (per-gate CV, idle GPU, NCU)

- **Per-gate stability (CV).** From the three authoritative gate logs
  (median/lo/hi), per-shape CV estimate is **0.43–0.74%** — far tighter than the
  ~2–3% HBM margins, so the 3-gate median >35% is robust. A 200-sample cold-L2
  re-run gives median 11.62 µs (36.7%) M16 / 12.03 µs (36.0%) M32; full-sample CV
  (incl. cold-start tail) 2.8%/2.0%. Recorded in
  `results/gates_fused/cv_idle_evidence.json`.
- **Idle-GPU evidence.** `run_gates.sh` selects the lowest-memory GPU; snapshots
  before/after show the selected GPU at 0% util / 120 MHz idle pre-run and
  boost-pinned (1965 MHz) during measurement, with the other three GPUs untouched
  (`cv_idle_evidence.json`).
- **NCU on the fused SM100 kernel.** `results/ncu/qb_decode_fused_m16.ncu-rep`
  (+ `..._metrics.txt`). Kernel signature confirms the fused path
  (`sm100_fp8_fp4_gemm_1d1d_impl<... kFuseScalePack=1 ...>`, extra
  `const float*, const float*, int×4` SF params), grid (128,1,1). Cold-L2 base
  clock: DRAM 26% of peak (Compute 12.5%, L2 25%) — still latency/overhead-bound,
  i.e. the warp-2 fused pack adds negligible DRAM traffic vs the 25% pre-fuse
  baseline (confirms the pack is nearly free and off the weight-stream path).
- Byte model (fixed, from `glm52_ops.cost`): M16 34,120,704 B, M32 34,678,784 B
  (32 MiB weight dominates); HBM% = byte model / CUPTI median span ÷ 8.0e12.

## Rollback

```bash
cd /home/qinhaiyan/DeepGEMM-GLM52
git checkout glm52-experiments
git revert 41c62355360c6ce07af1ca54eef612087a48f803     # or reset --hard 0b39e972...
/home/qinhaiyan/KDA-Pilot-Exp/llm/scripts/deepgemm_glm52/build_overlay.sh
```

The candidate auto-falls back to the pre-pack `fp8_gemm_nt` path (proven ~29%)
if a loaded overlay lacks `fp8_gemm_nt_fused`. Stock Harness package never
overwritten.
