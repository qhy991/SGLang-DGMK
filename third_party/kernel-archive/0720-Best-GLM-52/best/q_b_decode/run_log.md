# Run log — q_b DeepGEMM fork campaign

- 2026-07-19: Cloned `sgl-project/DeepGEMM` @ `v0.1.4`, branch `glm52-experiments`.
- Built isolated overlay with Harness `.venv` python; stock `sgl-deep-gemm 0.1.4` untouched.
- Dual smoke PASS (path/version/setter isolation + numeric parity).
- Unmodified fork vs stock on q_b M16/M32: bitwise equal.
- Packed+fork candidate: harness WIN ~29% HBM.
- NCU M16 packed GEMM: DRAM ~35%, grid 148 vs 128 tiles.
- Fork commit `0b39e97`: clamp single-wave launch SMS → 128.
- Three idle gates PASS; medians M16=14.576 µs (29.26%), M32=14.768 µs (29.35%).
- Registered shared variant `glm52-qb-sm100-sms-clamp`.

## 2026-07-19 — >35% campaign (fused UE8M0 pack)

- Reproduced fork baseline; span decomposition (CUPTI cold-L2): full = pack(~2.6µs)
  + gemm(~11.9µs); `gemm_only` alone = 35.9% (M16) / 35.7% (M32). The separate
  per-call pack kernel was the entire deficit to 35%.
- NCU (cold-L2, base): GEMM DRAM-bound but ~25% of DRAM peak → latency/overhead-
  bound with headroom (not a physical BW wall for this shape).
- Separate-kernel launch floor (~1.3–2µs) > the ~0.3µs pack budget ⇒ >35% needs
  the pack fused into the GEMM (no separate launch).
- Fork commit `4c2c22f`: opt-in `fp8_gemm_nt_fused` — device-side UE8M0 pack done
  in **warp 2** (UTCCP transposer, 32 lanes, off the weight-stream producer path),
  byte-identical pre-transpose SF smem. Naive warp-0 producer version was slower
  (M32 regressed to 23%); warp-2 version is nearly free.
- Correctness bit-exact (`calc_diff=0` vs stock oracle, both shapes). Dual smoke PASS.
- Three idle-GPU gates PASS on both shapes: medians M16=11.87µs (35.93%),
  M32=12.11µs (35.80%); speedup vs stock 2.87×. Stock package untouched.

## 2026-07-20 — round-1 (envelope correctness + AC-3 decomposition)

- Fixed `fp8_gemm_nt_fused` correctness for the full instantiable envelope: SFA
  now packs every activation row (branchless clamp for BLOCK_M<=32, per-row loop
  for BLOCK_M>32); SFB maps each tile row to its per-128 weight scale block.
  Bit-exact vs stock+packed for M16/M32/M64/M128 (BLOCK_M>32 exercised). Host
  asserts bound the fp8-e4m3/per-128/dense envelope.
- A first SFA `if(ar<shape_m)` branch regressed M32 ~0.6µs (warp-2 branch delayed
  the SF→MMA barrier); same-GPU A/B isolated it; branchless `min()` clamp fixed it.
- Added AC-3 span phase decomposition via opt-in `%globaltimer` probe behind a
  compile-time `kProfile` (prof=nullptr compiles it out; production unperturbed):
  prologue ~2.8%, producer-TMA ~59%, drain ~31%, CTA-tail ~1-6%.
- Fork commit `3b88389`; overlay rebuilt; dual smoke PASS; three idle gates both
  shapes PASS: medians M16=11.81µs (36.11%), M32=11.92µs (36.37%); per-gate CV
  0.43-0.74%. Registry + docs refreshed to `3b88389`.


