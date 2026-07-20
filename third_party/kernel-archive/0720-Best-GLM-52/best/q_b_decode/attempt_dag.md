# Attempt ledger — DeepGEMM fork q_b

| ID | Change | Evidence | Outcome |
|---|---|---|---|
| A0 | Bootstrap fork at v0.1.4, overlay build, dual loader | smoke_dual PASS; stock path site-packages | keep |
| A1 | Unmodified fork vs stock on q_b inputs | bitwise equal, calc_diff=0 | keep |
| A2 | Candidate = UE8M0 pack + fork `fp8_gemm_nt` | harness WIN ~29% HBM | keep |
| A3 | NCU packed GEMM M16 | DRAM~35%, grid=148, 128 tiles | diagnose |
| A4 | Clamp single-wave `num_sms` to last_wave_util | config shows 128; 3-gate median ~29.3% HBM | keep |
| A5 | Fuse UE8M0 pack in producer warp 0 (single lane) | M16 flat ~29%, M32 regressed to 23% (pack on weight-stream critical path) | reject |
| A6 | Fuse UE8M0 pack in warp 2 (UTCCP transposer, 32 lanes) | bit-exact; 3-gate median M16=35.93% M32=35.80% HBM | keep (round-0) |
| A7 | Round-1: make fused pack correct for all block_m/n (SFA clamp+loop, SFB per-128 map), host asserts, opt-in %globaltimer phase probe (kProfile compiled out) | bit-exact M16/32/64/128; branchless clamp avoids the warp-2 branch that regressed M32; 3-gate median M16=36.11% M32=36.37% | **ship** |

Round-2 (>35%): span decomposition showed `gemm_only`=35.9%/35.7%; the separate
per-call pack kernel (~2.6µs, launch-floor-bound) was the whole deficit. Fusing it
into the GEMM (A6) removes the separate launch → >35% on both shapes. Round-1 (A7)
made the public fused entry generally correct (block_m>32) and added the AC-3 phase
decomposition; the `shape_m` branch it first introduced regressed M32 (warp-2 branch
delayed the MMA) and was replaced by a branchless clamp. Fork commit `3b88389`,
opt-in `fp8_gemm_nt_fused`.

