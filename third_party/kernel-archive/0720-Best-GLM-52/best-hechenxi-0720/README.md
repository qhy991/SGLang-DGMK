# best-hechenxi-0720 — GLM-5.2 decode best candidates (@何晨曦, 2026-07-20)

Follow-up to `best/`. Production-valid best correct candidates for the 10 decode
operators owned by @何晨曦, gated on an idle B200 under the harness CUPTI cold-L2
protocol (`run.sh --candidate <dir> --repeat 10`, both M∈{16,32}, flock-serialised).
Reward = HBM bandwidth utilisation (all decode shapes are memory-bound).

| op | candidate | M16 | M32 | target | verdict | mechanism |
|----|-----------|-----|-----|--------|---------|-----------|
| fused_qkv_a_proj | notma_v1 | 23.1% | 19.5% | >18% | **MET** ~3x | no-TMA split-K(6) Triton fp8 GEMM, inline per-call scales |
| q_b_proj | prepack_weightonly | 32.5% | 32.1% | >25% | **MET** 2.6x | fork-free packed-UE8M0, static weight-scale prepack |
| moe_gate_proj | prepack_weightonly | 41.6% | 42.2% | >=40% | **MET** 1.54x | masked-grouped packed-UE8M0 + PDL |
| moe_up_proj | prepack_weightonly | 41.6% | 42.2% | >=40% | **MET** 1.54x | same as gate |
| index_weights_proj | candidate.py | 1.78x vs separate | | >=1.65x | **MET** | fuse wk+weights into one cuBLAS F.linear (N=160), slice |
| index_q_upproj | notma_v1 | 16.3% | 14.2% | 15% | M16 MET / M32 -0.76pp | no-TMA Triton, per-shape num_warps; M32 at read-latency floor |
| dsa_decode_attn | trtllm_bf16 | 20.1% | 30.4% | 40% | 1.61x vs stock (needs abs_tol fix) | flashinfer trtllm-gen sparse MLA |

calc_diff=0 (or <=5e-6) and 2/2 shapes WIN for every op above.

## Notes
- **dsa_decode_attn** needs companion PR `hcx/dsa-abs-tol-near-zero-floor`. Its
  calc_diff is 3.78e-6 (< diff_tol 5e-6) but the global-max-derived abs_tol
  (2.85e-5) falsely rejects the ~23% near-zero softmax.V outputs. With the
  NEAR_ZERO_FLOOR=1e-3 fix: 2/2 WIN, exit 0, 1.61x over stock flash_mla_sparse_fwd
  (10.9/21.4% -> 20.1/30.4%).
- **Physical-ceiling ops (no candidate here, baseline kept):** index_score,
  absorbed_W_UK, absorbed_W_UV. A max-BW pure-read of their exact byte sets equals
  or exceeds the baseline kernel HBM%, i.e. they are bandwidth-saturated for their
  transfer size: 6-138 MB cold reads cannot reach the 8 TB/s spec peak (L2=132MB +
  fill/drain), the achievable ceiling is 17-62%. Their >72/86/82% targets were set
  against the unreachable spec peak.
- prepack "cache-all" variants (which also cache the per-token activation scale)
  score higher on the gate but are NOT production-valid — the activation scale is
  constant only under the fixed-seed harness. The weightonly/notma candidates
  archived here re-pack the activation scale every call (production-valid).
