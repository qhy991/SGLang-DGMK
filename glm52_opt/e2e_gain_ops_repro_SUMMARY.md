# Reproduce: short/mid-context e2e_seq (prior gain scene)

- Out: `/home/ubuntu/wwxq/bench_results/e2e_gain_ops_repro_20260724_145934`
- Same as prior gain matrix: `serving_safe`, `ALLOW_ABI_ADAPTER=1`, M buckets 16|32
- Configs: opt0 / fused_qkv_a_proj / index_q_upproj / all_gain
- Focus: **TTFT ratio vs opt0** (prior win @ seq=2048 ≈0.91–0.93×)

| seq | config | TTFT p50 (s) | ×opt0 | TPOT p50 (ms) | ×opt0 | per_req tok/s | ×opt0 |
|----:|--------|-------------:|------:|--------------:|------:|--------------:|------:|
| 1024 | opt0 | 0.4040 | 1.0000 | 24.11 | 1.0000 | 42.70 | 1.0000 |
| 1024 | fused_qkv_a_proj | 0.4243 | 1.0501 | 24.19 | 1.0033 | 42.46 | 0.9943 |
| 1024 | index_q_upproj | 0.4120 | 1.0197 | 24.10 | 0.9999 | 42.59 | 0.9975 |
| 1024 | all_gain | 0.4163 | 1.0304 | 24.26 | 1.0062 | 42.54 | 0.9963 |
| 2048 | opt0 | 0.5165 | 1.0000 | 24.40 | 1.0000 | 40.66 | 1.0000 |
| 2048 | fused_qkv_a_proj | 0.5276 | 1.0215 | 24.50 | 1.0040 | 40.50 | 0.9962 |
| 2048 | index_q_upproj | 0.5136 | 0.9945 | 24.08 | 0.9869 | 41.19 | 1.0132 |
| 2048 | all_gain | 0.5817 | 1.1262 | 23.92 | 0.9803 | 40.45 | 0.9949 |
| 4096 | opt0 | 0.6392 | 1.0000 | 22.95 | 1.0000 | 41.74 | 1.0000 |
| 4096 | fused_qkv_a_proj | 0.6733 | 1.0533 | 22.65 | 0.9870 | 42.12 | 1.0089 |
| 4096 | index_q_upproj | 0.6542 | 1.0235 | 22.95 | 0.9999 | 41.91 | 1.0040 |
| 4096 | all_gain | 0.6563 | 1.0267 | 22.75 | 0.9913 | 42.16 | 1.0100 |

## Focus: seq=2048 TTFT (prior gain point)

- `fused_qkv_a_proj` TTFT ×opt0 = **1.0215** → flat/noise
- `index_q_upproj` TTFT ×opt0 = **0.9945** → flat/noise
- `all_gain` TTFT ×opt0 = **1.1262** → slower

## vs prior campaign (`e2e_gain_ops_full_metrics.md`)

| seq=2048 TTFT ×opt0 | prior | this repro |
|---------------------|------:|-----------:|
| fused_qkv_a_proj | **0.929** | 1.022 |
| index_q_upproj | **0.910** | 0.995 |
| all_gain | **0.927** | 1.126 |

**Prior TTFT gains at 2048 did not reproduce** under the same knobs (`serving_safe` + ABI adapter + M=16|32, conc=16). Current results are flat/noise or slower; TPOT remains ~±1–2%.

