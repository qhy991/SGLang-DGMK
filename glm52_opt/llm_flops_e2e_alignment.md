# e2e_candidates 官方评估（glm52-opt@12018ea99）

## 来源

- SGLang：`glm52-opt` @ `12018ea99`（`e2e_candidates` profile）
- INDEX：`glm52_opt/history/e2e_candidates_20260723/INDEX.md`
- Kernel-Harness：`harness-experience-bank`（拉取曾遇 HTTP2 失败；以 SGLang 归档为准）

## 测什么 / 不测什么

| 算子 | 动作 | 原因 |
|------|------|------|
| `o_proj` | **测** | goal-10 decode 图上 ~1.08–1.52×；eager 可能打平 |
| `moe_gate_proj` | **测** | goal-09 prefill PSUM ~1.05× |
| `moe_down_proj` | **测** | goal-08 prefill PSUM ~1.06× |
| `e2e_all` | **测** | 上述默认集合一起开 |
| `fused_qkv_a_proj` / `index_q_upproj` | **跳过** | 已官方 e2e，无收益 |
| q_b / indexer / dsa score / BM16 W2 decode / o_proj prefill | **跳过** | INDEX 标为 negative / 不安全 |

## 脚本

`run_e2e_candidates_official.sh` → `bench_results/e2e_candidates_official/`
