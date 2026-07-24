# GLM-5.2 MoE contig PSUM 路径矫正 e2e 详细分析

- 日期：2026-07-24
- 机器：B300 `10.24.0.55` / Docker `sglang_0512`
- 原始数据：`/home/ubuntu/wwxq/bench_results/e2e_psum_contig_prefill/`
- 本地摘要副本：`glm52_opt/e2e_psum_contig_prefill_SUMMARY.md`

---

## 1. 「vs stock」是什么意思？

在本仓库 / 本次实验语境里：

| 说法 | 含义 |
|------|------|
| **stock** | SGLang **默认 / 生产已接线** 的实现：DeepGEMM 库自带的 `fp8_gemm_nt` / `fp8_m_grouped_gemm_nt_*`、FlashMLA、`fp8_paged_mqa_logits` 等，**不经过** `glm52_opt` 的 archive cubin 替换 |
| **candidate / OPT1 / swap** | `glm52_opt` 从 kernel-archive 换上的候选 kernel（或带 PSUM kwargs 的同一条 DeepGEMM API） |
| **vs stock** | 相对上述默认路径的加速比或延时比；**不是**相对 kernel-harness 里那套偏慢的 f32-scale 参考 |

容易混淆的对照：

1. **Harness gate「WIN」**：经常是 candidate vs **慢参考**（~1.5–3×），并不等于 vs 线上 stock。
2. **本次 PSUM A/B 的 OPT0**：两边都是 `deepep=normal`（contig），OPT0 = stock contig DeepGEMM；OPT1 = 同一条 contig 路径上注入 PSUM layout kwargs。这是 **fair A/B**，故意偏离生产默认的 `low_latency`+masked。
3. **生产默认（见下文默认路径文档）**：`deepep=low_latency` → MoE **masked**，根本不会走到 contig+PSUM。

---

## 2. 为什么要改配置重测？

先前 `e2e_candidates` 在生产默认 `deepep=low_latency` 下测 `moe_gate/down`：

- 目标收益来自 archive **prefill contig + PSUM ~1.05×**
- 实际 HIT 却是 **`moe_masked` decode**（非目标路径）
- 因此「无 e2e 收益」可能是 **打错路**，需要路径矫正后再下结论

矫正方式：

- 两侧统一 `SGLANG_DEEPEP_MODE=normal` → `use_masked_gemm=False` → `_run_contiguous_gemm` + `expert_start_loc` PSUM
- OPT0：`serving_safe`，无 swap
- OPT1：`e2e_candidates` + `moe_gate_proj,moe_down_proj`
- 场景：增量 prefill，`S=65536`，`M∈{1024,2048,4096}`，DP8 batch=8

并增加 HIT 键 `moe_contig_psum:*`，确认打中 PSUM。

---

## 3. 结果总表

| M | opt0 TTFT (s) | moe_psum TTFT (s) | ratio (opt1/opt0，&lt;1 更快) | cache_hit | 判定 |
|--:|---:|---:|---:|---:|:---|
| 1024 | 2.5311 | 2.5889 | **1.0228** | 0.9846 / 0.9846 | 有效；略慢 ~2.3% |
| 2048 | 4.7052 | 4.7841 | **1.0168** | 0.9697 / 0.9697 | 有效；略慢 ~1.7% |
| 4096 | 9.1157 | 147.8153 | 16.22 | 0.9412 / **0.0000** | **无效**（OPT1 未命中 cache） |

有效点结论：**路径对了，但 e2e TTFT 没有收益，反而慢约 1.7–2.3%。**

---

## 4. HIT 校验（路径是否打中）

OPT1 `hits_moe_psum.json`：

- 大量 `moe_contig_psum:moe_gate_proj:{prefill,decode}:m*`
- 大量 `moe_contig_psum:moe_down_proj:{prefill,decode}:m*`
- **没有** `moe_masked:*`

→ 相对上次 e2e_candidates，**这次确实走了 contig+PSUM**，假阴性（打错路）已排除。

Prefill 大 M bucket（如 m3072–m4096）计数极高，与 chunked prefill / 多层 / 多请求累计一致，说明 PSUM kwargs 在真实 serving forward 中反复生效。

---

## 5. 为什么「叶级 ~1.05×」换不成 e2e？

### 5.1 Amdahl 上界

在 stock 层预算（B300，`llm_flops_style`，S=64k）里，prefill M=2048：

| 算子组 | 约占层时间 |
|--------|----------:|
| `dsa_prefill_attn` + `index_score` | **~56%** |
| `moe_gate` + `moe_up` + `moe_down` | **~25%** |
| 其余 GEMM/BMM | ~19% |

若仅 gate+down 各加速 1.05×（up 不动），MoE 三段合计近似：

\[
\text{层加速} \approx 1 / (1 - 0.25\cdot(1-1/1.05)\cdot\frac{2}{3}) \approx 1.008
\]

即层级上界大约 **&lt;1%**，再被 DeepEP / 调度 / 非计算开销稀释后，e2e 很难稳定超过噪声；实测甚至略慢，与「上界太薄 + 可能有 PSUM/dispatch 额外开销」一致。

### 5.2 对照基线是 stock contig，不是慢参考

本次 OPT0 已是生产级 packed DeepGEMM contig；archive PSUM 相对它只有 ~5% 叶级宣称，没有 harness 对慢参考那种 1.5×+ 的空间。

### 5.3 M=4096 异常点

`cache_hit_rate=0` → 整段 ~70k prefill，TTFT 147s 不可与 9s 的增量点比较。可能原因：前序大请求后 radix 未命中、warmup 抖动、或测量脚本在该点未复用 cache。分析时 **丢弃**。

### 5.4 与生产默认的关系

即使 contig+PSUM 将来略快，**默认 `low_latency` 也不走这条路**（见默认路径文档）。要在生产吃到 PSUM，需要改 DeepEP 模式或另做 masked 侧优化——本次 A/B **不能**外推为「线上开 OPT 就会变快」。

---

## 6. 方法学备注

| 项 | 说明 |
|----|------|
| 公平性 | OPT0/OPT1 同为 `deepep=normal`；不与 LL 默认混比 |
| 指标 | 官方 `one_batch_server` 的 `last_ttft` |
| Serve 启动 | `/usr/local/bin/sglang`；避免 cwd 下 `sglang/` 目录遮蔽 CLI |
| 收尾 | 测完恢复 `deepep=low_latency` + OPT0 |

---

## 7. 结论与建议

1. **路径矫正成功**：HIT=`moe_contig_psum`，此前「打到 masked」的假阴性已关闭。
2. **e2e 仍无收益**：有效点 TTFT ratio ≈ **1.017–1.023**（更慢）。
3. **不建议**把 MoE contig PSUM 作为生产 e2e 增益项；收益上界被 attn/score 主导，且默认路径不经过 contig。
4. 若继续挖 prefill TTFT：优先 **`dsa_attn` / `index_score`（flashmla_kv + paged MQA）**，并始终用 **vs stock（线上路径）** 而不是 vs harness 慢参考。

---

## 8. 相关文件

- 运行脚本：`/home/wwxq/run_e2e_psum_contig_prefill.sh`（及 `_continue.sh`）
- HIT 计数：`dispatch.record_psum_hit` → `moe_contig_psum:...`
- 默认路径算子/占比/代码入口：`glm52_opt/default_path_ops_and_shares.md`
