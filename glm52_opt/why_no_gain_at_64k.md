# 为什么短上下文有 e2e 收益，64k 上看不到？

- 日期：2026-07-24
- 对照数据：`e2e_gain_ops_full_metrics.md`（短序列 / 64k）、`nsys_prefill_64k_SUMMARY.md`、`default_path_ops_and_shares.md`

---

## 1. 先对齐「以前有收益」到底是什么

短/中上下文（全新 prompt ≈1024/2048/4096）上，相对稳定的点主要是：

| 场景 | 指标 | 大致结果 |
|------|------|----------|
| e2e_seq **2048** | TTFT | fused / index ≈ **0.91–0.93×**（约快 7–9%） |
| e2e_seq 1024/4096 | TTFT | 大多打平或变差 |
| 短序列 decode | TPOT | 基本 ±1–3%，几乎无增益 |
| 64k 场景 bs32 | TPOT | 曾见 ~0.87×（样本少）；**TTFT 无增益** |

也就是说：以前「有收益」主要是 **中等长度、以 compute GEMM 为主的 prefill TTFT**；不是「所有算子在所有场景都稳赚」。到 **64k 上下文**后，连这类收益也被冲掉。

---

## 2. 核心机制：瓶颈从「可换的 GEMM」变成「换不到的大头」

### 2.1 单卡层内（无通信）：64k 上 attn/score 吃掉大半

`llm_flops_style` stock（S=64k，prefill M=4096）：

| 组 | 约占层时间 |
|----|----------:|
| `dsa_attn` + `index_score` | **~58%** |
| MoE gate+up+down | ~26% |
| o_proj / fused / q_b / index_* 等「常被 swap 的 GEMM」 | 合计往往 **&lt;15%** |

短序列 prefill（M 小、S 也小）时，DSA/score 相对轻，**fused / index_q / o_proj 等在 TTFT 里的占比更高** → 叶级 1.1× 更容易冒出几个百分点的 e2e。

64k 后：即便增量 prefill 的 **新 token M 只有 1k–4k**，indexer/score/attn 仍要扫 **整段 64k KV**，计算侧大头钉在 DSA 路径上；你们 archive 里多数「赢」的却是 **小 GEMM**，Amdahl 上界天然变薄。

粗算：若某 GEMM 只占层 5%，叶级 1.20× → 层最多 ~1.01×；再被通信稀释，e2e 噪声内消失。

### 2.2 多卡 serving（含通信）：64k 增量 prefill 上 EP 更夸张

专门跑的 S=64k 增量 prefill nsys（默认 LL + OPT0）：

| 类别 | GPU kernel 时间 |
|------|----------------:|
| DeepEP（dispatch+combine） | **~78%** |
| DeepGEMM 等 | ~12% |
| DSA/MLA | ~2.8% |
| NCCL | ~1.4% |
| **通信合计** | **~79%** |

注意：DeepEP-LL dispatch 有 busy-wait，GPU% 会偏高；但方向明确——**多卡 64k 路径上，时间大量花在 EP 通信，不在你们替换的那几个 GEMM 上。**

短 decode / 短 prompt 的旧 nsys 里 NCCL 更显眼（~31%）、DeepEP ~21%；到 **64k 增量 prefill** 则 **DeepEP 主导**。场景一变，瓶颈就变，短上下文上「刚好够露头」的 GEMM 收益会直接被淹没。

### 2.3 一张对比示意

```text
短上下文 prefill TTFT（示意）
├─ 可 swap 的 GEMM ........ 相对可见（故 2048 上曾有 ~7–9% TTFT）
├─ MoE GEMM ............... 中等
├─ attn/score ............. 尚未按 64k 膨胀
└─ DeepEP/NCCL ............ 有，但未压成绝对主导

64k 上下文（增量 prefill / 长 decode）
├─ DeepEP (+NCCL) ......... 全链路里最大头之一（nsys ~70%+ kernel）
├─ dsa_attn + index_score . 计算侧层内 ~55–60%（随 S 涨）
├─ MoE masked GEMM ........ 有，但不是 archive PSUM 那条路
└─ fused/index_q/o_proj ... 叶级再快，e2e 也只剩噪声
```

---

## 3. 其它放大「64k 看不到收益」的因素

### 3.1 度量对象变了

| 说法 | 实际测到的 |
|------|------------|
| 「64k TTFT」旧 `decode_kv` | 常常是 **整段 64k 冷 prefill**（~110s），不是「KV 已 64k 再出首字」 |
| 真正对齐的增量 TTFT | S=64k cache + 增量 M（本次 ~2–4s） |

冷 64k prefill 几乎全是 **KV 构建 + 长上下文 DSA/score + EP**；swap 几个 decode 向 GEMM，TTFT 必然不动。

### 3.2 叶级「收益」参照系不对

Harness 常见 WIN 是 vs **慢 f32-scale 参考**；线上 OPT0 已是 packed DeepGEMM。相对 stock 往往只剩 **~1.00–1.05×**，在 64k 分母下不够。

### 3.3 路径打偏

- MoE 宣称收益在 **contig+PSUM prefill**；默认是 **LL masked**。  
- 矫正到 contig+PSUM 后 e2e 仍略慢（~1.02×）→ 即使打中目标路径，也过不了 64k 的 Amdahl/开销关。  
- DSA archive 多为 sparse；线上是 **flashmla_kv**，大头算子根本没换成「有收益的那版」。

### 3.4 调度与噪声

长上下文 + DP8：chunked prefill、cache 命中抖动、CUDA graph 覆盖不全（大 M eager）都会让 **&lt;3%** 的真收益淹没在 run-to-run 噪声里。短序列 TTFT 基数小（0.5s 级），同样绝对加速更容易读成「有收益」。

---

## 4. 一句话结论

**不是「64k 上算子变慢了」，而是时间结构变了：**

1. **短上下文**：TTFT 里一小撮 GEMM 占比够大 → 叶级小胜可以变成几个点的 e2e（如 2048 TTFT）。  
2. **64k**：时间改由 **DeepEP 通信 + 长上下文 DSA/score** 主导；你们主要优化的 GEMM/MoE 变成薄薄一层 → e2e 看不见是预期结果，不是偶然。

若目标是 64k e2e，应优先动 **DeepEP/通信路径** 与 **flashmla_kv + index_score**，而不是继续堆短上下文上曾「勉强露头」的那些 GEMM swap。
