# GLM-5.2-FP8 实验索引（标准脚本 / 数据 / 分析）

- 日期：2026-07-24
- 机器：B300 `10.24.0.55` / Docker `sglang_0512`
- 代码：本机 `/home/wwxq/SGLang-DGMK` ↔ 远端 `/home/ubuntu/wwxq/SGLang-DGMK`
- 脚本与 bench 产物：本机 `/home/wwxq/` ↔ 远端 `/home/ubuntu/wwxq/`（以下路径默认写远端）

---

## 0. 默认 serving 基线（所有「生产路径」实验共用）

| 项 | 值 |
|----|-----|
| 启动 | `/home/ubuntu/wwxq/run_glm52_dgmk.sh` |
| 环境文件 | `/home/ubuntu/wwxq/cache/sglang/glm52_opt.env` |
| 拓扑 | TP8 / DP8 / EP8 |
| DeepEP | **`low_latency`**（生产默认；PSUM 专用实验会改成 `normal`） |
| DSA | `flashmla_kv`（prefill + decode） |
| CUDA graph | `max_bs=16` |
| OPT 基线 | `SGLANG_GLM52_OPT=0`（无 archive swap） |
| API | `:30000`，模型 `GLM-5.2-FP8` |

> 「vs stock / vs opt0」= 相对上述生产 DeepGEMM/FlashMLA 路径，**不是** harness 慢路径 f32 ref。

---

## 1. 标准测试脚本（按场景选用）

### 1.1 底层 bench（被编排脚本调用）

| 脚本 | 用途 | 典型参数 |
|------|------|----------|
| `bench_glm52_decode_bs.py` | **短/中上下文 e2e**：并发 streaming chat，量 TTFT / TPOT / tok/s | `input-len`、`concurrency`、`output-len`、`warmup-waves`/`waves` |
| `run_one_batch_server_longtimeout.py` | **官方 one_batch**：64k 增量 prefill / decode 形状对齐 | `--cache-hit-rate`、`--input-len=S+M`、`--batch-size=DP` |
| `bench_glm52_64k_ttft.py` | 早期 64k TTFT 对比（较少用） | — |
| `bench_glm52_llm_flops_e2e.py` | llm_flops 形状对齐辅助 | — |

### 1.2 编排脚本（推荐入口）

| 场景 | 脚本 | 产出目录 | 说明 |
|------|------|----------|------|
| **短/中上下文 OPT A/B（历史主矩阵）** | `run_e2e_gain_ops_matrix.sh` | `bench_results/e2e_gain_ops_matrix/` | `e2e_seq` 1024/2048/4096 + 可选 64k 冷启动；`serving_safe` + ABI；fused / index_q / all_gain |
| **同上场景复现** | `run_e2e_gain_ops_repro.sh` | `bench_results/e2e_gain_ops_repro_<ts>/` | 只跑 e2e_seq；与 matrix 同源 bench |
| **官方 one_batch @ llm_flops 形状** | `run_e2e_official_llm_flops.sh` | `bench_results/e2e_official_llm_flops/` | S=64k 增量 prefill；S=32k decode；官方 `one_batch_server` |
| **e2e_candidates（o_proj / MoE PSUM 等）** | `run_e2e_candidates_official.sh` | `bench_results/e2e_candidates_official/` | 跳过已测无收益的 fused/index_q |
| **MoE contig PSUM（正确路径）** | `run_e2e_psum_contig_prefill.sh` (+ `_continue.sh`) | `bench_results/e2e_psum_contig_prefill/` | **两边都 `deepep=normal`**，才能走到 `moe_contig_psum` |
| **默认路径冒烟** | `run_default_path_one_batch.sh` | `bench_results/default_path_ops/` | OPT0 + LL；prefill M1024/2048 + decode M16 |
| **全机 nsys（短 decode 窗口）** | `run_nsys_op_single_ablation.sh` | `bench_results/nsys_op_single/` | 含 NCCL/DeepEP；短 prompt 偏 decode |
| **全机 nsys（S=64k 增量 prefill）** | `run_nsys_prefill_64k.sh` | `bench_results/nsys_prefill_64k/` | 先暖 64k cache，再 capture M=1024/2048 |

### 1.3 场景对照（避免混用结论）

| 名字 | 实际含义 | 适合回答 |
|------|----------|----------|
| `e2e_seq` / decode_bs | **全新**短 prompt（1024/2048/4096），conc=16 | 短上下文 TTFT/TPOT 是否有相对 opt0 收益 |
| `decode_kv` 64k | **全新** ~65k prompt（冷 radix） | 整段长 prefill 到首字，**不是**「已有 64k KV」 |
| `one_batch` + `cache_hit` | 暖到 S 后增量 M | 对齐 llm_flops 的 **增量 prefill / decode** |
| nsys serving | GPU kernel 时长占比（含通信） | 通信 vs compute 结构；注意 LL dispatch busy-wait |

---

## 2. 实验数据（远端 `bench_results/`）

| 目录 | 对应实验 | 关键文件 |
|------|----------|----------|
| `e2e_gain_ops_matrix/` | 首次「有收益」矩阵（2026-07-23） | `bench_*_e2e_seq*.json`、`bench_*_kv65536_*.json`、`hits_*.json`、`FULL_METRICS.md` |
| `e2e_gain_ops_repro_20260724_145934/` | 2048 收益场景复现（未复现） | `bench_*_e2e_seq*.json`、`SUMMARY.md` |
| `e2e_official_llm_flops/` | 官方 one_batch @ 64k/32k | `one_batch_*.jsonl`、`SUMMARY_OFFICIAL.md` |
| `e2e_candidates_official/` | o_proj / moe_gate / moe_down / e2e_all | `one_batch_*.jsonl`、`hits_*.json`、`SUMMARY.md` |
| `e2e_psum_contig_prefill/` | MoE PSUM 正确路径 A/B | `one_batch_opt0.jsonl`、`one_batch_moe_psum.jsonl`、`hits_moe_psum.json` |
| `default_path_ops/` | 默认路径冒烟 | `one_batch_prefill.jsonl`、`one_batch_decode.jsonl` |
| `nsys_op_single/` | 短窗口全机 nsys（opt0 + 各 op） | `opt0.nsys-rep`、`*_gpu_kern_sum.csv` |
| `nsys_prefill_64k/` | 64k 增量 prefill nsys | `prefill64k_opt0.nsys-rep`、`.sqlite`、`*_gpu_kern_sum.csv` |

本地分析文档在 `SGLang-DGMK/glm52_opt/`；原始 JSON/nsys 以远端为准（体积大）。

---

## 3. 分析文档（本地 `SGLang-DGMK/glm52_opt/`）

### 3.1 主结论（按阅读顺序）

| 文档 | 内容 |
|------|------|
| **本文** `EXPERIMENT_INDEX.md` | 脚本 / 数据 / 文档总索引 |
| `default_path_ops_and_shares.md` | 默认路径算子、层内占比、SGLang 入口；含通信总览 |
| `why_no_gain_at_64k.md` | 为何短上下文曾有点收益、64k 上看不到（Amdahl + DeepEP） |
| `e2e_gain_ops_full_metrics.md` | 首次 gain matrix 完整表（含 2048 TTFT ~0.91–0.93×） |
| `e2e_gain_ops_repro_SUMMARY.md` | 同 knobs 复现：绝对 TTFT≈不变，opt0 变快 → 相对收益消失 |
| `e2e_psum_contig_prefill_SUMMARY.md` + `_analysis.md` | PSUM 路径打通；e2e 仍 ~1.7–2.3% 更慢 |
| `nsys_prefill_64k_SUMMARY.md` + `_kern_categories.md` | 64k 增量 prefill：DeepEP ~78%，总通信 ~79% |
| `e2e_gpu_kern_categories.md` | 短 decode nsys：NCCL~31% + DeepEP~21% ≈ 通信 ~52% |
| `default_path_one_batch_SUMMARY.md` | 默认路径 one_batch 冒烟数字 |
| `llm_flops_e2e_alignment.md` / candidates 相关 | 官方形状对齐与 e2e_candidates 范围 |
| `history/e2e_candidates_20260723/INDEX.md` | 叶级候选正/负结果归档 |

### 3.2 一句话结论汇总

| 问题 | 结论 |
|------|------|
| 短/中 `e2e_seq@2048` 以前有 TTFT 收益？ | 首次矩阵有 fused/index ≈0.91–0.93×；**复现未稳住**（绝对 TTFT 接近，主要是 opt0 baseline 漂移） |
| 官方 64k 增量 prefill / PSUM 正确路径有 e2e 收益？ | **没有**（打平或略慢）；HIT 可对齐，Amdahl + 通信吃掉叶级加速 |
| 默认路径时间结构？ | 短 decode：通信 ~50%+；64k 增量 prefill：DeepEP 主导（~78% kernel 时长，含 busy-wait） |
| 继续换小 GEMM 能否拉 e2e？ | 64k serving 上概率低；瓶颈在 DeepEP / DSA+score，不在 archive 小 GEMM |

---

## 4. 常用命令速查

```bash
# 健康检查
curl -sf http://127.0.0.1:30000/health

# 写 env 后重启（在远端）
# 编辑 /home/ubuntu/wwxq/cache/sglang/glm52_opt.env
# 再跑 /home/ubuntu/wwxq/run_glm52_dgmk.sh

# 短上下文复现（推荐）
OUT=/home/ubuntu/wwxq/bench_results/e2e_gain_ops_repro_manual \
  bash /home/ubuntu/wwxq/run_e2e_gain_ops_repro.sh

# 64k 增量 prefill 官方 one_batch
bash /home/ubuntu/wwxq/run_e2e_official_llm_flops.sh

# 默认路径冒烟
bash /home/ubuntu/wwxq/run_default_path_one_batch.sh
```

---

## 5. 版本备注

- Gain matrix 与 repro 使用同一 `bench_glm52_decode_bs.py`（md5 一致）；参数 `out=64, warmup=1, waves=2, conc=16`。
- 复现失败**不是**脚本换错成 64k one_batch；见 `e2e_gain_ops_repro_SUMMARY.md` 与对话结论（opt0 绝对 TTFT 变快）。
- PSUM 实验必须 `deepep=normal`；与生产 `low_latency`（masked MoE）路径不同，**不可直接和 LL 数字横比「算子快慢」**。
