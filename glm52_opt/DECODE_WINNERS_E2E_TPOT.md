# GLM-5.2 Decode Winners — 端到端 TPOT 测试说明

面向协作伙伴：如何复现 **OPT0 vs decode winners** 的公平 decode TPOT 对比。

- **分支**：`docs/glm52-decode-winners-e2e-tpot`
- **机器假设**：8×B300，TP8 / DP8 / EP8，模型 `GLM-5.2-FP8`
- **主指标**：ITL / TPOT = `(latency − last_ttft) / output_len`（ms）

---

## 1. 本分支启用了什么（有 e2e 收益的算子）

| 算子 | 机制 | 说明 |
|---|---|---|
| `dsa_decode_attn`（`flashmla_sparse_decode`） | FlashMLA **P1+c2** | **不要用 r2a**（serving CUDA graph KV shape 会挂） |
| `o_proj` | fixed_nk graph-only | M∈{16,32} |
| `index_q_upproj` | fixed_nk graph-only | M∈{16,32} |
| MoE gate/up/down | `SGLANG_GLM52_INFINI_MOE_ALIGN=1` | 仅 decode 作用域；**禁止**对 prefill 开 |

### 明确不启用

- `fused_qkv_a_proj`：leaf 有加速，历史 e2e 接近打平/噪声大
- `dsa_prefill_attn` / 一切 prefill fixed_nk：本轮 prefill 无稳定 HIT/收益
- FlashMLA **r2a**
- `q_b_proj` / `index_k_proj` / `index_score`：无可用生产加速（或虚假 harness 加速）

Profile：`SGLANG_GLM52_OPT_PROFILE=combined_winners`。

---

## 2. 测试场景（伙伴复现口径）

| 项 | 值 |
|---|---|
| KV / input_len | **S = 32768** |
| 并行 | TP8 / DP8 / EP8，`--enable-dp-attention` |
| Global batch size | 推荐先 **128**（local_M=16）；可选再跑 **256**（local_M=32） |
| `cuda-graph-max-bs` | **32**（才能覆盖 local_M=32 → global 256） |
| output_len | 48 |
| 重复次数 | 每个 `(label × global_BS)` **N=3**（复现收益够用） |
| 统计 | 看 **median**（3 次中位）；脚本也会打 mean/stdev/p10/p90 |
| 对比 | `opt0`（`SGLANG_GLM52_OPT=0`）vs `winners`（上表算子） |

> 作者侧偶尔会用 `N_RUNS=100` 压噪声；**伙伴复现不需要**，默认 **3 次**即可。

### 重要：local_M vs global BS

| 表里常写的 BS / local_M | DP | **global batch_size** |
|---:|---:|---:|
| 16 | 8 | **128** |
| 32 | 8 | **256** |

历史 nsys 单次 e2e 大多是 **global BS=128**。

### 已有参考结果（口径不同勿直接对齐）

同一仓库工作区、decode-only、global BS=128、N=2（含 fused_qkv 的旧 winners 合集）中位 ITL 约：

- OPT0 ≈ **40.44 ms** → winners ≈ **38.61 ms**（约 **1.047×**）

去掉 fused_qkv 后的结果以你本地 `TPOT_SUMMARY.md` 为准。工作区示例：

`wwxq/bench_results/decode_tpot_n*_s32768_*`

---

## 3. 目录布局（伙伴侧）

推荐工作区（与本文作者机一致时可直接跑）：

```text
$WWXQ/                          # 工作根，含模型缓存、venv、bench_results
  venv_wwxq/                    # Python venv（已装 sglang 依赖）
  SGLang-DGMK/                  # 本仓库（checkout 本分支）
  run_glm52_dgmk.sh             # serve 入口（或复制 glm52_opt/scripts 用法）
  run_one_batch_server_longtimeout.py
  cache/sglang/glm52_opt.env
  bench_results/
  models -> /mnt/b300-shared/models/GLM-5.2-FP8
```

若伙伴机路径不同：导出 `ROOT` / `MODEL` / `PORT` 即可（见脚本头部）。

---

## 4. 一键跑法（推荐）

仓库内脚本：

- [`scripts/write_decode_winners_env.sh`](scripts/write_decode_winners_env.sh) — 写 winners env（无 fused_qkv）
- [`scripts/run_decode_tpot_n100_ab.sh`](scripts/run_decode_tpot_n100_ab.sh) — OPT0 vs winners，N 次 decode TPOT

```bash
# 1) checkout
cd $WWXQ/SGLang-DGMK
git fetch origin
git checkout docs/glm52-decode-winners-e2e-tpot

# 2) 环境（伙伴复现：N=3）
export ROOT=$WWXQ                    # 含 venv / bench_results / run_glm52_dgmk.sh
export PYTHONPATH=$PWD/python${PYTHONPATH:+:$PYTHONPATH}
export PATH=$ROOT/venv_wwxq/bin:$PATH
export MODEL=/path/to/GLM-5.2-FP8
export PORT=30002
export N_RUNS=3
export GLOBAL_BS_LIST="128"          # 可选再加 256："128 256"
export LABELS="opt0 winners"
export SGLANG_CUDA_GRAPH_MAX_BS=32
export MEM_FRACTION_STATIC=0.83

# 3) 跑
bash glm52_opt/scripts/run_decode_tpot_n100_ab.sh

# 4) 看结果
cat $ROOT/bench_results/decode_tpot_n*_s32768_*/TPOT_SUMMARY.md
```

脚本会：

1. 每个 label 起一次 serve（`cuda_graph_max_bs=32`）
2. 每个 global BS：flush + 预热 S=32k cache，再跑 **N** 次 decode（默认 3）
3. 写出 `decode_{opt0|winners}_bs{128|256}.jsonl` 与 `TPOT_SUMMARY.md`

### 可选：更严统计（作者侧）

```bash
N_RUNS=100 GLOBAL_BS_LIST="128 256" \
  bash glm52_opt/scripts/run_decode_tpot_n100_ab.sh
```

---

## 5. 手动 serve（调试 HIT）

```bash
bash glm52_opt/scripts/write_decode_winners_env.sh
# 确认 env 含：
#   OPT_PROFILE=combined_winners
#   OPT_OPS=flashmla_sparse_decode,o_proj,index_q_upproj,moe_gate_proj,moe_up_proj,moe_down_proj
#   GLM52_FLASHMLA_DECODE_STACK=p1_c2
#   SGLANG_GLM52_INFINI_MOE_ALIGN=1

export SGLANG_GLM52_ENV_FILE=$ROOT/cache/sglang/glm52_opt.env
export SGLANG_CUDA_GRAPH_MAX_BS=32
export SGLANG_EXTRA_SERVE_ARGS="--mem-fraction-static 0.83"
bash $ROOT/run_glm52_dgmk.sh
```

期望 HIT（global BS=128 / local_M=16）大致包括：

- `fp8_gemm/fixed_nk:o_proj:decode:m16`
- `fp8_gemm/fixed_nk:index_q_upproj:decode:m16`
- `hotspot_plugin/flashmla_sparse_decode:dsa_decode_attn:decode:m16`
- `moe_masked:moe_*:decode:m*`（align 路径）

**不应**再依赖 `fused_qkv_a` / `dsa_prefill` HIT。

---

## 6. 结果怎么读

`TPOT_SUMMARY.md` 示例列：

| label | global_BS | n | mean | median | stdev | p10 | p90 |
|---|---:|---:|---:|---:|---:|---:|---:|

关注：

- **median ITL** 为主结论（DeepEP 噪声大）
- mean 与 median 差大 → 看 p10/p90 / 是否 OOM 重试
- Amdahl：leaf 合计乐观 ~数 % forward；真实 e2e 常被通信稀释到 **~3–6%** 量级

---

## 7. 相关代码入口

- Profile / allowlist：`python/sglang/srt/layers/glm52_opt/config.py`（`combined_winners`）
- Registry：`python/sglang/srt/layers/glm52_opt/registry.py`
- Provider：`python/sglang/srt/layers/glm52_opt/hotspot_candidates/flashmla_accel_bundle_provider.py`
- MoE align：`python/sglang/srt/layers/glm52_opt/infini_moe_align.py` + `moe_masked.py`
- 启用总览：[`hotspot_accel_enable.md`](hotspot_accel_enable.md)

---

## 8. 给伙伴的最短 checklist

1. `git checkout docs/glm52-decode-winners-e2e-tpot`
2. 8 卡空闲，模型与 venv 就绪
3. `N_RUNS=3 GLOBAL_BS_LIST="128" bash glm52_opt/scripts/run_decode_tpot_n100_ab.sh`
4. 打开 `TPOT_SUMMARY.md`，对比 **BS=128 median**（opt0 vs winners）
5. （可选）再跑 `GLOBAL_BS_LIST="256"` 或 `N_RUNS=100` 做更严确认

问题可对照本文件 §2 场景表与 §5 HIT 列表。
