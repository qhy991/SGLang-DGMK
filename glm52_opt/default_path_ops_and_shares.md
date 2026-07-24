# GLM-5.2 默认 serving 路径：算子、占比、SGLang 入口

- 日期：2026-07-24
- 机器：B300 `10.24.0.55` / Docker `sglang_0512`
- **默认配置（当前线上一致）**：
  - `SGLANG_GLM52_OPT=0`（无 archive swap；`try_dispatch_*` 不生效）
  - `--deepep-mode low_latency`
  - `--dsa-prefill-backend flashmla_kv` / `--dsa-decode-backend flashmla_kv`
  - TP8 / DP8 / EP8，`cuda-graph-max-bs=16`
- e2e 冒烟：`/home/ubuntu/wwxq/bench_results/default_path_ops/`（脚本 `run_default_path_one_batch.sh`）
- 层内占比（计算侧）：同机 `llm_flops_style` stock 列；**单卡微基准，不含通信**
- **全模型 GPU profiling（含通信）**：`glm52_opt/e2e_gpu_kern_categories.md`（来自 `nsys_op_single/opt0.nsys-rep`）

> **重要**：若要看「整个推理过程」的时间结构，请看第 0 节（nsys 全机 kernel，含 NCCL/DeepEP）。  
> 第 3 节只回答「单层里各 compute op 谁大」，**故意不含通信**。

---

## 0. 全模型 serving GPU 时间（含通信）— 你真正要的 profiling

来源：默认路径 OPT0 在 nsys 下跑 serving 的 `cuda_gpu_kern_sum`（`bench_results/nsys_op_single/opt0.*`）。

| 大类 | GPU kernel 时间占比（约） | 对应路径 |
|------|--------------------------:|----------|
| **comm_nccl**（主要 AllGather） | **~31%** | DP/TP 集合通信（`ncclDevKernel_AllGather_*`） |
| **comm_deepep**（dispatch+combine） | **~21%** | DeepEP-LL `internode_ll::dispatch` / `combine` |
| **gemm_deepgemm**（含 MoE masked GEMM 等） | **~十几 %** | DeepGEMM `sm100_fp8_*` / nvjet |
| **dsa_attn_mla** | 若干 % | `flash_fwd_*_mla_*` / flashmla_kv |
| **moe_act_quant / quant / norm / router…** | 其余 | silu_mul_quant、per_token_group_quant、RMSNorm、topk… |

**Rollup：通信（NCCL + DeepEP）约占该次 capture 全部 GPU kernel 时间的 ~52%。**  
详细分类表与 top kernel 列表见：`e2e_gpu_kern_categories.md`。

Caveat：该 nsys 来自短 prompt、decode 偏多的 ablation 窗口；**64k 增量 prefill 的比例会变**（attn/score 会抬头），但默认 EP 路径上 **通信是一等公民**，不能从 llm_flops 层表里读出来。

若需要 **专门针对 S=64k incremental prefill** 的一版 nsys/Torch Profiler，需要再开一轮带 `cudaProfilerApi` 的 capture（会重启 serve）。

**已完成（2026-07-24）**：见 `glm52_opt/nsys_prefill_64k_SUMMARY.md` / 远端 `bench_results/nsys_prefill_64k/`。
该 capture 上 **通信（DeepEP+NCCL）≈79%** GPU kernel 时间（DeepEP dispatch 单独 ~72%，含 LL busy-wait；解读见该 SUMMARY）。

---

## 1. 「stock / 默认路径」一句话

**Stock** = SGLang 已接线、OPT=0 时实际执行的实现（DeepGEMM / FlashMLA / torch.bmm 等），不是 harness 慢参考。

默认与本次 PSUM 矫正实验的关键分叉：**MoE 在 LL 下走 masked GroupGEMM**，不走 contig+PSUM。

```text
deepep=low_latency
  → pre_permute_deepep_ll_to_deep_gemm(use_masked_gemm=True)
  → DeepGemmRunnerCore._run_masked_gemm
  → deep_gemm.fp8_m_grouped_gemm_nt_masked

deepep=normal（仅实验）
  → use_masked_gemm=False
  → _run_contiguous_gemm (+ 可选 PSUM kwargs)
```

---

## 2. 默认路径用到的算子 ↔ SGLang 代码路径

| 逻辑算子 | 默认 kernel / backend | SGLang 入口（OPT=0） |
|----------|----------------------|----------------------|
| **fused_qkv_a_proj** | DeepGEMM **contiguous** `fp8_gemm_nt` | `models/deepseek_v2.py` `prepare_qkv_latent` → FP8 `ReplicatedLinear` → `quantization/fp8_utils.py` `deepgemm_w8a8_block_fp8_linear_with_fallback` → `w8a8_block_fp8_matmul_deepgemm`（`try_dispatch_fp8_gemm` 返回 None） |
| **q_b_proj** | 同上 contiguous `fp8_gemm_nt` | `forward_mla.py` `forward_absorb_prepare` → `ColumnParallelLinear` → 同上 |
| **o_proj** | 同上 contiguous `fp8_gemm_nt` | `forward_mla.py` `forward_absorb_core` → `RowParallelLinear` → 同上 |
| **absorbed_W_UK / W_UV** | 默认 **`torch.bmm`（bf16，权重先 dequant）** | `forward_mla.py` 中 `w_kc` / `w_vc`；仅当 `SGL_USE_DEEPGEMM_BMM` 等条件满足才走 DeepGEMM BMM |
| **index_q_upproj** | contiguous `fp8_gemm_nt`（`wq_b`） | `attention/dsa/dsa_indexer.py` `Indexer` → `self.wq_b` |
| **index_k / index_weights** | 常 **fusion** 为一条 bf16 `wk_weights_proj`；否则 k 走 FP8 contiguous、weights 走 bf16 | `dsa_indexer.py` `_fused_k_weights` / `_get_q_k_bf16` |
| **index_score** | **`deep_gemm.fp8_paged_mqa_logits`**（`dsa_paged_mqa_logits_backend=auto/deepgemm`） | `dsa_indexer.py` `Indexer._get_topk_paged` |
| **dsa_attn** | **`sgl_kernel.flash_mla.flash_mla_with_kvcache`** | `attention/dsa_backend.py` `DeepseekSparseAttnBackend._forward_flashmla_kv`（prefill/decode 均 flashmla_kv） |
| **moe_gate + moe_up** | DeepGEMM **masked** `fp8_m_grouped_gemm_nt_masked`（w13 一次） | `moe/ep_moe/layer.py` DeepEP-LL → `moe_runner/deep_gemm.py` `pre_permute_deepep_ll_to_deep_gemm` → `_run_masked_gemm` |
| **moe_down_proj** | 同上 masked GroupGEMM（w2） | 同上 `_run_masked_gemm` |

关键源码锚点（masked）：

```789:795:python/sglang/srt/layers/moe/moe_runner/deep_gemm.py
    return DeepGemmRunnerInput(
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        use_masked_gemm=True,
        masked_m=masked_m,
        expected_m=expected_m,
    )
```

线性 FP8 在 OPT=0 时落到 stock：

```830:846:python/sglang/srt/layers/quantization/fp8_utils.py
    glm52_out = try_dispatch_fp8_gemm(...)
    if glm52_out is not None:
        return glm52_out...
    output = w8a8_block_fp8_matmul_deepgemm(...)
```

---

## 3. 层内时间占比（stock，单卡微基准）

来源：B300 `llm_flops_style` JSON 的 **`stock_ms`**，按层内求和归一化。  
**Caveat**：微基准 MoE 为 masked GroupGEMM shape，**无 DeepEP all-to-all**；serving 上 MoE/通信占比会更高。DSA 微基准接口是 `flash_mla_sparse_fwd`，与线上 `flashmla_kv` 布局不同，但 **「attn+score 主导 prefill」** 结论与 accept_layer / e2e 观察一致。

### 3.1 Prefill（S=64k）

| 算子 | M=1024 | M=2048 | M=4096 |
|------|-------:|-------:|-------:|
| **dsa_prefill_attn** | 27.0% | 29.1% | **29.8%** |
| **index_score** | 25.0% | 27.4% | **28.6%** |
| moe_down_proj | 8.1% | 8.4% | 9.0% |
| moe_gate_proj | 8.0% | 8.2% | 8.6% |
| moe_up_proj | 7.7% | 7.9% | 8.1% |
| o_proj | 6.9% | 7.1% | 7.0% |
| index_k_proj | 6.8% | 3.5% | 1.8% |
| q_b_proj | 3.4% | 2.9% | 2.7% |
| fused_qkv_a_proj | 2.0% | 1.5% | 1.2% |
| 其余（index_q / weights / absorbed） | ~5% | ~4% | ~3% |
| **dsa + index_score** | **52.0%** | **56.5%** | **58.4%** |
| **moe_gate+up+down** | **23.8%** | **24.5%** | **25.7%** |
| 层合计 stock (ms) | 1.20 | 2.36 | 4.65 |

### 3.2 Decode（S=64k；llm_flops `index_score` 本轮未计入有效时间）

| 算子 | M=16 | M=32 |
|------|-----:|-----:|
| dsa_decode_attn | 13.6% | 13.7% |
| moe_up / gate / down（各） | ~13.4% | ~13.3% |
| o_proj | 13.3% | **14.3%** |
| q_b_proj | 8.1% | 7.9% |
| fused_qkv_a_proj | 7.1% | 6.9% |
| index_k / index_q / weights | ~16% | ~16% |
| absorbed UK/UV | ~1.7% | ~1.7% |
| **moe 三段合计** | **~40%** | **~40%** |

补充（`accept_layer` backend，含 index_score，M=32）：`index_score` 可到层内第一（~12%），与 dsa/o_proj/moe 同量级——serving decode 仍应把 **paged MQA score** 算进预算。

---

## 4. 默认路径 e2e 冒烟（已跑完）

配置确认：`OPT=0`、`deepep_mode=low_latency`、health OK。  
远端：`/home/ubuntu/wwxq/bench_results/default_path_ops/` · 本地：`glm52_opt/default_path_one_batch_SUMMARY.md`

| run | last_ttft (s) | latency (s) | cache_hit | 说明 |
|-----|--------------:|------------:|----------:|------|
| `default_prefill_M1024` | **2.049** | 2.050 | 0.9846 | S=64k + 增量 1024，out=1 |
| `default_prefill_M2048` | **3.854** | 3.854 | 0.9697 | S=64k + 增量 2048，out=1 |
| `default_decode_M16` | 0.535 | **1.281** | 0.9998 | out=32；ITL≈23.3 ms |

与同场景 PSUM 实验的 OPT0（`deepep=normal`，TTFT 2.53/4.71）**不可直接比**——DeepEP 模式不同；LL 默认 prefill 明显更快。

---

## 5. 和优化工作的对应关系

| 想优化的点 | 默认是否热路径 | 备注 |
|------------|:-------------:|------|
| MoE contig PSUM | **否**（默认 masked） | 本次矫正实验证明：即使强行 contig+PSUM，e2e 仍无收益 |
| MoE masked decode archive | **是**（LL） | 先前 e2e_candidates HIT 在此；相对 stock 无稳定 e2e 增益 |
| o_proj / fused / index_q decode | **是**（contiguous stock） | 叶级有时赢 harness；vs 线上 stock / e2e 多打平或负 |
| **dsa_attn + index_score prefill** | **是，且最大** | 占 prefill 层 ~52–58%；优先挖这里才可能动 TTFT |

---

## 6. 原始数据指针

- Prefill/Decode stock 明细 JSON：`kernel-harness/.../llm_flops_style/results/glm5_{prefill,decode}_swapped_perf.json`（用 `stock_ms`）
- Harness 复现摘要：`/home/ubuntu/wwxq/bench_results/harness_reproduce_20260724_101710/SUMMARY.md`
- PSUM 矫正详细分析：`glm52_opt/e2e_psum_contig_prefill_analysis.md`
