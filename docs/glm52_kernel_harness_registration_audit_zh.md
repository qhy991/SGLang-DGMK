# GLM-5.2 Kernel-Harness 正收益算子与 SGLang 注册最终审计

审计日期：2026-07-30

审计分支：`agent/infini-kernel-glm52-verified-e2e`

审计提交：`ea65bfe8c`

本文把 Kernel-Harness 中曾经显示正收益的 GLM-5.2 算子，与后来按
SGLang 生产接口完成的 eager、CUDA Graph、完整调用边界和正确性实验合并
审查。它回答三个不同问题：

1. 历史 microbenchmark 是否曾经显示正收益；
2. 是否值得保留一个显式诊断注册；
3. 是否已经足够强，可以作为默认关闭的端到端候选或生产默认。

## 1. 最终结论

截至本次审计：

- **生产默认候选：0 个。**
- **值得带到真实 GLM-5.2 checkpoint 环境做端到端 A/B：3 个精确
  shape 路由。**
  - fused QKV-A prefill direct-N/K；
  - fused MoE W13 decode BM16 two-SM。
  - attention o_proj decode direct-N/K，仅 M16。
- **只值得保留显式诊断入口、不应宣传端到端加速：4 条路径。**
  - FlashMLA sparse decode PTX/SASS；
  - MoE W2 decode BM16；
  - W13 prefill PSUM；
  - W2 prefill PSUM。
- 其余历史候选不应进入生产候选注册。它们可以留在 archive 中用于复盘，
  但不应由 `serving_safe`、裸 `e2e_candidates` 或普通部署命令选中。

三个端到端候选如下：

| 优先级 | 算子与精确边界 | 本地最弱有效证据 | 当前建议 |
|---:|---|---:|---|
| P0 | fused QKV-A prefill，M4096/N2624/K6144，packed UE8M0 | 完整 eager apply 最弱 1.078588x；完整 projection region 最弱 1.083918x | L2：已经注册，默认关闭，去外部 checkpoint 环境做单算子 A/B |
| P1 | fused W13 decode，E32/slab1024/K6144/N4096，BM16 two-SM | 16 个 eager/graph lane 全过；完整 region 最弱 1.034255x | L2：已经注册，默认关闭，去外部 TP8/DP8/EP8 环境做单算子 A/B |
| P2 | attention o_proj decode，M16/N6144/K16384，direct-N/K | M16 eager 三组最弱 estimator 1.0520x；graph 三组最弱 1.3481x | L2：将现有注册收窄到 M16；M32 必须在调用前回退 stock |

这里的“已经注册”不等于“已经生产加速”。三者都还缺少真实 checkpoint、
多 rank 命中一致性、服务 TTFT/TPOT/throughput 和 rank-max latency 验收。

## 2. 本审计采用的注册等级

| 等级 | 含义 | 允许的启用方式 |
|---|---|---|
| L0 | 不注册为运行时候选 | 只保留报告、源码或 archive |
| L1 | 显式诊断注册 | 必须指定 profile、op、精确 provider；永不随裸 profile 自动选择 |
| L2 | 默认关闭的端到端候选 | 精确 shape/ABI fail-closed；允许去 checkpoint 环境做单算子 A/B |
| L3 | 生产默认 | 真实 checkpoint、多 rank 和服务指标全部通过后才允许 |

这次没有任何 L3 算子。

L2 的支持集合可以只有一个 shape。一个算子不需要所有 M bucket 都变快；
需要满足的是：**每个被注册为 candidate 的 shape 自己通过全部必需边界，而
所有未通过或未测试的 shape 在 candidate 调用前回退 stock。** 因此
`o_proj M16` 通过而 `M32` 失败时，正确做法是注册 M16，不是同时拒绝 M16，
也不是让 M32 一起命中。

## 3. 为什么旧 Kernel-Harness 的正收益不能直接注册

主仓库目前有 273 份 `runs/glm52/*/*/result.json`。其中：

- 179 份写有 `performance_ok=true`；
- 覆盖 14 个历史 task；
- **0 份是 official evidence**；
- 全部还是 schema 1.0/provisional，部分运行记录为 dirty worktree。

更关键的是，旧 harness 的 FP8 GEMM 基线使用 FP32 block scale，而当前
SGLang/DeepGEMM 生产路径使用 packed int32 UE8M0 scale。旧基线额外承担
scale 转换或走了较慢的实现，因此小于约 1.6x 的旧收益本身并不足以证明超过
当前 SGLang。即使旧数字超过 1.6x，也仍需排除以下情况：

- candidate 把 scale packing 移出计时，而 stock 留在计时内；
- 只计目标 `__global__`，没有计 Python/C++ dispatch、descriptor 和 enqueue；
- eager 提交看起来更快，但生产 decode 实际走 CUDA Graph replay；
- 单独 gate/up/down GEMM 与 SGLang 的 fused W13/W2 grouped-GEMM ABI 不同；
- candidate 输出虽然满足 leaf 容差，却改变了 score/top-k 等直接消费者；
- candidate 实际回退到 stock，或 graph capture 后 replay 的不是新 kernel；
- 只做一轮或只看 pooled median，收益落在约 ±4% 的噪声区。

DeepGEMM 的 production ABI 是 fine-grained FP8：activation 采用 1x128 scale，
weight 采用 128x128 scale；Blackwell 路径由 `tcgen05.mma` 原生消费 packed
UE8M0。MoE decode 还要求 masked grouped layout 才能保持静态地址并兼容
CUDA Graph。方法依据可参照 KernelWiki 的 `kernel-deepgemm` 和
`kernel-grouped-gemm` 页面。

## 4. 历史 14 个正收益 task 的逐项结论

下表的“旧 best”是旧 schema 1.0 文件中最好的 conservative speedup，只能
用于发现候选，不能作为注册结论。

| 历史 task | 旧 best | 生产接口复核 | 最终等级与明确结论 |
|---|---:|---|---|
| `fused_qkv_a_decode` | 3.3389x | M16 leaf eager 约 1.0405x、graph 约 1.3162x，但包含 quantize/dispatch 的 apply 只有 0.9762x；M32 稳定性也未过门槛 | **L0。无端到端加速，不注册为 E2E 候选。** 旧 legacy 路径只能用于复盘 |
| `q_b_decode` | 5.5020x | 旧大数字主要是 FP32 scale/packing 分母；生产 stock 已是 two-SM，计划中的 one-SM 比较前提无效 | **L0。没有可晋级 candidate。** |
| `o_proj_decode` | 1.6075x | exact SGLang apply 的 M16 eager 和 graph 全部通过；M32 eager 未在每个 series 达到 1.03x | **L2，仅 M16。** M16 值得外部 E2E；M32 必须回退 stock |
| `o_proj_prefill` | 1.0666x | production packed UE8M0 下 DeepGEMM 约 0.980–0.993x；CUTLASS 最好约 0.312x | **L0。无加速，不注册。** |
| `index_q_upproj_decode` | 3.4513x | production WQ-B eager leaf 1.3147–1.4742x，但 graph 只有 0.8454–0.9266x；score/top-k 直接消费者有 3371/32768 个元素不同 | **L0。性能和直接消费者正确性都失败；从 E2E registry 移除。** |
| `index_k_proj_decode` | 1.3648x | genuine one-SM `tcgen05` topology 未通过精确 BF16 正确性 | **L0。错误结果，不注册。** |
| `index_k_prefill` | 1.1561x | packed production ABI 下 leaf 约 0.784–0.786x，containing region 约 0.883–0.890x | **L0。明确回退，不注册。** |
| `index_score_decode` | 1.0019x | 唯一旧正结果只有 repeat=1，属于噪声；重复 1.03x 门槛失败 | **L0。无可复现加速，不注册。** |
| `moe_gate_proj_decode` | 2.0703x | SGLang 生产并不单独调用 gate N2048，而是 fused W13 N4096 grouped GEMM | **L0 对旧算子。** 不能按旧 ABI 注册；仅由下文精确 W13 candidate 替代 |
| `moe_up_proj_decode` | 2.2265x | 与 gate 相同，生产是 fused W13；旧单算子收益包含 ABI/packing 差异 | **L0 对旧算子。** 不单独注册 |
| `moe_down_proj_decode` | 1.4932x | 精确 W2 candidate 的 device kernel 1.138x，但完整 selected API 为 0.801642x | **L1 对精确 W2。** 仅诊断 enqueue/dispatch gap；不是端到端候选 |
| `moe_gate_proj_prefill` | 1.0568x | 生产应按 fused W13/PSUM 边界测试；单独 gate 结果不可达 | **L0 对旧算子。** 不单独注册 |
| `moe_up_proj_prefill` | 1.1784x | 旧 best 来自 repeat=1；生产 fused W13 PSUM component 虽约 1.05x，但完整 region 有 series 低于 1.03x | **L1 对 fused W13 PSUM。** 显式诊断，不进入 E2E 默认 |
| `moe_down_proj_prefill` | 1.0150x | 精确 W2 PSUM leaf 约 1.058x，但 containing region 约 1.011x | **L1。** 显式 PSUM 诊断，不进入 E2E 默认 |

因此，旧 14 个 task 的 archive candidate 仍然不能直接继承旧数字注册；
但是 `o_proj_decode` 后来重新实现并验证的 exact M16 direct-N/K 路由可以成为
L2。QKV-A prefill 和 W13 decode 也都是重新对齐 production ABI、重新构建
公平 stock/candidate、重新跑完整边界后产生的新实现，不是对旧数字的直接
继承。

## 5. 后续 production-native 实验的结论

后来的 campaign 不只看旧 14 个 task，还测试了 fusion、fixed-N/K、
tcgen05 one/two-SM、CUTLASS、DeepGEMM 参数和 PTX/SASS 等路径。

| 新实验 | 最有利的局部结果 | 失败或通过的强边界 | 最终结论 |
|---|---:|---|---|
| fused QKV-A prefill direct-N/K | GEMM leaf 1.173894x | eager apply 1.091133x，完整 region 1.114767x，所有 series 最弱 1.078588x | **L2，外部 E2E 候选** |
| fused W13 decode BM16 two-SM | leaf 约 1.043x | 16/16 eager/graph/region lane 通过，最弱 1.034255x | **L2，外部 E2E 候选** |
| attention o_proj decode M16 direct-N/K | eager 最弱 1.0520x；graph 最弱 1.3481x | exact `Fp8LinearMethod.apply`、正确性、capture/replay 和 fail-closed 全过 | **L2，仅 M16；M32 回退 stock** |
| W13 decode BM32 two-SM | pooled graph region 1.036620x | 单个必需 estimator 1.028125x | **L0 对 BM32 实现；不能用 pooled 值覆盖失败** |
| FlashMLA sparse decode PTX/SASS | leaf eager 1.14–1.26x | graph leaf 最低 1.0056/1.0082x；containing eager 约 0.76–0.87x | **L1，显式诊断；no-replacement** |
| W2 decode BM16 | device kernel 77.659→68.223 us，1.138x | 完整 API 93.760→116.960 us，0.801642x | **L1，显式诊断；no-replacement** |
| q_b prefill fixed-N/K | eager region 1.071–1.118x | graph region 仅 1.0128–1.0196x | **L0，不注册为生产候选** |
| dense gate/up decode fusion | 无 | graph 完整 fused region 0.8219–0.8770x | **L0，明确回退** |
| dense down decode | isolated kernel 约 1.695x | production boundary 的必需 estimator 低于 1.0x | **L0，局部收益被完整路径消除** |
| dense gate/up prefill | 无 | CUTLASS leaf 约 0.43–0.46x，fused region 约 0.74–0.75x | **L0，明确回退** |
| dense down prefill | 无 | DeepGEMM 约 0.994–1.008x；CUTLASS 约 0.335–0.608x | **L0，无稳定收益** |
| router FP32 GEMM prefill M4096 | graph region 1.0903x | eager leaf 0.3371x | **当前 L1。** 若增加只在 graph capture 选择 candidate、eager 调用前回退的 mode-selective dispatcher，可重新评为 L2 |

FlashMLA 的 PTX/SASS 修改确实进入了最终机器码，但“机器码不同”只证明新
kernel 被编译出来，不证明生产 critical path 更快。FlashMLA 是带 page table、
KV layout、LSE 和 combine 的完整 attention 接口；必须以实际 graph node 和
containing region 判定。方法依据可参照 KernelWiki 的 `kernel-flashmla` 页面。

## 6. 当前 SGLang 注册代码审计

当前分支已经具备三类注册：

1. `_DECODE` / `_PREFILL_FULL`：0720 archive 的 legacy 路径；
2. `_E2E_DECODE` / `_E2E_PREFILL`：fixed-N/K 路径；
3. `_HOTSPOT_DECODE`：FlashMLA、W13、W2 的精确 provider hook。

其中：

- 默认 `serving_safe` 不隐式选择算子，这一点是正确的；
- QKV-A prefill direct-N/K 是精确 L2 注册，保持默认关闭是正确的；
- W13 BM16 two-SM 是精确 L2 注册，保持默认关闭是正确的；
- FlashMLA 和 W2 只允许显式选择，这一点是正确的；
- legacy `decode_max/full` 只能视为 archive/诊断 profile，不能作为部署建议。

但 `config.py` 当前仍有一个需要修正的语义问题：

```python
_E2E_DEFAULT_OPS = frozenset({"o_proj", "moe_gate_proj", "moe_down_proj"})
```

新证据已经表明：

- `o_proj` decode 的 M16 通过，但 M32 eager 失败；
- `moe_gate_proj` 对应的旧 prefill PSUM 只有 component win；
- `moe_down_proj` prefill 的完整 region 约 1.011x。

因此裸 `e2e_candidates` 可以保留 `o_proj`，但必须把它收窄到 M16；另外两个
op 不应再自动选择。按本审计标准，正确状态应是：

```python
_E2E_DEFAULT_OPS = frozenset({"o_proj"})
_E2E_EXPLICIT_OPS = frozenset({"fused_qkv_a_proj"})

_E2E_DECODE["o_proj"].m_values = (16,)
```

M32、W13/W2 prefill PSUM 等旧路径必须显式写 op allowlist，并标成 L1
diagnostic。`index_q_upproj` 也不应再称为 E2E candidate；其 graph 和直接
消费者正确性已经失败。

这次审计只记录结论，没有擅自改变运行时代码。代码修正应作为一个独立提交，
同时更新 registry/config 单元测试和部署文档。

## 7. 三个值得做的外部端到端测试

外部环境不要一次同时打开多个候选。按以下顺序单算子 A/B：

### 7.1 QKV-A prefill

- 固定真实 checkpoint、请求集、TP/DP/EP、并发和随机种子；
- 只打开 `SGLANG_OPT_GLM52_FUSED_QKV_A_PREFILL_DIRECT_NK=1`；
- 核对 M4096/N2624/K6144、packed UE8M0、TP1 replicated projection；
- 比较 TTFT、prefill throughput、rank-max projection region；
- Nsys 中必须出现 `infini_kernel_glm52_fused_qkv_a_prefill_nk`。

### 7.2 W13 decode

- 只打开 `hotspot_candidates + moe_w13`；
- 核对 M16/M32、expected-M 4/5/8/9、E32/slab1024 和 two-SM provider；
- 比较 TPOT、decode throughput、rank-max W13→SwiGLU/quant→W2 region；
- 核对所有 rank 的 candidate hit/fallback 计数；
- Nsys 中必须出现 `infini_kernel_glm52_moe_w13_decode...`。

### 7.3 attention o_proj decode M16

- 注册集合必须只有 local M16，不得把 M32 一并打开；
- 核对 N6144/K16384、packed UE8M0、attention TP1、非 MTP decode；
- M16 candidate 必须同时覆盖 eager 和独立 capture/replay graph；
- M32、`TARGET_VERIFY`、prefill、TP8/local K2048 等调用必须记录为
  pre-invocation stock fallback；
- Nsys 中必须出现 `infini_kernel_glm52_attn_o_decode_nk`。

任一算子如果整模结果为 1.00x、轻微回退或波动，也应记录为有效结论，而不是
回头只引用 leaf microbenchmark。真正的结论应写成：

```text
该实现对精确 kernel/region 有局部收益，但在真实 checkpoint、并行拓扑和服务
负载下未得到可复现的端到端收益，因此保持 default-off / no-replacement。
```

## 8. 证据索引

仓库内已有的主要报告：

- [`glm52_verified_kernel_registration_zh.md`](glm52_verified_kernel_registration_zh.md)：
  当前 QKV-A、W13、FlashMLA、W2 注册与加速边界；
- [`../glm52_opt/e2e_gain_ops_repro_SUMMARY.md`](../glm52_opt/e2e_gain_ops_repro_SUMMARY.md)：
  旧 TTFT 收益未复现；
- [`../glm52_opt/history/e2e_candidates_20260723/INDEX.md`](../glm52_opt/history/e2e_candidates_20260723/INDEX.md)：
  旧候选归档；
- [`../glm52_opt/history/e2e_candidates_20260723/10_attn_o_decode_fixed_nk/REPORT.md`](../glm52_opt/history/e2e_candidates_20260723/10_attn_o_decode_fixed_nk/REPORT.md)：
  o_proj fixed-N/K no-replacement；
- [`../glm52_opt/history/e2e_candidates_20260723/08_moe_w2_prefill_psum/FINAL_REPORT.md`](../glm52_opt/history/e2e_candidates_20260723/08_moe_w2_prefill_psum/FINAL_REPORT.md)：
  W2 prefill PSUM component win；
- [`../glm52_opt/history/e2e_candidates_20260723/09_moe_w13_prefill_psum/FINAL_REPORT.md`](../glm52_opt/history/e2e_candidates_20260723/09_moe_w13_prefill_psum/FINAL_REPORT.md)：
  W13 prefill PSUM component win。

本机更严格的 production-native 原始证据位于：

```text
/home/qinhaiyan/glm52-v2-goal-runs/worktrees/*/kernel-harness/serving_native/evidence/
/home/qinhaiyan/glm52-hotspot-goal-runs/worktrees/*/kernel-harness/serving_native/evidence/
/home/qinhaiyan/Kernel-Harness/runs/glm52/
```

## 9. 一句话判定

**一个 shape 的稳定正收益足以注册该 shape。按此标准，当前 QKV-A prefill
M4096、精确 W13 decode 和 attention o_proj decode M16 值得做外部端到端
A/B；所有未通过的 shape 必须 fail-closed 回退 stock，且没有任何算子已经
具备生产默认资格。**
