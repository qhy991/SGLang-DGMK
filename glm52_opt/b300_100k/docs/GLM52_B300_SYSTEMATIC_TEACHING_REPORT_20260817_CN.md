# GLM-5.2 在 B300 上的端到端推理优化：系统实验报告与 CUDA 入门讲解

**报告日期：** 2026-08-17
**实验环境：** 8 × NVIDIA B300 SXM6
**模型：** GLM-5.2-FP8
**公开证据：** 本仓库中的 compact evidence bundle；原始 profile、主机元数据、模型资产和私有归档位置有意排除
**报告目的：** 用可审计证据总结 B300 上做过的优化，并让 CUDA 与多 GPU 推理经验较少的读者理解每一项修改为什么做、GPU 上发生了什么、结果是否足以进入生产。

---

## 0. 一页结论

### 0.1 最终结论

当前唯一正式晋级的方案是 **N6**：

- balanced static expert placement；
- router 使用 physical expert ID；
- DeepEP SM 数从 136 调整为 120。

在冻结的 100K 长上下文、fresh-server、request-rate=888888 的饱和 cached-prefill server E2E/TTFT 测试中：

| 指标 | Baseline | N6 | 变化 |
|---|---:|---:|---:|
| P50 TTFT | 2042.12 ms | 1933.67 ms | -5.31% |
| P90 TTFT | 3035.39 ms | 2891.13 ms | -4.75% |
| 总 token 吞吐 | 452544.29 token/s | 486640.21 token/s | +7.53% |

独立 holdout 再次得到：

- P50 -5.46%；
- P90 -6.73%；
- 吞吐 +7.40%；
- P50 为 5/5 paired wins；
- 输出 token 完全一致。

因此 N6 同时满足正确性、重复性和历史绝对性能门槛，是这个冻结 workload cell 的当前 accepted baseline/candidate；尚不能外推为所有部署 workload 的 production replacement。

### 0.2 扩大优化边界后的结论

按照“允许预先声明的局部回退、扩大搜索空间”的思路，v5 temporal placement 在 78 行 map 中替换 25 层，其余 53 层保留 N6；按 active MoE 计为 25/75，另有 3 个 dense/inactive 行不变。它通过了离线约束、外部 replay 和模型正确性，并在 CPU 严重降频的退化宿主机上相对 control 获得：

- P50 -68.89%；
- P90 -79.38%；
- 吞吐 +257.53%。

但它没有达到健康主机的 P50 与吞吐绝对门槛，所以状态是：

**RELATIVE_WIN_BLOCKED_BY_HOST_ANCHOR**

这表示“研究信号非常强，但不能替换 N6”，而不是“已经快了 3 倍”。

### 0.3 最重要的技术认识

1. 本项目真正影响端到端时间的主要不是某一个 CUDA 计算 kernel，而是：
   - 不必要的 MoE 通信轮次；
   - expert placement 造成的跨 rank 到达偏斜；
   - DeepEP notify/dispatch/combine 等待长尾；
   - CPU 进程调度与 affinity。
2. FlashMLA、QKV、MoE 等 leaf kernel 可以明显变快，但如果它们不在关键路径上，端到端 TTFT 可能不变甚至变差。
3. 最早的 chunk 修复取得 3.3× 到 6.56×，本质是“删除多余通信轮次”；这比单纯让某个 kernel 快 10% 更有价值。
4. 当前最近的正式测试是 **prefill/TTFT**，不是 decode；历史上另有一组独立 decode/MTP 测试取得约 1.32×。

### 0.4 公开证据边界

本报告只提交可审阅的代码、配置、专家 map、无 profiler 正式摘要、配对样本以及派生的因果分析。原始 Nsight 报告、服务日志、容器导出、模型权重、内部主机信息和迁移运维资料保留在私有证据归档中，不进入公开 Git 历史。

---

## 1. 本报告采用的判断规则

### 1.1 目标

目标不是让某个 kernel 的 microbenchmark 数字更漂亮，而是在冻结 workload 上：

- 降低首 token 延迟；
- 提高完整服务吞吐；
- 保持模型语义与输出正确；
- 能说明为什么更快；
- 能从归档证据中复现；
- 新候选失败时可以退回最后一个 accepted 版本。

### 1.2 不变量

任何正式 A/B 对比都必须保持以下条件不变：

- 模型权重与量化方式；
- 请求内容和随机 seed；
- cached/new/output token 分解；
- TP、DP、EP、attention-DP/CP 进程组；
- KV page、chunk、memory fraction；
- eager/graph、overlap 和 attention backend；
- 服务请求数、并发与到达率；
- 计时边界；
- 正确性门槛。

### 1.3 明确不做的外推

本报告不会：

- 用 100K prefill 结果代表 decode；
- 用单算子收益代表 TTFT 收益；
- 用 Nsys 下的时间代表无 profiler 的正式性能；
- 把 v5 在退化主机上的相对收益当成健康主机收益；
- 把不同日期、不同容器、不同 cache hit 的数字直接横向相减；
- 为没有正式 summary 的探索分支补造性能数字。

### 1.4 四个正交证据门

这些证据不是一条“越往后越高级”的单轴阶梯；候选必须同时通过四类彼此独立的门：

1. **语义/正确性门**
   - operator、layer、model 输出；
   - exact token 与 logprob tolerance；
   - shape、rank、fallback 和 side effect。
2. **正式性能门**
   - 无 profiler 的相邻 A/B 或 A-B-A；
   - P50/P90/吞吐、paired wins、漂移；
   - microbenchmark 只能支持迭代，不能替代该门。
3. **因果解释门**
   - Nsys 检查 timeline、rank skew、stream 和 collective；
   - 必要时用 NCU 回答具体 kernel 内部问题；
   - profile 不是比无 profiler E2E 更高等级的性能数字。
4. **复现/归档门**
   - revision、dirty state、命令、环境、原始样本；
   - source contract、hash、profile 和限制。

代码可编译、leaf 更快、server smoke 完成只是进入这些正式门之前的开发证据。

---

## 2. CUDA 初学者需要先理解的概念

## 2.1 CPU、GPU、kernel、SM 和 stream

可以把 GPU 想象成一座有很多车间的工厂：

- **CPU** 是总调度员，准备参数并发出工作指令；
- **CUDA kernel** 是一次工作指令，例如做 attention、矩阵乘或搬运 token；
- **thread block** 是一组并行工人；
- **SM** 是执行 thread block 的车间；
- **CUDA stream** 是按顺序提交工作的队列；
- **HBM** 是 GPU 的大容量高速内存；
- **L2/shared memory/register** 是不同层级、更靠近计算单元的存储。

一个 kernel 占用更多 SM 不一定让服务更快。如果通信 kernel 占满 SM，它可能挡住 GEMM 或 attention；反过来，给通信留得太少，又会延长跨 GPU 等待。因此 DeepEP 的 SM 数是一个系统级资源分配问题，不是“越大越好”。

不同 stream 理论上可以并发，但只有同时满足以下条件才会真正重叠：

- 数据依赖允许；
- 没有 event 或同步阻塞；
- SM、寄存器、shared memory 等资源还有余量；
- collective 没有等待最慢 rank；
- kernel launch 已及时到达 GPU。

所以“代码里创建了两个 stream”不是重叠证据，必须看时间线。

## 2.2 Prefill 与 decode

### Prefill

Prefill 是处理输入 token 的阶段。模型需要计算 hidden states、attention、MoE、KV cache 和 output head，为第一个输出 token 做准备。

当前正式 workload 有：

~~~text
90K 已缓存公共前缀
      +
10K 新 suffix
      ↓
增量 prefill 读取长 KV
      ↓
生成第 1 个 token
~~~

这不是重算完整 100K。90K 前缀已经进入 KV cache，但新 suffix 的 attention 仍会访问长上下文 KV。

### Decode

Decode 是首 token 之后逐 token 生成。每一步通常只增加少量 token，常受以下因素影响：

- kernel launch latency；
- KV 读取带宽；
- CUDA Graph；
- 小 batch 的低并行度；
- speculative decoding 的接受率。

当前正式实验 output_len=1，所以它测 TTFT，不能测稳态 TPOT。

## 2.3 TTFT、P50、P90、吞吐和 TPOT

- **TTFT**：请求到达测量边界后，到第一个 token 返回的时间。
- **P50**：一半请求比这个值快，一半比它慢。
- **P90**：90% 请求比这个值快，用来观察尾延迟。
- **吞吐**：单位时间内系统处理的 token 数。
- **TPOT**：首 token 之后每个输出 token 的平均时间。

TTFT 不等于某个 kernel 时间，也不等于所有 GPU kernel 时间之和。多 GPU、多 stream 的 kernel 会重叠；不同请求也会并发。

## 2.4 TP、DP、EP 和 CP

| 并行方式 | 拆分对象 | 直观理解 | 常见代价 |
|---|---|---|---|
| TP | 同一层的矩阵/张量 | 多张 GPU 合作完成同一个大算子 | all-reduce/all-gather |
| DP | 请求或 attention 数据 | 不同 rank 处理不同数据 | 负载和调度偏斜 |
| EP | MoE experts | 每张 GPU 保存一部分 experts | dispatch/combine A2A |
| CP | 一个请求的上下文位置/KV | 沿序列维拆分长上下文 | attention 通信与同步 |

TP=8、DP=8、EP=8 不代表需要 512 张 GPU。process group 可以在同一组 8 张 GPU 上重叠。

本项目必须特别说明：

- 用户目标口径是 CP=8、EP=8；
- 实际日志是 TP=8、DP=8、EP=8；
- 实际 SGLang 字段 attn_cp_size=1。

所以准确说法是“8 路 DP-attention/上下文分发 rank + EP8”，不能写成 attention CP8。

## 2.5 MoE 的 router、dispatch、expert compute 和 combine

MoE 可以理解成“每个 token 只去少数几个专家”：

~~~mermaid
flowchart LR
    A["输入 token"] --> B["Router 选择 Top-K expert"]
    B --> C["Dispatch：把 token 发到 expert 所在 GPU"]
    C --> D["Expert GEMM + activation"]
    D --> E["Combine：发回并加权合并"]
    E --> F["Residual / 下一层"]
~~~

### Router

Router 为 token 选择 expert，并产生组合权重。

- logical expert ID 是模型语义中的编号；
- physical expert ID 是经过 placement 后实际放到某张 GPU 的编号。

### Dispatch

Dispatch 需要：

1. 统计每个目标 rank 的 token；
2. 打包 token；
3. 跨 GPU 发送；
4. 等待各 rank 数据与通知到齐。

### Combine

Combine 将 expert 输出送回原 token 所属位置并按权重合并。

因此 MoE 关键路径是：

~~~text
router → dispatch → 等待最慢 rank → expert compute
       → combine  → 再次等待       → 下一层
~~~

只看每张 GPU 的总 token 数还不够。即使整场总量均衡，只要某一层、某一时间片有大量 token 同时涌向一张 GPU，所有 rank 仍会等待它。

## 2.6 FlashMLA 与 DeepEP

- **FlashMLA** 负责 MLA attention，包括读取长 KV、计算稀疏 attention 和输出 attention 结果。
- **DeepEP** 负责 MoE expert-parallel 通信，包括 dispatch、rank 间通知与 combine。

v5 的 Nsys 结果显示 FlashMLA 单次耗时几乎不变，而 DeepEP notify 长尾下降。因此 v5 改善的是通信/控制平衡，不是 attention 算得更快。

## 2.7 Nsys、NCU 与正式计时

| 工具 | 用途 | 不能做什么 |
|---|---|---|
| 无 profiler server timing | 正式判断 TTFT/吞吐是否变快 | 不能单独解释因果 |
| Nsight Systems / Nsys | 看 CPU/GPU/rank/stream 时间线与等待 | 不能用采集后的 TTFT 替代正式性能 |
| Nsight Compute / NCU | 看选定 kernel 的寄存器、occupancy、访存和 stall | 不适合盲扫整个服务 |

正确顺序是：

1. 无 profiler 判断是否真的有 E2E 变化；
2. Nsys 找到关键路径变化；
3. 只有定位到具体 kernel 后再用 NCU。

当前 v5 的差异定位在跨 rank wait/control-plane，而不是单 kernel，所以没有进行无目标的 NCU 扫描。

## 2.8 Amdahl 定律：为什么 leaf 快了，服务可能不快

若某部分占原时间比例为 p，这部分加速 s 倍，则理论总加速：

~~~text
总加速 = 1 / ((1 - p) + p / s)
~~~

如果一个 kernel 只占总时间 2%，即使它快 2 倍，E2E 理论上也只改善约 1%。如果它还与通信重叠，实际收益可能更小。

相反，删除每层重复的通信轮次，或者缩短最慢 rank 的等待，即使没有任何 GEMM 变快，也可能显著改善 TTFT。

## 2.9 局部回退与 fail-closed

两者并不矛盾：

- **局部回退**：设计阶段明确规定某些层/shape 使用新路径，其余继续用 accepted 路径。
- **fail-closed**：运行时进入未声明、未验证的配置时直接报错，不悄悄换路径。

v5 的 25 层选择属于显式组合；FlashMLA 的八个 M shape allowlist 也属于显式组合。未知 shape 若已经进入 selected provider，则必须报错，不能静默再跑一次 stock kernel。

---

## 3. 冻结的正式实验合同

## 3.1 硬件与模型

| 项目 | 值 |
|---|---|
| Host | 8×B300 test host |
| GPU | 8 × NVIDIA B300 SXM6 |
| 模型 | ${GLM52_MODEL_PATH} |
| 量化 | FP8 |
| 服务框架 | SGLang 派生仓库 |

模型路径以及部分 config/index/dataset hash 已记录；但 full weight shards、tokenizer、quantization artifacts 和稳定 model revision 的完整 hash 尚未闭环，因此还不能证明未来重跑使用逐文件相同的模型资产。

## 3.2 并行与服务

| 项目 | 值 |
|---|---|
| TP | 8 |
| DP / 上下文分发 rank | 8 |
| EP | 8 |
| attn_cp_size | 1 |
| Attention KV backend | FlashMLA KV |
| MoE backend | DeepEP normal |
| CUDA Graph | 关闭 |
| overlap | 关闭 |
| memory fraction | 0.82 |

## 3.3 请求

| 项目 | 值 |
|---|---|
| shared prefix | 90000 token |
| suffix | 10000 token |
| output | 1 token |
| requests | 110 |
| concurrency | 11 |
| request rate | 888888 |
| formal client primary seed | 0 |
| holdout client seed | 20260813 |
| server seed | 565849983 |
| KV page size | 64 |
| 真实 cached prefix | 89984 token |
| 重新计算的 prefix tail | 16 token |
| chunked prefill | 总 80384 / 每 rank 10048 |

由于 page size=64，90000 不是完整 page 的整数倍，所以实际 cache hit 是 89984，剩余 16 个 prefix token 会重算。这一细节解释了日志中 observed new token 数并非永远严格等于 10000。

request-rate=888888 表示尽快压入请求、并发上限为 11。它是饱和 server E2E，不是来自真实线上到达分布的 continuous-batching 评测；deployment 的 P95/P99、动态长度分布和排队 goodput 仍未覆盖。

## 3.4 晋级门槛

- 成功请求 110/110；
- median TTFT ≤ 2000 ms；
- P90 TTFT ≤ 5000 ms；
- total-token throughput ≥ 438000 token/s；
- 输出正确性通过；
- screen 至少五对相邻 baseline/candidate；
- 必须保留原始样本、paired 顺序、server 参数和哈希。

## 3.5 为什么使用 A-B 或 A-B-A

机器状态会随温度、后台进程、CPU/GPU 频率变化。只先跑完 baseline 再跑 candidate，可能把时间漂移误认为优化。

- A-B：相邻交替，减少慢漂移影响。
- A-B-A：candidate 前后各跑一次 control，可以检测 control 自身是否漂移。

如果 A 与最后一个 A 相差很大，就不能仅凭中间 B 说“优化成功”。

---

## 4. Accepted N6：逐项修改分析

N6 可以从概念上分成三项变化，但实现与实验上 N4 已经是“balanced placement + direct physical-ID router”的原子组合，N6=N4+DeepEP120。正式结果不能把总收益拆给某一项，也不能把早期 screen 的百分比相加。

## 4.1 修改一：balanced static expert placement

### 原问题

GLM-5.2 的 router 会把 token 分配给不同 experts。如果热点 experts 集中在少数 GPU：

- 热点 rank 收到更多 token；
- 其 expert GEMM 更晚完成；
- dispatch/combine 等待最慢 rank；
- 其他 GPU 即使已经完成，也不能进入下一层。

### 修改

根据观察到的 expert 使用分布，将 logical experts 重新映射到 physical slots，使热点 expert 尽量分散到 8 个 EP rank。

accepted map：

<code>glm52_opt/glm52_100k_x11_static_expert_map.json</code>

SHA256 前缀：

<code>36d132...</code>

### GPU 上发生什么

该修改不改变 expert GEMM 的数学公式，也不减少模型层数。它改变的是 token 在 GPU 之间的去向：

- 单个热点 rank 的峰值 token 负载下降；
- dispatch 后各 rank 更可能接近同时开始/结束；
- collective 与 notify 的尾部等待可能缩短。

### 风险与正确性

expert relocation 必须是每层完整 permutation：

- 每个 expert 仍存在且只出现一次；
- 每个 rank 持有的 expert 数不变；
- router ID 与物理权重位置一致；
- 权重、scale 和 expert metadata 必须一起映射。

否则可能出现“token 选中了 expert 17，却执行了另一份权重”的严重语义错误。

### 结论

它单独作为 N1 做过 screen：

- identity placement 的 median/max 负载比约为 1.253/1.686；
- balanced placement 约为 1.0007/1.0272；
- P50 2051.07 → 1962.35 ms，改善 4.33%，5/5 P50 wins；
- throughput +4.24%；
- 但 P90 3055.06 → 3336.21 ms，恶化 9.20%。

因此 **N1 单独被拒绝**：平均负载更均衡、P50 更好，并不代表尾延迟一定更好。它后来只作为 N6 的基础 primitive 重新参与组合验证。

## 4.2 N4 原子组合：balanced placement + router 输出 physical expert ID

### 原问题

静态 placement 之后，logical ID 与 physical ID 不再相同。如果 router 输出 logical ID 后，下游再做重复转换：

- 增加 transform/metadata 路径；
- 可能产生额外内存访问；
- 更容易让 dispatch 与权重布局的 SSOT 不一致；
- 可能使 profile 中“路由完成”和“通信开始”之间出现额外控制开销。

### 修改

N4 不是“只改 router”的独立 treatment；它同时启用 N1 的 balanced placement，并让路由结果直接使用与该 placement 一致的 physical expert ID，使 dispatch 直接消费真实物理位置。

### GPU/控制路径影响

目标不是让 router GEMM 本身更快，而是缩短：

~~~text
Top-K 结果 → ID 映射/转换 → dispatch metadata → DeepEP
~~~

这也是“让生产者直接写消费者需要的布局”的典型优化。

### 风险

physical ID 必须成为唯一事实源。若一部分代码使用 logical ID、另一部分使用 physical ID，会导致：

- token 发错 GPU；
- combine 权重顺序错误；
- correctness 可能偶发失败；
- profile 难以解释。

### 结论

N4 原子组合 screen：

- P50 2035.57 → 1993.82 ms，改善 2.05%，5/5；
- P90 恶化 0.53%；
- arm-median throughput 478593.75 → 465113.32，下降 2.82%；
- paired throughput 的中位方向曾为 +2.03%，与 arm median 相矛盾，暴露出非平稳噪声。

profile 还显示每 GPU 删除了约 316 到 466 次 post-router padded-ID mask，但减少小 kernel 次数本身不是晋级依据。

因此 **N4 被拒绝作为当时的独立候选**：吞吐回退超过允许范围。reviewed 原子实现被保留，随后只增加 DeepEP120，形成 N6 重新正式测量。N4 的 E2E 数字不能归因于 physical-ID router 单项。

## 4.3 修改三：DeepEP SM 136 → 120

### 原问题

DeepEP 通信 kernel 使用 SM 来做 token 搬运、通知和同步。分配太多 SM：

- 可能抢占 expert GEMM/attention 的执行资源；
- 破坏原有计算-通信重叠；
- 让局部 dispatch 更快但整层更慢。

分配太少 SM 又会使 dispatch/combine 本身过慢。

### 修改

在当前 100K workload 中把 DeepEP SM 数从 136 降为 120。

### 为什么减少资源可能更快

候选机制假设是资源平衡，而不是单 kernel 极值：

~~~text
DeepEP 更少独占 SM
        ↓
GEMM / attention 有更稳定的执行空间
        ↓
整条 MoE 关键路径更短
~~~

现有 matched Nsys 是 N6 组合候选，并没有只隔离 120-SM treatment；因此上图是合理机制假设，正式 E2E 只能证明 N4+DeepEP120 组合收益，不能单独证明“减少 SM 必然让 GEMM/attention 更稳定”。

### 风险

最佳 SM 数依赖：

- B300 的 SM 总量与调度；
- token 数；
- expert 分布；
- overlap 是否开启；
- prefill/decode 阶段；
- DeepEP 模式。

因此 120 不能被外推成其他 workload 的通用最优值。

### 结论

development sweep 中：

- 120 相对两个 136 anchor 中位：P50 -1.50%、P90 -12.80%、吞吐 +9.74%；
- 144：P50 +0.68%，更差；
- 112：P50 +2.12%，更差；
- 128：P50 +6.29%、吞吐 -8.08%，拒绝。

还尝试过让 dispatch 与 combine 使用不同 SM 数，但 DeepEP 的共享 QP/buffer layout 需要一致配置，启动期断言 fail-closed；没有绕过该不变量继续计时。

120 最终随 N6 组合晋级。早期 8K 实验中的 DeepEP 24 SM 属于不同版本/模式/workload，不能与这里的 120 直接比较。

## 4.4 N6 正式结果

### 五对 P50 原始样本

Baseline：

<code>[2039.98, 2042.12, 2068.38, 2056.02, 2040.72] ms</code>

N6：

<code>[1887.18, 1910.80, 1933.67, 2057.02, 1963.03] ms</code>

正式中位数：

| 指标 | Baseline | N6 | 变化 | paired wins |
|---|---:|---:|---:|---:|
| P50 | 2042.12 | 1933.67 | -5.31% | 4/5 |
| P90 | 3035.39 | 2891.13 | -4.75% | 3/5 |
| throughput | 452544.29 | 486640.21 | +7.53% | 4/5 |

N6 的 P90 CV 约为 10.8%，说明尾部仍有噪声。独立 holdout 仍取得 P50 -5.46%、P90 -6.73%、吞吐 +7.40%，显著增强了可信度。

### N6 matched Nsys 因果摘要

该 profile 只用于解释组合候选，不参与正式计时：

- dispatch progress event 的每 device 中位数约 616 → 654，+6.17%；
- cached_notify_combine P50/P90 分别下降约 43.65%/45.91%；
- combine 主 kernel P50 基本不变；
- direct physical-ID 路径每 GPU 移除了约 316 到 466 个 padded-ID mask。

这些观察支持“更少 post-router 控制工作、跨 rank combine 等待更短”的组合解释，但不能把变化分别归因给 balanced placement、physical-ID router 或 DeepEP120。

### 决策

**PROMOTE / KEEP ACCEPTED IN THE FROZEN CELL**

理由：

- 正确性通过；
- 正式 median 通过绝对门槛；
- P90 通过；
- 吞吐通过；
- screen 有稳定 paired wins；
- holdout 重复；
- accepted map 与结果哈希归档。

---

## 5. 扩大优化边界：temporal expert placement

静态 placement 只优化“整场总量”。但多 GPU MoE 的等待由逐层、逐时间片的最慢 rank 决定。v1 到 v5 的演进是在回答：

> 能否利用每 token、每层、每个时间片的路由轨迹，选择更能降低瞬时最慢 rank 的 placement，同时保留 N6 作为局部回退？

v1/v2/v5 的后期 development bracket 均发生在 CPU 频域已经退化的主机上。它们的相对 bracket 可用于筛选和诊断，但绝对值不能与健康时期的 N6 formal 直接横比。

## 5.1 基础工具：per-token route recorder

### 修改

在 router 的真实 Top-K 位置记录每个 token 选中的 expert ID，并验证记录：

- 覆盖预期层；
- shape 正确；
- expert ID 范围合法；
- 每个 step 与请求路径对应；
- seed 0/1/2 可以精确 replay。

### 为什么 router 是 SSOT

曾检查过 DeepEP hook，但 DeepEP 已经处在 ID 转换和 dispatch 之后，不适合作为“模型最初选择了谁”的唯一真值。router Top-K 是更靠近语义源头的 SSOT。

### 成本与风险

记录工具用于实验，不能默认留在生产热路径：

- 可能引入 D2H、同步或大文件写入；
- 改变 timing；
- 记录量可能非常大。

因此 recorder 用于离线建模与 replay，不用它的运行时间做正式性能。

## 5.2 v1：逐层 temporal hill-climbing

### 修改

以 N6 map 为起点，在每层交换不同 rank 上的 experts，目标函数关注：

- 时间片 critical-rank ratio；
- P90；
- 最大值；
- aggregate imbalance 上限。

只有训练窗口改善的层才替换；无活跃数据或无收益的层局部回退 N6。每层仍必须是完整 permutation。

### 设计优点

- 搜索空间大于静态总量平衡；
- 不要求所有层都采用新方案；
- accepted N6 始终是局部安全底座；
- holdout 不参与交换选择。

### 局限

局部 proxy 不等于真实通信时间。它没有完整模拟：

- DeepEP channel 和 packet；
- 每个 rank 的真实到达时间；
- CPU scheduler；
- 多层串联；
- GPU resource overlap。

### E2E 结果

相对 control 中位：

- P50 恶化 8.60%；
- P90 恶化 6.26%；
- 吞吐下降 2.64%。

### 决策

**REVERT**

它说明“离线 critical-rank proxy 改善”不足以保证服务变快，不能通过更换 workload 或门槛来挽救。

## 5.3 v2：communication-safe hybrid

### 修改

不直接接受所有离线获胜层，而是选择一部分层，增加通信安全预算：

- send 数不能恶化；
- remote send 不能明显增加；
- rank 级目标受约束；
- 未选层回退 N6。

### E2E 结果

相对 control：

- P50 恶化约 5.95%；
- P90 有改善；
- 吞吐有改善；
- 仍未达到绝对门槛。

### 为什么尾部和吞吐改善仍不能接受

当前冻结合同以 median TTFT、P90 和吞吐共同约束。不能为某个候选事后只选择它表现好的指标。P50 变差意味着典型用户体验退化。

### 决策

**REVERT**

## 5.4 v5：P50/rank-constrained whole-layer selection

### 核心修改

v5 不再只看全局平均，而是用 whole-layer 二进制选择：

- 每一层要么采用 proposal map；
- 要么完整回退 N6；
- 不在层内部混合未验证状态。

MILP 目标综合：

- compute mean；
- compute P50；
- compute P90。

约束覆盖：

- source rank compute；
- source rank send；
- remote send；
- destination compute；
- destination channel；
- 每个 rank 的最大预算。

训练/holdout 隔离：

- seed 1 用于训练/选择；
- seed 0 检查；
- seed 2 在解冻结后做外部 exact replay；
- holdout 不参与 solver 求解。

### 最终 25 个替换层

<code>3, 5, 7, 10, 11, 14, 19, 20, 22, 23, 27, 32, 34, 37, 43, 46, 50, 57, 58, 61, 64, 67, 68, 69, 71</code>

在完整 78 行 map 中，其余 53 层保留 N6；按 active MoE 计为 25 个 selected + 50 个 active fallback，另 3 个 dense/inactive 行不变。

v5 map SHA256：

<code>570e58026ae890bfd294a34786a095142ca71d14747c87606a536b20c415620a</code>

### 离线与外部 replay

seed 2 外部 replay：

- compute proxy mean 约 -1.05%；
- max-channel proxy mean 约 -0.505%；
- send counts 不增加。

这些是方向证据，不是 TTFT。

### 正确性

- 对两个 reference，generated tokens 均为 11/11 exact；
- max logprob absolute error = 1.5565e-4，小于 1e-3；
- mean logprob absolute error = 3.4867e-5，小于 1e-4。

### 退化宿主机 A-B-A

| Arm | P50 | P90 | throughput |
|---|---:|---:|---:|
| Control A | 10457.46 ms | 18401.27 ms | 87524.63 |
| v5 | 3119.99 ms | 3755.22 ms | 320683.49 |
| Control C | 9600.52 ms | 18028.12 ms | 91865.47 |
| Control median | 10028.99 ms | 18214.70 ms | 89695.05 |

相对 control median：

- P50 -68.89%；
- P90 -79.38%；
- throughput +257.53%。

Control A/C drift：

- P50 8.54%；
- P90 2.05%；
- throughput 4.84%。

相对 bracket gate 通过，但 v5：

- P50 3119.99 ms > 2000 ms；
- throughput 320683.49 < 438000；
- 只有 P90 通过绝对门槛。

### 决策

**CONTINUE AS RESEARCH; DO NOT PROMOTE**

它很可能修复了退化环境中的非线性 rank imbalance，但健康主机上的真实增量尚未测出。

## 5.5 v5 的 Nsys 因果解释

匹配 control/candidate trace span 只差 -0.189%，可进行结构性比较：

先强调计时边界：早期 N6 的 10 秒 Nsys capture 对应 profiled P50 约 11.55 秒、throughput 约 74.9k token/s；这些数字受 profiler、capture window 和退化主机影响，只用于查看时间线，绝不能与正式无 profiler 的 1.93 秒、486.6k token/s 横向比较。

| 观察项 | 变化 |
|---|---:|
| DeepEP dispatch calls/GPU | 294 → 340，+15.87% progress rate |
| dispatch notify wait sum/call | -17.23% |
| dispatch wait P90 | -27.81% |
| dispatch wait P99 | -65.02% |
| dispatch wait max | -21.18% |
| 大于 50ms 的等待时间 | -37.10% |
| combine notify sum/call | -37.17% |
| combine notify P90 | -39.73% |
| combine notify P99 | -43.10% |
| FlashMLA per-call P50 | +0.0008% |
| FlashMLA per-call P90 | -0.0145% |

解释：

- 在相同服务合同和近似相同的 10 秒 capture 窗口中，candidate 观察到更多 DeepEP dispatch progress events；结合 path/workload marker，它是 profiler 下“进展速率更高”的信号，但不严格等价于完成了更多相同 layer/step，更不能替代正式吞吐；
- notify kernel 的时间主要包含“等别人到达”，不等于它自己做了很多算术；
- v5 缩短的是跨 rank 到达与控制等待长尾；
- FlashMLA per-call 分布几乎不变，不支持“attention 单次 kernel 变快是主要原因”的解释；调用排布和 overlap 仍需通过 timeline 一并判断。

因此没有针对 v5 进行无目标的 NCU 采集：当前假设不在单 kernel 内部。其他历史 kernel 候选仍保留了 NCU reports。

---

## 6. 从 DeepSeek-V4-Flash/KDA 迁移到 GLM-5.2

## 6.1 迁移的不是一段代码，而是一种边界设计

参考策略的核心是：

- 用模型/shape 专用路径覆盖真正的热点；
- 预绑定固定资源；
- 对经过验证的 layer/shape 建立显式 allowlist；
- 未选择区域保留可靠路径；
- 不把所有阶段强行塞进一个 mega-kernel；
- 用 E2E 关键路径而不是 kernel 数量决定成败。

归档中保留了 DeepSeek-V4-Flash clustered MQA 参考源，以及面向 GLM 的 H32/SM103/Q16-Q32 specialization。

但 DeepSeek 与 GLM 的：

- head 数；
- head dimension；
- KV layout；
- page/index 结构；
- 稀疏模式；
- B300 tile/SM 资源；

并不相同，所以不能把参考 kernel 直接复制后宣称服务收益。

## 6.2 KDA/clustered MQA 状态

clustered MQA 的想法是让相邻 query CTA 共享同一个 KV tile，减少重复 KV 读取。迁移版本针对 GLM 的 H32/D128、page64 和 Q16/Q32 cluster 做了专用实现。

在真实 M=10048、context=100K 的 direct-paged leaf 测试中：

| 顺序 | Stock | Candidate | 结果 |
|---|---:|---:|---:|
| 顺序一 | 5.764 ms | 6.907 ms | candidate 慢约 19.8% |
| 顺序二 | 5.806 ms | 6.843 ms | candidate 慢约 17.9% |

selected-score multiset 正确，但 candidate 明显更慢。可能原因包括额外 shared-memory、寄存器、cluster 调度或 KV 复用不足；由于没有针对该候选运行 targeted NCU，不能把回退归因于其中任何一项。按照当时的 Amdahl 估算，它需要约 28% 的 leaf 降幅才值得进入服务，而实测方向相反。

### 决策

**REVERT, KEEP REFERENCE**

教训是：可以迁移优化原理，不能照搬为另一种 head/layout/并行合同设计的物理 kernel。

---

## 7. FlashMLA B3+B5 production band

这一方向不是一次完成的，而是经历了 exact-M10048、indexer、pipeline stage、对齐修复，最后才收敛为八个 shape 的 production band。

## 7.1 N23：exact-M10048 FlashMLA

### 修改

直接优化真实服务中最常见的 sparse-prefill 主 shape M=10048，并保持 stock/candidate 相同 ABI、输入和输出。

### 正确性

- 329252864 个 BF16 output bit-exact；
- 643072 个 FP32 LSE bit-exact；
- 模型生成 11/11 token exact；
- max/mean logprob error = 1.68e-4 / 2.30e-5；
- 服务 selected path 命中 702 次。

### 性能

- leaf 4.171 → 3.802 ms，改善 8.85%；
- E2E anchor：P50 1899.37 ms、P90 3123.55 ms、throughput 485845.87；
- N23：P50 1881.34 ms、P90 2795.49 ms、throughput 504490.82；
- P50 只改善 0.949%，P90 -10.50%，吞吐 +3.84%。

预先声明的 primary P50 门槛是至少 1%。它只差 0.051 个百分点，但不能在看到结果后四舍五入或修改门槛。

### 决策

**REJECT AT DEVELOPMENT GATE; RETAIN AS ITERATION BASE**

它没有进入 five-pair/formal promotion。

## 7.2 N24：N23 + indexer-Q 128 → 256 threads

### 修改

增加 CUDA block threads，让 indexer Q 阶段并行处理更多元素。

### 局部结果

- 6 个 operator case bit-exact；
- 真实 B=10048 的 indexer kernel 改善 26.5% 到 29.1%；
- 但 N6 trace 中该 kernel 总计只有约 9.17 ms/10s，理论 E2E 上限只有约 2.7 ms。

### E2E

development P50 曾改善 1.272%，但 formal 三对：

1. 1972.94 → 1898.91 ms，胜；
2. 1921.16 → 1968.19 ms，负；
3. 1977.07 → 2031.23 ms，负。

只得到 1/3。即使剩余两对都赢，最多也只有 3/5，因此数学 early-stop。

### 决策

**REVERT**

一次 development win 不能替代重复性。

## 7.3 N32/N35：NoPE producer staggering

N32 在 N23 上将 NoPE producer 的工作错开，leaf 再改善约 0.914%，保持 bit-exact，形成 N35。

N35 E2E candidate 为：

- P50 1888.87 ms；
- P90 2814.94 ms；
- throughput 503693。

但两次 bracket anchor 漂移；重试漂移 3.060%，超过预设 3% 门槛 0.060 个百分点。虽然表面 P50 方向约有 2.096% 改善，也必须判为 **NO DECISION**，不能申报。

## 7.4 N36/N37：三 stage index ring 与 16B 对齐

N36 把 index/scale/validity ring 从 2 stage 增加到 3 stage，希望让 producer/consumer pipeline 更充分。

但新增 24B validity array 使后续 tma_coord 失去 16B 对齐，首个同步 kernel 报 misaligned address。正确性门直接失败，因此没有性能值。

N37 显式加入 16B 对齐：

- shared-memory plan 232432B；
- B300 上限约 232448B，仅余 16B；
- output/LSE 全部 bit-exact；
- leaf 再改善 0.8157%。

这说明对齐不是“微小性能细节”，而是 TMA 能否正确执行的接口不变量。

## 7.5 N38：三 stage 版本 E2E

| 指标 | Control | N38 | 变化 |
|---|---:|---:|---:|
| P50 | 1929.865 | 1931.17 | +0.068%，更慢 |
| P90 | 3189.005 | 2826 | -11.38% |
| throughput | 484953.095 | 502162.49 | +3.55% |

虽然 leaf、P90 和吞吐有信号，primary P50 没有改善，故拒绝。更深 pipeline 还把 shared memory 推到硬件上限附近，维护与鲁棒性成本更高。

## 7.6 最终八 shape production band：原问题

GLM-5.2 的 100K 增量 prefill 在每 rank 上会出现一组接近 10K 的真实 M：

<code>9616, 9728, 9792, 9856, 9920, 9984, 10016, 10048</code>

通用 FlashMLA kernel 需要覆盖大量 shape，而 B300 对固定 shape 的 tile、pipeline 和工作区可以更激进地专用化。

## 7.7 修改

- 为八个实际生产 M 建立 B3+B5 CUDA-13.2 预编译 candidate；
- registry 只在 exact shape、phase、head、page、dtype 等条件匹配时选择；
- provider 进入后若 M 不在 allowlist，直接报错；
- .so 通过 SHA256 manifest 校验；
- 每 GPU 预分配最大 workspace，运行时只切 view，避免热路径反复分配；
- 未被 registry 选中的 M 继续使用 stock FlashMLA。

## 7.8 CUDA 层面的意义

专用 kernel 可以：

- 选择更合适的 tile；
- 减少边界判断；
- 调整 pipeline stage；
- 提高 B300 Tensor Core/SM 利用；
- 避免每次调用重新分配大工作区。

但它也可能：

- 增加寄存器或 shared memory；
- 降低 occupancy；
- 破坏与 DeepEP/GEMM 的重叠；
- 在未测 shape 上错误。

因此只允许八个通过测试的 shape。

## 7.9 正确性与 leaf 性能

- 两种测量顺序；
- 八个 shape；
- output 与 LSE 逐 shape bit-exact；
- leaf eager median 相对 stock 改善 8.82% 到 12.49%。

LSE 是 log-sum-exp，attention softmax 稳定计算的重要中间量。只比 output 不够；LSE 一致增强了 attention 数值正确性证据。

## 7.10 为什么没有晋级

8.82% 到 12.49% 是 attention leaf latency，不是 TTFT。后期饱和 server 测试受 CPU 退化污染，没有形成可以替换 N6 的健康主机 formal E2E win。

### 决策

**OPERATOR/LEAF-ADMITTED DEVELOPMENT CANDIDATE; NOT E2E-PROMOTED**

---

## 8. DeepEP dispatch 参数探索

## 8.1 原问题

DeepEP dispatch 的 packet/send/recv 配置会影响：

- 每次发送粒度；
- channel 利用率；
- 通知次数；
- 最慢 rank 长尾；
- 与 expert GEMM 的 overlap。

## 8.2 探索

测试包含 send 7/8/12/16/20，以及 recv 128/192/256 等组合。代表性 8-rank leaf probe：

- tokens=10048；
- hidden=6144；
- experts=256；
- Top-K=8；
- num_sms=120；
- combine 保持默认。

## 8.3 N17：default send 6 → 7

正确性 11/11 exact，但 E2E：

| 指标 | Control | send7 | 变化 |
|---|---:|---:|---:|
| P50 | 1907.925 | 1973.37 | +3.43% |
| P90 | 2829.96 | 3368.38 | +19.03% |
| throughput | 502826 | 482359 | -4.07% |

小幅增加 chunk 并没有减少尾部，反而显著放大 P90，直接拒绝。

## 8.4 N19：dispatch/combine 联合参数

候选设置 dispatch send/recv=32/256、combine=16/256：

- correctness exact；
- development P50 曾改善 1.18%；
- formal 前两对分别慢 1.23% 和 2.56%；
- 0/2 后已不可能满足最终 paired-win 门槛，数学 early-stop。

### 决策

**REVERT**

## 8.5 N20：只修改 combine send 6 → 16

development 曾显示：

- P50 -1.77%；
- P90 -10.73%；
- throughput +3.96%。

但 formal 只有 1/3 wins，后两对吞吐明显下降，提前停止并拒绝。这是“development 看起来很好、formal 无法重复”的典型案例。

## 8.6 后续 send16 real-shape leaf 结果

| 指标 | Baseline | send16 | 变化 |
|---|---:|---:|---:|
| 最慢 rank median dispatch | 1.29894 ms | 1.26709 ms | -2.45% |
| wall | 2.41001 ms | 2.37317 ms | -1.53% |
| 最慢 rank P90 dispatch | 1.39760 ms | 1.28387 ms | -8.14% |
| P90 wall | 2.50307 ms | 2.39083 ms | -4.48% |

send8/12 只有小幅信号；send20 和 recv 扩展没有形成正式晋级。

## 8.7 Server E2E 结论

send16/20 在退化 host 上出现过相对 P50 信号，但 control anchor 漂移和绝对门槛失败，因此不能接受。

### 决策

**REVERT / RETAIN NEGATIVE EVIDENCE**

---

## 9. Equal DP all-gather

## 9.1 先区分两个候选

这一家族包含两个不同 treatment：

1. **N5 variable-length allgatherv + symmetric reduce-scatterv**
   - 目标是按各 rank 实际长度传输；
   - 静态路径审计即发现它不覆盖冻结 cell 的高价值边界；
   - 没有进入 GPU/E2E 计时。
2. **后续 exact-fill equal-chunk all-gather**
   - 只处理八个 rank 行数完全相同的情况；
   - CPU admission test 通过；
   - 在线路径实际没有 material change，表现为 no-op。

以下原理和实现主要描述第二个候选。

## 9.2 原路径

SUM_LEN 模式下，常见路径把每个 rank 的 token 放进全局 buffer 的不同 slice，其他位置补零，然后 all-reduce。因为各 rank 的非零 slice 不重叠，all-reduce 的“加法”实际上被用作 gather。

这会做一些并不需要的事情：

- 对零填充区域做 reduction；
- 传输与计算 reduction 语义；
- 可能增加 buffer 处理。

## 9.3 修改

当且仅当：

- TP size 与 DP size 匹配；
- attention TP size=1；
- SUM_LEN；
- 每个 rank 的 authoritative aligned row count 完全相等；
- 行数之和精确填满 global buffer；

则使用 native all-gather 填充全局 buffer；combine 仍保持已验证的 stock reduce-scatter。

不等长、tail 或条件不满足时，使用 stock 路径。

## 9.4 为什么理论上可能更好

all-gather 表达的是“收集不同数据”，比“把互斥非零 slice 做求和”更贴近真实语义，可减少无用 arithmetic 和流量。

## 9.5 正确性风险

MoE 下游会读取整个 global buffer。如果各 rank 行数之和没有精确填满 buffer：

- 尾部可能未初始化；
- expert 输入出现垃圾值；
- 错误可能只在某些 tail shape 出现。

因此 exact-fill 是硬条件，CPU-only 三 case admission test 覆盖 equal/unequal/tail。

## 9.6 路径与 E2E 结果

静态路径审计发现，这个候选选错了高价值边界：

- 冻结 cell 的 DeepEP sparse layers 与 dense layers 使用 SCATTERED 路径；
- attention TP=1，使 78 层 attention→MLP 边界的相关通信已经接近 trivial；
- 剩余 DP gather 主要只在 1-token terminal LM-head/logits 边界。

也就是说，即使 equal all-gather 本身完全正确，它在该 workload 中的调用频率和关键路径占比也接近零。实际运行没有形成 material path change，表现为 no-op/equivalence，没有可申报服务收益。

### 决策

**REVERT / NO E2E EFFECT**

---

## 10. Scheduler、overlap 与 prefill delayer

## 10.1 Native SBO：shared expert 与 routed expert overlap

### 假设与修改

MoE 中 shared expert 处理所有 token，routed experts 只处理 router 选中的 token。若两条分支独立，可以在不同 stream 上执行并在第一次共同消费者处 join。

N6 已经较早启动 shared expert；SBO 候选调整 event/stream 依赖，尝试把 shared expert 与 routed-down GEMM/combine 更晚地重叠。

### 正确性

- 11/11 generated tokens exact；
- max/mean logprob error = 1.68e-4 / 3.03e-5。

### E2E

| 指标 | Control | SBO | 变化 |
|---|---:|---:|---:|
| P50 | 1917.79 | 1951.58 | +1.76% |
| P90 | 3438.36 | 3500.84 | +1.82% |
| throughput | 467911.32 | 467556.12 | -0.08% |

### 决策

**REVERT**

数学正确不等于物理 overlap 有效；新的 join 时机没有缩短关键路径。

## 10.2 Native overlap scheduler

### 假设

GPU 执行当前 batch 时，让 CPU scheduler 提前准备下一 batch，可能隐藏 host 调度时间。两臂唯一 server-args 差异是 disable-overlap-schedule true → false；同时关闭一个实际零命中的 clustered-MQA guard，避免 kernel path 混入差异。

### 风险

- host 与 device 的 current/next batch 推进顺序变化；
- DP8 各 scheduler 可能在不同时间准备或提交 batch；
- cached-prefill 的同步 progression 被破坏；
- queue、波次或 collective 到达顺序改变。

该候选没有对应因果 profile，因此不能把回退归因于某一个机制。

### 结果

- correctness 11/11 exact；
- max/mean logprob error = 2.73e-4 / 5.80e-5；
- candidate P50 2687.84 ms；
- anchor P50 1908.96 ms，candidate 恶化约 40.8%；
- throughput 391109 vs 477659，下降约 18.1%；
- P90 3181.75 → 2734.50，看似改善。

这里更低的 P90 不是健康的 tail win：整个请求分布被压到约 2.7 秒的较慢窄区间，P50 与吞吐均大幅恶化。

### 决策

**REVERT**

E2E 只证明在这个同步 DP8 cached-prefill cell 中，启用 native overlap scheduler 明显回退；确切内部原因未知。双 CUDA stream 的资源竞争解释只适用于前面的 SBO 类候选，不能直接套用到这里。

## 10.3 Synchronous PrefillDelayer

### 假设

对 prefill 启动时机做小幅延迟或同步，可能让 rank 更整齐地进入 collective，减少尾部。

具体修改移植了一个小型 sync-scheduler 兼容补丁，复用已有的五整数 DP all-gather negotiation。每次调度判断增加一次小型跨 rank 协商：它可能减少错位，也会给正常 median 路径增加同步。

### 正确性

- 4-process distributed test 通过 mixed → delay → wait_timeout；
- 模型 11/11 exact；
- max/mean logprob error = 2.19e-4 / 3.46e-5。

### 结果

- P50 2067.43 vs 1945.88，恶化约 6.25%；
- P90 改善约 6.72%；
- throughput 约 485212，接近不变。

### 决策

**REVERT**

不能为了 P90 改善而接受典型请求 P50 退化。

## 10.4 其他 overlap 探索

- BLACKWELL_OVERLAP_SHARED_EXPERTS_OUTSIDE_SBO 在 Blackwell + DeepGEMM + DeepEP-normal 下只注册空 hook，是 no-op，在 timing 前即拒绝。
- 历史 TBO M10048/x11 最好 P50 约 2178.96 ms、request throughput 3.62 到 3.88 req/s，弱于 N6，未进入当前正式复测。

---

## 11. CPU affinity 与宿主机频率异常

## 11.1 发现的硬件/平台异常

主机暴露 256 个逻辑 CPU：

- socket 0 的全部 128 个逻辑 CPU约 500MHz；
- socket 1 低半 64 个逻辑 CPU约 500MHz；
- 只有 socket 1 高半 64 个逻辑 CPU，也就是 32 个物理核及 SMT sibling，稳定约 3.6GHz。

已检查但未发现：

- thermal throttling 证据；
- RAPL 功耗限制解释；
- GPU Xid；
- NVLink/GPU health 故障。

因此根因只能准确写成：

**CPU frequency-domain/platform fault**

不能进一步断言是 BIOS、驱动或某个硬件部件。

## 11.2 旧 affinity 的问题

外层 taskset/cpuset 本来只允许进程使用健康 CPU，但旧 set_gpu_proc_affinity() 会根据全局 CPU 编号重新计算 worker mask，子进程可能逃出父 mask，重新跑到 500MHz 区域。

这违反了一个重要的系统不变量：

> 父进程授予的 CPU mask 是子进程可用 CPU 的上限。

## 11.3 修改

隔离候选实现：

- 如果父 affinity 已受限，则父 mask 是 SSOT；
- 从 sysfs 读取 physical_package_id、core_id 和 SMT sibling；
- 同一物理核的 sibling 放在同一组；
- 8 个 GPU worker 获得平衡、互不重叠的物理核分区；
- 物理核不足、拓扑非法或分区失败时 fail closed；
- 父 mask 未受限时保留历史行为。

## 11.4 为什么按物理核分组

两个 SMT logical CPU 共享同一物理核的执行资源。若把同一核的两个 sibling 分给两个重负载 worker：

- 看似分配了两个 CPU；
- 实际仍争抢一个物理核；
- latency 会抖动。

按物理核分组可以避免不同 GPU worker 共享 sibling。

## 11.5 验证

- 2 个 focused unit tests PASS；
- 23-file runtime source contract VALID；
- process-tree affinity audit PASS；
- accepted N6 仓库未被修改，候选保存在隔离 repo。

这些验证证明 affinity 分区合同和进程树约束正确，不等于完整模型输出 correctness。下面是退化 host 上的 **single-arm diagnostic**；treatment 同时包含 v5 map、cpuset patch 与 topology partition，不是 paired promotion，也不能估计 affinity 的独立增量：

| Treatment | P50 | P90 | throughput | 结论 |
|---|---:|---:|---:|---|
| 32 healthy physical cores + SMT | 2716.96 | 2899.52 | 412371.30 | P50/吞吐失败 |
| physical-only，一核一线程 | 2752.50 | 2851.30 | 412677.09 | P50/吞吐失败 |

去掉 SMT 没有恢复历史门槛，支持“纯 SMT contention 不是充分解释”；由于缺少健康 host paired treatment，不能进一步量化物理核数量/频域与 affinity 各自的贡献。

### 决策

**CORRECTNESS/ROBUSTNESS RESEARCH CANDIDATE; NOT PERFORMANCE-PROMOTED**

---

## 12. 更早的 B300 实验：不同 workload，不能与 100K 正式数字混合

本节数字来自较早的 B300 调查与 handoff，使用不同请求、阶段、容器或计时合同；它们用于说明已经验证过的机制，不属于最近 N1–N40 的同合同 promotion 序列。这里的“正结果”不能改变“当前只有 N6 正式晋级”的结论。

## 12.1 Chunked prefill 的 DP 二次除法修复

### 原问题

配置 chunk=1024 后，DP-attention 路径又按 DP8 除一次，导致每 rank runtime chunk=128。

较小 chunk 不只是多启动几次 attention，它会让约 75 个 active MoE layers 的 A2A dispatch/combine 轮次约增加 8 倍：

~~~text
相同 token 总量
chunk 1024 → 较少的 forward/MoE 轮次
chunk 128  → 约 8 倍 forward/MoE 轮次
~~~

### 修改

把每 rank runtime chunk 从 128 恢复到 1024，使配置含义与真正执行一致。

### 结果

| Workload | 修改前 | 修改后 | 改善 |
|---|---:|---:|---:|
| full 8192 prefill TTFT | 79.05 s | 15.86 s | 4.98× |
| cached 64K + 1024 incremental | 2.14 s | 0.65 s | 3.3× |

再结合该早期 cell 的 DeepEP normal/num_sms=24，full 8192 降到 12.05 s，相对原始约 6.56×。

### 为什么这是最大收益

它没有把某个 kernel 加速 6 倍，而是删除了大量重复的全模型层执行与跨 GPU 通信轮次。这符合“先删除工作，再加速剩余工作”的原则。

### 决策

**POSITIVE HISTORICAL CELL; NOT CURRENT PROMOTION**

但它的绝对值不能与当前 100K/N6 横向比较。

## 12.2 Decode MTP/EAGLE

### 原理

speculative decoding 使用一个 draft 路径一次预测多个 token，再由主模型验证。若接受率高，主模型每次昂贵 forward 可以提交多个 token。

### 配置

- 64K context；
- batch=8；
- output=512；
- EAGLE；
- 1 speculative step；
- Top-K=1；
- 2 draft tokens。

### 结果

| 指标 | Baseline | MTP | 变化 |
|---|---:|---:|---:|
| TPOT | 32.15 ms | 24.33 ms | 改善 |
| throughput | 248 token/s | 329 token/s | +约 32% |
| acceptance length | 1 | 1.87 | 相对单 token 提交 +87% |

E2E 约 1.32×。

最大 draft token 数为 2，因此 acceptance ratio 约为 1.87/2=93.5%；这里的 93.5% 是两枚 draft 的接受比例，不是相对 baseline 的“额外 token 百分比”。

### 决策

**POSITIVE HISTORICAL DECODE CELL; NOT CURRENT PROMOTION**

这是本批实验中真正的 decode 结果，与 output=1 的 100K prefill 完全不同。

## 12.3 Fused QKV-A

### 原理

把相邻 projection 或准备步骤融合，减少：

- kernel launch；
- 中间 tensor；
- HBM 读写；
- layout 转换。

### 结果

- leaf 约 2.4×；
- bit-exact；
- decode E2E TPOT 33.26 → 34.44 ms，基本无收益且略退化。

历史 trace 呈现明显的通信主导现象，但跨 GPU/stream 的累计 kernel duration 不能当作 critical-path 占比，也不能据此计算可靠的 Amdahl 比例。leaf 约 2.4×、E2E TPOT 却从 33.26 ms 变为 34.44 ms，已经足以证明局部收益没有转化。

### 决策

**LEAF WIN, E2E REVERT**

这是 Amdahl 定律最典型的例子。

## 12.4 MoE masked alignment 128 → 16

### 原理

expert GEMM 常把 token 行数补齐到 tile/alignment。小 expert load 如果从 5 行补到 128，会计算大量 padding；改成 16 可减少浪费。

### 结果

- 低 expected_m 的 decode leaf 约 1.13 到 1.14×；
- 全局 align16 会让 prefill 只剩原性能的 0.24 到 0.56×；
- 即 prefill 变慢约 2 到 4 倍。

### 为什么阶段不同

decode 的每 expert token 数通常很小，padding 浪费突出；prefill token 多，较大 tile 可能更适合 Tensor Core、访存和 occupancy。一个全局常量无法同时最优。

### 决策

**DO NOT SET GLOBALLY**

未来只能做 phase/shape bucketed guard，并为每个 bucket 单独验证。

## 12.5 手写 PTX

### 原理

PTX 是更低层的 NVIDIA GPU 中间指令。手写 PTX 可以精细控制 load、Tensor Core 和寄存器，但维护与正确性风险很高。

### 结果

候选约 152 到 329 µs，参考约 74 µs，并存在错误。

### 决策

**ABANDON**

低层代码不天然更快；编译器、pipeline、寄存器压力和边界条件都可能使手写版本更差。

## 12.6 MoE PSUM / fusion

### 假设

把 partial sum、激活或相邻写回融合，减少中间 tensor 和 HBM 流量。

### 结果

没有 E2E win；代表性 PSUM 约慢 1.7% 到 2.3%。

### 决策

**REVERT**

可能原因包括额外指令、寄存器压力、较差 tile 或破坏 overlap；现有证据不支持继续扩张该物理 fusion。

## 12.7 MoK megakernel：不要与 MTP 混淆

### 概念区别

- MTP 是多 token 预测，属于 speculative decode。
- MoK 是把 MoE 的多个计算/通信阶段做成更大的 GPU 执行边界，使用 MXFP8、symmetric memory 等；它不是多 token 预测。

### 解决过的稳定性问题

旧 MoK 在 ragged DP rank 切换 DeepEP/MoK 时出现 timeout/OOM。修复包括：

- all-rank arm consensus；
- transition drain；
- EP barrier；
- 约 1.08GiB workspace 预分配。

修复后：

- x10：100/100 稳定；
- x12：120/120 稳定；
- x16：160/160 稳定。

### 性能和资源

- x10 P50 恶化约 8.40%，throughput -14.42%；
- x12 P50 恶化约 7.55%，throughput -8.36%；
- x16 两次结果不重复；
- 每 rank 内存约从 109.45GiB 增至 207GiB；
- KV capacity 从约 2103808 降至 205888。

这说明“更大的物理 kernel”可能解决一部分 launch/边界问题，也可能带来巨额 workspace、容量下降、同步与 residency 风险。

### 决策

**STABILITY/DEBUG SUCCESS, PERFORMANCE REVERT**

## 12.8 其他 reviewed N-series 候选

### N2：single-D2H control read

**假设：** 将多个细碎 GPU→CPU 控制值读取合并为一次 D2H，减少 host synchronization 和 PCIe/NVLink 控制往返。

**结果：**

- P50 +0.23%；
- P90 +3.16%；
- throughput -5.94%；
- 1/5 paired wins。

**解释与决策：** 控制读取减少并没有缩短服务关键路径，吞吐明显回退。**REVERT**。

### N3：DP-attention local control broadcast

**假设：** 在一个 rank 形成控制决定，再向 DP-attention 组广播，删除每 rank 重复的 control construction。

**正确性与结果：**

- correctness exact；
- P50 +1.88%；
- P90 +5.24%；
- throughput -3.59%；
- 1/5 paired wins。

广播自身、rank 到达与同步成本超过了删除重复控制的收益。**REVERT**。

### N10：FlashInfer Top-K 替换

**假设：** 用另一套通用 Top-K kernel 替换 SGLang 当前 router Top-K。

**leaf：**

- SGLang：2.4461 ms；
- FlashInfer：4.294 ms。

candidate 明显更慢，未进入 server。通用库不一定适合 GLM 当前 shape、dtype 和 layout。**LEAF REVERT**。

### N11：显式 flashmla_sparse 路径

**假设：** 强制选择 explicit sparse FlashMLA 路径，验证 backend dispatch 或隐式选择是否留下性能。

| 指标 | Control | Candidate | 变化 |
|---|---:|---:|---:|
| P50 | 1891.15 | 2090.68 | +10.55% |
| P90 | 2826.85 | 3266.26 | +15.54% |
| throughput | 499383 | 445728 | -10.74% |

**REVERT**。显式命中某个 attention backend 不代表它在完整 DP8/EP8 服务路径更快。

### N12：ninth-route shared expert

**概念：** shared expert 处理所有 token，不应简单等同为 router 额外选择的第九个 routed expert。将其塞进同一 route/dispatch 表示可能改变 ownership、通信和 combine。

历史候选：

- ninth-route P50 2078.67 ms、throughput 442544；
- balanced nonfused P50 1909.35 ms、throughput 501987。

候选明显更差，且语义边界更复杂。**REVERT**。

### N15：Top-K workspace reuse

**假设：** 复用 router Top-K 临时 buffer，减少 75 层反复 allocation。

测得相关机会总量约 0.002462 ms/75 layers，远低于可影响 TTFT 的量级。无需承担额外生命周期和复用风险，按 Amdahl 在进入 server 前拒绝。**STATIC/LEAF REJECT**。

### N28：更新 DeepGEMM wheel

**假设：** 使用更新 wheel 获得更好的 GEMM kernel 或 Blackwell 支持。

实际在 packing/hot-path ABI 即失败，说明二进制接口、layout 或调用约定不匹配。正确性/启动合同未通过，因此没有性能请求。**FAIL-CLOSED; NO TIMING**。

### N40：contiguous SwiGLU + FP8 fusion

**原理：** 将 MoE 中相邻的 contiguous layout、SwiGLU activation 和 FP8 量化/写回融合，减少中间 tensor 与 HBM 流量。

**局部证据：**

- leaf 改善 59.1%；
- 加入 zero-M guard 后 correctness 通过。

**E2E：**

- P50 +1.56%；
- P90 +18.61%；
- throughput -6.01%。

**解释：** 大幅 leaf win 仍可能因调用占比、额外资源、rank tail、layout 边界或丢失 overlap 而在 E2E 回退。没有针对性 profile 时不能任选一个原因。**REVERT**。

---

## 13. 探索分支/代码资产总表

以下资产已迁移，但“目录存在”不等于“已有正式收益”。对没有 compact formal summary 的分支，本报告只记录探索方向，不创造数字。

| 分支/目录家族 | 主要探索方向 | 当前证据等级/结论 |
|---|---|---|
| SGLang-DGMK-router-fusion-reviewed | static placement + physical router path | 进入 N6 组合，accepted |
| router-fusion-cpuset-reviewed | N6 + cpuset-safe affinity | source contract/test 通过，research |
| flashmla-prefill-band-reviewed | 八个生产 M 的 B3+B5 FlashMLA | leaf admitted，未 E2E 晋级 |
| flashmla-equal-allgather-reviewed | equal-chunk gather | no-op/equivalence，拒绝 |
| dp-gatherv | variable-length DP gather | 有实现探索，无正式晋级 |
| dp-sync-single-d2h | N2 single-D2H 控制读取 | throughput -5.94%、1/5，拒绝 |
| m10048-base/q256/b2/indexbuf3/contig-swiglu | M≈10048 attention/index/融合变体 | 叶子/实现探索，无 formal promotion |
| indexer-q256 | indexer 量化/shape 专用 | 无 formal promotion |
| moe-swiglu variants | MoE activation/fusion | 无 formal promotion |
| o-proj | output projection kernel | leaf candidate，无 formal promotion |
| prefill-delayer-sync | rank 启动对齐 | P90信号但P50退化，拒绝 |
| route-complete | router/route 完整路径实验 | 部分思想进入 N6，独立收益未证明 |
| temporal placement v1/v2/v5 | 逐时间 expert placement | v1/v2拒绝；v5 research |
| CPU affinity candidate | 父 cpuset 与物理核分区 | correctness通过，未性能晋级 |
| mixture-of-kittens / MoK | MoE megakernel、MXFP8、symmetric memory | 稳定性修复成功，性能/容量失败 |

这些负结果应继续保留摘要，以免未来重复进行已经关闭的探索。

---

## 14. 所有主要候选的最终决策矩阵

| 候选 | 正确性 | 局部性能 | E2E | 决策 |
|---|---|---|---|---|
| N1 balanced placement | permutation/组合正确性通过 | 负载比显著均衡 | P50赢但P90+9.20% | REVERT AS SINGLE |
| N4 balanced+physical-ID原子组合 | 通过 | 删除padded-ID mask | P50赢但吞吐arm median回退 | REVERT AS CANDIDATE |
| N6 三项组合 | 通过 | 有方向证据 | 正式+holdout通过 | **PROMOTE IN CELL** |
| N2 single-D2H | 可运行 | 减少control read假设 | throughput -5.94%，1/5 | REVERT |
| N3 local control broadcast | exact | 删除重复control假设 | P50/P90/throughput均退化 | REVERT |
| N10 FlashInfer Top-K | operator可比 | 4.294ms慢于2.4461ms | 未进入E2E | REVERT |
| N11 explicit flashmla_sparse | 可运行 | backend显式路径 | P50 +10.55%、thr -10.74% | REVERT |
| N12 ninth-route shared expert | 路径可运行 | 扩大route边界 | 明显慢于balanced nonfused | REVERT |
| N15 Top-K workspace reuse | 静态可行 | 机会仅0.002462ms/75层 | Amdahl前置拒绝 | REVERT |
| v1 temporal | 通过开发门槛 | proxy改善 | P50/P90/吞吐均退化 | REVERT |
| v2 communication-safe | 通过开发门槛 | proxy改善 | P50退化、绝对失败 | REVERT |
| v5 P50/rank constrained | 11/11 exact，logprob通过 | replay proxy改善 | 退化host相对大胜，绝对失败 | CONTINUE |
| KDA clustered MQA | selected score正确 | leaf慢17.9%–19.8% | 未进入服务 | REVERT |
| N23 exact-M10048 | output/LSE bit-exact | leaf -8.85% | P50仅-0.949%，差门槛0.051pp | REVERT/ITERATE |
| N24 indexer-Q256 | operator/model通过 | indexer -26.5%到-29.1% | formal 1/3，early-stop | REVERT |
| N35 NoPE stagger | bit-exact | leaf再改善 | bracket drift 3.060% | NO DECISION |
| N36 3-stage未对齐 | 失败 | 无合法性能值 | 未进入E2E | REVERT |
| N38 3-stage对齐 | bit-exact | leaf改善 | P50 +0.068%更慢 | REVERT |
| N28 newer DeepGEMM wheel | ABI失败 | 无合法leaf | 未计时 | FAIL-CLOSED |
| N40 contig SwiGLU+FP8 | zero-M guard后通过 | leaf +59.1% | P50/P90/thr均退化 | REVERT |
| FlashMLA B3+B5 band | output/LSE bit-exact | 8.82%–12.49% | 无健康formal win | CONTINUE |
| DeepEP send16 | leaf正确 | dispatch有小幅收益 | anchor漂移/绝对失败 | REVERT |
| Equal all-gather | 单测通过 | 理论减少无用reduce | 实际no-op | REVERT |
| Native SBO | 11/11 exact | overlap假设 | P50/P90均退化 | REVERT |
| Overlap scheduler | 可运行 | 未证明有效重叠 | P50/吞吐大幅退化 | REVERT |
| Prefill delayer | 可运行 | 尾部有信号 | P50退化 | REVERT |
| cpuset-safe affinity | 单测/审计通过 | 调度更安全 | 绝对门槛失败 | CONTINUE |
| Chunk division fix | 通过 | 删除重复轮次 | 历史cell 3.3×–6.56× | POSITIVE HISTORICAL CELL |
| Decode MTP | token路径有效 | acceptance≈1.87 | decode约1.32× | POSITIVE HISTORICAL CELL |
| Fused QKV-A | bit-exact | leaf≈2.4× | TPOT不升反降 | REVERT |
| MoE align16 global | 局部正确 | decode leaf 1.13× | prefill严重退化 | REVERT/GUARD |
| 手写 PTX | 存在错误 | 慢于参考 | 未进入E2E | ABANDON |
| MoE PSUM | 可运行 | 未形成可靠win | 慢1.7%–2.3% | REVERT |
| MoK megakernel | 稳定性修复后可长跑 | 大边界/大workspace | P50、吞吐、KV容量均退化 | REVERT |

---

## 15. 如何理解本项目的真正瓶颈

## 15.1 三条执行图

### 计算图

~~~text
norm/projection → attention → router → expert GEMM
→ activation → expert GEMM → residual/output
~~~

### 通信图

~~~text
DP gather / attention communication
→ MoE dispatch A2A
→ expert compute
→ MoE combine
→ TP/DP reduction
~~~

### 控制图

~~~text
Python scheduler → batch/shape metadata → kernel launch
→ event/stream wait → process affinity → rank progress
~~~

早期 chunk bug 主要放大了通信图；N6/v5 主要优化 communication + control；CPU 异常污染 control；FlashMLA/QKV 主要优化 compute。

## 15.2 不同 workload 下的瓶颈证据必须分开

历史 decode Nsys 的累计分类统计曾显示：

- 通信约 88.1%；
- dense GEMM 约 5.7%；
- MLA 约 2.2%。

这些比例只支持对应 decode cell，而且若来自累计 kernel duration，不能直接当作 wall-clock critical-path 百分比。

最近 100K cached-prefill 的 v5 matched Nsys 则显示：

- FlashMLA per-call 分布几乎不变；
- DeepEP dispatch/combine wait 长尾显著下降。

因此对当前 prefill cell，证据支持“v5 的主要差异与 rank arrival/control wait 相关，而不是 attention 单次 kernel 变快”。它不证明所有 GLM-5.2 workload 都由通信主导；decode、短 prefill、不同 batch 需要各自 profile。

## 15.3 不能把 kernel sum 当 wall time

不同 GPU、不同 stream 的 kernel 会并发。例如八张 GPU 各跑 10 ms，不代表请求用了 80 ms；同一张 GPU 两个 stream 的 kernel 也可能重叠。

正确做法是：

- 看 critical-path interval；
- 看最早必须开始和最晚必须结束的依赖；
- 看哪个 rank 阻塞全局前进；
- 用无 profiler wall time 判断真实效果。

---

## 16. 公开证据边界

公开目录采用仓库相对路径，只保存能够复核结论的小型证据：工作负载合同、N6 formal 与 holdout 的配对样本和摘要、accepted map、v5 冻结 research map、正确性结论和派生因果摘要。raw `.nsys-rep`、`.ncu-rep`、SQLite、日志、环境转储、路由 recorder 张量、模型文件和基础设施信息均被排除。

这意味着公开材料足以复核本报告已经给出的统计和晋级决策，但若要重新探索 Nsight 时间线或完全从零复跑，仍需访问私有原始证据与等价模型／运行环境。

---

## 17. 在新 B300 上应该如何继续

## 17.1 第一步：先验证机器，不要马上跑候选

检查：

- 每个 socket/core 的持续频率；
- CPU governor/turbo；
- nvidia-smi 拓扑；
- NVLink；
- GPU clock/power；
- Xid；
- 父 cpuset 与完整进程树；
- driver/CUDA/容器版本。

若健康 baseline 都不能恢复，就不解释 candidate 绝对值。

## 17.2 第二步：恢复 N6 历史 anchor

1. 固定相同模型身份；
2. 恢复 90K cached + 10K suffix；
3. 核验实际 89984 cache hit；
4. 110 请求、并发 11、output=1；
5. overlap/graph 关闭；
6. 核验 110/110、cached/new/output、rank、path marker 与 server args；
7. 先跑 baseline-to-baseline 测噪声和 control drift；
8. 再跑 N6 五对或更多相邻 A/B；
9. 只有 N6 恢复 P50≤2s、P90≤5s、吞吐≥438k，且 paired/noise 稳定，才进入 v5。

## 17.3 第三步：v5 正式复验

建议：

- N6 / v5 只改变 expert map；
- 同一 affinity；
- A-B-A 或 AB/BA；
- 完整 correctness；
- 至少五对 screen，健康稳定机器最好扩展正式 pair 数；
- 保存每个 raw sample；
- 若获胜，再跑 matched Nsys。

目标是回答：

> v5 在健康 anchor 上是否仍比 N6 快，而不是它是否能拯救退化 control。

## 17.4 第四步：FlashMLA band

在 v5 与 N6 的系统结论稳定后，再测试 FlashMLA：

- 先证明八个 M 的 selected marker；
- 比较 stock 与 band 的 E2E；
- 检查未 admitted M 走预声明路径；
- 若 TTFT 获胜，用 Nsys 确认 attention interval 缩短且未破坏 DeepEP overlap；
- 只有出现具体 kernel 内部疑问时再运行 NCU。

## 17.5 停止规则

以下情况应停止或回退：

- correctness 失败；
- optimized path marker 不完整；
- control drift 超门槛；
- E2E 无收益；
- 只有 microbenchmark 赢；
- Nsys 显示收益来自 workload/path 改变；
- 三次针对性迭代仍无关键路径改善。

---

## 18. 对 CUDA 初学者最重要的十个提醒

1. “CP=8”不等于本实验真的使用 attention CP8；实际 attn_cp_size=1。
2. TP8、DP8、EP8 不需要 512 张 GPU，process group 可以重叠。
3. output=1 的测试主要是 prefill/TTFT，不是 decode/TPOT。
4. 90K prefix 已缓存，但 attention 仍需访问长 KV。
5. TTFT 是完整服务关键路径，不是 attention kernel 时间。
6. microbenchmark 快不代表服务快，必须看 E2E。
7. Nsys 下的时间不能替代正式无 profiler 时间。
8. 没有 NCU 不等于分析不完整；先要有具体单 kernel 假设。
9. DeepEP 使用更多 SM 不一定更快，可能破坏计算通信平衡。
10. v5 相对 control 大胜不代表已经替换 N6，因为 host anchor 已严重退化。

---

## 19. 最终交付状态

### 已经明确

- N6 是冻结 100K cached-prefill cell 的唯一 accepted baseline/candidate；
- v5 是最值得在健康机器复验的 research candidate；
- FlashMLA B3+B5 是已通过 leaf admission 的第二优先候选；
- cpuset-safe affinity 是正确性/鲁棒性修复，但尚未性能晋级；
- 主要瓶颈位于通信轮次、rank arrival skew 与控制等待；
- 主 100K 实验文件在 rsync quick-check 下已迁齐；单一旧 tar 例外不影响 N6/v5/cpuset 主结论，但会影响对应旧 M3/64K cell 的完整复跑。

### 不能宣称

- 不能说 v5 已经比 N6 快 3 倍；
- 不能说 FlashMLA 让 TTFT 快 8.82% 到 12.49%；
- 不能说用户口径的 CP8 已被 attention CP8 正式验证；
- 不能说所有本地分支已经推到 GitHub；
- 不能说当前归档已经具备脱离模型/容器来源的从零复现能力。

### 一句话结论

在 8×B300、EP8、8 路 DP/上下文分发、实际 attn_cp_size=1 的 GLM-5.2 100K fresh-server 饱和 cached-prefill TTFT 测试中，N6 以 P50 -5.31%、P90 -4.75%、吞吐 +7.53% 成为该 cell 唯一正式晋级方案；v5 通过正确性并显著压缩 DeepEP 跨 rank 等待，但因宿主 CPU 降频导致绝对门槛失败，仍需在健康新机器上复验。

---

## 20. 公开代码与证据索引

本节中的路径都相对于 `glm52_opt/b300_100k/`。

| 内容 | 路径 | 状态 |
|---|---|---|
| 本系统教学报告 | `docs/GLM52_B300_SYSTEMATIC_TEACHING_REPORT_20260817_CN.md` | 权威说明 |
| 冻结 workload 合同 | `workload_contract.json` | SSOT |
| N6 正式摘要与配对样本 | `evidence/n6/formal/` | accepted evidence |
| N6 独立 holdout | `evidence/n6/holdout/` | accepted evidence |
| N6 expert map | `../glm52_100k_x11_static_expert_map.json` | accepted runtime map |
| v5 决策／正确性／Nsys 摘要 | `evidence/v5/` | research evidence |
| v5 冻结 map | `../research/temporal_placement/maps/` | not promoted |
| temporal 纯分析工具 | `../research/temporal_placement/tools/` | research only |

正式服务性能以无 profiler 的 N6 formal 与 holdout 为准；Nsys 文件只解释因果，不参与晋级计时。公开目录不包含 raw profiler、服务器日志或私有迁移资料。
