# GLM-5.2 真 CP8 / EP8 Prefill 优化报告（教学版）

更新时间：2026-08-18

## 1. 先说结论

这轮实验把 GLM-5.2 的长上下文 prefill 从旧的“8 路 DP-attention、attention CP=1”切换成了真正的：

- TP=8：模型张量计算使用 8 张 GPU；
- DP=1：只有一份请求流，不再把请求复制成 8 份数据并行；
- attention CP=8：一条长请求的上下文由 8 张 GPU 合作处理；
- EP=8：MoE 的专家分布在 8 张 GPU；
- attention TP=1：每个 CP rank 在 attention 内不是再做一层 TP；
- DeepEP normal dispatch/combine 使用 120 SM；
- 继续使用此前 N6 的 balanced expert map 和 physical expert ID router。

在冻结的 100K cached-prefill 测试单元中，新实现把每层 indexer 的两个 626-row 调用合成一个 1252-row 调用。concurrency=1 的五个相邻 AB/BA 配对结果是：

- P50 TTFT：5/5 胜，中位改善 2.50%；
- P90 TTFT：4/5 胜，中位改善 2.43%；
- 总 token 吞吐：4/5 胜，中位提升 2.70%；
- 五个 control 的 P50 范围漂移只有 0.90%；
- 11/11 生成 token 完全一致；
- logprob 最大绝对误差 1.13e-4，小于 1e-3 门；
- logprob 平均绝对误差 2.46e-5，小于 1e-4 门；
- Nsys 证明 DeepGEMM MQA 和 fused top-k 的调用次数都精确减半。

随后使用 concurrency=11、每臂 110 请求做同样的五对测试：

- P50、P90、吞吐全部 5/5 paired wins；
- P50 中位改善 3.30%；
- P90 中位改善 3.32%；
- 总 token 吞吐中位提升 3.19%；
- control 的 P50/P90/吞吐范围分别为 1.40%/2.995%/1.65%；
- 每臂 110/110 成功，8 个 rank 都命中 M1252 candidate。

当前状态应写成：

> 在冻结的 TP8 / DP1 / attention-CP8 / EP8、100K cached-prefill、concurrency=1 和 concurrency=11 单元中通过配对性能、模型正确性和因果 profile 的研究候选；尚未证明可以替换所有线上工作负载，也未设为仓库默认路径。

还必须同时说明一个更高层的结论：**这个候选优化了 CP8 路径，但当前 CP8 路径本身没有超过此前的 CP1 / DP-attention N6。** 在相同的 100K、concurrency=11、每臂 110 请求单元中：

| 配置 | P50 TTFT | P90 TTFT | total-token throughput |
|---|---:|---:|---:|
| 此前 accepted N6：attention CP1 / DP-attention 8 路 | 1,933.67 ms | 2,891.13 ms | 486,640 token/s |
| true CP8 control：attention CP8 / DP1 | 3,755.57 ms | 3,797.85 ms | 291,300 token/s |
| true CP8 combined-indexer candidate | 3,634.06 ms | 3,664.88 ms | 302,172 token/s |

这些数是各自五组 arm 结果的中位数；表内 candidate 相对 control 的 arm-median 方向与下面的 paired 统计一致，但正式 treatment 百分比仍以相邻配对的 3.30% / 3.32% / 3.19% 为准。

直观解释是：当前每条请求只需要重算约 10K token，而并发为 11。此时把不同请求分给 8 个 attention 数据并行 rank，可以保留较强的请求级并行；让 8 张卡共同切一条请求，会增加跨卡协调并减少请求级独立推进。这个解释符合观测，但没有单独 profile 出全部架构差异，所以应写成“工作假设”，不能当成已隔离的单一因果。

因此，本轮正确的晋级关系是：

- combined-indexer 是 **true CP8 内部的 research winner**；
- true CP8 仍是 **未超过现有 N6 的架构实验**；
- 现有 N6 不应被 true CP8 替换。

## 2. 这个测试到底是什么

### 2.1 它是 prefill / TTFT，不是 decode

请求由三部分组成：

1. 90,000 个逻辑 shared-prefix token；
2. 10,000 个 suffix token；
3. 只生成 1 个输出 token。

因此用户看到的指标主要是 TTFT（Time To First Token，首 token 等待时间）。因为只输出 1 个 token，这里几乎没有可研究的 decode/TPOT 阶段。

### 2.2 为什么缓存命中不是整 90,000

KV cache 的页大小是 64 token。能复用的完整页数是：

```text
floor(90000 / 64) * 64 = 89984
```

所以每条正式请求会命中 89,984 个 token，prefix 最后的 16 个 token 与 10,000-token suffix 一起重算，总共 10,016 个真实 extend token。

### 2.3 为什么日志写 10048，而主要 kernel 是 M1252

SGLang 的调度日志会显示对齐和通信所用的 padded token 数，因此每个 rank 能看到 `#new-token=10048`。但真正的 zigzag CP8 会把 10,016 个真实 token 分成 16 个片段：

```text
10016 / (2 * 8) = 626
```

每个 CP rank 拿一个前片段和一个后片段，所以实际 indexer 查询行数是：

```text
626 + 626 = 1252
```

服务日志中的 hotspot shape 也分别出现：

- M10048：90K warmup 的第一大块；
- M2452：90K warmup 的尾块；
- M1252：正式 10,016-token suffix。

这说明不能只看一条调度日志就判断 CUDA kernel 的真实 M。

## 3. 为什么改用精确 token 输入

通用 benchmark 客户端会执行：

```text
token IDs -> decode 成文本 -> 服务端重新 tokenize
```

GLM tokenizer 的这个往返不是严格保长。第二条 100K 请求曾被缩短成 99,412 token，局部 kernel shape 从 M1252 变成 M1180，并触发了异步 illegal-memory-access。

正式 runner 因此直接向 `/generate` 发送 `input_ids`，并让每个 suffix 的第一个普通 token 唯一。这样每条请求都满足：

- 输入长度严格 100,000；
- 只能复用共同的 89,984-token 页对齐前缀；
- 正式 kernel shape 固定 M1252；
- 不会因为文本往返悄悄改变工作负载。

对于不能被 `2 * CP` 整除的真实 extend 长度，DSA CP 现在局部回退普通非 CP 路径。这个回退只影响不安全的请求；冻结的 90K warmup和 10,016-token suffix 都能整除 16，仍使用 CP8。

## 4. 正式优化：合并 zigzag 的前后两个 indexer 调用

### 4.1 indexer 是什么

GLM-5.2 的 DSA attention 不会让每个 query 看完整 100K KV。它先用一个 indexer 给历史 token 打分，再选出 top-k=2048 个位置交给 sparse attention。

可以把它类比成：

- indexer 是图书馆检索系统；
- sparse attention 是只阅读检索出的 2,048 页；
- 如果检索本身太慢，即使后面的阅读很省，TTFT 仍会被拖慢。

### 4.2 stock 路径做了什么

zigzag CP rank 有两个 query 块：

- 前半：626 行；
- 后半：626 行。

stock 路径每层执行：

```text
前半 DeepGEMM MQA -> 前半 fused top-k
后半 DeepGEMM MQA -> 后半 fused top-k
```

75 个 active indexer layer 意味着一条请求每个 rank 有 150 次 MQA 和 150 次 top-k 调用。

### 4.3 candidate 路径做了什么

DeepGEMM MQA 的 ABI 已经接受逐行 `ks` / `ke`，也就是每一行可见 KV 的起点和终点。前后两半虽然可见终点不同，但属于同一条请求，可以合并成：

```text
一次 1252-row DeepGEMM MQA
  - 前 626 行使用前半的 ke
  - 后 626 行使用后半的 ke
一次 1252-row fused top-k
```

数学上没有让任何 query 看到本来不可见的 token，只是把两张任务单合并成一次 kernel launch。

### 4.4 为什么只允许 M1252

本轮 leaf 证据只覆盖正式 suffix 的 M1252。90K warmup 的 M10048/M2452 不自动外推，显式走 stock fallback。

这是“扩大优化边界，同时允许局部回退”的具体实现：

- 扩大的边界：从单个 kernel 内部，扩大到两个 CP 半段和 top-k 的联合调用；
- 局部回退：非 M1252、非 CP8、非 H32/D128、启用 overlap 等情况都不选择 treatment；
- selected path 出错时不会静默变成 baseline。

## 5. CUDA leaf 证据怎么读

### 5.1 DeepGEMM 部分

在 8 个 CP rank 的真实 KV 终点上，两种测量顺序都显示：

- stock 两次 626-row 调用合计约 0.73 ms；
- candidate 一次 1252-row 调用约 0.56 ms；
- 不同 rank 的降幅约 22.9%–24.0%；
- 有效 logits 逐元素完全相同。

这个 23% 只属于 indexer 的 MQA 子步骤，不能写成 TTFT 快 23%。

### 5.2 fused top-k 的 tie

改变 top-k launch shape 后，边界等分项的 ID 可能不同：

- 10,016 行中有 57 行出现 selected-score multiset 非 bit-exact；
- 最大 selected-score 绝对差 3.11e-5；
- 最大平均差约 4.03e-9；
- 单行 top-k ID 重合最低 99.316%。

因此 operator 证据只允许进入模型正确性测试，不能单凭 leaf 宣称正确。

模型门最终通过：11/11 token exact，logprob 误差远低于阈值。

## 6. 为什么第一次五轮 baseline 不能直接使用

第一次 same-server 五轮 P50 是：

```text
419.64, 417.02, 419.01, 351.97, 356.31 ms
```

把 50 条原始 TTFT 展开后，最佳变点在第 27 条请求之后：

- 变点前中位：419.29 ms；
- 变点后中位：354.62 ms；
- 后半比前半快 15.42%。

这说明进程在请求 27 附近进入了第二个性能稳态。五轮总中位 417.02 ms 混合了两个状态，不能作为正式锚点。

后续所有 A/B 都先跑 40 条明确排除的 control 稳定化请求。

## 7. 无 profiler E2E 结果

### 7.1 A–B–A development

首次稳定化后的 A–B–A：

- P50 方向改善 3.26%；
- P90 方向改善 2.61%；
- 吞吐方向提升 3.10%；
- 但 control A/C 的 P50 漂移 3.195%，超过 3% 门 0.195 个百分点。

所以这轮是“正向但 no-decision”，没有晋级。

### 7.2 五个相邻 AB/BA pair

顺序预注册为：

```text
AB, BA, AB, BA, AB
```

结果：

| 指标 | paired wins | 中位改善 |
|---|---:|---:|
| P50 TTFT | 5/5 | 2.50% |
| P90 TTFT | 4/5 | 2.43% |
| total-token throughput | 4/5 | 2.70% |

control 跨五对的范围：

- P50：0.90%；
- P90：1.57%；
- throughput：1.26%。

这组数据通过了配对门。

### 7.3 concurrency=11 / 110-request 配对

为了验证收益不是 concurrency=1 特例，服务把 `max-running-requests` 提高到 128，客户端每次并发 11 条，总共发 110 条 exact-token 请求。仍使用相同的五对顺序：

```text
AB, BA, AB, BA, AB
```

结果：

| 指标 | paired wins | 中位改善 |
|---|---:|---:|
| P50 TTFT | 5/5 | 3.30% |
| P90 TTFT | 5/5 | 3.32% |
| total-token throughput | 5/5 | 3.19% |

五个 control 的范围：

- P50：1.40%；
- P90：2.995%；
- throughput：1.65%。

control 的 P50 大致在 3.71–3.76 秒，candidate 大致在 3.59–3.64 秒。这里的 TTFT 包含 concurrency=11 下的排队/服务推进，不能直接和 concurrency=1 的约 0.35 秒横向比较。

所有 arm 都是 110/110 成功，candidate 的 8 个 rank 都命中 M1252 合并路径。之后的独立 correctness 仍为 11/11 token exact，logprob 最大/平均误差 1.58e-4/3.45e-5。

## 8. Nsys 因果证据

control 和 candidate 分别使用 8 秒 cudaProfilerApi 窗口。profiled TTFT 含 profiler 启动开销，不能与无 profiler 数字比较。

两个报告的工作覆盖锚点一致：

- sparse-attention kernel：两边都是 6,240 calls；
- dense FP8 GEMM：两边都是 44,880 calls；
- DeepEP combine：两边都是 6,000 calls。

治疗相关调用：

| kernel family | control calls | candidate calls | 调用数变化 | 累计 kernel 时间变化 |
|---|---:|---:|---:|---:|
| DeepGEMM MQA | 3,360 | 1,680 | -50% | -24.40% |
| fused top-k | 3,360 | 1,680 | -50% | -14.60% |
| sparse attention | 6,240 | 6,240 | 0% | 约 0% |
| dense FP8 GEMM | 44,880 | 44,880 | 0% | 约 0% |

这证明收益来自 indexer 调用合并，不是 attention kernel 自己变快。

累计 kernel duration 不是关键路径 wall time，不能把表内百分比相加。

NCU 没有继续跑，原因是这里的机制已经定位为跨调用消除，不是一个无法解释的单 kernel 指令/访存问题。

## 9. 继续探索但拒绝的方向

### 9.1 combined MQA 的 SM 数量

扫了 96/112/120/128/136/144/148 SM：

- 112 SM 最快，最慢 rank leaf 比 148 SM 快 2.42%；
- 96 SM 慢 26.8%；
- 112 SM 每层只省约 0.013 ms；
- 75 层理论只省约 0.97 ms；
- 相对约 350 ms TTFT 的 Amdahl 上限只有约 0.28%。

低于 1% 服务门，且 overlap 关闭，故不跑服务测试。

### 9.2 pack 后再做 KV all-gather

通信 leaf 中：

- MLA KV 的 BF16 all-gather 后 pack，改成先 pack 再 uint8 all-gather，局部约 1.96 倍；
- indexer-K 同类改法反而只有约 0.85 倍，变慢。

对应 x11 服务 control 与 candidate 都在约 3.72–3.77 s TTFT，三轮没有清晰 E2E 收益，P90 还存在回退；当前归类为中性/拒绝，不能用局部通信 1.96 倍宣称服务更快。

## 10. 失败尝试教会了什么

### 10.1 `num_splits` shape 错误

CP 拆分 query 后，FlashMLA page table 已是 local rows，但 eager metadata 仍是 global rows。修复是在显式 CP mismatch 时从 local `indices >= 0` 重建 cache lengths 和 FlashMLA metadata；非 CP mismatch 仍 fail closed。

### 10.2 第二条文本请求 illegal memory access

根因边界是文本往返把真实 extend 改成不能整除 16 的长度，触发未覆盖 M1180。处理方式不是忽略报错，而是：

- 正式测试改成 exact input IDs；
- 非 segment-aligned DSA CP 请求局部回退；
- 不把首条 819.58 ms 的不稳定结果作为 baseline。

### 10.3 fused top-k 缺 CUDA `cu_seqlens`

第一次 combined candidate 在 top-k wiring 处报：

```text
RuntimeError: cu_seqlens_q must be a CUDA tensor
```

修复是把两个 CP 半段表述成同一条 1,252-query 逻辑序列，传入 CUDA 上的长度张量 `[1252]`，同时保持所有 index offset 为零。这里的参数保存“每条序列各有多少 query”，不是常见的 `[0, 1252]` 累积边界表示。

### 10.4 profiler 双窗口缺一个 STOP

Nsys `repeat-shutdown:2` 第二段有一个 worker 卡在 `cudaProfilerStop`，因此没有形成第二份报告。该 artifact 被标记 `causal_claim=false`；随后使用可靠的一次性 candidate report，与已落盘的 control report 做比较。

## 11. 当前代码边界

主要变化：

1. `dsa_backend.py`
   - CP-local FlashMLA metadata 重建；
   - 只处理显式 local/global row mismatch。
2. `dsa/utils.py`
   - in-sequence CP 只接受真实 extend 长度能填满 `2 * CP` segments；
   - 其他长度局部回退。
3. `dsa_indexer.py`
   - default-off combined-indexer treatment；
   - request-boundary arm file；
   - 只 admit CP8/H32/D128/M1252/overlap-off；
   - 其他 shape 保留 stock 两调用路径。
4. exact-token runner
   - 固定输入哈希；
   - 固定 cache hit 和 shape；
   - 保存每条 TTFT；
   - fail-closed 检查 8-rank 日志。

## 12. 下一步

1. 在新的健康 B300 host 上复现 true-CP8 control steady-state；
2. 解决 DeepGEMM 每次进程启动扫描 0..131071 全 M 的问题；
3. 增加不同 prefix/suffix 长度、到达率和 continuous-batching 组合；
4. 只有 broader workload 和新 host 都通过，才考虑把 treatment 从 research gate 提升为默认候选；
5. 不把 x1 的 2.50% 或 x11 的 3.30% 外推成所有线上流量的收益。
