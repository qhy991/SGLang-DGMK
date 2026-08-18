# GLM-5.2 真 CP8 / EP8：从问题、修改到证据

这是一份面向 CUDA 初学者的入口。完整教学报告见
[`../docs/GLM52_TRUE_CP8_OPTIMIZATION_20260818_CN.md`](../docs/GLM52_TRUE_CP8_OPTIMIZATION_20260818_CN.md)，机器可读数字见
[`evidence/public_evidence.json`](evidence/public_evidence.json)。

## 一句话结论

我们把 GLM-5.2 的 attention 真正改成了 CP8，并把每层两个 626 行的 indexer 调用合成一个 1252 行调用；这个修改在 CP8 内部让并发 11 的 P50、P90、吞吐分别稳定改善 3.30%、3.32%、3.19%，但 CP8 整体仍明显慢于此前的 CP1 / DP-attention N6，因此它是研究候选，不是新的默认方案。

## CP8 和 EP8 各是什么意思

- CP8（Context Parallel 8）：一条长请求的上下文被切成 8 份，8 张 GPU 合作完成 attention。
- EP8（Expert Parallel 8）：MoE 的专家权重分散在 8 张 GPU，token 会被发送到拥有目标专家的 GPU。
- DP1：只有一份请求流；这里不是此前“8 个 attention 数据并行 rank 各自推进请求”的 DP8。
- TP8：模型的张量计算仍跨 8 张 GPU。TP、CP、EP 可以使用同一组 GPU，它们不是要相乘成 512 张卡。

## 为什么每张卡拿到两个 626 行的块

逻辑 shared prefix 是 90,000 token，KV cache 以 64 token 为一页，所以实际完整命中 89,984 token。prefix 最后 16 token 加上 10,000 token suffix，一共要重算 10,016 token。

zigzag CP8 把它切成 `2 × 8 = 16` 段：

```text
10,016 / 16 = 626
```

每个 CP rank 拿一个靠前的 626-token 段和一个靠后的 626-token 段，因此 indexer 的本地行数是 1,252。

## 修改前后发生了什么

修改前，每层的本地 indexer 做两套工作：

```text
前 626 行 -> DeepGEMM MQA -> fused top-k
后 626 行 -> DeepGEMM MQA -> fused top-k
```

修改后，把两张任务单合成一张：

```text
前后合计 1,252 行 -> 一次 DeepGEMM MQA -> 一次 fused top-k
```

每一行仍保留自己能看到的 KV 终点，所以没有扩大 attention 的可见范围，只减少重复的 kernel 启动和固定开销。

## 代码安全边界

该路径默认关闭。只有同时满足以下条件才会选择：

1. 用户显式开启；
2. NVIDIA CUDA 的 e4m3fn FP8；
3. DSA attention CP8；
4. zigzag/in-sequence 切分；
5. 32 个 query head、每个 head 128 维；
6. overlap schedule 关闭；
7. 当前本地 query 行数恰好为 1,252。

M10048、M2452 和其他动态形状保留原来的两调用路径。真实 extend 长度不能整除 `2 × CP` 时，整条请求局部回退非 CP 路径，避免把 padding 行当成真实 token。

## 证据怎样一层层建立

1. leaf 测试先证明 DeepGEMM 子步骤约快 23.52%，有效 logits 完全一致。
2. top-k 有极少边界并列项变化，所以 leaf 只能允许进入模型正确性测试，不能直接宣布正确。
3. 独立模型测试得到 11/11 生成 token 完全一致，logprob 误差通过预设门。
4. 不带 profiler 的五对 AB/BA 服务测试证明 x1 和 x11 都有稳定收益。
5. Nsys 证明 MQA 和 top-k 的调用数都从 3,360 降到 1,680；attention 和 dense GEMM 的调用数不变。

## 为什么“CP8 内部赢了”仍不能替换 N6

在相同 100K、concurrency=11 单元里：

| 配置 | P50 TTFT | P90 TTFT | 吞吐 |
|---|---:|---:|---:|
| accepted N6，CP1 / DP-attention | 1,933.67 ms | 2,891.13 ms | 486,640 token/s |
| true CP8 control | 3,755.57 ms | 3,797.85 ms | 291,300 token/s |
| true CP8 candidate | 3,634.06 ms | 3,664.88 ms | 302,172 token/s |

所以，本轮修改确实让 CP8 更快，但没有证明 CP8 是这个工作负载更好的并行架构。10K 左右的重算长度加上 11 路并发，更适合保留请求级并行；这也是后续需要研究“CP 在什么长度和并发下才值得”的原因。

## 发布状态

- 冻结运行时 commit `1571b72d...`：B300 E2E、正确性和 Nsys 已验证。
- 当前 GitHub `main`：只移植了默认关闭的最小代码路径和测试，尚未在 B300 上重新跑完整 E2E。
- 使用前必须先复现 control，再按 [`repro/README_CN.md`](repro/README_CN.md) 的顺序验证。

不要把本目录的 3.30% 写成所有 prefill 的通用收益，也不要把 leaf 的 23.52% 写成 TTFT 收益。
