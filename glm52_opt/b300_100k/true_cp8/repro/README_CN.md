# 真 CP8 复验顺序

这不是一条可直接复制到任意机器的神奇启动命令，而是一份必须逐项核对的实验合同。模型路径、容器、CPU mask 和端口应由新机器自己的配置提供，不写死在公开仓库。

## 1. 先冻结工作负载

- 模型：GLM-5.2-FP8；
- GPU：8 张同型 B300；
- TP8、DP1、attention CP8、attention TP1、EP8；
- DSA prefill CP 使用 zigzag/in-sequence；
- N6 static expert map、physical-ID router、DeepEP normal 120 SM；
- overlap schedule 与 CUDA Graph 都关闭；
- KV page size 64；
- 90,000 shared prefix + 10,000 suffix + 1 output token；
- 客户端直接发送精确 `input_ids`；
- suffix 的第一个普通 token 每条请求唯一，确保只有 89,984-token 公共完整页被复用。

## 2. 先跑 control，不要先跑 candidate

1. 不设置 `SGLANG_GLM52_CP8_COMBINE_INDEXER_HALVES`；
2. 确认日志和运行时审计显示 attention CP8、DP1、EP8；
3. 确认 8 个 rank 都正常，DeepEP normal dispatch/combine 为 120 SM；
4. 先发送至少 40 条不计入结果的稳定化请求；x11 历史复验使用 44 条；
5. 正式请求开始后，确认真实 extend 是 10,016，主要 candidate shape 将是 M1252。

如果 control 自己不稳定、GPU/CPU 频率异常、出现 Xid、NCCL 错误或请求失败，应停止，不解释 candidate 百分比。

## 3. 开启 candidate

静态实验可设置：

```bash
export SGLANG_GLM52_CP8_COMBINE_INDEXER_HALVES=1
```

同一服务进程内做 request-boundary A/B 时，可设置一个 arm 文件路径：

```bash
export SGLANG_GLM52_CP8_COMBINE_INDEXER_HALVES_ARM_FILE=/path/to/candidate.arm
```

该文件存在时，新请求选择 candidate；不存在时选择 control。每个 batch 只读取并缓存一次选择，不能在一个请求执行到一半时切臂。

必须看到 8 个 rank 都记录：

```text
e2e_prefill/cp8_combined_indexer_halves ... m=1252
```

如果 marker 缺失，本轮不能算 candidate 测量。

## 4. 无 profiler 配对测试

预先固定顺序：

```text
AB, BA, AB, BA, AB
```

每对相邻运行只改变 candidate arm。至少保存：

- 每条请求的 TTFT；
- 每臂成功请求数；
- P50、P90、总 token 吞吐；
- 8-rank path marker；
- 进程树 CPU affinity；
- server args 差异；
- 输入 token 哈希。

先跑 concurrency=1、每臂 10 请求，再跑 concurrency=11、每臂 110 请求。不要把两个并发单元混成一个分布。

## 5. 独立正确性

使用两个不依赖 candidate 输出生成的 reference，比较 11 条请求：

- generated token 必须完全一致；
- max absolute logprob error 必须小于 `1e-3`；
- mean absolute logprob error 必须小于 `1e-4`。

top-k leaf 不是模型正确性的替代品，因为边界并列项可能选择不同 ID。

## 6. Nsys 只解释原因

用相同覆盖窗口分别抓 control 和 candidate，核对工作覆盖锚点相等，再比较：

- DeepGEMM MQA 调用数；
- fused top-k 调用数；
- sparse attention 调用数；
- dense FP8 GEMM 调用数；
- DeepEP combine 调用数。

profiled TTFT 不参与性能晋级。累计 kernel duration 也不是 wall time，不能把各项百分比相加。

## 7. 当前晋级标准

当前公开证据只支持：

> combined-indexer 是冻结 true-CP8 单元内的研究候选。

当前不支持：

- 在新 `main` 上已经复现；
- true CP8 已超过 N6；
- 任意 prefix、suffix、并发和 continuous batching 都有 3.3% 收益；
- 默认开启该环境变量。
