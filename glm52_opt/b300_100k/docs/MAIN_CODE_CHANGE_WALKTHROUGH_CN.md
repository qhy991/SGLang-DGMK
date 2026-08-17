# 当前 GitHub main 的修改：逐文件、逐职责讲解

## 1. 发布状态

仓库：`qhy991/SGLang-DGMK`

本轮教学文档发布前的 `main` base：`38fc37ce5f39292d72a37b95d7683a4991510c2f`

| Commit | 内容 | 精确规模 |
|---|---|---:|
| `01749bb27781c5044898fd62ac4b70ced98c8335` | N6 router runtime path + 2 个 CUDA tests | 5 files，+391/-9 |
| `644547a95421c12547d3e27c99dd40f688a96163` | accepted N6 map、v5 research map 和分析工具 | 13 files，+22922 |
| `38fc37ce5f39292d72a37b95d7683a4991510c2f` | workload contract、报告与 compact evidence | 16 files，+4157/-6 |

基线父提交是 `6d65a33549f96687794adbb3b3069284b21d51c6`。上表三个提交已经在 Git 历史中，不需要另建一份 patch 作为第二事实源。

## 2. 设计的最小 primitive

当前 main 没有把整棵实验 worktree 搬进来，而只提交四类正交 primitive：

1. router kernel 能选择性读取 logical→physical map；
2. router kernel 能选择性把 padded row 的 ID 写成 `-1`；
3. router kernel 能选择性直接写 DeepEP 需要的 int64 ID；
4. 上层用严格条件选择该 path，并跳过已完成的重复 remap/mask。

Expert placement map 是数据 SSOT；DeepEP 120 SM 是启动参数；v5 是 research data/tooling。三者没有被隐藏成另一个平行实现。

## 3. `moe_fused_gate.py`

仓库文件：[`python/sglang/jit_kernel/moe_fused_gate.py`](../../../python/sglang/jit_kernel/moe_fused_gate.py)

### 3.1 原路径

Router/Top-K 先生成 logical expert ID。随后另一个路径：

1. 把 logical ID 映射到 physical ID；
2. 对 CUDA graph 或 batch padding 产生的无效行做 mask；
3. 必要时把 ID 转成 DeepEP ABI 所需的 int64。

这些步骤的算术不重，但会产生小 kernel、额外内存读写和两套“谁拥有最终 expert ID”的语义来源。

### 3.2 新路径

JIT/Triton router 增加可选输入：

- `logical_to_physical_map`：一维、CUDA、连续、int64；
- `num_token_non_padded`：CUDA int32 scalar；
- `output_ids_int64`：直接建立 int64 输出。

Top-K 已经得到 `selected_idx` 后，kernel 在同一次执行里：

1. 对 routed columns 读取 physical ID；
2. 保持 shared expert 列的既有语义；
3. 对 `row >= num_token_non_padded` 写入 `-1`；
4. 一次 store 输出最终 ID。

这删除的是中间 materialization 和 post-router 小 kernel，不改变 router 的 Top-K 数学、权重归一化或专家函数。

### 3.3 不变量

- map shape 必须与 routed expert 数一致；
- map dtype 必须是 int64，且在 CUDA 上连续；
- padded ID 是 `-1`，padded weight 的既有处理不能被错误改变；
- shared expert slot 不得被 routed map 重写；
- output ID ABI 必须和 DeepEP 消费方一致。

## 4. `topk.py`

仓库文件：[`python/sglang/srt/layers/moe/topk.py`](../../../python/sglang/srt/layers/moe/topk.py)

这个文件承担两个职责：admission gate 和“避免重复工作”。

### 4.1 Admission gate

Static placement fusion 只有在 GLM-5.2 对应的 CUDA/JIT/DeepEP/static-placement 组合以及 expert 数、Top-K、router 语义等窄条件成立时才会选择。它还要求冲突的 recorder、EPLB 动态行为、其他 routing backend 或模拟路径关闭。

但这里要准确区分两件事：

- 被选中后，内部 shape/dtype 不变量有 assertion；
- 如果外层 gate 不匹配，很多情况会回到已有路径，而不是统一报错。

所以 benchmark 的运行合同必须要求 selected marker。没有 marker 的“正常跑完”只说明 fallback 能跑，不能说明测到了 N6 router fusion。

### 4.2 跳过重复 remap/mask

上层通过 `static_placement_already_fused`、`padded_region_already_masked` 记录 router 已完成的工作。后处理据此不再：

- 第二次 logical→physical remap；
- 第二次 padded ID mask；
- 对最终 int64 ID 做无意义转换。

这体现了 SSOT 原则：最终 expert ID 只在一个 canonical stage 生成。

### 4.3 Selected marker

```text
GLM-5.2 router static-placement fusion selected:
logical-to-physical + optional padded-row mask + int64 IDs
```

复验脚本应把该 marker、map SHA、DeepEP config 和 server args 一起归档。

## 5. `environ.py`

仓库文件：[`python/sglang/srt/environ.py`](../../../python/sglang/srt/environ.py)

main 只增加三个 default-off flag：

```text
SGLANG_GLM52_ROUTER_PAD_MASK_FUSION
SGLANG_GLM52_ROUTER_DEEPEP_IDS_FUSION
SGLANG_GLM52_ROUTER_STATIC_PLACEMENT_FUSION
```

冻结 N6 使用统一 static-placement fusion；前两个 flag 是更细粒度的相关 primitive。默认关闭保护其他模型和 workload。发布时没有整文件搬运实验 worktree 中的 shared-expert、SwiGLU、profile-default 等未晋级改动。

## 6. 两个 CUDA 测试

### 6.1 Static placement 等价

[`test/registered/jit/test_moe_fused_gate_static_placement.py`](../../../test/registered/jit/test_moe_fused_gate_static_placement.py)

覆盖 `(M, valid)=(1,1),(17,13),(10048,10000)`。它比较：

- 原 router 输出 logical ID 后再 reference remap；
- 新 router 直接输出 physical ID。

这覆盖单 token、小 ragged batch 和冻结真实大 M。

### 6.2 Padded ID / int64 ABI

[`test/registered/jit/test_moe_fused_gate_pad_mask.py`](../../../test/registered/jit/test_moe_fused_gate_pad_mask.py)

覆盖 valid=0/1/2/7/12/15/16，验证：

- 有效行 ID 不变；
- padded 行 ID 为 `-1`；
- int64 输出 ABI；
- 不同边界下没有越界。

测试证明的是 kernel contract，不等于在当前 main 上已经复现 N6 的 server 性能。

## 7. Accepted map

路径：[`glm52_opt/glm52_100k_x11_static_expert_map.json`](../../glm52_100k_x11_static_expert_map.json)

SHA-256：`36d13233672288317fd69495d4cedb46844b8ae99033d184d84aff0c99c68f09`

每一行是 256 个 expert 的 permutation。它改变“哪个 physical rank 拥有哪个 logical expert”，不改变权重或模型语义。这个 map 只对冻结 GLM-5.2/EP8/100K-x11 cell 有证据，不能作为其他模型、EP size 或流量的通用默认。

## 8. DeepEP 120 为什么不是代码改动

N6 的第三部分通过启动参数传入：

```json
{
  "normal_dispatch": {"num_sms": 120},
  "normal_combine": {"num_sms": 120}
}
```

不需要修改 `deepep.py`。这很重要：实验 worktree 里的 SBO/旧 wheel 兼容代码不是 N6 treatment，已从 main 白名单排除。

## 9. v5 为什么在 research 目录

v5 map SHA-256：
`570e58026ae890bfd294a34786a095142ca71d14747c87606a536b20c415620a`

它使用同一 N6 runtime，仅改变 25 个 active MoE layer 的 placement；其余层局部回退 N6。由于健康 host 上尚无正式绝对 anchor，它放在 `glm52_opt/research/temporal_placement/`，不会被 production import 或设为默认。

## 10. 没有合入 main 的修改

FlashMLA band、DeepEP send/recv sweep、equal all-gather、overlap/SBO、PrefillDelayer、cpuset runtime patch、clustered MQA、contiguous SwiGLU、MoK 等均未作为 accepted runtime path 合入。负结果保留在报告和证据中，而不是把失败代码长期并存于主路径。

## 11. 当前 main 仍需完成的验证

冻结 formal runtime 有一个 22 文件 source contract。新 main 中只有 3 个文件字节相同，11 个存在但不同，8 个缺失；所以源代码移植审查通过不等于二进制/runtime 等价。新 B300 上必须先：

1. 构建当前 main；
2. 运行两个 CUDA tests；
3. 校验 map SHA；
4. 检查 selected marker；
5. 恢复 baseline 绝对 anchor；
6. 重新做 N6 A/B 与 holdout。
