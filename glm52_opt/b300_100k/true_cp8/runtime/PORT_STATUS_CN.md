# 运行时代码的来源和状态

## 已在 B300 验证的版本

- 冻结运行时 commit：`1571b72db014603ebfaad6d59cbbde1740b23b3e`；
- 被测 `dsa_indexer.py` SHA256：`9eddd376384b05ae97f64dd3f4cba1ebce85fa30c5b134a9c5c6fcccc8b43021`；
- 37 文件运行时合同 SHA256：`9a017e29dfc8b3036def9802486a5ddc7c55ca537e9622e28229b1fb51e50e18`；
- B300 上已完成 leaf、x1 五对、x11 五对、独立正确性和 Nsys 因果验证。

完整相对路径和哈希见 [`../evidence/frozen_runtime_contract.json`](../evidence/frozen_runtime_contract.json)。

## 当前 main 的移植

当前提交把三个互相独立的最小 primitive 移植到 `main`：

1. DSA CP local/global 行数明确错配时，重建 local FlashMLA metadata；
2. zigzag CP 的真实 extend 长度不能填满 `2 × CP` 个 segment 时，局部回退；
3. 默认关闭的 CP8 M1252 combined-indexer 路径。

[`main_true_cp8_runtime.patch`](main_true_cp8_runtime.patch) 保存了相对移植前 `main` 的审查补丁。

- 移植基底：`9e439b4a9bdb6339a9ede363b19d18d4f3192b8f`；
- 补丁 SHA256：`1f4f6e3e748886cf1754d4f73b52c9f919de01466b2d67290fc73f273e37eeca`；
- `git apply --reverse --check` 已通过，证明补丁与当前工作树中的三个运行时文件精确对应。

## 不能混淆的两句话

- “冻结运行时已经在 B300 验证”是真的。
- “当前 main 已经在 B300 验证”目前是假的。

移植后的代码通过了补丁应用、语法和 CPU-only segment predicate 测试；还在 B300 推理容器中实际导入了当前 `main` 的 `dsa_indexer` / `dsa_backend`，并确认未设置环境变量时 treatment 为关闭。它们是移植 smoke，不是完整 B300 E2E；正式服务 A/B 仍必须在新 revision 上重跑。
