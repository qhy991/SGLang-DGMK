# GLM-5.2 FlashMLA flat-token-index（实验性、默认关闭）

B300 集成分支：`opt/b300-glm52-integrated`  
锚定提交（不可变测试 release）：`fdc61029f8fa435d487e599666f48c358a61afd7`

本路径已注册并接入 sgl-kernel，但作为 **实验性、默认关闭** 的编译选项，避免把微小收益直接变成生产默认。

## 开启方式

构建 `sgl-kernel` 时显式打开：

```bash
-DSGL_FLASHMLA_GLM52_FLAT_TOKEN_INDEX=ON
```

默认 `OFF`。未开启时扩展返回 dispatch state `0`（未编译专用核）。

## 命中 / 回退契约

| 条件 | 行为 |
|------|------|
| 精确 GLM-5.2 **H64 / page64 / topk2048** 形状 | 命中专用核（specialized hit） |
| H128 | 自动回退原生 FlashMLA（generic fallback） |
| topk128 | 自动回退原生 FlashMLA |
| 非标准 stride | 自动回退原生 FlashMLA |

B300 烟测确认：目标形状返回 specialized hit；三个非目标形状均返回 generic fallback。

## 代码入口

- `sgl-kernel/cmake/flashmla.cmake` — `SGL_FLASHMLA_GLM52_FLAT_TOKEN_INDEX` option + patch 应用
- `sgl-kernel/csrc/flashmla_extension.cc` — dispatch 探测（编译期宏守卫）
- `sgl-kernel/cmake/patches/flashmla/glm52_v32_flat_token_index.patch`
- 契约 / 基准：`sgl-kernel/benchmark/flashmla_glm52_contract.py`、
  `sgl-kernel/tests/test_flashmla_glm52_contract.py`

## 性能结果（B300）

| 场景 | 结果 |
|------|------|
| B16 CUDA Graph p50 | 快约 **0.21%–0.32%** |
| B32 CUDA Graph p50 | 快约 **0.09%–0.19%** |
| metadata + main + combine 完整区间 | 基本持平 |
| 目标与回退正确性测试 | 全部通过 |
| REGRESS 标签 | 主要来自 **topk128 回退路径** 的一次噪声样本；该路径 **未命中** 优化核 |

**结论：收益不足以支持 8 卡实验；保持默认关闭。**

## 部署状态（B300）

- 不可变测试 release：`B300-OPT-GLM52/releases/glm-lead/fdc61029f8fa435d487e599666f48c358a61afd7/`
- **未**修改正式 checkout（`wwxq/SGLang-DGMK` 工作树）
- 正式生产默认仍不启用该 CMake option
