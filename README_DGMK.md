## SGLang GLM-5.2 优化分支（DGMK）

本仓库分支将 **SGLang** 与 **GLM-5.2 phase-aware 内核**、以及实验性 **DeepGEMM-GLM52** overlay 打包在一起，用于在 Blackwell（SM100）上做 decode/prefill 分阶段内核调度。默认不影响库存 `deep_gemm`；实验 overlay 需显式构建并 opt-in。

## 本 fork 是什么

- **SGLang 运行时**：在 `python/sglang/srt/layers/glm52_opt/` 中按 phase（decode / prefill）路由到归档内核或 DeepGEMM 实验 fork。
- **Kernel 归档**：`third_party/kernel-archive/0720-Best-GLM-52`（原 Kernel-Harness 战役产物，已 vendored）。
- **DeepGEMM-GLM52**：`third_party/DeepGEMM-GLM52` 源码树；构建产物写入本地 `overlays/<commit>/`，通过 `deep_gemm_experimental` 加载，**不覆盖** site-packages 里的 stock `deep_gemm`。

## 目录布局

```
sglang/
├── glm52_opt/manifest.json          # 归档与 overlay 路径（相对仓库根）
├── README.md                        # 本说明（GitHub 落地页）
├── README.upstream.md               # 上游 sgl-project/sglang README
├── README_DGMK.md                   # 与落地页同内容的副本
├── docs/glm52_opt_deploy.md         # 部署与校验清单
├── python/sglang/srt/layers/glm52_opt/   # phase 检测、registry、dispatch
├── scripts/glm52_opt_*.sh           # smoke / route_check / validate
└── third_party/
    ├── kernel-archive/0720-Best-GLM-52/  # 归档内核与 bench 脚本
    ├── DeepGEMM-GLM52/                   # DeepGEMM 实验 fork 源码
    │   └── overlays/                     # 本地构建产物（gitignore）
    └── deepgemm_glm52/                   # build_overlay / loader / smoke
```

`glm52_opt/manifest.json` 中的路径为相对仓库根；运行时由 `config.py` 解析为绝对路径。

## 构建 DeepGEMM overlay

```bash
cd /path/to/sglang/third_party/deepgemm_glm52

# 可选：指定 Python（需能 import torch，且与部署环境 ABI 一致）
export HARNESS_PYTHON=/path/to/venv/bin/python

./build_overlay.sh
```

要点：

- 默认 `FORK_ROOT` 为脚本旁的 `../DeepGEMM-GLM52`（可用 `DEEPGEMM_GLM52_ROOT` 覆盖）。
- 产物在 `third_party/DeepGEMM-GLM52/overlays/<full-commit>/`，并写绝对路径到 `third_party/deepgemm_glm52/manifest.json` 供 loader 使用。
- **不会** `pip install` 进当前 venv；stock `deep_gemm` 保持不动。

说明：vendored 的 `third_party/DeepGEMM-GLM52` **通常没有独立 `.git`**。此时 `build_overlay.sh` 从 `GLM52_OPT_COMMIT.txt`（或环境变量 `DEEPGEMM_GLM52_COMMIT`）读取 commit，用于 overlay 分区目录名。

双路隔离冒烟：

```bash
CUDA_VISIBLE_DEVICES=0 "$HARNESS_PYTHON" ./smoke_dual.py
```

## 环境变量与启动示例

| 变量 | 含义 |
|------|------|
| `SGLANG_GLM52_OPT=1` | 打开 glm52_opt 调度 |
| `SGLANG_GLM52_OPT_PROFILE` | `decode_max`（默认）或 `full` |
| `SGLANG_GLM52_DEEPGEMM_VARIANT` | 可选；manifest 中有 `deepgemm_variant` / commit |
| `SGLANG_GLM52_MANIFEST` | 覆盖默认 `glm52_opt/manifest.json` |
| `SGLANG_GLM52_ARCHIVE` | 覆盖内核归档根目录 |
| `SGLANG_GLM52_DEEPGEMM_OVERLAY` | 覆盖 deepgemm 工具目录 |

```bash
export SGLANG_GLM52_OPT=1
export SGLANG_GLM52_OPT_PROFILE=decode_max

python -m sglang.launch_server \
  --model-path <glm-5.2-checkpoint> \
  ...
```

## Profile：`decode_max` vs `full`

| Profile | Decode | Prefill |
|---------|--------|---------|
| `decode_max`（默认） | 启用归档/实验 decode 赢家算子 | 走 stock |
| `full` | 同上 | 额外启用 fused_qkv_a、q_b、index_* 等 prefill 赢家 |

部分算子（如 prefill `moe_gate`、`dsa_prefill_attn`）刻意留在 stock，以避免 CUDA Graph 回退。细节见 `docs/glm52_opt_deploy.md`。

## 端到端测试计划

> **硬件**：DeepGEMM fused 实验路径需要 **NVIDIA B200 / SM100（Blackwell）**。无 GPU 或非 SM100 时可跑 smoke/route_check 的路由与加载逻辑，但 fused overlay / 完整 serving 对比需在 B200 上执行。

### 1. Smoke

```bash
./scripts/glm52_opt_smoke.sh
```

### 2. 路由检查

确认 decode/prefill 落到 archive / experimental / stock：

```bash
./scripts/glm52_opt_route_check.sh
```

可选更完整校验：

```bash
./scripts/glm52_opt_validate.sh
```

### 3. build_overlay + smoke_dual

构建 DeepGEMM-GLM52 overlay 后做双路隔离冒烟（stock `deep_gemm` vs `deep_gemm_experimental`）：

```bash
cd third_party/deepgemm_glm52
export HARNESS_PYTHON="${HARNESS_PYTHON:-$(which python)}"
./build_overlay.sh
CUDA_VISIBLE_DEVICES=0 "$HARNESS_PYTHON" ./smoke_dual.py
```

### 4. Optional：Kernel-Harness / llm_flops_style benches

若本机另有 Kernel-Harness，或直接使用已 vendored 的归档：

```bash
# 归档内 layer FLOPs 风格 bench（路径相对本仓库）
ARCHIVE=third_party/kernel-archive/0720-Best-GLM-52
python "$ARCHIVE/llm_flops_style/bench_decode.py"   # decode shapes
python "$ARCHIVE/llm_flops_style/bench_prefill.py"  # prefill shapes
```

外部 Kernel-Harness 时，也可对 `archive/0720-Best-GLM-52`（或本仓库对应 vendored 路径）跑 per-op `run.sh`。

### 5. Serving e2e（TTFT / TPOT）

同一模型与 workload 下对比 **opt off vs on**：

```bash
# Baseline
SGLANG_GLM52_OPT=0 python -m sglang.launch_server \
  --model-path <glm-5.2-checkpoint> \
  ...

# Opt-in（默认 decode_max；可改 full）
SGLANG_GLM52_OPT=1 SGLANG_GLM52_OPT_PROFILE=decode_max \
  python -m sglang.launch_server \
  --model-path <glm-5.2-checkpoint> \
  ...
```

压测后对比 **TTFT** 与 **TPOT**（`SGLANG_GLM52_OPT=0` vs `1`）。更细的部署清单见 `docs/glm52_opt_deploy.md`。

## 依赖说明

- **Stock DeepGEMM（`deep_gemm`）保持原样**；本分支只通过 `deep_gemm_experimental` 加载 overlay。
- Overlay 为 **opt-in**：未构建或未设置 `SGLANG_GLM52_OPT` 时，不走实验路径。
- 内核归档已 vendored 在 `third_party/kernel-archive/`；完整 Kernel-Harness 仓库仅在跑原始战役脚本时需要。
