#!/usr/bin/env bash
# Write glm52_opt.env for e2e-proven decode winners only:
#   FlashMLA P1+c2 + o_proj + index_q_upproj fixed_nk + MoE M-tile align
# Excludes: fused_qkv_a, dsa_prefill, r2a
# Usage:
#   ROOT=/path/to/wwxq bash glm52_opt/scripts/write_decode_winners_env.sh
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DEFAULT=$(cd "$SCRIPT_DIR/../.." && pwd)
ROOT=${ROOT:-$(cd "$REPO_DEFAULT/.." && pwd)}
REPO=${REPO:-$REPO_DEFAULT}
ENV_FILE=${SGLANG_GLM52_ENV_FILE:-$ROOT/cache/sglang/glm52_opt.env}
HIT_FILE=${SGLANG_GLM52_OPT_HIT_FILE:-$ROOT/cache/sglang/glm52_opt_hits.json}
PROVIDER=$REPO/python/sglang/srt/layers/glm52_opt/hotspot_candidates/flashmla_accel_bundle_provider.py

mkdir -p "$(dirname "$ENV_FILE")"
cat > "$ENV_FILE" <<EOF
SGLANG_GLM52_ALLOW_ABI_ADAPTER=0
SGLANG_GLM52_INFINI_KERNEL_NVTX=0
SGLANG_GLM52_OPT_HIT_FILE=$HIT_FILE
SGLANG_GLM52_MANIFEST=$REPO/glm52_opt/manifest.json
SGLANG_GLM52_DEEPGEMM_VARIANT=41c6235
SGLANG_GLM52_DEEPGEMM_OVERLAY=$REPO/third_party/deepgemm_glm52
SGLANG_OPT_GLM52_FUSED_QKV_A_PREFILL_DIRECT_NK=0
SGLANG_OPT_GLM52_FUSED_QKV_A_DECODE_DIRECT_NK=0
SGLANG_GLM52_OPT=1
SGLANG_GLM52_OPT_PROFILE=combined_winners
SGLANG_GLM52_OPT_OPS=flashmla_sparse_decode,o_proj,index_q_upproj,moe_gate_proj,moe_up_proj,moe_down_proj
SGLANG_GLM52_OPT_M_BUCKETS=dsa_decode_attn:16|32,o_proj:16|32,index_q_upproj:16|32
SGLANG_GLM52_HOTSPOT_MODULE=$PROVIDER
SGLANG_GLM52_FLASHMLA_GRAPH_ONLY=1
SGLANG_GLM52_O_PROJ_GRAPH_ONLY=1
SGLANG_GLM52_INDEX_Q_UPPROJ_GRAPH_ONLY=1
SGLANG_GLM52_INFINI_MOE_ALIGN=1
GLM52_FLASHMLA_USE_PREBUILT=1
GLM52_FLASHMLA_DECODE_STACK=p1_c2
EOF
echo "[OK] wrote $ENV_FILE"
cat "$ENV_FILE"
