#!/bin/bash
set -euo pipefail
ROOT=${ROOT:-/mnt/b300-shared/home/qinhaiyan/wwxq}
REPO=${REPO:-$ROOT/SGLang-DGMK}
MODEL=${MODEL:-/mnt/b300-shared/models/GLM-5.2-FP8}
cd "$ROOT"

model_path=$MODEL
model_name=GLM-5.2-FP8
port="${PORT:-30000}"
max_bs="${SGLANG_CUDA_GRAPH_MAX_BS:-16}"
max_running="${SGLANG_MAX_RUNNING_REQUESTS:-$((8 * max_bs))}"

mkdir -p "$ROOT/cache/"{flashinfer,deep_gemm,triton,sglang}
ENV_FILE=$ROOT/cache/sglang/glm52_opt.env
export SGLANG_GLM52_ENV_FILE="$ENV_FILE"
while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "${line//[[:space:]]/}" || "$line" =~ ^[[:space:]]*# ]] && continue
  key="${line%%=*}"; val="${line#*=}"
  key="${key%"${key##*[![:space:]]}"}"; key="${key#"${key%%[![:space:]]*}"}"
  val="${val#"${val%%[![:space:]]*}"}"; val="${val%"${val##*[![:space:]]}"}"
  if [[ "$val" == \'*\' ]]; then val="${val:1:-1}"; elif [[ "$val" == \"*\" ]]; then val="${val:1:-1}"; fi
  export "$key=$val"
done < "$ENV_FILE"

export NCCL_NVLS_ENABLE=1 PYTHONUNBUFFERED=1 SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 NCCL_CUMEM_ENABLE=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024
export SGLANG_CACHE_DIR=$ROOT/cache/sglang FLASHINFER_CACHE_DIR=$ROOT/cache/flashinfer
export SGLANG_DG_CACHE_DIR=${SGLANG_DG_CACHE_DIR:-$ROOT/cache/deep_gemm} TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$ROOT/cache/triton}
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1
export PYTHONPATH="$REPO/python${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_DEEPGEMM_PDL=1

DEEPEP_MODE="${SGLANG_DEEPEP_MODE:-auto}"
# normal-mode dispatch/combine with tuned num_sms=24 (only read in normal/auto, not LL)
DEEPEP_CONFIG="${SGLANG_DEEPEP_CONFIG:-$REPO/glm52_opt/deepep/ep8_joint_config_seed.json}"
DEEPEP_ARGS=(--deepep-mode "$DEEPEP_MODE")
[[ -n "$DEEPEP_CONFIG" ]] && DEEPEP_ARGS+=(--deepep-config "$DEEPEP_CONFIG")

echo "[INFO] repo=$REPO model=$model_path OPT=$SGLANG_GLM52_OPT deepep=$DEEPEP_MODE"
exec sglang serve \
  --model-path "$model_path" --served-model-name "$model_name" --trust-remote-code \
  --json-model-override-args "{\"qk_rope_head_dim\":64,\"qk_nope_head_dim\":192}" \
  --host 0.0.0.0 --port "$port" --tp-size 8 --dp-size 8 --enable-dp-attention \
  --watchdog-timeout 1800 --context-length 131072 --page-size 64 --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.78 --max-running-requests "$max_running" --cuda-graph-max-bs "$max_bs" \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE:-8192}" --max-prefill-tokens "${MAX_PREFILL_TOKENS:-8192}" \
  --dsa-prefill-backend flashmla_kv --dsa-decode-backend flashmla_kv \
  --moe-a2a-backend deepep "${DEEPEP_ARGS[@]}" --moe-dense-tp-size 1 --enable-dp-lm-head \
  --disable-overlap-schedule --allow-auto-truncate \
  --reasoning-parser glm45 --tool-call-parser glm47 --enable-metrics --decode-log-interval 10 \
  ${SGLANG_EXTRA_SERVE_ARGS:-} \
  2>&1 | tee "$ROOT/logs/glm52_dgmk.log"
