#!/usr/bin/env bash
set -euo pipefail

adapter_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
sglang_root="${SGLANG_ROOT:-$(cd -- "${adapter_root}/../.." && pwd)}"
: "${MOK_ROOT:?Set MOK_ROOT to a built mixture-of-kittens checkout}"
: "${MODEL_PATH:?Set MODEL_PATH to the GLM-5.2 FP8 checkpoint}"
mok_root="${MOK_ROOT}"
model_root="${MODEL_PATH}"
python_bin="${PYTHON_BIN:-/usr/bin/python3}"
sglang_bin="${SGLANG_BIN:-/usr/local/bin/sglang}"

export PYTHONPATH="${adapter_root}:${sglang_root}/python:${mok_root}${PYTHONPATH:+:${PYTHONPATH}}"
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=1
export MOK_SGLANG_PREFILL="${MOK_SGLANG_PREFILL:-1}"
export MOK_SGLANG_LAYERS="${MOK_SGLANG_LAYERS:-all}"
export MOK_SGLANG_VALIDATE_LAYER="${MOK_SGLANG_VALIDATE_LAYER:--1}"
export MOK_SGLANG_PREFILL_TOKENS="${MOK_SGLANG_PREFILL_TOKENS:-1024}"
export MOK_SGLANG_FWD_COMM_SMS="${MOK_SGLANG_FWD_COMM_SMS:-32}"
export MOK_SGLANG_MINIBATCH_SIZE="${MOK_SGLANG_MINIBATCH_SIZE:-2560}"
export MOK_SGLANG_MACROBATCH_SIZE="${MOK_SGLANG_MACROBATCH_SIZE:-20480}"
export MOK_SGLANG_SCHEDULE_CAPACITY_MULTIPLIER="${MOK_SGLANG_SCHEDULE_CAPACITY_MULTIPLIER:-1.0}"
export MOK_SGLANG_MODEL_PATH="${MOK_SGLANG_MODEL_PATH:-${model_root}}"

log_path="${MOK_SGLANG_LOG_PATH:-/tmp/mok_glm52_prefill_server.log}"
mkdir -p "$(dirname -- "${log_path}")"

exec "${python_bin}" "${sglang_bin}" serve \
  --model-path "${model_root}" \
  --served-model-name GLM-5.2-FP8 \
  --trust-remote-code \
  --json-model-override-args '{"qk_rope_head_dim":64,"qk_nope_head_dim":192}' \
  --host 0.0.0.0 \
  --port 30000 \
  --watchdog-timeout 1800 \
  --context-length 8192 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.78 \
  --max-total-tokens 16384 \
  --max-running-requests 16 \
  --chunked-prefill-size 8192 \
  --max-prefill-tokens 8192 \
  --tp-size 8 \
  --dp-size 8 \
  --ep-size 8 \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --load-balance-method round_robin \
  --dsa-prefill-backend flashmla_kv \
  --dsa-decode-backend flashmla_kv \
  --moe-a2a-backend deepep \
  --deepep-mode normal \
  --moe-dense-tp-size 1 \
  --disable-overlap-schedule \
  --disable-radix-cache \
  --disable-cuda-graph \
  --allow-auto-truncate \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --enable-metrics \
  >"${log_path}" 2>&1
