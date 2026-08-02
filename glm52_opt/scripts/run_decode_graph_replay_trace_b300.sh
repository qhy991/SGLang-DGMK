#!/usr/bin/env bash
# Unprofiled host-side cudaGraph replay trace under the exact fixed-KV lane.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DEFAULT=$(cd "$SCRIPT_DIR/../.." && pwd)
ROOT=${ROOT:-$(cd "$REPO_DEFAULT/.." && pwd)}
REPO=${REPO:-$REPO_DEFAULT}
VENV=${VENV:-$ROOT/venv_wwxq}
PY=${PY:-$VENV/bin/python3}
SERVER_LAUNCHER=${SERVER_LAUNCHER:-$REPO/glm52_opt/scripts/run_b300_repo_server.sh}
TRIGGER_RUNNER=${TRIGGER_RUNNER:-$REPO/glm52_opt/scripts/run_one_batch_fixedkv_nsys.py}
HARNESS_ROOT=${HARNESS_ROOT:-$ROOT/Kernel-Harness-moe-swiglu-b300-test}
ANALYZER=${ANALYZER:-$HARNESS_ROOT/testbench/bin/analyze_decode_graph_replay_host_trace.py}
MODEL=${MODEL:-/mnt/b300-shared/models/GLM-5.2-FP8}
PORT=${PORT:-30002}
S=${S:-32768}
GLOBAL_BS=${GLOBAL_BS:-128}
OUT_LEN=${OUT_LEN:-48}
REPLAY_SAMPLES=${REPLAY_SAMPLES:-40}
RUN_ID=${RUN_ID:-decode_graph_replay_host_b300_$(date -u +%Y%m%dT%H%M%SZ)}
OUT=$ROOT/bench_results/$RUN_ID
TRACE_DIR=$OUT/replay_trace
TRIGGER=$ROOT/cache/sglang/replay_trace_trigger_$RUN_ID
ENV_FILE=${SGLANG_GLM52_ENV_FILE:-$ROOT/cache/sglang/glm52_opt.env}
HIT_FILE=$ROOT/cache/sglang/glm52_opt_hits.json
ENV_BACKUP=$OUT/original_glm52_opt.env
ENV_ABSENT_MARKER=$OUT/original_glm52_opt.env.absent
LAUNCH_PID_FILE=$OUT/launch_pid.txt
SERVER_PID_FILE=$OUT/server_pid.txt
PROVIDER=$REPO/python/sglang/srt/layers/glm52_opt/hotspot_candidates/flashmla_accel_bundle_provider.py

if [[ "$GLOBAL_BS" -ne 128 || "$S" -ne 32768 ]]; then
  echo "[ERR] audited replay trace requires S=32768 and global BS=128" >&2
  exit 2
fi
for path in "$PY" "$SERVER_LAUNCHER" "$TRIGGER_RUNNER" "$ANALYZER" \
  "$PROVIDER" "$REPO/python/sglang/srt/model_executor/runner_utils/replay_trace.py"; do
  [[ -e "$path" ]] || { echo "[ERR] missing $path" >&2; exit 2; }
done

mkdir -p "$OUT" "$TRACE_DIR" "$ROOT/cache/sglang" "$ROOT/logs"
exec > >(tee -a "$OUT/run.log") 2>&1

restore_env() {
  if [[ -f "$ENV_BACKUP" ]]; then
    cp -f "$ENV_BACKUP" "$ENV_FILE"
  elif [[ -f "$ENV_ABSENT_MARKER" ]]; then
    rm -f "$ENV_FILE"
  fi
}

cleanup_ours() {
  rm -f "$TRIGGER"
  local launch_pid=""
  if [[ -f "$LAUNCH_PID_FILE" ]]; then
    launch_pid=$(cat "$LAUNCH_PID_FILE")
  fi
  if [[ -n "$launch_pid" ]] && kill -0 "$launch_pid" 2>/dev/null; then
    kill -TERM -- "-$launch_pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$launch_pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$launch_pid" 2>/dev/null; then
      kill -KILL -- "-$launch_pid" 2>/dev/null || true
    fi
  fi
  local server_pid=""
  if [[ -f "$SERVER_PID_FILE" ]]; then
    server_pid=$(cat "$SERVER_PID_FILE")
  fi
  if [[ -n "$server_pid" && -r "/proc/$server_pid/cmdline" ]]; then
    local cmdline pgid
    cmdline=$(tr '\0' ' ' < "/proc/$server_pid/cmdline")
    if [[ "$cmdline" == *sglang* && "$cmdline" == *"--port $PORT"* ]]; then
      pgid=$(ps -o pgid= -p "$server_pid" | tr -d ' ')
      [[ -z "$pgid" ]] || kill -TERM -- "-$pgid" 2>/dev/null || true
    fi
  fi
}

cleanup_on_exit() {
  local rc=$?
  trap - EXIT INT TERM
  cleanup_ours
  restore_env
  exit "$rc"
}
trap cleanup_on_exit EXIT INT TERM

if [[ -f "$ENV_FILE" ]]; then
  cp -f "$ENV_FILE" "$ENV_BACKUP"
else
  : > "$ENV_ABSENT_MARKER"
fi
if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
  echo "[ERR] port $PORT is already in use" >&2
  exit 2
fi
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
max_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
  awk 'BEGIN{m=0} {if($1+0>m)m=$1+0} END{print m}')
if [[ "$apps" -ne 0 || "$max_used" -ge 2048 ]]; then
  echo "[ERR] GPUs busy: apps=$apps max_used_mib=$max_used" >&2
  exit 2
fi

cat > "$ENV_FILE" <<EOF
SGLANG_GLM52_ALLOW_ABI_ADAPTER=0
SGLANG_GLM52_INFINI_KERNEL_NVTX=0
SGLANG_GLM52_NSYS_GATE=0
SGLANG_GLM52_OPT_HIT_FILE=$HIT_FILE
SGLANG_GLM52_MANIFEST=$REPO/glm52_opt/manifest.json
SGLANG_GLM52_DEEPGEMM_VARIANT=41c6235
SGLANG_GLM52_DEEPGEMM_OVERLAY=$REPO/third_party/deepgemm_glm52
SGLANG_OPT_GLM52_FUSED_QKV_A_PREFILL_DIRECT_NK=0
SGLANG_OPT_GLM52_FUSED_QKV_A_DECODE_DIRECT_NK=0
SGLANG_GLM52_OPT=1
SGLANG_GLM52_OPT_PROFILE=combined_winners
SGLANG_GLM52_OPT_OPS=flashmla_sparse_decode,o_proj,index_q_upproj,moe_gate_proj,moe_up_proj,moe_down_proj,moe_swiglu_quant
SGLANG_GLM52_OPT_M_BUCKETS=dsa_decode_attn:16|32,o_proj:16|32,index_q_upproj:16|32
SGLANG_GLM52_HOTSPOT_MODULE=$PROVIDER
SGLANG_GLM52_FLASHMLA_GRAPH_ONLY=1
SGLANG_GLM52_O_PROJ_GRAPH_ONLY=1
SGLANG_GLM52_INDEX_Q_UPPROJ_GRAPH_ONLY=1
SGLANG_GLM52_INFINI_MOE_ALIGN=1
SGLANG_OPT_MOE_SWIGLU_QUANT_VARIANT=cuda_valid_cta
GLM52_FLASHMLA_USE_PREBUILT=1
GLM52_FLASHMLA_DECODE_STACK=p1_c2
SGLANG_DECODE_GRAPH_REPLAY_TRACE_DIR=$TRACE_DIR
SGLANG_DECODE_GRAPH_REPLAY_TRACE_TRIGGER=$TRIGGER
SGLANG_DECODE_GRAPH_REPLAY_TRACE_SAMPLES=$REPLAY_SAMPLES
EOF

cat > "$OUT/README.md" <<EOF
# B300 unprofiled decode CUDA-graph replay trace

- commit: $(git -C "$REPO" rev-parse HEAD)
- contract: real weights, TP8/DP8/EP8, DeepEP low_latency
- decode: KV=32768/request, global BS=128, local M=16, output=$OUT_LEN
- prefill allocation: chunked=8192, max prefill tokens=8192, mem fraction=0.78
- trace: $REPLAY_SAMPLES asynchronous host replay calls/rank, armed after KV warmup
- clock: time.perf_counter_ns on one host; no CUDA synchronization and no nsys
- purpose: test whether nsys's every-fourth-replay millisecond tail exists in production
EOF
git -C "$REPO" status --short > "$OUT/git_status.txt"
git -C "$REPO" log -1 --oneline > "$OUT/git_rev.txt"
cp -f "$ENV_FILE" "$OUT/candidate.env"
rm -f "$TRIGGER" "$HIT_FILE"

echo "[INFO] launching unprofiled server at $(date -Is); OUT=$OUT"
setsid env \
  PATH="$VENV/bin:$PATH" \
  ROOT="$ROOT" REPO="$REPO" MODEL="$MODEL" PORT="$PORT" \
  SGLANG_GLM52_ENV_FILE="$ENV_FILE" \
  SGLANG_CUDA_GRAPH_MAX_BS=16 \
  SGLANG_MAX_RUNNING_REQUESTS=128 \
  CHUNKED_PREFILL_SIZE=8192 \
  MAX_PREFILL_TOKENS=8192 \
  SGLANG_DEEPEP_MODE=low_latency \
  SGLANG_EXTRA_SERVE_ARGS="--mem-fraction-static 0.78" \
  bash "$SERVER_LAUNCHER" > "$OUT/server.log" 2>&1 &
launch_pid=$!
echo "$launch_pid" > "$LAUNCH_PID_FILE"

http_code() {
  curl -s -o /dev/null -w '%{http_code}' --connect-timeout 2 \
    "http://127.0.0.1:$PORT/v1/models" 2>/dev/null || echo 000
}
ready=0
for i in $(seq 1 180); do
  code=$(http_code); code=${code: -3}
  echo "$(date +%H:%M:%S) ready_wait=$i code=$code"
  if [[ "$code" == 200 ]]; then
    server_pid=$(ss -ltnp 2>/dev/null |
      sed -n "s/.*:${PORT} .*pid=\\([0-9][0-9]*\\).*/\\1/p" | head -1)
    [[ -n "$server_pid" ]] || { echo "[ERR] cannot resolve server PID" >&2; exit 2; }
    echo "$server_pid" > "$SERVER_PID_FILE"
    ready=1
    break
  fi
  if ! kill -0 "$launch_pid" 2>/dev/null; then
    echo "[ERR] server exited before ready" >&2
    tail -120 "$OUT/server.log"
    exit 2
  fi
  if grep -qE "CUDA out of memory|Capture cuda graph failed|Address already in use" \
    "$OUT/server.log" 2>/dev/null; then
    echo "[ERR] server failure signature" >&2
    tail -120 "$OUT/server.log"
    exit 2
  fi
  sleep 10
done
[[ "$ready" -eq 1 ]] || { echo "[ERR] server readiness timeout" >&2; exit 2; }

for bucket in 1 2 4 8 12 16; do
  count=$(grep -F -c \
    "shape=(32, 8192, 4096) stride=(33554432, 4096, 1) routed_m=$bucket topk=8" \
    "$OUT/server.log" || true)
  [[ "$count" -eq 8 ]] || {
    echo "[ERR] expected M=$bucket candidate selection on 8 ranks; got $count" >&2
    exit 2
  }
done

rate=$($PY -c "print(($S - 64) / float($S))")
result=$OUT/decode_candidate_bs${GLOBAL_BS}.jsonl
: > "$result"
GLM52_NSYS_TRIGGER="$TRIGGER" GLM52_NSYS_ARM_DELAY=2 \
"$PY" "$TRIGGER_RUNNER" \
  --model None \
  --base-url "http://127.0.0.1:$PORT" \
  --local-tokenizer-path "$MODEL" \
  --batch-size "$GLOBAL_BS" \
  --input-len "$S" \
  --output-len "$OUT_LEN" \
  --cache-hit-rate "$rate" \
  --dataset-name random-ids \
  --result-filename "$result" \
  --run-name candidate_replay_host_decode_bs${GLOBAL_BS} \
  --show-report \
  --no-append-to-github-summary \
  --skip-warmup

trace_count=0
for i in $(seq 1 60); do
  trace_count=$(find "$TRACE_DIR" -maxdepth 1 -type f \
    -name 'decode-graph-replay-tp-rank-*.json' | wc -l)
  [[ "$trace_count" -eq 8 ]] && break
  echo "$(date +%H:%M:%S) trace_wait=$i files=$trace_count"
  sleep 1
done
[[ "$trace_count" -eq 8 ]] || { echo "[ERR] expected 8 replay traces" >&2; exit 2; }
if find "$TRACE_DIR" -maxdepth 1 -type f -name '*.tmp' | grep -q .; then
  echo "[ERR] incomplete replay trace temporary files remain" >&2
  exit 2
fi
"$PY" "$ANALYZER" --trace-dir "$TRACE_DIR" --world-size 8 --period 4 \
  --output "$OUT/replay_trace_analysis.json"

if [[ -f "$HIT_FILE" ]]; then
  cp -f "$HIT_FILE" "$OUT/hits_candidate.json"
fi
sha256sum "$result" "$OUT/replay_trace_analysis.json" \
  "$TRACE_DIR"/decode-graph-replay-tp-rank-*.json > "$OUT/SHA256SUMS"
echo "[DONE] unprofiled replay trace=$OUT/replay_trace_analysis.json"

cleanup_ours
restore_env
trap - EXIT INT TERM
