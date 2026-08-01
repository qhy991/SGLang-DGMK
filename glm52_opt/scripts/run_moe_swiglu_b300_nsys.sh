#!/usr/bin/env bash
# Capture one fixed-KV B300 decode trace with stock or valid-CTA SwiGLU.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DEFAULT=$(cd "$SCRIPT_DIR/../.." && pwd)
ROOT=${ROOT:-$(cd "$REPO_DEFAULT/.." && pwd)}
REPO=${REPO:-$REPO_DEFAULT}
VENV=${VENV:-$ROOT/venv_wwxq}
PY=${PY:-$VENV/bin/python3}
NSYS=${NSYS:-/usr/local/cuda/bin/nsys}
SERVER_LAUNCHER=${SERVER_LAUNCHER:-$REPO/glm52_opt/scripts/run_b300_repo_server.sh}
MODEL=${MODEL:-/mnt/b300-shared/models/GLM-5.2-FP8}
PORT=${PORT:-30002}
DP=${DP:-8}
S=${S:-32768}
GLOBAL_BS=${GLOBAL_BS:-128}
OUT_LEN=${OUT_LEN:-48}
NSYS_DURATION=${NSYS_DURATION:-30}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.83}
SGLANG_CUDA_GRAPH_MAX_BS=${SGLANG_CUDA_GRAPH_MAX_BS:-16}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-2048}
MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-16384}
SWIGLU_MODE=${SWIGLU_MODE:-candidate}
RUN_ID=${RUN_ID:-nsys_moe_swiglu_${SWIGLU_MODE}_$(date -u +%Y%m%dT%H%M%SZ)}
OUT=$ROOT/bench_results/$RUN_ID
ENV_FILE=${SGLANG_GLM52_ENV_FILE:-$ROOT/cache/sglang/glm52_opt.env}
HIT_FILE=$ROOT/cache/sglang/glm52_opt_hits.json
TRIGGER=$ROOT/cache/sglang/nsys_trigger_$RUN_ID
TRIGGER_RUNNER=$REPO/glm52_opt/scripts/run_one_batch_fixedkv_nsys.py
PROVIDER=$REPO/python/sglang/srt/layers/glm52_opt/hotspot_candidates/flashmla_accel_bundle_provider.py
REP_BASE=$OUT/$SWIGLU_MODE
NSYS_PID_FILE=$OUT/nsys_pid.txt
SERVER_PID_FILE=$OUT/server_pid.txt
ENV_BACKUP=$OUT/original_glm52_opt.env
ENV_ABSENT_MARKER=$OUT/original_glm52_opt.env.absent

if [[ "$GLOBAL_BS" -ne $((DP * 16)) ]]; then
  echo "[ERR] audited trace requires global BS=$((DP * 16)); got $GLOBAL_BS" >&2
  exit 2
fi
if [[ "$SWIGLU_MODE" != candidate && "$SWIGLU_MODE" != stock ]]; then
  echo "[ERR] SWIGLU_MODE must be candidate or stock; got $SWIGLU_MODE" >&2
  exit 2
fi
for path in "$PY" "$NSYS" "$SERVER_LAUNCHER" "$TRIGGER_RUNNER" "$PROVIDER"; do
  [[ -e "$path" ]] || { echo "[ERR] missing $path" >&2; exit 2; }
done

mkdir -p "$OUT" "$ROOT/cache/sglang" "$ROOT/logs"
LOG=$OUT/run.log
exec > >(tee -a "$LOG") 2>&1

restore_env() {
  if [[ -f "$ENV_BACKUP" ]]; then
    cp -f "$ENV_BACKUP" "$ENV_FILE"
  elif [[ -f "$ENV_ABSENT_MARKER" ]]; then
    rm -f "$ENV_FILE"
  fi
}

cleanup_ours() {
  rm -f "$TRIGGER"
  if [[ -f "$NSYS_PID_FILE" ]]; then
    local pid
    pid=$(cat "$NSYS_PID_FILE")
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM -- "-$pid" 2>/dev/null || true
      for _ in $(seq 1 20); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
      done
      if kill -0 "$pid" 2>/dev/null; then
        kill -KILL -- "-$pid" 2>/dev/null || true
      fi
    fi
  fi

  # Nsight may put the instrumented server in a child session instead of the
  # nsys process group.  Reap only the service that this runner recorded on its
  # already-exclusive test port; never use a broad sglang/process-name kill.
  local server_pid=""
  if [[ -f "$SERVER_PID_FILE" ]]; then
    server_pid=$(cat "$SERVER_PID_FILE")
  else
    server_pid=$(ss -ltnp 2>/dev/null |
      sed -n "s/.*:${PORT} .*pid=\\([0-9][0-9]*\\).*/\\1/p" | head -1)
  fi
  if [[ -n "$server_pid" && -r "/proc/$server_pid/cmdline" ]]; then
    local cmdline pgid
    cmdline=$(tr '\0' ' ' < "/proc/$server_pid/cmdline")
    if [[ "$cmdline" == *sglang* && "$cmdline" == *"--port $PORT"* ]]; then
      pgid=$(ps -o pgid= -p "$server_pid" | tr -d ' ')
      if [[ -n "$pgid" ]]; then
        kill -TERM -- "-$pgid" 2>/dev/null || true
        for _ in $(seq 1 20); do
          kill -0 "$server_pid" 2>/dev/null || break
          sleep 1
        done
        if kill -0 "$server_pid" 2>/dev/null; then
          kill -KILL -- "-$pgid" 2>/dev/null || true
        fi
      fi
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
  echo "[ERR] GPUs busy before trace: apps=$apps max_used_mib=$max_used" >&2
  exit 2
fi

ops=flashmla_sparse_decode,o_proj,index_q_upproj,moe_gate_proj,moe_up_proj,moe_down_proj
if [[ "$SWIGLU_MODE" == candidate ]]; then
  ops=$ops,moe_swiglu_quant
fi

cat > "$ENV_FILE" <<EOF
SGLANG_GLM52_ALLOW_ABI_ADAPTER=0
SGLANG_GLM52_INFINI_KERNEL_NVTX=1
SGLANG_GLM52_NSYS_GATE=1
SGLANG_GLM52_NSYS_TRIGGER=$TRIGGER
SGLANG_GLM52_NSYS_SECONDS=$NSYS_DURATION
SGLANG_GLM52_OPT_HIT_FILE=$HIT_FILE
SGLANG_GLM52_MANIFEST=$REPO/glm52_opt/manifest.json
SGLANG_GLM52_DEEPGEMM_VARIANT=41c6235
SGLANG_GLM52_DEEPGEMM_OVERLAY=$REPO/third_party/deepgemm_glm52
SGLANG_OPT_GLM52_FUSED_QKV_A_PREFILL_DIRECT_NK=0
SGLANG_OPT_GLM52_FUSED_QKV_A_DECODE_DIRECT_NK=0
SGLANG_GLM52_OPT=1
SGLANG_GLM52_OPT_PROFILE=combined_winners
SGLANG_GLM52_OPT_OPS=$ops
SGLANG_GLM52_OPT_M_BUCKETS=dsa_decode_attn:16|32,o_proj:16|32,index_q_upproj:16|32
SGLANG_GLM52_HOTSPOT_MODULE=$PROVIDER
SGLANG_GLM52_FLASHMLA_GRAPH_ONLY=1
SGLANG_GLM52_O_PROJ_GRAPH_ONLY=1
SGLANG_GLM52_INDEX_Q_UPPROJ_GRAPH_ONLY=1
SGLANG_GLM52_INFINI_MOE_ALIGN=1
GLM52_FLASHMLA_USE_PREBUILT=1
GLM52_FLASHMLA_DECODE_STACK=p1_c2
EOF
if [[ "$SWIGLU_MODE" == candidate ]]; then
  echo "SGLANG_OPT_MOE_SWIGLU_QUANT_VARIANT=cuda_valid_cta" >> "$ENV_FILE"
fi

{
  echo "# B300 $SWIGLU_MODE SwiGLU containing-region nsys"
  echo
  echo "- commit: $(git -C "$REPO" rev-parse HEAD)"
  echo "- workload: S=$S, global BS=$GLOBAL_BS, local M=16, output=$OUT_LEN"
  echo "- cache target: only the last 64 prompt tokens uncached (99.8% hit)"
  echo "- topology: TP8/DP8/EP8; CUDA graph max BS=$SGLANG_CUDA_GRAPH_MAX_BS"
  echo "- capture: cuda,nvtx; cudaProfilerApi; ${NSYS_DURATION}s"
  if [[ "$SWIGLU_MODE" == candidate ]]; then
    echo "- SwiGLU: cuda_valid_cta; stock body; grid=128 rather than 65,536"
  else
    echo "- SwiGLU: stock production denominator; grid=65,536"
  fi
} > "$OUT/README.md"
git -C "$REPO" status --short > "$OUT/git_status.txt"
git -C "$REPO" log -1 --oneline > "$OUT/git_rev.txt"
"$NSYS" --version > "$OUT/nsys_version.txt"
cp -f "$ENV_FILE" "$OUT/${SWIGLU_MODE}.env"
rm -f "$HIT_FILE" "$TRIGGER" "$REP_BASE.nsys-rep" "$REP_BASE.sqlite"

echo "[INFO] launching nsys at $(date -Is); OUT=$OUT"
setsid env \
  PATH="$VENV/bin:$PATH" \
  ROOT="$ROOT" REPO="$REPO" MODEL="$MODEL" PORT="$PORT" \
  SGLANG_GLM52_ENV_FILE="$ENV_FILE" \
  SGLANG_CUDA_GRAPH_MAX_BS="$SGLANG_CUDA_GRAPH_MAX_BS" \
  SGLANG_MAX_RUNNING_REQUESTS=128 \
  CHUNKED_PREFILL_SIZE="$CHUNKED_PREFILL_SIZE" \
  MAX_PREFILL_TOKENS="$MAX_PREFILL_TOKENS" \
  SGLANG_EXTRA_SERVE_ARGS="--mem-fraction-static $MEM_FRACTION_STATIC" \
  "$NSYS" profile \
    --force-overwrite=true \
    -o "$REP_BASE" \
    --trace=cuda,nvtx \
    --sample=none \
    --cpuctxsw=none \
    --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop-shutdown \
    --kill=none \
    bash "$SERVER_LAUNCHER" > "$OUT/nsys_launch.log" 2>&1 &
nsys_pid=$!
echo "$nsys_pid" > "$NSYS_PID_FILE"

http_code() {
  curl -s -o /dev/null -w '%{http_code}' --connect-timeout 2 \
    "http://127.0.0.1:$PORT/v1/models" 2>/dev/null || echo 000
}

ready=0
for i in $(seq 1 180); do
  code=$(http_code); code=${code: -3}
  gates=$(grep -c "nsys gate armed" "$OUT/nsys_launch.log" 2>/dev/null || true)
  echo "$(date +%H:%M:%S) ready_wait=$i code=$code gates=$gates"
  if [[ "$code" == 200 ]]; then
    server_pid=$(ss -ltnp 2>/dev/null |
      sed -n "s/.*:${PORT} .*pid=\\([0-9][0-9]*\\).*/\\1/p" | head -1)
    [[ -n "$server_pid" ]] || { echo "[ERR] cannot resolve test server PID" >&2; exit 2; }
    echo "$server_pid" > "$SERVER_PID_FILE"
    ready=1
    break
  fi
  if ! kill -0 "$nsys_pid" 2>/dev/null; then
    echo "[ERR] nsys/server exited before ready" >&2
    tail -120 "$OUT/nsys_launch.log"
    exit 2
  fi
  if grep -qE "CUDA out of memory|Capture cuda graph failed|Address already in use" \
    "$OUT/nsys_launch.log" 2>/dev/null; then
    echo "[ERR] server failure signature" >&2
    tail -120 "$OUT/nsys_launch.log"
    exit 2
  fi
  sleep 10
done
[[ "$ready" -eq 1 ]] || { echo "[ERR] server readiness timeout" >&2; exit 2; }

selection_pattern="GLM-5.2 masked SwiGLU quant selected: variant=cuda_valid_cta capability=(10, 3) shape=(32, 8192, 4096)"
selection_count=$(grep -c "$selection_pattern" "$OUT/nsys_launch.log" || true)
if [[ "$SWIGLU_MODE" == candidate ]]; then
  graph_buckets=(1 2 4 8 12 16)
  if [[ "$SGLANG_CUDA_GRAPH_MAX_BS" -ge 32 ]]; then
    graph_buckets+=(32)
  fi
  for bucket in "${graph_buckets[@]}"; do
    bucket_count=$(grep -F -c \
      "shape=(32, 8192, 4096) stride=(33554432, 4096, 1) routed_m=$bucket topk=8" \
      "$OUT/nsys_launch.log" || true)
    if [[ "$bucket_count" -ne "$DP" ]]; then
      echo "[ERR] expected M=$bucket candidate selection on $DP ranks, observed $bucket_count" >&2
      exit 2
    fi
  done
  expected_selection_count=$((DP * ${#graph_buckets[@]}))
  if [[ "$selection_count" -ne "$expected_selection_count" ]]; then
    echo "[ERR] expected $expected_selection_count total selections, observed $selection_count" >&2
    exit 2
  fi
elif [[ "$selection_count" -ne 0 ]]; then
  echo "[ERR] stock denominator selected candidate on $selection_count ranks" >&2
  exit 2
fi
echo "[VALID] SwiGLU mode=$SWIGLU_MODE selection_count=$selection_count"

rate=$($PY -c "print(($S - 64) / float($S))")
result=$OUT/decode_${SWIGLU_MODE}_bs${GLOBAL_BS}.jsonl
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
  --run-name ${SWIGLU_MODE}_nsys_decode_bs${GLOBAL_BS} \
  --show-report \
  --no-append-to-github-summary \
  --skip-warmup

"$PY" - "$result" <<'PY'
import json
import sys
from pathlib import Path

rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
if len(rows) != 1:
    raise SystemExit(f"expected one result row, found {len(rows)}")
row = rows[0]
if row.get("batch_size") != 128 or row.get("input_len") != 32768:
    raise SystemExit(f"unexpected measured shape: {row}")
if float(row.get("cache_hit_rate", -1)) < 0.99:
    raise SystemExit(f"invalid fixed-KV cache hit: {row.get('cache_hit_rate')}")
print("[VALID] fixed-KV result", json.dumps(row, sort_keys=True))
PY

sleep "$((NSYS_DURATION + 5))"
rm -f "$TRIGGER"
for i in $(seq 1 180); do
  if ! kill -0 "$nsys_pid" 2>/dev/null; then
    break
  fi
  echo "$(date +%H:%M:%S) nsys_flush_wait=$i"
  sleep 5
done
wait "$nsys_pid" || true

[[ -s "$REP_BASE.nsys-rep" ]] || {
  echo "[ERR] missing nsys report $REP_BASE.nsys-rep" >&2
  exit 2
}
"$NSYS" stats --force-export=true --report cuda_gpu_kern_sum --format csv \
  -o "$OUT/${SWIGLU_MODE}_kern" "$REP_BASE.nsys-rep"
[[ -s "$REP_BASE.sqlite" ]] || { echo "[ERR] nsys sqlite export missing" >&2; exit 2; }
if [[ -f "$HIT_FILE" ]]; then
  cp -f "$HIT_FILE" "$OUT/hits_${SWIGLU_MODE}.json"
fi
sha256sum "$REP_BASE.nsys-rep" "$REP_BASE.sqlite" "$result" > "$OUT/SHA256SUMS"
echo "[DONE] trace=$REP_BASE.nsys-rep sqlite=$REP_BASE.sqlite"

cleanup_ours
restore_env
trap - EXIT INT TERM
