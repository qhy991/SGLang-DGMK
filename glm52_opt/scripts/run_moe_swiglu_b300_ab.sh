#!/usr/bin/env bash
# Fair decode TPOT A/B: compare one candidate inside the existing decode winners.
# Supports valid-CTA SwiGLU plus order-bracketed FlashMLA and router/DeepEP runs.
#
# Workload: S=32k KV, global BS ∈ {128,256} (local_M=16/32, DP=8).
# Each measured run asks one_batch_server to build the exact per-request KV
# prefixes first, then times only the incremental tail plus decode.  Keep
# --enable-multi-batch off: SGLang documents TTFT/ITL as invalid in that mode.
set -euo pipefail

# ROOT = workspace containing venv_wwxq/, run_glm52_dgmk.sh, bench_results/
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DEFAULT=$(cd "$SCRIPT_DIR/../.." && pwd)   # .../SGLang-DGMK
ROOT=${ROOT:-$(cd "$REPO_DEFAULT/.." && pwd)}
REPO=${REPO:-$REPO_DEFAULT}
VENV=${VENV:-$ROOT/venv_wwxq}
PY=${PY:-$VENV/bin/python}
ENV_FILE=${SGLANG_GLM52_ENV_FILE:-$ROOT/cache/sglang/glm52_opt.env}
HIT_FILE=${SGLANG_GLM52_OPT_HIT_FILE:-$ROOT/cache/sglang/glm52_opt_hits.json}
SERVE_LOG=${SERVE_LOG:-$ROOT/logs/glm52_dgmk.log}
PORT=${PORT:-30002}
DP=${DP:-8}
S=${S:-32768}
OUT_LEN=${OUT_LEN:-48}
N_RUNS=${N_RUNS:-3}
GLOBAL_BS_LIST=${GLOBAL_BS_LIST:-"128"}
LABELS=${LABELS:-"winners swiglu"}
FLASHMLA_R2A_SO=${FLASHMLA_R2A_SO:-}
# Opt in for dense repeated measurements after one audited prefix-cache build.
# The default preserves the historical, independently flushed protocol.
FIXED_KV_SERIES=${FIXED_KV_SERIES:-0}
FIXED_KV_WARMUP_RUNS=${FIXED_KV_WARMUP_RUNS:-2}
# Start, capture and audit selection without constructing the expensive KV
# prefix or recording latency. Useful for clean-artifact serving ABI smoke.
VALIDATE_ONLY=${VALIDATE_ONLY:-0}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.83}
# Need 32 for global BS=256 (local_M=32).
SGLANG_CUDA_GRAPH_MAX_BS=${SGLANG_CUDA_GRAPH_MAX_BS:-32}
# DP attention divides this by DP.  2048 -> 256 tokens/rank preserves enough
# KV capacity for BS128 x S32k; 16384 -> 2048/rank reserves a much larger
# prefill workspace and reduced the observed capacity to only 67,904/rank.
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-2048}
MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-16384}
MODEL=${MODEL:-/mnt/b300-shared/models/GLM-5.2-FP8}
RUN_ID=${RUN_ID:-moe_swiglu_b300_n${N_RUNS}_s${S}_$(date -u +%Y%m%dT%H%M%SZ)}
OUT=$ROOT/bench_results/$RUN_ID
SERVER_LAUNCHER=${SERVER_LAUNCHER:-$REPO/glm52_opt/scripts/run_b300_repo_server.sh}
FIXED_KV_RUNNER=${FIXED_KV_RUNNER:-$REPO/glm52_opt/scripts/run_fixed_kv_decode_series.py}
FIXED_KV_ABA_ANALYZER=${FIXED_KV_ABA_ANALYZER:-$REPO/glm52_opt/scripts/analyze_fixed_kv_aba.py}
ENV_BACKUP=$OUT/original_glm52_opt.env
ENV_ABSENT_MARKER=$OUT/original_glm52_opt.env.absent
ACTIVE_SERVER_PID_FILE=$OUT/active_server_pid.txt

PROVIDER=$REPO/python/sglang/srt/layers/glm52_opt/hotspot_candidates/flashmla_accel_bundle_provider.py

export PATH=$VENV/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=$REPO/python${PYTHONPATH:+:$PYTHONPATH}
unset CUDA_VISIBLE_DEVICES || true
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset SGLANG_EXTRA_SERVE_ARGS || true
export SGLANG_CUDA_GRAPH_MAX_BS
export SGLANG_MAX_RUNNING_REQUESTS=${SGLANG_MAX_RUNNING_REQUESTS:-$((8 * SGLANG_CUDA_GRAPH_MAX_BS))}

mkdir -p "$OUT" "$ROOT/cache/sglang" "$ROOT/logs" /tmp
if [[ "$FIXED_KV_SERIES" != 0 && "$FIXED_KV_SERIES" != 1 ]]; then
  echo "[ERR] FIXED_KV_SERIES must be 0 or 1; got $FIXED_KV_SERIES" >&2
  exit 2
fi
if [[ "$VALIDATE_ONLY" != 0 && "$VALIDATE_ONLY" != 1 ]]; then
  echo "[ERR] VALIDATE_ONLY must be 0 or 1; got $VALIDATE_ONLY" >&2
  exit 2
fi
if [[ "$FIXED_KV_WARMUP_RUNS" -lt 0 ]]; then
  echo "[ERR] FIXED_KV_WARMUP_RUNS must be non-negative" >&2
  exit 2
fi
if [[ "$FIXED_KV_SERIES" == 1 && ! -f "$FIXED_KV_RUNNER" ]]; then
  echo "[ERR] missing fixed-KV series runner: $FIXED_KV_RUNNER" >&2
  exit 2
fi
if [[ "$FIXED_KV_SERIES" == 1 \
      && ( "$LABELS" == "p1_before r2a p1_after" \
        || "$LABELS" == "router_before router_ids router_after" ) \
      && ! -f "$FIXED_KV_ABA_ANALYZER" ]]; then
  echo "[ERR] missing fixed-KV A-B-A analyzer: $FIXED_KV_ABA_ANALYZER" >&2
  exit 2
fi
if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
  echo "[ERR] test port $PORT is already in use" >&2
  exit 2
fi
if [[ -f "$ENV_FILE" ]]; then
  cp -f "$ENV_FILE" "$ENV_BACKUP"
else
  : > "$ENV_ABSENT_MARKER"
fi
LOG=$OUT/run.log
exec > >(tee -a "$LOG") 2>&1

echo "======== decode TPOT N=$N_RUNS S=$S BS={$GLOBAL_BS_LIST} $(date -Is) ========"
echo "OUT=$OUT LABELS=$LABELS graph_max_bs=$SGLANG_CUDA_GRAPH_MAX_BS mem=$MEM_FRACTION_STATIC"
echo "FIXED_KV_SERIES=$FIXED_KV_SERIES FIXED_KV_WARMUP_RUNS=$FIXED_KV_WARMUP_RUNS"
echo "VALIDATE_ONLY=$VALIDATE_ONLY"
cd "$REPO"
git rev-parse --short HEAD | tee "$OUT/git_rev.txt"
git branch --show-current | tee -a "$OUT/git_rev.txt"
git log -1 --oneline | tee -a "$OUT/git_rev.txt"
test -f "$PROVIDER" || { echo "[ERR] missing $PROVIDER"; exit 1; }
cd "$ROOT"

cat > "$OUT/README.md" <<MD
# B300 MoE SwiGLU+quant decode A/B (N=${N_RUNS}, S=${S})

## Winners (e2e-proven only)
- FlashMLA sparse decode **P1+c2** (not r2a)
- \`o_proj\` / \`index_q_upproj\` fixed_nk graph-only (M16|M32)
- MoE gate/up/down via \`SGLANG_GLM52_INFINI_MOE_ALIGN=1\`

## Optional candidates
- \`swiglu\`: same winners stack plus the valid-CTA masked activation
- \`r2a\`: same winners stack with only FlashMLA P1 replaced by r2a;
  requires \`FLASHMLA_R2A_SO\`
- \`p1_before r2a p1_after\`: brackets r2a with two independently started
  P1 servers so host/service drift is visible
- \`router_before router_ids router_after\`: brackets direct masked-int64
  router output with two independently started stock router/DeepEP servers

## Excluded
- \`fused_qkv_a_proj\` (leaf win, e2e historically flat/noisy)
- \`dsa_prefill_attn\`, \`q_b\` / \`index_k\` / \`index_score\`

## Protocol
- Per label: one serve (cuda_graph_max_bs=${SGLANG_CUDA_GRAPH_MAX_BS})
- DeepEP mode: ${SGLANG_DEEPEP_MODE:-auto}; chunked prefill=${CHUNKED_PREFILL_SIZE}; max prefill tokens=${MAX_PREFILL_TOKENS}; mem fraction=${MEM_FRACTION_STATIC}
- Fixed-KV series mode: ${FIXED_KV_SERIES}; decode-shaped warmup runs excluded from statistics: ${FIXED_KV_WARMUP_RUNS}
- Validate-only mode: ${VALIDATE_ONLY}; when enabled, stop after graph capture and all-rank module selection proof
- When fixed-KV series mode is 0, every run independently flushes stale radix state and rebuilds the same deterministic random-id prefixes
- When fixed-KV series mode is 1, every label/BS cell flushes once, builds the deterministic ${S}-64-token prefixes once, and then sends paired requests whose unique 64-token tails prevent measured-suffix cache reuse
- Required measured cache-hit rate: at least 0.99 (target: 0.998); multi-batch mode is disabled so TTFT/ITL remain valid
- Runs per label and global BS: **${N_RUNS}**
- Metric: TPOT/ITL = (latency − last_ttft) / output_len × 1000 (ms)
- Report: mean / median / std / p10 / p90 / min / max, winners vs candidate
MD

write_env() {
  local mode=$1
  {
    echo "SGLANG_GLM52_ALLOW_ABI_ADAPTER=0"
    echo "SGLANG_GLM52_INFINI_KERNEL_NVTX=0"
    echo "SGLANG_GLM52_OPT_HIT_FILE=$HIT_FILE"
    echo "SGLANG_GLM52_MANIFEST=$REPO/glm52_opt/manifest.json"
    echo "SGLANG_GLM52_DEEPGEMM_VARIANT=41c6235"
    echo "SGLANG_GLM52_DEEPGEMM_OVERLAY=$REPO/third_party/deepgemm_glm52"
    echo "SGLANG_OPT_GLM52_FUSED_QKV_A_PREFILL_DIRECT_NK=0"
    echo "SGLANG_OPT_GLM52_FUSED_QKV_A_DECODE_DIRECT_NK=0"
    case "$mode" in
      opt0)
        echo "SGLANG_GLM52_OPT=0"
        echo "SGLANG_GLM52_OPT_PROFILE=serving_safe"
        ;;
      winners|swiglu|p1_before|p1_after|r2a|router_before|router_ids|router_after)
        echo "SGLANG_GLM52_OPT=1"
        echo "SGLANG_GLM52_OPT_PROFILE=combined_winners"
        # e2e-proven only — no fused_qkv_a, no dsa_prefill
        local ops="flashmla_sparse_decode,o_proj,index_q_upproj,moe_gate_proj,moe_up_proj,moe_down_proj"
        if [[ "$mode" == "swiglu" ]]; then
          ops="$ops,moe_swiglu_quant"
          echo "SGLANG_OPT_MOE_SWIGLU_QUANT_VARIANT=cuda_valid_cta"
        fi
        echo "SGLANG_GLM52_OPT_OPS=$ops"
        echo "SGLANG_GLM52_OPT_M_BUCKETS=dsa_decode_attn:16|32,o_proj:16|32,index_q_upproj:16|32"
        echo "SGLANG_GLM52_HOTSPOT_MODULE=$PROVIDER"
        echo "SGLANG_GLM52_FLASHMLA_GRAPH_ONLY=1"
        echo "SGLANG_GLM52_O_PROJ_GRAPH_ONLY=1"
        echo "SGLANG_GLM52_INDEX_Q_UPPROJ_GRAPH_ONLY=1"
        echo "SGLANG_GLM52_INFINI_MOE_ALIGN=1"
        echo "SGLANG_GLM52_ROUTER_PAD_MASK_FUSION=0"
        if [[ "$mode" == "router_ids" ]]; then
          echo "SGLANG_GLM52_ROUTER_DEEPEP_IDS_FUSION=1"
        else
          echo "SGLANG_GLM52_ROUTER_DEEPEP_IDS_FUSION=0"
        fi
        echo "GLM52_FLASHMLA_USE_PREBUILT=1"
        if [[ "$mode" == "r2a" ]]; then
          [[ -n "$FLASHMLA_R2A_SO" && -f "$FLASHMLA_R2A_SO" ]] || {
            echo "[ERR] r2a requires an existing FLASHMLA_R2A_SO" >&2
            return 1
          }
          echo "GLM52_FLASHMLA_DECODE_STACK=r2a"
          echo "GLM52_FLASHMLA_PREBUILT_SO=$FLASHMLA_R2A_SO"
        else
          echo "GLM52_FLASHMLA_DECODE_STACK=p1_c2"
        fi
        ;;
      *) echo "[ERR] unknown mode=$mode"; return 1 ;;
    esac
  } > "$ENV_FILE"
  echo "[INFO] env ($mode):"; cat "$ENV_FILE"
}

cleanup_ours() {
  local server_pid="" recorded_pid=0
  if [[ -f "$ACTIVE_SERVER_PID_FILE" ]]; then
    server_pid=$(cat "$ACTIVE_SERVER_PID_FILE")
    recorded_pid=1
  fi
  if [[ -z "$server_pid" || ! -r "/proc/$server_pid/cmdline" ]]; then
    server_pid=$(ss -ltnp 2>/dev/null |
      sed -n "s/.*:${PORT} .*pid=\\([0-9][0-9]*\\).*/\\1/p" | head -1)
    recorded_pid=0
  fi
  if [[ -n "$server_pid" && -r "/proc/$server_pid/cmdline" ]]; then
    local cmdline pgid
    cmdline=$(tr '\0' ' ' < "/proc/$server_pid/cmdline")
    if [[ "$recorded_pid" == 1 \
          || ( "$cmdline" == *sglang* && "$cmdline" == *"--port $PORT"* ) ]]; then
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
    else
      echo "[ERR] refusing to clean unowned listener on port $PORT: $cmdline" >&2
      return 1
    fi
  fi
  rm -f "$ACTIVE_SERVER_PID_FILE"
  sleep 4
}

restore_env() {
  if [[ -f "$ENV_BACKUP" ]]; then
    cp -f "$ENV_BACKUP" "$ENV_FILE"
  elif [[ -f "$ENV_ABSENT_MARKER" ]]; then
    rm -f "$ENV_FILE"
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

wait_gpus_free() {
  local i=0
  while true; do
    local n max_used
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
    max_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'BEGIN{m=0} {if($1+0>m)m=$1+0} END{print m}')
    if [ "${n:-0}" -eq 0 ] && [ "${max_used:-0}" -lt 2048 ]; then
      echo "[INFO] GPUs free apps=$n max_used_mib=$max_used"
      sleep 10
      return 0
    fi
    i=$((i+1))
    echo "[WAIT] busy apps=$n max_used_mib=$max_used loop=$i $(date +%H:%M:%S)"
    sleep 20
  done
}

http_code() {
  curl -s -o /dev/null -w "%{http_code}" --connect-timeout 2 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null || echo 000
}

launch_serve() {
  local label=$1
  # A previous campaign version made SERVE_LOG a symlink to the per-label
  # archive. Truncating that path for the next label then destroyed the prior
  # archive. Remove only that legacy symlink; the launcher may keep writing its
  # ordinary current-service log while stdout is archived independently below.
  if [[ -L "$SERVE_LOG" ]]; then
    rm -f "$SERVE_LOG"
  fi
  : > "$SERVE_LOG" || true
  rm -f "$HIT_FILE"
  (
    cd "$ROOT"
    export SGLANG_GLM52_ENV_FILE="$ENV_FILE"
    export PORT CHUNKED_PREFILL_SIZE MAX_PREFILL_TOKENS
    export SGLANG_CUDA_GRAPH_MAX_BS SGLANG_MAX_RUNNING_REQUESTS
    export SGLANG_DG_CACHE_DIR=${SGLANG_DG_CACHE_DIR:-/tmp/glm52_deep_gemm_cache}
    export TMPDIR=${TMPDIR:-/tmp/glm52_tmpdir}
    export MEM_FRACTION_STATIC
    # The per-label env file is authoritative.  Do not let an inherited shell
    # override leak the r2a SO into either P1 bracket (or vice versa).
    unset GLM52_FLASHMLA_USE_PREBUILT GLM52_FLASHMLA_DECODE_STACK
    unset GLM52_FLASHMLA_PREBUILT_SO
    mkdir -p "$SGLANG_DG_CACHE_DIR" "$TMPDIR"
    export SGLANG_EXTRA_SERVE_ARGS="--mem-fraction-static ${MEM_FRACTION_STATIC}"
    ROOT="$ROOT" REPO="$REPO" MODEL="$MODEL" exec setsid bash "$SERVER_LAUNCHER"
  ) > "$OUT/serve_${label}.log" 2>&1 &
  echo $! > "$ACTIVE_SERVER_PID_FILE"
  cp -f "$ACTIVE_SERVER_PID_FILE" "$OUT/serve_pid_${label}.txt"
  echo "[INFO] launched serve pid=$(cat "$OUT/serve_pid_${label}.txt") label=$label"
}

wait_ready() {
  local label=$1 i=0
  while [ $i -lt 180 ]; do
    i=$((i+1))
    local code; code=$(http_code); code=${code: -3}
    echo "$(date +%H:%M:%S) wait=$i code=$code label=$label"
    if [ "$code" = "200" ]; then
      rg -n "hotspot provider ready|glm52_opt|combined_winners|INFINI_MOE|enabled=True|FlashMLA" \
        "$OUT/serve_${label}.log" 2>/dev/null | tail -20 || true
      return 0
    fi
    if ! kill -0 "$(cat "$OUT/serve_pid_${label}.txt")" 2>/dev/null; then
      echo "[ERR] exited early"; tail -120 "$OUT/serve_${label}.log"; return 1
    fi
    if grep -qE "CUDA out of memory|Capture cuda graph failed|Address already in use" \
      "$OUT/serve_${label}.log" 2>/dev/null; then
      echo "[ERR] launch failure"; tail -80 "$OUT/serve_${label}.log"; return 1
    fi
    sleep 10
  done
  echo "[ERR] timeout ready"; tail -80 "$OUT/serve_${label}.log"; return 1
}

run_decode_once() {
  local tag=$1
  local gbs=$2
  local i=$3
  local outp=$4
  local lm=$((gbs / DP))
  local RATE
  RATE=$("$PY" -c "print(max(0.0, ($S - 64) / float($S)))")
  "$PY" "$ROOT/run_one_batch_server_longtimeout.py" \
    --model None \
    --base-url "http://127.0.0.1:$PORT" \
    --local-tokenizer-path "$MODEL" \
    --batch-size "$gbs" \
    --input-len "$S" \
    --output-len "$OUT_LEN" \
    --cache-hit-rate "$RATE" \
    --dataset-name random-ids \
    --result-filename "$outp" \
    --run-name "${tag}_decode_bs${gbs}_i${i}" \
    --show-report \
    --no-append-to-github-summary \
    --skip-warmup

  "$PY" - "$outp" "$tag" "$gbs" "$i" <<'PY'
import json
import sys
from pathlib import Path

path, tag, batch_size, run_index = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
if len(rows) < run_index:
    raise SystemExit(f"missing result row {run_index} in {path}")
row = rows[run_index - 1]
expected_name = f"{tag}_decode_bs{batch_size}_i{run_index}"
if row.get("run_name") != expected_name:
    raise SystemExit(f"unexpected run_name: {row.get('run_name')!r} != {expected_name!r}")
if row.get("batch_size") != batch_size:
    raise SystemExit(f"unexpected batch size: {row.get('batch_size')} != {batch_size}")
hit = row.get("cache_hit_rate")
if hit is None or float(hit) < 0.99:
    raise SystemExit(f"invalid fixed-KV run: cache_hit_rate={hit!r}, expected >= 0.99")
print(f"[VALID] {expected_name} cache_hit_rate={float(hit):.4f}")
PY
}

run_label() {
  local label=$1
  echo "======== LABEL=$label $(date -Is) ========"
  write_env "$label"
  cleanup_ours
  wait_gpus_free
  launch_serve "$label"
  wait_ready "$label"
  if [[ "$label" == "r2a" ]]; then
    local r2a_real loaded_ranks gpu_pid
    r2a_real=$(readlink -f "$FLASHMLA_R2A_SO")
    loaded_ranks=0
    while read -r gpu_pid; do
      [[ -n "$gpu_pid" && -r "/proc/$gpu_pid/maps" ]] || continue
      if grep -F -q "$r2a_real" "/proc/$gpu_pid/maps"; then
        loaded_ranks=$((loaded_ranks + 1))
      fi
    done < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sort -u)
    if [[ "$loaded_ranks" -ne "$DP" ]]; then
      echo "[ERR] expected r2a SO on all $DP ranks; observed $loaded_ranks" >&2
      tail -160 "$OUT/serve_${label}.log"
      return 1
    fi
    sha256sum "$FLASHMLA_R2A_SO" > "$OUT/flashmla_r2a_so.sha256"
    echo "[VALID] r2a SO loaded_ranks=$loaded_ranks path=$r2a_real"
  fi
  local router_ids_selection_count
  router_ids_selection_count=$(grep -F -c \
    "GLM-5.2 router DeepEP-ID fusion selected" \
    "$OUT/serve_${label}.log" || true)
  if [[ "$label" == "router_ids" ]]; then
    if [[ "$router_ids_selection_count" -ne "$DP" ]]; then
      echo "[ERR] expected router DeepEP-ID fusion on all $DP ranks; observed $router_ids_selection_count" >&2
      tail -160 "$OUT/serve_${label}.log"
      return 1
    fi
  elif [[ "$router_ids_selection_count" -ne 0 ]]; then
    echo "[ERR] baseline label=$label selected router DeepEP-ID fusion on $router_ids_selection_count ranks" >&2
    return 1
  fi
  echo "[VALID] router DeepEP-ID label=$label selection_count=$router_ids_selection_count"
  if [[ "$label" == "swiglu" ]]; then
    local graph_buckets=(1 2 4 8 12 16)
    if [[ "$SGLANG_CUDA_GRAPH_MAX_BS" -ge 32 ]]; then
      graph_buckets+=(32)
    fi
    local bucket selection_count
    for bucket in "${graph_buckets[@]}"; do
      selection_count=$(grep -F -c \
        "shape=(32, 8192, 4096) stride=(33554432, 4096, 1) routed_m=$bucket topk=8" \
        "$OUT/serve_${label}.log" || true)
      if [[ "$selection_count" -ne "$DP" ]]; then
        echo "[ERR] expected B300/T=8192 M=$bucket selection on all $DP ranks; observed $selection_count"
        tail -160 "$OUT/serve_${label}.log"
        return 1
      fi
      echo "[VALID] B300/T=8192 SwiGLU candidate selected for M=$bucket on all $selection_count ranks"
    done
  fi

  if [[ "$VALIDATE_ONLY" == 1 ]]; then
    if [ -f "$HIT_FILE" ]; then
      cp -f "$HIT_FILE" "$OUT/hits_${label}_capture.json"
    fi
    echo "[VALID] validate-only graph capture and selection completed for label=$label"
    cleanup_ours
    return 0
  fi

  for gbs in $GLOBAL_BS_LIST; do
    local lm=$((gbs / DP))
    echo "---- $label global_bs=$gbs local_M=$lm N=$N_RUNS ----"
    local outp="$OUT/decode_${label}_bs${gbs}.jsonl"
    : > "$outp"
    if [[ "$FIXED_KV_SERIES" == 1 ]]; then
      "$PY" "$FIXED_KV_RUNNER" \
        --base-url "http://127.0.0.1:$PORT" \
        --model-path "$MODEL" \
        --label "$label" \
        --result-filename "$outp" \
        --batch-size "$gbs" \
        --input-len "$S" \
        --prefix-len $((S - 64)) \
        --output-len "$OUT_LEN" \
        --runs "$N_RUNS" \
        --warmup-runs "$FIXED_KV_WARMUP_RUNS" \
        --seed 42
      "$PY" - "$outp" "$label" "$gbs" "$N_RUNS" <<'PY'
import json
import sys
from pathlib import Path

path, label, batch_size, expected_n = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
if len(rows) != expected_n:
    raise SystemExit(f"expected {expected_n} fixed-KV rows in {path}, found {len(rows)}")
target = (int(rows[0]["input_len"]) - 64) / float(rows[0]["input_len"])
for index, row in enumerate(rows, 1):
    expected_name = f"{label}_decode_bs{batch_size}_i{index}"
    if row.get("run_name") != expected_name:
        raise SystemExit(f"unexpected run_name: {row.get('run_name')!r} != {expected_name!r}")
    if row.get("protocol") != "fixed-kv-decode-series-v1":
        raise SystemExit(f"unexpected protocol in row {index}: {row.get('protocol')!r}")
    if row.get("batch_size") != batch_size:
        raise SystemExit(f"unexpected batch size in row {index}: {row.get('batch_size')}")
    hit = row.get("cache_hit_rate")
    if hit is None or abs(float(hit) - target) > 0.001:
        raise SystemExit(
            f"invalid fixed-KV row {index}: cache_hit_rate={hit!r}, target={target:.6f}"
        )
    if not row.get("prompt_set_id"):
        raise SystemExit(f"missing prompt_set_id in row {index}")
print(f"[VALID] fixed-KV series rows={len(rows)} target_cache_hit={target:.6f}")
PY
    else
      local i
      for i in $(seq 1 "$N_RUNS"); do
        echo "[RUN] $label bs=$gbs i=$i/$N_RUNS $(date +%H:%M:%S)"
        run_decode_once "$label" "$gbs" "$i" "$outp"
      done
    fi
    if [ -f "$HIT_FILE" ]; then
      cp -f "$HIT_FILE" "$OUT/hits_${label}_bs${gbs}.json"
    fi
  done

  cleanup_ours
}

summarize() {
  export OUT N_RUNS S OUT_LEN LABELS GLOBAL_BS_LIST FIXED_KV_SERIES
  "$PY" - <<'PY'
import json, math, statistics
from pathlib import Path
import os

out = Path(os.environ["OUT"])
n_runs = int(os.environ["N_RUNS"])
S = int(os.environ["S"])
out_len = int(os.environ["OUT_LEN"])
labels = os.environ["LABELS"].split()
batch_sizes = [int(x) for x in os.environ["GLOBAL_BS_LIST"].split()]
fixed_kv_series = bool(int(os.environ["FIXED_KV_SERIES"]))

def pct(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)

def load_itls(path):
    itls, lats, ttfts, prompt_ids = [], [], [], []
    if not path.exists():
        return itls, lats, ttfts, prompt_ids
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        lat, ttft = r.get("latency"), r.get("last_ttft")
        ol = r.get("output_len") or out_len
        if lat is None or ttft is None:
            continue
        itl = (float(lat) - float(ttft)) / float(ol) * 1000.0
        itls.append(itl)
        lats.append(float(lat))
        ttfts.append(float(ttft))
        prompt_ids.append(r.get("prompt_set_id"))
    return itls, lats, ttfts, prompt_ids

def stats(xs):
    if not xs:
        return None
    return {
        "n": len(xs),
        "mean": statistics.mean(xs),
        "median": statistics.median(xs),
        "stdev": statistics.stdev(xs) if len(xs) > 1 else 0.0,
        "p10": pct(xs, 10),
        "p90": pct(xs, 90),
        "min": min(xs),
        "max": max(xs),
    }

rows = []
summary = {"S": S, "out_len": out_len, "n_runs_target": n_runs, "cells": {}}
series = {}
for label in labels:
    for gbs in batch_sizes:
        p = out / f"decode_{label}_bs{gbs}.jsonl"
        itls, lats, ttfts, prompt_ids = load_itls(p)
        series[f"{label}_bs{gbs}"] = {"itls": itls, "prompt_ids": prompt_ids}
        st = stats(itls)
        summary["cells"][f"{label}_bs{gbs}"] = {
            "itl_ms": st,
            "latency_s": stats(lats),
            "ttft_s": stats(ttfts),
            "path": str(p),
            "prompt_set_ids": prompt_ids if fixed_kv_series else None,
        }
        if st:
            rows.append(
                (
                    label,
                    gbs,
                    st["n"],
                    st["mean"],
                    st["median"],
                    st["stdev"],
                    st["p10"],
                    st["p90"],
                    st["min"],
                    st["max"],
                )
            )

lines = [
    f"# Decode TPOT summary (S={S}, out_len={out_len}, N≈{n_runs})",
    "",
    f"Measured labels, in server-start order: {' -> '.join(labels)}.",
    "",
    "| label | global_BS | n | mean ITL (ms) | median ITL (ms) | stdev | p10 | p90 | min | max |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    lines.append(
        f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.3f} | {r[4]:.3f} | {r[5]:.3f} | {r[6]:.3f} | {r[7]:.3f} | {r[8]:.3f} | {r[9]:.3f} |"
    )

lines += ["", f"## Comparisons against first label ({labels[0] if labels else 'none'})", ""]
if labels:
    for gbs in batch_sizes:
        baseline = summary["cells"].get(f"{labels[0]}_bs{gbs}", {}).get("itl_ms")
        for label in labels[1:]:
            candidate = summary["cells"].get(f"{label}_bs{gbs}", {}).get("itl_ms")
            if not baseline or not candidate:
                lines.append(f"- BS={gbs}, {label}: incomplete")
                continue
            speed = baseline["median"] / candidate["median"] if candidate["median"] else float("nan")
            delta = baseline["median"] - candidate["median"]
            reduction_pct = (
                delta / baseline["median"] * 100
                if baseline["median"]
                else float("nan")
            )
            lines.append(
                f"- **BS={gbs}, {labels[0]} vs {label}**: "
                f"{baseline['median']:.3f} → {candidate['median']:.3f} ms "
                f"(**{speed:.4f}×**, {delta:+.3f} ms, {reduction_pct:+.2f}%); "
                f"mean {baseline['mean']:.3f} → {candidate['mean']:.3f} ms"
            )

if fixed_kv_series and labels:
    lines += ["", "## Fixed-KV request pairing", ""]
    summary["fixed_kv_request_pairing"] = {}
    for gbs in batch_sizes:
        reference_key = f"{labels[0]}_bs{gbs}"
        reference_ids = series.get(reference_key, {}).get("prompt_ids", [])
        if len(reference_ids) != n_runs or any(not value for value in reference_ids):
            raise SystemExit(f"invalid fixed-KV prompt IDs in {reference_key}")
        matched = []
        for label in labels[1:]:
            key = f"{label}_bs{gbs}"
            candidate_ids = series.get(key, {}).get("prompt_ids", [])
            if candidate_ids != reference_ids:
                raise SystemExit(f"fixed-KV request sequence mismatch: {reference_key} vs {key}")
            matched.append(label)
        summary["fixed_kv_request_pairing"][str(gbs)] = {
            "reference": labels[0],
            "matched_labels": matched,
            "n_exact_prompt_sets": len(reference_ids),
        }
        lines.append(
            f"- BS={gbs}: all {len(reference_ids)} measured prompt sets match exactly "
            f"across {' -> '.join(labels)} by protocol hash."
        )

if labels == ["p1_before", "r2a", "p1_after"]:
    lines += ["", "## P1/r2a/P1 drift bracket", ""]
    summary["aba_bracket"] = {}
    for gbs in batch_sizes:
        before = summary["cells"].get(f"p1_before_bs{gbs}", {}).get("itl_ms")
        candidate = summary["cells"].get(f"r2a_bs{gbs}", {}).get("itl_ms")
        after = summary["cells"].get(f"p1_after_bs{gbs}", {}).get("itl_ms")
        if not before or not candidate or not after:
            lines.append(f"- BS={gbs}: incomplete")
            continue
        bracket_mid = (before["median"] + after["median"]) / 2.0
        reduction = bracket_mid - candidate["median"]
        reduction_pct = reduction / bracket_mid * 100.0
        speedup = bracket_mid / candidate["median"]
        drift = after["median"] - before["median"]
        lower_than_both = candidate["median"] < min(before["median"], after["median"])
        before_series = series[f"p1_before_bs{gbs}"]["itls"]
        candidate_series = series[f"r2a_bs{gbs}"]["itls"]
        after_series = series[f"p1_after_bs{gbs}"]["itls"]
        paired_reductions = [
            (left + right) / 2.0 - middle
            for left, middle, right in zip(before_series, candidate_series, after_series)
        ]
        paired = stats(paired_reductions)
        summary["aba_bracket"][str(gbs)] = {
            "p1_before_median_itl_ms": before["median"],
            "r2a_median_itl_ms": candidate["median"],
            "p1_after_median_itl_ms": after["median"],
            "p1_bracket_midpoint_median_itl_ms": bracket_mid,
            "p1_drift_ms": drift,
            "r2a_reduction_ms": reduction,
            "r2a_reduction_pct": reduction_pct,
            "r2a_speedup": speedup,
            "r2a_lower_than_both_p1_brackets": lower_than_both,
            "paired_p1_midpoint_minus_r2a_itl_ms": paired,
            "paired_positive_fraction": (
                sum(value > 0 for value in paired_reductions) / len(paired_reductions)
                if paired_reductions
                else None
            ),
        }
        lines.append(
            f"- **BS={gbs}**: P1 {before['median']:.3f} → r2a {candidate['median']:.3f} "
            f"→ P1 {after['median']:.3f} ms; bracket midpoint {bracket_mid:.3f} ms, "
            f"r2a **{speedup:.4f}×** ({reduction:+.3f} ms, {reduction_pct:+.2f}%), "
            f"P1 drift {drift:+.3f} ms, lower-than-both={str(lower_than_both).lower()}"
        )
        if paired:
            lines.append(
                f"  Paired per-request-sequence P1-midpoint minus r2a: "
                f"mean {paired['mean']:+.4f} ms, median {paired['median']:+.4f} ms, "
                f"p10/p90 {paired['p10']:+.4f}/{paired['p90']:+.4f} ms; "
                f"positive {sum(value > 0 for value in paired_reductions)}/{len(paired_reductions)}."
            )

(out / "TPOT_SUMMARY.md").write_text("\n".join(lines) + "\n")
(out / "TPOT_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
print("\n".join(lines))
PY
}

# ---- main ----
for label in $LABELS; do
  run_label "$label"
done

if [[ "$VALIDATE_ONLY" == 1 ]]; then
  echo "======== VALIDATION DONE $(date -Is) OUT=$OUT ========"
  ls -lah "$OUT" | head -40
  exit 0
fi

summarize
if [[ "$FIXED_KV_SERIES" == 1 \
      && ( "$LABELS" == "p1_before r2a p1_after" \
        || "$LABELS" == "router_before router_ids router_after" ) ]]; then
  if [[ "$LABELS" == "p1_before r2a p1_after" ]]; then
    aba_before=p1_before
    aba_candidate=r2a
    aba_after=p1_after
  else
    aba_before=router_before
    aba_candidate=router_ids
    aba_after=router_after
  fi
  for gbs in $GLOBAL_BS_LIST; do
    "$PY" "$FIXED_KV_ABA_ANALYZER" \
      --before "$OUT/decode_${aba_before}_bs${gbs}.jsonl" \
      --candidate "$OUT/decode_${aba_candidate}_bs${gbs}.jsonl" \
      --after "$OUT/decode_${aba_after}_bs${gbs}.jsonl" \
      --output "$OUT/ABA_BOOTSTRAP_bs${gbs}.json" \
      --expected-runs "$N_RUNS" \
      --before-label "$aba_before" \
      --candidate-label "$aba_candidate" \
      --after-label "$aba_after"
  done
fi
echo "======== ALL DONE $(date -Is) OUT=$OUT ========"
ls -lah "$OUT" | head -40
cat "$OUT/TPOT_SUMMARY.md"
