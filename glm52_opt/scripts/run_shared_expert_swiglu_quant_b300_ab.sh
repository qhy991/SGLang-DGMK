#!/usr/bin/env bash
# Fair decode TPOT A/B: existing winners vs BF16-exact shared-expert
# SwiGLU+packed-UE8M0 quant fusion.
#   Include: FlashMLA P1+c2 + o_proj + index_q_upproj fixed_nk + MoE M-tile align
#   Exclude: fused_qkv_a (e2e historically flat/noisy), dsa_prefill, r2a, q_b/index_k/score
#
# Default workload: S=32k KV, global BS=128 (local_M=16, DP=8).
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
OUT_LEN=${OUT_LEN:-240}
N_RUNS=${N_RUNS:-3}
GLOBAL_BS_LIST=${GLOBAL_BS_LIST:-"128"}
LABELS=${LABELS:-"shared_stock shared_fused"}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.78}
# Need 32 for global BS=256 (local_M=32).
SGLANG_CUDA_GRAPH_MAX_BS=${SGLANG_CUDA_GRAPH_MAX_BS:-16}
# DP attention divides this by DP.  2048 -> 256 tokens/rank preserves enough
# KV capacity for BS128 x S32k; 16384 -> 2048/rank reserves a much larger
# prefill workspace and reduced the observed capacity to only 67,904/rank.
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-8192}
MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-8192}
MODEL=${MODEL:-/mnt/b300-shared/models/GLM-5.2-FP8}
RUN_ID=${RUN_ID:-shared_expert_swiglu_quant_b300_n${N_RUNS}_s${S}_$(date -u +%Y%m%dT%H%M%SZ)}
OUT=$ROOT/bench_results/$RUN_ID
SERVER_LAUNCHER=${SERVER_LAUNCHER:-$REPO/glm52_opt/scripts/run_b300_repo_server.sh}
ENV_BACKUP=$OUT/original_glm52_opt.env
ENV_ABSENT_MARKER=$OUT/original_glm52_opt.env.absent

PROVIDER=$REPO/python/sglang/srt/layers/glm52_opt/hotspot_candidates/flashmla_accel_bundle_provider.py

export PATH=$VENV/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=$REPO/python${PYTHONPATH:+:$PYTHONPATH}
unset CUDA_VISIBLE_DEVICES || true
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset SGLANG_EXTRA_SERVE_ARGS || true
export SGLANG_CUDA_GRAPH_MAX_BS
export SGLANG_MAX_RUNNING_REQUESTS=${SGLANG_MAX_RUNNING_REQUESTS:-$((8 * SGLANG_CUDA_GRAPH_MAX_BS))}

mkdir -p "$OUT" "$ROOT/cache/sglang" "$ROOT/logs" /tmp
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
cd "$REPO"
git rev-parse --short HEAD | tee "$OUT/git_rev.txt"
git branch --show-current | tee -a "$OUT/git_rev.txt"
git log -1 --oneline | tee -a "$OUT/git_rev.txt"
test -f "$PROVIDER" || { echo "[ERR] missing $PROVIDER"; exit 1; }
cd "$ROOT"

cat > "$OUT/README.md" <<MD
# B300 shared-expert SwiGLU+quant decode A/B (N=${N_RUNS}, S=${S})

## Winners (e2e-proven only)
- FlashMLA sparse decode **P1+c2** (not r2a)
- \`o_proj\` / \`index_q_upproj\` fixed_nk graph-only (M16|M32)
- MoE gate/up/down via \`SGLANG_GLM52_INFINI_MOE_ALIGN=1\`

## Shared-expert candidate
- Same winners stack and workload, including the B300 valid-CTA routed-expert kernel
- Replaces only contiguous shared-expert \`SiluAndMul -> per-token FP8 quant\`
- One fused kernel preserves the intermediate BF16 rounding and packed UE8M0 layout

## Excluded
- \`fused_qkv_a_proj\` (leaf win, e2e historically flat/noisy)
- \`dsa_prefill_attn\`, FlashMLA r2a, \`q_b\` / \`index_k\` / \`index_score\`

## Protocol
- Per label: one serve (cuda_graph_max_bs=${SGLANG_CUDA_GRAPH_MAX_BS})
- DeepEP mode: ${SGLANG_DEEPEP_MODE:-auto}; chunked prefill=${CHUNKED_PREFILL_SIZE}; max prefill tokens=${MAX_PREFILL_TOKENS}; mem fraction=${MEM_FRACTION_STATIC}
- Each run flushes stale radix state, builds the same deterministic random-id prefixes, then times an S=${S} request with only the last 64 prompt tokens uncached
- Required measured cache-hit rate: at least 0.99 (target: 0.998); multi-batch mode is disabled so TTFT/ITL remain valid
- Runs per label and global BS: **${N_RUNS}**
- Metric: TPOT/ITL = (latency − last_ttft) / output_len × 1000 (ms)
- Report: mean / median / std / p10 / p90 / min / max, stock vs fused
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
      shared_stock|shared_fused|router_stock|router_fused)
        echo "SGLANG_GLM52_OPT=1"
        echo "SGLANG_GLM52_OPT_PROFILE=combined_winners"
        # e2e-proven only — no fused_qkv_a, no dsa_prefill
        local ops="flashmla_sparse_decode,o_proj,index_q_upproj,moe_gate_proj,moe_up_proj,moe_down_proj,moe_swiglu_quant"
        echo "SGLANG_OPT_MOE_SWIGLU_QUANT_VARIANT=cuda_valid_cta"
        if [[ "$mode" == "shared_fused" ]]; then
          echo "SGLANG_GLM52_SHARED_EXPERT_SWIGLU_QUANT=1"
        else
          echo "SGLANG_GLM52_SHARED_EXPERT_SWIGLU_QUANT=0"
        fi
        if [[ "$mode" == "router_fused" ]]; then
          echo "SGLANG_GLM52_ROUTER_PAD_MASK_FUSION=1"
        else
          echo "SGLANG_GLM52_ROUTER_PAD_MASK_FUSION=0"
        fi
        echo "SGLANG_GLM52_OPT_OPS=$ops"
        echo "SGLANG_GLM52_OPT_M_BUCKETS=dsa_decode_attn:16|32,o_proj:16|32,index_q_upproj:16|32"
        echo "SGLANG_GLM52_HOTSPOT_MODULE=$PROVIDER"
        echo "SGLANG_GLM52_FLASHMLA_GRAPH_ONLY=1"
        echo "SGLANG_GLM52_O_PROJ_GRAPH_ONLY=1"
        echo "SGLANG_GLM52_INDEX_Q_UPPROJ_GRAPH_ONLY=1"
        echo "SGLANG_GLM52_INFINI_MOE_ALIGN=1"
        echo "GLM52_FLASHMLA_USE_PREBUILT=1"
        echo "GLM52_FLASHMLA_DECODE_STACK=p1_c2"
        ;;
      *) echo "[ERR] unknown mode=$mode"; return 1 ;;
    esac
  } > "$ENV_FILE"
  echo "[INFO] env ($mode):"; cat "$ENV_FILE"
}

cleanup_ours() {
  fuser -k "${PORT}/tcp" 2>/dev/null || true
  sleep 4
  fuser -k "${PORT}/tcp" 2>/dev/null || true
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
    export GLM52_FLASHMLA_USE_PREBUILT=1
    export GLM52_FLASHMLA_DECODE_STACK=${GLM52_FLASHMLA_DECODE_STACK:-p1_c2}
    mkdir -p "$SGLANG_DG_CACHE_DIR" "$TMPDIR"
    export SGLANG_EXTRA_SERVE_ARGS="--mem-fraction-static ${MEM_FRACTION_STATIC}"
    ROOT="$ROOT" REPO="$REPO" MODEL="$MODEL" bash "$SERVER_LAUNCHER"
  ) > "$OUT/serve_${label}.log" 2>&1 &
  echo $! > "$OUT/serve_pid_${label}.txt"
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
  if [[ "$label" == "shared_stock" || "$label" == "shared_fused" || "$label" == "router_stock" || "$label" == "router_fused" ]]; then
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
      echo "[VALID] B300/T=8192 routed SwiGLU winner selected for M=$bucket on all $selection_count ranks"
    done

    if [[ "$label" == "shared_stock" || "$label" == "shared_fused" ]]; then
      local shared_selection_count
      shared_selection_count=$(grep -F -c \
        "GLM-5.2 shared-expert SwiGLU quant selected: round_to_bf16=True" \
        "$OUT/serve_${label}.log" || true)
      if [[ "$label" == "shared_fused" ]]; then
        if [[ "$shared_selection_count" -ne "$DP" ]]; then
          echo "[ERR] expected shared-expert fusion on all $DP ranks; observed $shared_selection_count"
          tail -160 "$OUT/serve_${label}.log"
          return 1
        fi
      elif [[ "$shared_selection_count" -ne 0 ]]; then
        echo "[ERR] stock denominator selected shared-expert fusion on $shared_selection_count ranks"
        return 1
      fi
      echo "[VALID] shared-expert fusion label=$label selection_count=$shared_selection_count"
    fi
    if [[ "$label" == "router_stock" || "$label" == "router_fused" ]]; then
      local router_selection_count
      router_selection_count=$(grep -F -c \
        "GLM-5.2 router padded-ID mask fusion selected" \
        "$OUT/serve_${label}.log" || true)
      if [[ "$label" == "router_fused" ]]; then
        if [[ "$router_selection_count" -ne "$DP" ]]; then
          echo "[ERR] expected router mask fusion on all $DP ranks; observed $router_selection_count"
          tail -160 "$OUT/serve_${label}.log"
          return 1
        fi
      elif [[ "$router_selection_count" -ne 0 ]]; then
        echo "[ERR] stock denominator selected router mask fusion on $router_selection_count ranks"
        return 1
      fi
      echo "[VALID] router mask fusion label=$label selection_count=$router_selection_count"
    fi
  fi

  for gbs in $GLOBAL_BS_LIST; do
    local lm=$((gbs / DP))
    echo "---- $label global_bs=$gbs local_M=$lm N=$N_RUNS ----"
    local outp="$OUT/decode_${label}_bs${gbs}.jsonl"
    : > "$outp"
    local i
    for i in $(seq 1 "$N_RUNS"); do
      echo "[RUN] $label bs=$gbs i=$i/$N_RUNS $(date +%H:%M:%S)"
      run_decode_once "$label" "$gbs" "$i" "$outp"
    done
    if [ -f "$HIT_FILE" ]; then
      cp -f "$HIT_FILE" "$OUT/hits_${label}_bs${gbs}.json"
    fi
  done

  cleanup_ours
}

summarize() {
  export OUT N_RUNS S OUT_LEN GLOBAL_BS_LIST LABELS
  "$PY" - <<'PY'
import json, math, statistics
from pathlib import Path
import os

out = Path(os.environ["OUT"])
n_runs = int(os.environ["N_RUNS"])
S = int(os.environ["S"])
out_len = int(os.environ["OUT_LEN"])
global_bs_list = [int(value) for value in os.environ["GLOBAL_BS_LIST"].split()]
labels = os.environ["LABELS"].split()
if len(labels) != 2:
    raise SystemExit(f"expected exactly two A/B labels, got {labels!r}")
baseline_label, candidate_label = labels
boundary = (
    "shared-expert activation+quant boundary"
    if candidate_label == "shared_fused"
    else "router padded-ID mask boundary"
)

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
    itls, lats, ttfts = [], [], []
    if not path.exists():
        return itls, lats, ttfts
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
    return itls, lats, ttfts

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
for label in labels:
    for gbs in global_bs_list:
        p = out / f"decode_{label}_bs{gbs}.jsonl"
        itls, lats, ttfts = load_itls(p)
        st = stats(itls)
        summary["cells"][f"{label}_bs{gbs}"] = {
            "itl_ms": st,
            "latency_s": stats(lats),
            "ttft_s": stats(ttfts),
            "path": str(p),
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
    f"Both labels use the same winner stack; {candidate_label} changes only the {boundary}.",
    "",
    "| label | global_BS | n | mean ITL (ms) | median ITL (ms) | stdev | p10 | p90 | min | max |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    lines.append(
        f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.3f} | {r[4]:.3f} | {r[5]:.3f} | {r[6]:.3f} | {r[7]:.3f} | {r[8]:.3f} | {r[9]:.3f} |"
    )

lines += ["", f"## {baseline_label} vs {candidate_label} (median ITL)", ""]
for gbs in global_bs_list:
    baseline = summary["cells"].get(f"{baseline_label}_bs{gbs}", {}).get("itl_ms")
    candidate = summary["cells"].get(f"{candidate_label}_bs{gbs}", {}).get("itl_ms")
    if not baseline or not candidate:
        lines.append(f"- BS={gbs}: incomplete")
        continue
    speed = baseline["median"] / candidate["median"] if candidate["median"] else float("nan")
    delta = baseline["median"] - candidate["median"]
    pct = delta / baseline["median"] * 100 if baseline["median"] else float("nan")
    lines.append(
        f"- **BS={gbs}**: stock median {baseline['median']:.3f} → fused {candidate['median']:.3f} ms "
        f"(**{speed:.4f}×**, {delta:+.3f} ms, {pct:+.2f}%); "
        f"mean {baseline['mean']:.3f} → {candidate['mean']:.3f} ms"
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

summarize
echo "======== ALL DONE $(date -Is) OUT=$OUT ========"
ls -lah "$OUT" | head -40
cat "$OUT/TPOT_SUMMARY.md"
