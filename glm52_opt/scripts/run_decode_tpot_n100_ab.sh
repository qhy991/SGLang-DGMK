#!/usr/bin/env bash
# Fair decode TPOT A/B: OPT0 vs e2e-proven decode winners only.
#   Include: FlashMLA P1+c2 + o_proj + index_q_upproj fixed_nk + MoE M-tile align
#   Exclude: fused_qkv_a (e2e historically flat/noisy), dsa_prefill, r2a, q_b/index_k/score
#
# Workload: S=32k KV, global BS ∈ {128,256} (local_M=16/32, DP=8).
# Partner repro default: N=3 decode runs per (label × BS); set N_RUNS=100 for
# tighter stats. Report mean/median/p10/p90 vs OPT0.
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
LABELS=${LABELS:-"opt0 winners"}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.83}
# Need 32 for global BS=256 (local_M=32).
SGLANG_CUDA_GRAPH_MAX_BS=${SGLANG_CUDA_GRAPH_MAX_BS:-32}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-16384}
MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-16384}
MODEL=${MODEL:-/mnt/b300-shared/models/GLM-5.2-FP8}
RUN_ID=${RUN_ID:-decode_tpot_n${N_RUNS}_s${S}_$(date -u +%Y%m%dT%H%M%SZ)}
OUT=$ROOT/bench_results/$RUN_ID

PROVIDER=$REPO/python/sglang/srt/layers/glm52_opt/hotspot_candidates/flashmla_accel_bundle_provider.py

export PATH=$VENV/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=$REPO/python${PYTHONPATH:+:$PYTHONPATH}
unset CUDA_VISIBLE_DEVICES || true
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset SGLANG_EXTRA_SERVE_ARGS || true
export SGLANG_CUDA_GRAPH_MAX_BS
export SGLANG_MAX_RUNNING_REQUESTS=${SGLANG_MAX_RUNNING_REQUESTS:-$((8 * SGLANG_CUDA_GRAPH_MAX_BS))}

mkdir -p "$OUT" "$ROOT/cache/sglang" "$ROOT/logs" /tmp
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
# Decode TPOT A/B (N=${N_RUNS}, S=${S})

## Winners (e2e-proven only)
- FlashMLA sparse decode **P1+c2** (not r2a)
- \`o_proj\` / \`index_q_upproj\` fixed_nk graph-only (M16|M32)
- MoE gate/up/down via \`SGLANG_GLM52_INFINI_MOE_ALIGN=1\`

## Excluded
- \`fused_qkv_a_proj\` (leaf win, e2e historically flat/noisy)
- \`dsa_prefill_attn\`, FlashMLA r2a, \`q_b\` / \`index_k\` / \`index_score\`

## Protocol
- Per label: one serve (cuda_graph_max_bs=${SGLANG_CUDA_GRAPH_MAX_BS})
- Per global BS ∈ {${GLOBAL_BS_LIST}}: warm S=${S} cache once, then **${N_RUNS}** decode runs
- Metric: TPOT/ITL = (latency − last_ttft) / output_len × 1000 (ms)
- Report: mean / median / std / p10 / p90 / min / max vs OPT0
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
      winners)
        echo "SGLANG_GLM52_OPT=1"
        echo "SGLANG_GLM52_OPT_PROFILE=combined_winners"
        # e2e-proven only — no fused_qkv_a, no dsa_prefill
        echo "SGLANG_GLM52_OPT_OPS=flashmla_sparse_decode,o_proj,index_q_upproj,moe_gate_proj,moe_up_proj,moe_down_proj"
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
  pkill -9 -f "run_glm52_dgmk|sglang serve.*GLM-5.2|glm52_tpot_${RUN_ID}" 2>/dev/null || true
  sleep 4
  fuser -k "${PORT}/tcp" 2>/dev/null || true
  sleep 4
}

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
    bash "$ROOT/run_glm52_dgmk.sh"
  ) > "$OUT/serve_${label}.log" 2>&1 &
  echo $! > "$OUT/serve_pid_${label}.txt"
  ln -sfn "$OUT/serve_${label}.log" "$SERVE_LOG" 2>/dev/null || true
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

warm_cache() {
  local tag=$1
  local gbs=$2
  local RATE
  RATE=$("$PY" -c "print(max(0.0, ($S - 64) / float($S)))")
  echo "[WARM] tag=$tag global_bs=$gbs S=$S"
  curl -s -X POST "http://127.0.0.1:$PORT/flush_cache" >/dev/null || true
  sleep 1
  "$PY" "$ROOT/run_one_batch_server_longtimeout.py" \
    --model None \
    --base-url "http://127.0.0.1:$PORT" \
    --local-tokenizer-path "$MODEL" \
    --batch-size "$gbs" \
    --input-len "$S" \
    --output-len 1 \
    --cache-hit-rate "$RATE" \
    --dataset-name random-ids \
    --result-filename "$OUT/warm_${tag}_bs${gbs}.jsonl" \
    --run-name "${tag}_warm_bs${gbs}" \
    --show-report \
    --enable-multi-batch \
    --no-append-to-github-summary \
    --skip-warmup || true
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
    --enable-multi-batch \
    --no-append-to-github-summary \
    --skip-warmup || true
}

run_label() {
  local label=$1
  echo "======== LABEL=$label $(date -Is) ========"
  write_env "$label"
  cleanup_ours
  wait_gpus_free
  launch_serve "$label"
  wait_ready "$label"

  for gbs in $GLOBAL_BS_LIST; do
    local lm=$((gbs / DP))
    echo "---- $label global_bs=$gbs local_M=$lm N=$N_RUNS ----"
    warm_cache "$label" "$gbs"
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
  export OUT N_RUNS S OUT_LEN
  "$PY" - <<'PY'
import json, math, statistics
from pathlib import Path
import os

out = Path(os.environ["OUT"])
n_runs = int(os.environ["N_RUNS"])
S = int(os.environ["S"])
out_len = int(os.environ["OUT_LEN"])

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
for label in ("opt0", "winners"):
    for gbs in (128, 256):
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
    "Winners = FlashMLA P1+c2 + o_proj + index_q_upproj + MoE M-tile align (no fused_qkv).",
    "",
    "| label | global_BS | n | mean ITL (ms) | median ITL (ms) | stdev | p10 | p90 | min | max |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    lines.append(
        f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.3f} | {r[4]:.3f} | {r[5]:.3f} | {r[6]:.3f} | {r[7]:.3f} | {r[8]:.3f} | {r[9]:.3f} |"
    )

lines += ["", "## OPT0 vs winners (median ITL)", ""]
for gbs in (128, 256):
    o = summary["cells"].get(f"opt0_bs{gbs}", {}).get("itl_ms")
    w = summary["cells"].get(f"winners_bs{gbs}", {}).get("itl_ms")
    if not o or not w:
        lines.append(f"- BS={gbs}: incomplete")
        continue
    speed = o["median"] / w["median"] if w["median"] else float("nan")
    delta = o["median"] - w["median"]
    pct = delta / o["median"] * 100 if o["median"] else float("nan")
    lines.append(
        f"- **BS={gbs}**: OPT0 median {o['median']:.3f} → winners {w['median']:.3f} ms "
        f"(**{speed:.4f}×**, {delta:+.3f} ms, {pct:+.2f}%); "
        f"mean {o['mean']:.3f} → {w['mean']:.3f} ms"
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
