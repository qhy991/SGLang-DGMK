#!/usr/bin/env bash
set -euo pipefail

# Five adjacent fresh-server pairs for the CP8 direct packed MLA-KV candidate.
# Each arm runs one exact-input x11 point.  The runner itself flushes the cache,
# performs a 90K warmup, then measures 110 requests at concurrency 11.

ROOT=/mnt/b300-shared/home/qinhaiyan/wwxq
EVIDENCE=$ROOT/bench_results/sglang_kda_to_glm52_cp8_migration_20260818
BUNDLE=$ROOT/bench_results/glm52_mok_followup_candidates_20260813
CONTROL_REPO=$ROOT/SGLang-DGMK-cp8-packed-kv-control-20260818
CANDIDATE_REPO=$ROOT/SGLang-DGMK-cp8-packed-kv-comm-20260818
LAUNCHER=$CANDIDATE_REPO/glm52_opt/scripts/run_glm52_100k_cp8_ep8_packed_review_host.sh
RUNNER=$CANDIDATE_REPO/glm52_opt/scripts/run_glm52_100k_prefill_cp8_exact_concurrent.py
CONTROL_CONTRACT=$EVIDENCE/runtime_source_contract_cp8_packed_kv_control.json
CANDIDATE_CONTRACT=$EVIDENCE/runtime_source_contract_cp8_direct_packed_kv.json
CONTROL_ENV=$EVIDENCE/host_e2e_prefill_cp8_packed_kv_baseline.env
CANDIDATE_ENV=$CANDIDATE_REPO/glm52_opt/host_e2e_prefill_cp8_direct_packed_kv.env
REFERENCE_LOG=$ROOT/bench_results/glm52_e2e_prefill_attention_backend/20260818T033243Z_flashmla_kv_cp8_dp1_m10048_deepep120_sharedfusion0_record0_mapbalanced_routermap1_mok0_stable_mem0p83/server.log
CORRECTNESS_PROBE=$ROOT/bench_results/glm52_cp8_ep8_20260817/probe_glm52_100k_first_token_cp8.py
COMPARE=$BUNDLE/candidates/compare_server_args.py
CPUSET=65,67,69,71,73,75,77,79,81,83,85,87,89,91,93,95,97,99,101,103,105,107,109,111,113,115,117,119,121,123,125,127,193,195,197,199,201,203,205,207,209,211,213,215,217,219,221,223,225,227,229,231,233,235,237,239,241,243,245,247,249,251,253,255

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT=$EVIDENCE/direct_packed_fresh_abba_$STAMP
mkdir -p "$OUT"
MANIFEST=$OUT/runs.tsv
printf 'label\tarm\tlauncher_log\tresult_json\tserver_log\n' > "$MANIFEST"

check_clear() {
  local gpu_pids port_pid
  gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    | sed '/^$/d')
  port_pid=$(docker exec sglang_0515_optimized bash -lc \
    'lsof -nP -iTCP:30000 -sTCP:LISTEN -t 2>/dev/null || true')
  if [[ -n "$gpu_pids" || -n "$port_pid" ]]; then
    echo "[ERR] machine is not clear: gpu_pids=${gpu_pids:-none} port=${port_pid:-none}" >&2
    return 2
  fi
}

wait_clear() {
  for _ in $(seq 1 60); do
    if check_clear >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  check_clear
}

run_arm() {
  local label=$1 arm=$2 repo contract env_file log static_map
  if [[ "$arm" == control ]]; then
    repo=$CONTROL_REPO
    contract=$CONTROL_CONTRACT
    env_file=$CONTROL_ENV
  elif [[ "$arm" == direct ]]; then
    repo=$CANDIDATE_REPO
    contract=$CANDIDATE_CONTRACT
    env_file=$CANDIDATE_ENV
  else
    echo "[ERR] unknown arm: $arm" >&2
    return 2
  fi
  static_map=$repo/glm52_opt/glm52_100k_x11_static_expert_map.json
  log=$OUT/${label}_${arm}.log
  wait_clear
  echo "[RUN] label=$label arm=$arm log=$log"
  env \
    GLM52_RUNTIME_REPO="$repo" \
    GLM52_EXPERIMENT_BUNDLE_ROOT="$BUNDLE" \
    GLM52_RUNTIME_SOURCE_CONTRACT="$contract" \
    GLM52_CORRECTNESS_PROBE_PATH="$CORRECTNESS_PROBE" \
    GLM52_COMPARE_SERVER_ARGS_PATH="$COMPARE" \
    GLM52_ENV_FILE="$env_file" \
    GLM52_DSA_PREFILL_BACKEND=flashmla_kv \
    GLM52_DEEPEP_SMS=120 \
    GLM52_STATIC_EXPERT_MAP="$static_map" \
    GLM52_ROUTER_STATIC_PLACEMENT_FUSION=1 \
    GLM52_MEM_FRACTION_STATIC=0.83 \
    GLM52_MAX_RUNNING_REQUESTS=128 \
    GLM52_REFERENCE_SERVER_LOG="$REFERENCE_LOG" \
    GLM52_RUNNER_PATH="$RUNNER" \
    GLM52_EXACT_X1_RUNNER_PATH="$RUNNER" \
    GLM52_RUNNER_ARM=phase1 \
    GLM52_PROCESS_CPUSET="$CPUSET" \
    GLM52_SET_CPU_AFFINITY=0 \
    bash "$LAUNCHER" 11 > "$log" 2>&1

  local server_out result_dir result_json server_log
  server_out=$(awk -F= '/^\[INFO\] OUT=/{print $2}' "$log" | tail -1)
  result_dir=$(sed -n 's/.*"run_dir": "\(.*\)",/\1/p' "$log" | tail -1)
  result_json=$result_dir/result.json
  server_log=$server_out/server.log
  [[ -f "$result_json" && -f "$server_log" ]] || {
    echo "[ERR] arm artifacts missing: result=$result_json server=$server_log" >&2
    return 2
  }
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$label" "$arm" "$log" "$result_json" "$server_log" >> "$MANIFEST"
  wait_clear
  echo "[PASS] label=$label arm=$arm result=$result_json"
}

# Alternate both pair order and campaign arm adjacency.
run_arm p1a control
run_arm p1b direct
run_arm p2a direct
run_arm p2b control
run_arm p3a control
run_arm p3b direct
run_arm p4a direct
run_arm p4b control
run_arm p5a control
run_arm p5b direct

echo "[DONE] $OUT"
