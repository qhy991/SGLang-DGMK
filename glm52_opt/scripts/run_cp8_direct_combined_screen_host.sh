#!/usr/bin/env bash
set -euo pipefail

# Matched source-control vs direct-multimem + combined-indexer composition.
# Each arm records exact x1, three exact x11 repeats, and an independent
# 11-request token/logprob correctness probe.

ROOT=/mnt/b300-shared/home/qinhaiyan/wwxq
EVIDENCE=$ROOT/bench_results/sglang_kda_to_glm52_cp8_migration_20260818
BUNDLE=$ROOT/bench_results/glm52_mok_followup_candidates_20260813
CONTROL_REPO=$ROOT/SGLang-DGMK-cp8-packed-kv-control-20260818
CANDIDATE_REPO=$ROOT/SGLang-DGMK-cp8-packed-kv-comm-20260818
LAUNCHER=$CANDIDATE_REPO/glm52_opt/scripts/run_glm52_100k_cp8_ep8_packed_review_host.sh
RUNNER=$CANDIDATE_REPO/glm52_opt/scripts/run_glm52_100k_prefill_cp8_exact_concurrent.py
CONTROL_CONTRACT=$EVIDENCE/runtime_source_contract_cp8_packed_kv_control.json
CANDIDATE_CONTRACT=$EVIDENCE/runtime_source_contract_cp8_direct_packed_kv_augmented.json
CONTROL_ENV=$EVIDENCE/host_e2e_prefill_cp8_packed_kv_baseline.env
CANDIDATE_ENV=$CANDIDATE_REPO/glm52_opt/host_e2e_prefill_cp8_direct_packed_kv_combined_indexer.env
REFERENCE_LOG=$ROOT/bench_results/glm52_e2e_prefill_attention_backend/20260818T033243Z_flashmla_kv_cp8_dp1_m10048_deepep120_sharedfusion0_record0_mapbalanced_routermap1_mok0_stable_mem0p83/server.log
CORRECTNESS_PROBE=$ROOT/bench_results/glm52_cp8_ep8_20260817/probe_glm52_100k_first_token_cp8.py
COMPARE=$BUNDLE/candidates/compare_server_args.py
CPUSET=65,67,69,71,73,75,77,79,81,83,85,87,89,91,93,95,97,99,101,103,105,107,109,111,113,115,117,119,121,123,125,127,193,195,197,199,201,203,205,207,209,211,213,215,217,219,221,223,225,227,229,231,233,235,237,239,241,243,245,247,249,251,253,255

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT=$EVIDENCE/direct_plus_combined_$STAMP
mkdir -p "$OUT"

wait_clear() {
  for _ in $(seq 1 60); do
    gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^$/d')
    port=$(docker exec sglang_0515_optimized bash -lc \
      'lsof -nP -iTCP:30000 -sTCP:LISTEN -t 2>/dev/null || true')
    [[ -z "$gpu" && -z "$port" ]] && return 0
    sleep 2
  done
  return 2
}

run_arm() {
  local label=$1 repo=$2 contract=$3 env_file=$4 map log
  map=$repo/glm52_opt/glm52_100k_x11_static_expert_map.json
  log=$OUT/$label.log
  wait_clear
  echo "[RUN] $label"
  env \
    GLM52_RUNTIME_REPO="$repo" \
    GLM52_EXPERIMENT_BUNDLE_ROOT="$BUNDLE" \
    GLM52_RUNTIME_SOURCE_CONTRACT="$contract" \
    GLM52_CORRECTNESS_PROBE_PATH="$CORRECTNESS_PROBE" \
    GLM52_COMPARE_SERVER_ARGS_PATH="$COMPARE" \
    GLM52_ENV_FILE="$env_file" \
    GLM52_DSA_PREFILL_BACKEND=flashmla_kv \
    GLM52_DEEPEP_SMS=120 \
    GLM52_STATIC_EXPERT_MAP="$map" \
    GLM52_ROUTER_STATIC_PLACEMENT_FUSION=1 \
    GLM52_MEM_FRACTION_STATIC=0.83 \
    GLM52_MAX_RUNNING_REQUESTS=128 \
    GLM52_REFERENCE_SERVER_LOG="$REFERENCE_LOG" \
    GLM52_RUNNER_PATH="$RUNNER" \
    GLM52_EXACT_X1_RUNNER_PATH="$RUNNER" \
    GLM52_RUNNER_ARM=phase1 \
    GLM52_PROCESS_CPUSET="$CPUSET" \
    GLM52_SET_CPU_AFFINITY=0 \
    GLM52_CORRECTNESS_OUTPUT="$OUT/${label}_correctness.json" \
    GLM52_CORRECTNESS_LABEL="$label" \
    bash "$LAUNCHER" 1 11 11 11 > "$log" 2>&1
  echo "[PASS] $label"
}

run_arm control "$CONTROL_REPO" "$CONTROL_CONTRACT" "$CONTROL_ENV"
run_arm direct_combined "$CANDIDATE_REPO" "$CANDIDATE_CONTRACT" "$CANDIDATE_ENV"
echo "[DONE] $OUT"
