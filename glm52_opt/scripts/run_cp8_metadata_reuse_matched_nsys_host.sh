#!/usr/bin/env bash
set -euo pipefail

# Matched exact-x1 Nsight Systems attribution for the accepted c01 integrated
# baseline and the per-forward CP-local FlashMLA metadata-reuse candidate. The
# five-pair no-profiler campaign remains the latency authority.

ROOT=/mnt/b300-shared/home/qinhaiyan/wwxq
EVIDENCE=$ROOT/bench_results/ariadne_glm52_cp8_prefill_20260819
BUNDLE=$ROOT/bench_results/glm52_mok_followup_candidates_20260813
BASELINE_REPO=$ROOT/SGLang-DGMK-cp8-packed-kv-comm-20260818
CANDIDATE_REPO=$ROOT/SGLang-DGMK-cp8-metadata-reuse-ariadne-20260819
LAUNCHER=$CANDIDATE_REPO/glm52_opt/scripts/run_glm52_100k_cp8_ep8_packed_review_host.sh
RUNNER=$CANDIDATE_REPO/glm52_opt/scripts/run_glm52_100k_prefill_cp8_exact_concurrent.py
ANALYZER=$CANDIDATE_REPO/glm52_opt/scripts/analyze_cp8_metadata_reuse_nsys.py
BASELINE_CONTRACT=$ROOT/bench_results/sglang_kda_to_glm52_cp8_migration_20260818/runtime_source_contract_cp8_integrated.json
CANDIDATE_CONTRACT=$EVIDENCE/contracts/runtime_source_contract_metadata_reuse_v1.json
BASELINE_ENV=$BASELINE_REPO/glm52_opt/host_e2e_prefill_cp8_integrated.env
CANDIDATE_ENV=$CANDIDATE_REPO/glm52_opt/host_e2e_prefill_cp8_integrated_metadata_reuse.env
REFERENCE_LOG=$ROOT/bench_results/glm52_e2e_prefill_attention_backend/20260818T033243Z_flashmla_kv_cp8_dp1_m10048_deepep120_sharedfusion0_record0_mapbalanced_routermap1_mok0_stable_mem0p83/server.log
CORRECTNESS_PROBE=$ROOT/bench_results/glm52_cp8_ep8_20260817/probe_glm52_100k_first_token_cp8.py
COMPARE=$BUNDLE/candidates/compare_server_args.py
CPUSET=65,67,69,71,73,75,77,79,81,83,85,87,89,91,93,95,97,99,101,103,105,107,109,111,113,115,117,119,121,123,125,127,193,195,197,199,201,203,205,207,209,211,213,215,217,219,221,223,225,227,229,231,233,235,237,239,241,243,245,247,249,251,253,255

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT=${GLM52_MATCHED_NSYS_OUT:-$EVIDENCE/metadata_reuse_matched_nsys_$STAMP}
mkdir -p "$OUT"

contract_commit() {
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["expected_git_commit"])' "$1"
}

wait_clear() {
  for _ in $(seq 1 60); do
    local gpu port
    gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^$/d')
    port=$(docker exec sglang_0515_optimized bash -lc \
      'lsof -nP -iTCP:30000 -sTCP:LISTEN -t 2>/dev/null || true')
    [[ -z "$gpu" && -z "$port" ]] && return 0
    sleep 2
  done
  return 2
}

run_arm() {
  local label=$1 repo=$2 contract=$3 env_file=$4
  local map log report trigger expected_commit
  map=$repo/glm52_opt/glm52_100k_x11_static_expert_map.json
  log=$OUT/$label.log
  report=$OUT/$label
  trigger=$OUT/$label.trigger
  expected_commit=$(contract_commit "$contract")
  wait_clear
  echo "[RUN] label=$label commit=$expected_commit"
  env \
    GLM52_RUNTIME_REPO="$repo" \
    GLM52_EXPERIMENT_BUNDLE_ROOT="$BUNDLE" \
    GLM52_RUNTIME_SOURCE_CONTRACT="$contract" \
    GLM52_EXPECTED_SGLANG_COMMIT="$expected_commit" \
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
    GLM52_NSYS_CAPTURE_TRIGGER="$trigger" \
    GLM52_NSYS_REPORT_BASE="$report" \
    GLM52_NSYS_BIN=/usr/local/bin/nsys \
    bash "$LAUNCHER" 1 > "$log" 2>&1
  [[ -s "$report.nsys-rep" ]] || {
    echo "[ERR] missing Nsight report: $report.nsys-rep" >&2
    return 2
  }
  docker exec sglang_0515_optimized /usr/local/bin/nsys export \
    --type sqlite --force-overwrite true \
    --output "$report.sqlite" "$report.nsys-rep" >/dev/null
  [[ -s "$report.sqlite" ]] || {
    echo "[ERR] missing Nsight SQLite export: $report.sqlite" >&2
    return 2
  }
  echo "[PASS] label=$label report=$report.nsys-rep sqlite=$report.sqlite"
}

run_arm baseline "$BASELINE_REPO" "$BASELINE_CONTRACT" "$BASELINE_ENV"
run_arm candidate "$CANDIDATE_REPO" "$CANDIDATE_CONTRACT" "$CANDIDATE_ENV"
python3 "$ANALYZER" \
  --baseline-sqlite "$OUT/baseline.sqlite" \
  --candidate-sqlite "$OUT/candidate.sqlite" \
  --baseline-log "$OUT/baseline.log" \
  --candidate-log "$OUT/candidate.log" \
  --output "$OUT/attribution.json" \
  > "$OUT/analyzer.stdout.json"
sha256sum "$OUT/attribution.json" > "$OUT/attribution.sha256"
sha256sum "$OUT"/*.nsys-rep "$OUT"/*.sqlite "$OUT"/*.log > "$OUT/artifacts.sha256"
echo "[DONE] $OUT"
