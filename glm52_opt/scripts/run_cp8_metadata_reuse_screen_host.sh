#!/usr/bin/env bash
set -euo pipefail

# One fresh-server baseline/candidate screen. Each arm runs exact x1, one exact
# x11 point, and the frozen 11-input token/logprob probe. This is not formal
# promotion evidence; it only decides whether five-pair budget is warranted.

ROOT=/mnt/b300-shared/home/qinhaiyan/wwxq
EVIDENCE=$ROOT/bench_results/ariadne_glm52_cp8_prefill_20260819
BUNDLE=$ROOT/bench_results/glm52_mok_followup_candidates_20260813
BASELINE_REPO=$ROOT/SGLang-DGMK-cp8-packed-kv-comm-20260818
CANDIDATE_REPO=$ROOT/SGLang-DGMK-cp8-metadata-reuse-ariadne-20260819
BASELINE_CONTRACT=$ROOT/bench_results/sglang_kda_to_glm52_cp8_migration_20260818/runtime_source_contract_cp8_integrated.json
CANDIDATE_CONTRACT=$EVIDENCE/contracts/runtime_source_contract_metadata_reuse_v1.json
BASELINE_ENV=$BASELINE_REPO/glm52_opt/host_e2e_prefill_cp8_integrated.env
CANDIDATE_ENV=$CANDIDATE_REPO/glm52_opt/host_e2e_prefill_cp8_integrated_metadata_reuse.env
LAUNCHER=$CANDIDATE_REPO/glm52_opt/scripts/run_glm52_100k_cp8_ep8_packed_review_host.sh
RUNNER=$CANDIDATE_REPO/glm52_opt/scripts/run_glm52_100k_prefill_cp8_exact_concurrent.py
SUMMARIZER=$CANDIDATE_REPO/glm52_opt/scripts/summarize_cp8_metadata_reuse_screen.py
REFERENCE_LOG=$ROOT/bench_results/glm52_e2e_prefill_attention_backend/20260818T033243Z_flashmla_kv_cp8_dp1_m10048_deepep120_sharedfusion0_record0_mapbalanced_routermap1_mok0_stable_mem0p83/server.log
CORRECTNESS_PROBE=$ROOT/bench_results/glm52_cp8_ep8_20260817/probe_glm52_100k_first_token_cp8.py
COMPARE=$BUNDLE/candidates/compare_server_args.py
CPUSET=65,67,69,71,73,75,77,79,81,83,85,87,89,91,93,95,97,99,101,103,105,107,109,111,113,115,117,119,121,123,125,127,193,195,197,199,201,203,205,207,209,211,213,215,217,219,221,223,225,227,229,231,233,235,237,239,241,243,245,247,249,251,253,255

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT=$EVIDENCE/metadata_reuse_screen_$STAMP
mkdir -p "$OUT"
MANIFEST=$OUT/runs.tsv
printf 'label\tarm\tlauncher_log\tresult_json\tcorrectness_json\tserver_log\n' > "$MANIFEST"

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
  local label=$1 arm=$2 repo=$3 contract=$4 env_file=$5
  local map log correctness_json server_out result_dir result_json server_log
  map=$repo/glm52_opt/glm52_100k_x11_static_expert_map.json
  log=$OUT/${label}_${arm}.log
  correctness_json=$OUT/${label}_${arm}_correctness.json
  wait_clear
  env \
    GLM52_RUNTIME_REPO="$repo" \
    GLM52_EXPERIMENT_BUNDLE_ROOT="$BUNDLE" \
    GLM52_RUNTIME_SOURCE_CONTRACT="$contract" \
    GLM52_EXPECTED_SGLANG_COMMIT="$(git -C "$repo" rev-parse HEAD)" \
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
    GLM52_CORRECTNESS_OUTPUT="$correctness_json" \
    GLM52_CORRECTNESS_LABEL="${label}_${arm}" \
    bash "$LAUNCHER" 1 11 > "$log" 2>&1
  server_out=$(awk -F= '/^\[INFO\] OUT=/{print $2}' "$log" | tail -1)
  result_dir=$(sed -n 's/.*"run_dir": "\(.*\)",/\1/p' "$log" | tail -1)
  result_json=$result_dir/result.json
  server_log=$server_out/server.log
  [[ -f "$result_json" && -f "$correctness_json" && -f "$server_log" ]] || {
    echo "[ERR] missing artifacts for $arm" >&2
    return 2
  }
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$label" "$arm" "$log" "$result_json" "$correctness_json" "$server_log" >> "$MANIFEST"
  wait_clear
}

run_arm s1a baseline "$BASELINE_REPO" "$BASELINE_CONTRACT" "$BASELINE_ENV"
run_arm s1b candidate "$CANDIDATE_REPO" "$CANDIDATE_CONTRACT" "$CANDIDATE_ENV"
python3 "$SUMMARIZER" --manifest "$MANIFEST" --output "$OUT/summary.json"
sha256sum "$OUT/summary.json" > "$OUT/summary.sha256"
echo "[DONE] $OUT"
