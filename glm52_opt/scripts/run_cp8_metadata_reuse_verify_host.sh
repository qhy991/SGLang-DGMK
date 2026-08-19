#!/usr/bin/env bash
set -euo pipefail

# Correctness-only exact-x1 verification of the Ariadne per-forward CP-local
# FlashMLA metadata-reuse invariant. The environment intentionally enables
# per-layer valid-length comparison and must not be used for performance.

ROOT=/mnt/b300-shared/home/qinhaiyan/wwxq
REPO=$ROOT/SGLang-DGMK-cp8-metadata-reuse-ariadne-20260819
EVIDENCE=$ROOT/bench_results/ariadne_glm52_cp8_prefill_20260819
BUNDLE=$ROOT/bench_results/glm52_mok_followup_candidates_20260813
CONTRACT=$EVIDENCE/contracts/runtime_source_contract_metadata_reuse_v1.json
LAUNCHER=$REPO/glm52_opt/scripts/run_glm52_100k_cp8_ep8_packed_review_host.sh
RUNNER=$REPO/glm52_opt/scripts/run_glm52_100k_prefill_cp8_exact_concurrent.py
ENV_FILE=$REPO/glm52_opt/host_e2e_prefill_cp8_integrated_metadata_reuse_verify.env
MAP=$REPO/glm52_opt/glm52_100k_x11_static_expert_map.json
REFERENCE_LOG=$ROOT/bench_results/glm52_e2e_prefill_attention_backend/20260818T033243Z_flashmla_kv_cp8_dp1_m10048_deepep120_sharedfusion0_record0_mapbalanced_routermap1_mok0_stable_mem0p83/server.log
CORRECTNESS_PROBE=$ROOT/bench_results/glm52_cp8_ep8_20260817/probe_glm52_100k_first_token_cp8.py
COMPARE=$BUNDLE/candidates/compare_server_args.py
CPUSET=65,67,69,71,73,75,77,79,81,83,85,87,89,91,93,95,97,99,101,103,105,107,109,111,113,115,117,119,121,123,125,127,193,195,197,199,201,203,205,207,209,211,213,215,217,219,221,223,225,227,229,231,233,235,237,239,241,243,245,247,249,251,253,255

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT=$EVIDENCE/metadata_reuse_verify_$STAMP
mkdir -p "$OUT"

gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^$/d')
port_pid=$(docker exec sglang_0515_optimized bash -lc \
  'lsof -nP -iTCP:30000 -sTCP:LISTEN -t 2>/dev/null || true')
if [[ -n "$gpu_pids" || -n "$port_pid" ]]; then
  echo "[ERR] machine is not clear: gpu=${gpu_pids:-none} port=${port_pid:-none}" >&2
  exit 2
fi

env \
  GLM52_RUNTIME_REPO="$REPO" \
  GLM52_EXPERIMENT_BUNDLE_ROOT="$BUNDLE" \
  GLM52_RUNTIME_SOURCE_CONTRACT="$CONTRACT" \
  GLM52_EXPECTED_SGLANG_COMMIT="$(git -C "$REPO" rev-parse HEAD)" \
  GLM52_CORRECTNESS_PROBE_PATH="$CORRECTNESS_PROBE" \
  GLM52_COMPARE_SERVER_ARGS_PATH="$COMPARE" \
  GLM52_ENV_FILE="$ENV_FILE" \
  GLM52_DSA_PREFILL_BACKEND=flashmla_kv \
  GLM52_DEEPEP_SMS=120 \
  GLM52_STATIC_EXPERT_MAP="$MAP" \
  GLM52_ROUTER_STATIC_PLACEMENT_FUSION=1 \
  GLM52_MEM_FRACTION_STATIC=0.83 \
  GLM52_MAX_RUNNING_REQUESTS=128 \
  GLM52_REFERENCE_SERVER_LOG="$REFERENCE_LOG" \
  GLM52_RUNNER_PATH="$RUNNER" \
  GLM52_EXACT_X1_RUNNER_PATH="$RUNNER" \
  GLM52_RUNNER_ARM=phase1 \
  GLM52_PROCESS_CPUSET="$CPUSET" \
  GLM52_SET_CPU_AFFINITY=0 \
  GLM52_CORRECTNESS_OUTPUT="$OUT/correctness.json" \
  GLM52_CORRECTNESS_LABEL=metadata_reuse_verify \
  bash "$LAUNCHER" 1 2>&1 | tee "$OUT/launcher.log"

echo "[DONE] $OUT"
