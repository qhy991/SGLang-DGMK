#!/usr/bin/env bash
set -euo pipefail

# Host orchestrator for Docker-only GLM-5.2 DSA prefill validation.  The host
# may inspect files and coordinate lifecycle, but the server, warmup, measured
# workload and correctness client all execute inside one frozen container.
if [[ -f /.dockerenv ]]; then
  echo "[ERR] run the orchestrator on the host; runtime work is entered through docker exec" >&2
  exit 2
fi

ROOT=/mnt/b300-shared/home/qinhaiyan/wwxq
REPO=${GLM52_RUNTIME_REPO:-$ROOT/SGLang-DGMK-router-fusion-reviewed}
VENV=$ROOT/venv_wwxq
MODEL=/mnt/b300-shared/models/GLM-5.2-FP8
RUNTIME_CONTAINER=${GLM52_RUNTIME_CONTAINER:-sglang_0515_optimized}
EXPECTED_CONTAINER_IMAGE_ID=${GLM52_EXPECTED_CONTAINER_IMAGE_ID:-sha256:4484ab841baa40eb89a7ee187110877982c44cd7da06f8a8469c8e0059e28bdd}
CONTAINER_USER=${GLM52_RUNTIME_CONTAINER_USER:-0:0}
CONTAINER_HOME=${GLM52_RUNTIME_CONTAINER_HOME:-/root}
SCRIPT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUNDLE_ROOT=${GLM52_EXPERIMENT_BUNDLE_ROOT:-$(cd "$SCRIPT_ROOT/../.." && pwd)}
EXPECTED_SGLANG_COMMIT=${GLM52_EXPECTED_SGLANG_COMMIT:-1571b72db014603ebfaad6d59cbbde1740b23b3e}
RUNTIME_SOURCE_CONTRACT=${GLM52_RUNTIME_SOURCE_CONTRACT:-$SCRIPT_ROOT/runtime_source_contract_reviewed.json}
RUNTIME_SOURCE_VERIFY=${GLM52_RUNTIME_SOURCE_VERIFY:-$BUNDLE_ROOT/candidates/verify_runtime_source_contract.py}
MOK_RUNTIME_SOURCE_CONTRACT=${GLM52_MOK_RUNTIME_SOURCE_CONTRACT:-$BUNDLE_ROOT/candidates/mok_runtime_source_contract.json}
CONTAINER_RUNTIME_WRAPPER=$BUNDLE_ROOT/candidates/run_runtime_command_in_container.sh
PORT=30000
DEEPEP_SMS=${GLM52_DEEPEP_SMS:-136}
DEEPEP_CONFIG_JSON=${GLM52_DEEPEP_CONFIG_JSON:-"{\"normal_dispatch\":{\"num_sms\":${DEEPEP_SMS}},\"normal_combine\":{\"num_sms\":${DEEPEP_SMS}}}"}
DSA_PREFILL_BACKEND=${GLM52_DSA_PREFILL_BACKEND:-tilelang}
GLOBAL_CHUNKED_PREFILL_SIZE=${GLM52_GLOBAL_CHUNKED_PREFILL_SIZE:-80384}
ENFORCE_SHARED_EXPERT_FUSION=${GLM52_ENFORCE_SHARED_EXPERT_FUSION:-0}
RECORD_EXPERT_DISTRIBUTION=${GLM52_RECORD_EXPERT_DISTRIBUTION:-0}
STATIC_EXPERT_MAP=${GLM52_STATIC_EXPERT_MAP:-}
ROUTER_STATIC_PLACEMENT_FUSION=${GLM52_ROUTER_STATIC_PLACEMENT_FUSION:-0}
MOK_PREFILL=${GLM52_MOK_PREFILL:-0}
MOK_ROOT=${GLM52_MOK_ROOT:-/mnt/b300-shared/home/qinhaiyan/mixture-of-kittens}
MOK_ADAPTER_PATH=${GLM52_MOK_ADAPTER_PATH:-$REPO/glm52_opt/mok_prefill}
MOK_CANDIDATE=${GLM52_MOK_CANDIDATE:-stable}
MOK_EXPECTED_COMMIT=0af9e80e67767af5cc2ecd1a64179abbe0fc41af
MOK_TK_EXPECTED_COMMIT=1c3920d993404dd49a6d4c7267ea11d583bd5c68
MEM_FRACTION_STATIC=${GLM52_MEM_FRACTION_STATIC:-0.78}
MAX_RUNNING_REQUESTS=${GLM52_MAX_RUNNING_REQUESTS:-128}
SERVER_SEED=${GLM52_SERVER_SEED:-565849983}
CORRECTNESS_PROBE_ONLY=${GLM52_CORRECTNESS_PROBE_ONLY:-0}
CORRECTNESS_OUTPUT=${GLM52_CORRECTNESS_OUTPUT:-}
CORRECTNESS_LABEL=${GLM52_CORRECTNESS_LABEL:-}
EXECUTOR_PREWARM=${GLM52_EXECUTOR_PREWARM:-0}
PROCESS_CPUSET=${GLM52_PROCESS_CPUSET:-}
SET_CPU_AFFINITY=${GLM52_SET_CPU_AFFINITY:-}
DP_SIZE=1
ATTN_CP_SIZE=8
if [[ -n "$PROCESS_CPUSET" && ! "$PROCESS_CPUSET" =~ ^[0-9,-]+$ ]]; then
  echo "[ERR] GLM52_PROCESS_CPUSET must be a Linux CPU-list string" >&2
  exit 2
fi
if [[ -n "$SET_CPU_AFFINITY" && "$SET_CPU_AFFINITY" != 0 && "$SET_CPU_AFFINITY" != 1 ]]; then
  echo "[ERR] GLM52_SET_CPU_AFFINITY must be empty, 0, or 1" >&2
  exit 2
fi
command -v docker >/dev/null || {
  echo "[ERR] Docker CLI is required on the host" >&2
  exit 2
}
[[ "$(docker inspect -f '{{.State.Running}}' "$RUNTIME_CONTAINER" 2>/dev/null || true)" == true ]] || {
  echo "[ERR] runtime container is not running: $RUNTIME_CONTAINER" >&2
  exit 2
}
CONTAINER_ID=$(docker inspect -f '{{.Id}}' "$RUNTIME_CONTAINER")
CONTAINER_IMAGE_ID=$(docker inspect -f '{{.Image}}' "$RUNTIME_CONTAINER")
CONTAINER_IMAGE_REF=$(docker inspect -f '{{.Config.Image}}' "$RUNTIME_CONTAINER")
if [[ "$CONTAINER_IMAGE_ID" != "$EXPECTED_CONTAINER_IMAGE_ID" ]]; then
  echo "[ERR] runtime image drift: expected $EXPECTED_CONTAINER_IMAGE_ID, got $CONTAINER_IMAGE_ID" >&2
  exit 2
fi
docker exec --user "$CONTAINER_USER" -e HOME="$CONTAINER_HOME" "$RUNTIME_CONTAINER" \
  test -x "$VENV/bin/python" || {
    echo "[ERR] frozen Python is not executable inside $RUNTIME_CONTAINER: $VENV/bin/python" >&2
    exit 2
  }
docker exec --user "$CONTAINER_USER" -e HOME="$CONTAINER_HOME" "$RUNTIME_CONTAINER" \
  test -r "$MODEL/config.json" || {
    echo "[ERR] model is not visible inside $RUNTIME_CONTAINER: $MODEL" >&2
    exit 2
  }
if [[ -n "$PROCESS_CPUSET" ]]; then
  docker exec --user "$CONTAINER_USER" -e HOME="$CONTAINER_HOME" \
    "$RUNTIME_CONTAINER" bash -lc 'command -v taskset' >/dev/null || {
      echo "[ERR] taskset is required for GLM52_PROCESS_CPUSET" >&2
      exit 2
    }
fi
container_exec() {
  docker exec --user "$CONTAINER_USER" -e HOME="$CONTAINER_HOME" \
    --workdir "$REPO" "$RUNTIME_CONTAINER" "$@"
}
case "$DSA_PREFILL_BACKEND" in
  tilelang|fa3|flashmla_kv|flashmla_sparse|trtllm) ;;
  *)
    echo "[ERR] unsupported isolated prefill backend: $DSA_PREFILL_BACKEND" >&2
    exit 2
    ;;
esac
if [[ "$DSA_PREFILL_BACKEND" == trtllm ]]; then
  DSA_DECODE_BACKEND=trtllm
else
  DSA_DECODE_BACKEND=flashmla_kv
fi
if [[ ! "$GLOBAL_CHUNKED_PREFILL_SIZE" =~ ^[0-9]+$ ]] \
  || (( GLOBAL_CHUNKED_PREFILL_SIZE < DP_SIZE * ATTN_CP_SIZE * 64 )) \
  || (( GLOBAL_CHUNKED_PREFILL_SIZE % (DP_SIZE * ATTN_CP_SIZE * 64) != 0 )); then
  echo "[ERR] global chunk must yield a page-aligned positive per-CP-rank chunk" >&2
  exit 2
fi
EFFECTIVE_CHUNKED_PREFILL_SIZE=$((GLOBAL_CHUNKED_PREFILL_SIZE / DP_SIZE / ATTN_CP_SIZE))
MOK_TARGET_M=${GLM52_MOK_TARGET_M:-$((EFFECTIVE_CHUNKED_PREFILL_SIZE - 32))}
case "$ENFORCE_SHARED_EXPERT_FUSION" in
  0|1) ;;
  *)
    echo "[ERR] GLM52_ENFORCE_SHARED_EXPERT_FUSION must be 0 or 1" >&2
    exit 2
    ;;
esac
case "$RECORD_EXPERT_DISTRIBUTION" in
  0|1) ;;
  *)
    echo "[ERR] GLM52_RECORD_EXPERT_DISTRIBUTION must be 0 or 1" >&2
    exit 2
    ;;
esac
if [[ -n "$STATIC_EXPERT_MAP" && ! -f "$STATIC_EXPERT_MAP" ]]; then
  echo "[ERR] static expert map does not exist: $STATIC_EXPERT_MAP" >&2
  exit 2
fi
case "$CORRECTNESS_PROBE_ONLY" in
  0|1) ;;
  *)
    echo "[ERR] GLM52_CORRECTNESS_PROBE_ONLY must be 0 or 1" >&2
    exit 2
    ;;
esac
case "$EXECUTOR_PREWARM" in
  0|1) ;;
  *)
    echo "[ERR] GLM52_EXECUTOR_PREWARM must be 0 or 1" >&2
    exit 2
    ;;
esac
case "$ROUTER_STATIC_PLACEMENT_FUSION" in
  0|1) ;;
  *)
    echo "[ERR] GLM52_ROUTER_STATIC_PLACEMENT_FUSION must be 0 or 1" >&2
    exit 2
    ;;
esac
case "$MOK_PREFILL" in
  0|1) ;;
  *)
    echo "[ERR] GLM52_MOK_PREFILL must be 0 or 1" >&2
    exit 2
    ;;
esac
case "$MOK_CANDIDATE" in
  stable|a|b|d|e|f|g|h|i|j) ;;
  *)
    echo "[ERR] GLM52_MOK_CANDIDATE must be stable or a/b/d/e/f/g/h/i/j; candidate c was retired in favor of integrated j" >&2
    exit 2
    ;;
esac
if [[ "$MOK_PREFILL" == 0 && "$MOK_CANDIDATE" != stable ]]; then
  echo "[ERR] a MoK candidate cannot be selected when GLM52_MOK_PREFILL=0" >&2
  exit 2
fi
if [[ "$MOK_PREFILL" == 1 && "$MOK_CANDIDATE" != stable ]] \
  && [[ "$MOK_ADAPTER_PATH" == "$REPO/glm52_opt/mok_prefill" ]]; then
  echo "[ERR] candidate arm requires an explicit GLM52_MOK_ADAPTER_PATH" >&2
  exit 2
fi
if ! [[ "$MEM_FRACTION_STATIC" =~ ^0\.[0-9]+$ ]]; then
  echo "[ERR] GLM52_MEM_FRACTION_STATIC must be a decimal in (0, 1)" >&2
  exit 2
fi
if ! [[ "$MAX_RUNNING_REQUESTS" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERR] GLM52_MAX_RUNNING_REQUESTS must be a positive integer" >&2
  exit 2
fi
if ! [[ "$SERVER_SEED" =~ ^[0-9]+$ ]]; then
  echo "[ERR] GLM52_SERVER_SEED must be a non-negative integer" >&2
  exit 2
fi
LAYER_AB_PROBE_ITERS=${GLM52_MOK_LAYER_AB_PROBE_ITERS:-0}
MOK_VALIDATE_LAYER=${GLM52_MOK_VALIDATE_LAYER:--1}
MOK_NVTX=${GLM52_MOK_NVTX:-0}
MOK_PRUNE_DUMMY_ROUTES=${GLM52_MOK_PRUNE_DUMMY_ROUTES:-0}
MOK_CONCISE_RUNTIME_EVIDENCE=${GLM52_MOK_CONCISE_RUNTIME_EVIDENCE:-0}
MOK_USE_SCHEDULER_EP_PLAN=${GLM52_MOK_USE_SCHEDULER_EP_PLAN:-0}
if ! [[ "$LAYER_AB_PROBE_ITERS" =~ ^[0-9]+$ ]]; then
  echo "[ERR] GLM52_MOK_LAYER_AB_PROBE_ITERS must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "$MOK_VALIDATE_LAYER" =~ ^-?[0-9]+$ ]]; then
  echo "[ERR] GLM52_MOK_VALIDATE_LAYER must be an integer" >&2
  exit 2
fi
case "$MOK_NVTX" in
  0|1) ;;
  *)
    echo "[ERR] GLM52_MOK_NVTX must be 0 or 1" >&2
    exit 2
    ;;
esac
case "$MOK_PRUNE_DUMMY_ROUTES" in
  0|1) ;;
  *)
    echo "[ERR] GLM52_MOK_PRUNE_DUMMY_ROUTES must be 0 or 1" >&2
    exit 2
    ;;
esac
case "$MOK_CONCISE_RUNTIME_EVIDENCE" in
  0|1) ;;
  *)
    echo "[ERR] GLM52_MOK_CONCISE_RUNTIME_EVIDENCE must be 0 or 1" >&2
    exit 2
    ;;
esac
case "$MOK_USE_SCHEDULER_EP_PLAN" in
  0|1) ;;
  *)
    echo "[ERR] GLM52_MOK_USE_SCHEDULER_EP_PLAN must be 0 or 1" >&2
    exit 2
    ;;
esac
if [[ "$MOK_CONCISE_RUNTIME_EVIDENCE" == 1 \
      && "$MOK_CANDIDATE" != g && "$MOK_CANDIDATE" != h \
      && "$MOK_CANDIDATE" != i && "$MOK_CANDIDATE" != j ]]; then
  echo "[ERR] concise runtime evidence is isolated to candidates g/h/i/j" >&2
  exit 2
fi
if [[ "$EXECUTOR_PREWARM" == 1 && "$MOK_PREFILL" == 1 \
      && "$MOK_CONCISE_RUNTIME_EVIDENCE" != 1 ]]; then
  echo "[ERR] MoK executor prewarm requires concise evidence so warmup cannot consume measured HIT markers" >&2
  exit 2
fi
if [[ ( "$MOK_CANDIDATE" == g || "$MOK_CANDIDATE" == h \
        || "$MOK_CANDIDATE" == i || "$MOK_CANDIDATE" == j ) \
      && "$MOK_CONCISE_RUNTIME_EVIDENCE" != 1 ]]; then
  echo "[ERR] candidates g/h/i/j require GLM52_MOK_CONCISE_RUNTIME_EVIDENCE=1" >&2
  exit 2
fi
if [[ "$MOK_PRUNE_DUMMY_ROUTES" == 1 \
      && "$MOK_CANDIDATE" != e && "$MOK_CANDIDATE" != f \
      && "$MOK_CANDIDATE" != g && "$MOK_CANDIDATE" != h \
      && "$MOK_CANDIDATE" != i && "$MOK_CANDIDATE" != j ]]; then
  echo "[ERR] dummy-route pruning is isolated to candidates e/f/g/h/i/j" >&2
  exit 2
fi
if [[ ( "$MOK_CANDIDATE" == e || "$MOK_CANDIDATE" == f \
        || "$MOK_CANDIDATE" == g || "$MOK_CANDIDATE" == h \
        || "$MOK_CANDIDATE" == i || "$MOK_CANDIDATE" == j ) \
      && "$MOK_PRUNE_DUMMY_ROUTES" != 1 ]]; then
  echo "[ERR] candidates e/f/g/h/i/j require GLM52_MOK_PRUNE_DUMMY_ROUTES=1" >&2
  exit 2
fi
if [[ "$MOK_CANDIDATE" == f || "$MOK_CANDIDATE" == h ]]; then
  if [[ "${GLM52_MOK_MACROBATCH_SIZE:-}" != 131072 \
        || "${GLM52_MOK_CAPACITY_MULTIPLIER:-}" != 0.75 ]]; then
    echo "[ERR] candidate $MOK_CANDIDATE requires macro=131072 and capacity=0.75" >&2
    exit 2
  fi
fi
if (( LAYER_AB_PROBE_ITERS > 0 )) \
    && [[ "$MOK_CANDIDATE" != b && "$MOK_CANDIDATE" != g \
          && "$MOK_CANDIDATE" != i && "$MOK_CANDIDATE" != j ]]; then
  echo "[ERR] the identical-input layer A/B probe is isolated to candidates b/g/i/j" >&2
  exit 2
fi
if [[ ( "$MOK_CANDIDATE" == i || "$MOK_CANDIDATE" == j ) \
      && "$MOK_USE_SCHEDULER_EP_PLAN" != 1 ]]; then
  echo "[ERR] candidates i/j require GLM52_MOK_USE_SCHEDULER_EP_PLAN=1" >&2
  exit 2
fi
if [[ "$MOK_CANDIDATE" != i && "$MOK_CANDIDATE" != j \
      && "$MOK_USE_SCHEDULER_EP_PLAN" != 0 ]]; then
  echo "[ERR] scheduler EP-plan reuse is isolated to candidates i/j" >&2
  exit 2
fi
MOK_DIRECT_SYMMETRIC_INPUT=${GLM52_MOK_DIRECT_SYMMETRIC_INPUT:-0}
case "$MOK_DIRECT_SYMMETRIC_INPUT" in
  0|1) ;;
  *)
    echo "[ERR] GLM52_MOK_DIRECT_SYMMETRIC_INPUT must be 0 or 1" >&2
    exit 2
    ;;
esac
if [[ "$MOK_CANDIDATE" == j \
      && "$MOK_DIRECT_SYMMETRIC_INPUT" != 1 ]]; then
  echo "[ERR] candidate j requires GLM52_MOK_DIRECT_SYMMETRIC_INPUT=1" >&2
  exit 2
fi
if [[ "$MOK_CANDIDATE" != j \
      && "$MOK_DIRECT_SYMMETRIC_INPUT" != 0 ]]; then
  echo "[ERR] direct symmetric input is isolated to candidate j" >&2
  exit 2
fi
if (( LAYER_AB_PROBE_ITERS > 0 )) && [[ "$CORRECTNESS_PROBE_ONLY" != 1 ]]; then
  echo "[ERR] layer A/B probe requires GLM52_CORRECTNESS_PROBE_ONLY=1" >&2
  exit 2
fi
if (( MOK_VALIDATE_LAYER >= 0 )) && [[ "$CORRECTNESS_PROBE_ONLY" != 1 ]]; then
  echo "[ERR] layer validation is forbidden in performance mode" >&2
  exit 2
fi
if [[ "$MOK_PREFILL" == 1 ]] \
  && { ! [[ "$MOK_TARGET_M" =~ ^[0-9]+$ ]] || (( MOK_TARGET_M < 512 )); }; then
  echo "[ERR] GLM52_MOK_TARGET_M must be an integer >= 512" >&2
  exit 2
fi
if [[ "$ROUTER_STATIC_PLACEMENT_FUSION" == 1 && -z "$STATIC_EXPERT_MAP" ]]; then
  echo "[ERR] router static-placement fusion requires GLM52_STATIC_EXPERT_MAP" >&2
  exit 2
fi
if [[ -n "$CORRECTNESS_OUTPUT" && -z "$CORRECTNESS_LABEL" ]]; then
  echo "[ERR] GLM52_CORRECTNESS_OUTPUT requires GLM52_CORRECTNESS_LABEL" >&2
  exit 2
fi
if [[ "$CORRECTNESS_PROBE_ONLY" == 1 && -z "$CORRECTNESS_OUTPUT" ]]; then
  echo "[ERR] probe-only mode requires GLM52_CORRECTNESS_OUTPUT" >&2
  exit 2
fi
if [[ ! "$DEEPEP_SMS" =~ ^[0-9]+$ ]] \
  || (( DEEPEP_SMS < 2 || DEEPEP_SMS > 146 || DEEPEP_SMS % 2 != 0 )); then
  echo "[ERR] internal DeepEP SM value is invalid" >&2
  exit 2
fi
container_exec "$VENV/bin/python" -c \
  'import json,sys; value=json.loads(sys.argv[1]); assert set(value)=={"normal_dispatch","normal_combine"}' \
  "$DEEPEP_CONFIG_JSON"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
MAP_LABEL=$([[ -n "$STATIC_EXPERT_MAP" ]] && echo balanced || echo identity)
MEM_LABEL=${MEM_FRACTION_STATIC/./p}
OUT=$ROOT/bench_results/glm52_e2e_prefill_attention_backend/${STAMP}_${DSA_PREFILL_BACKEND}_cp${ATTN_CP_SIZE}_dp${DP_SIZE}_m${EFFECTIVE_CHUNKED_PREFILL_SIZE}_deepep${DEEPEP_SMS}_sharedfusion${ENFORCE_SHARED_EXPERT_FUSION}_record${RECORD_EXPERT_DISTRIBUTION}_map${MAP_LABEL}_routermap${ROUTER_STATIC_PLACEMENT_FUSION}_mok${MOK_PREFILL}_${MOK_CANDIDATE}_mem${MEM_LABEL}
SERVER_LOG=$OUT/server.log
MOK_ARM_FILE=$OUT/mok_measured.arm
MOK_EVIDENCE_ARM_FILE=$OUT/mok_measured_evidence.arm
EXPERT_RECORD_DIR=$OUT/expert_distribution
ENV_FILE=${GLM52_ENV_FILE:-$REPO/glm52_opt/host_e2e_prefill_tbo.env}
NSYS_CAPTURE_TRIGGER=${GLM52_NSYS_CAPTURE_TRIGGER:-}
NSYS_REPORT_BASE=${GLM52_NSYS_REPORT_BASE:-}
NSYS_BIN=${GLM52_NSYS_BIN:-/usr/local/cuda/bin/nsys}
RUNNER=${GLM52_RUNNER_PATH:-$REPO/glm52_opt/scripts/run_glm52_100k_prefill_arm.py}
EXACT_X1_RUNNER=${GLM52_EXACT_X1_RUNNER_PATH:-}
RUNNER_ARM=${GLM52_RUNNER_ARM:-phase1}
CORRECTNESS_PROBE=${GLM52_CORRECTNESS_PROBE_PATH:-$REPO/glm52_opt/scripts/probe_glm52_100k_first_token.py}
COMPARE=${GLM52_COMPARE_SERVER_ARGS_PATH:-$REPO/glm52_opt/scripts/compare_server_args.py}
case "$RUNNER_ARM" in
  baseline|phase1|phase2|all) ;;
  *) echo "[ERR] invalid GLM52_RUNNER_ARM=$RUNNER_ARM" >&2; exit 2 ;;
esac
CANDIDATE_VERIFY=${GLM52_MOK_CANDIDATE_VERIFY:-}
REFERENCE_LOG=${GLM52_REFERENCE_SERVER_LOG-$ROOT/logs/glm52_route_complete_chunk10048_ws0_deepep136_aligned_warm_20260812_v28.log}
RESULT_ROOT=$ROOT/bench_results/glm52_e2e_prefill_100k_runs

for required_input in "$RUNNER" "$CORRECTNESS_PROBE" "$COMPARE" "$ENV_FILE" \
  "$RUNTIME_SOURCE_CONTRACT" "$RUNTIME_SOURCE_VERIFY" \
  "$MOK_RUNTIME_SOURCE_CONTRACT" "$CONTAINER_RUNTIME_WRAPPER"; do
  [[ -f "$required_input" ]] || {
    echo "[ERR] missing immutable experiment input: $required_input" >&2
    exit 2
  }
done
if [[ -n "$EXACT_X1_RUNNER" && ! -f "$EXACT_X1_RUNNER" ]]; then
  echo "[ERR] missing exact-x1 runner: $EXACT_X1_RUNNER" >&2
  exit 2
fi
git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1 || {
  echo "[ERR] frozen SGLang checkout does not exist: $REPO" >&2
  exit 2
}
ACTUAL_SGLANG_COMMIT=$(git -C "$REPO" rev-parse HEAD)
if [[ "$ACTUAL_SGLANG_COMMIT" != "$EXPECTED_SGLANG_COMMIT" ]]; then
  echo "[ERR] SGLang commit drift: expected $EXPECTED_SGLANG_COMMIT, got $ACTUAL_SGLANG_COMMIT" >&2
  exit 2
fi
RUNTIME_SOURCE_CONTRACT_RESULT=$(
  container_exec "$VENV/bin/python" "$RUNTIME_SOURCE_VERIFY" \
    --repo "$REPO" --contract "$RUNTIME_SOURCE_CONTRACT"
)
echo "$RUNTIME_SOURCE_CONTRACT_RESULT"
if [[ -n "$REFERENCE_LOG" && ! -f "$REFERENCE_LOG" ]]; then
  echo "[ERR] reference server log does not exist: $REFERENCE_LOG" >&2
  exit 2
fi

if [[ "$MOK_PREFILL" == 1 && "$MOK_CANDIDATE" != stable ]] \
  && [[ ! -f "$CANDIDATE_VERIFY" ]]; then
  echo "[ERR] candidate arm requires GLM52_MOK_CANDIDATE_VERIFY" >&2
  exit 2
fi
if [[ -n "$NSYS_REPORT_BASE" ]]; then
  container_exec test -x "$NSYS_BIN" || {
    echo "[ERR] nsys is not executable: $NSYS_BIN" >&2
    exit 2
  }
  [[ -n "$NSYS_CAPTURE_TRIGGER" ]] || {
    echo "[ERR] GLM52_NSYS_REPORT_BASE requires GLM52_NSYS_CAPTURE_TRIGGER" >&2
    exit 2
  }
  mkdir -p "$(dirname "$NSYS_REPORT_BASE")"
  rm -f "$NSYS_REPORT_BASE.nsys-rep" "$NSYS_REPORT_BASE.sqlite"
fi

if [[ "$#" -eq 0 && "$CORRECTNESS_PROBE_ONLY" == 0 ]]; then
  set -- 1
fi

mkdir -p "$OUT"
cd "$REPO"
export PYTHONPATH=$REPO/python
export SGLANG_GLM52_ROUTER_STATIC_PLACEMENT_FUSION=$ROUTER_STATIC_PLACEMENT_FUSION
if [[ "$MOK_PREFILL" == 1 ]]; then
  [[ -d "$MOK_ADAPTER_PATH" && -f "$MOK_ADAPTER_PATH/sitecustomize.py" ]] || {
    echo "[ERR] MoK adapter path is incomplete: $MOK_ADAPTER_PATH" >&2
    exit 2
  }
  git -C "$MOK_ROOT" rev-parse --git-dir >/dev/null 2>&1 || {
    echo "[ERR] MoK checkout does not exist: $MOK_ROOT" >&2
    exit 2
  }
  [[ "$(git -C "$MOK_ROOT" rev-parse HEAD)" == "$MOK_EXPECTED_COMMIT" ]] || {
    echo "[ERR] unexpected MoK commit" >&2
    exit 2
  }
  [[ "$(git -C "$MOK_ROOT/third_party/ThunderKittens" rev-parse HEAD)" == "$MOK_TK_EXPECTED_COMMIT" ]] || {
    echo "[ERR] unexpected ThunderKittens commit" >&2
    exit 2
  }
  [[ -f "$MOK_ROOT/mok/_C.cpython-312-x86_64-linux-gnu.so" ]] || {
    echo "[ERR] built MoK CPython 3.12 extension is missing" >&2
    exit 2
  }
  MOK_RUNTIME_SOURCE_CONTRACT_RESULT=$(
    container_exec "$VENV/bin/python" "$RUNTIME_SOURCE_VERIFY" \
      --repo "$MOK_ROOT" --contract "$MOK_RUNTIME_SOURCE_CONTRACT"
  )
  echo "$MOK_RUNTIME_SOURCE_CONTRACT_RESULT"
  MOK_EXTENSION_SHA256=$(sha256sum \
    "$MOK_ROOT/mok/_C.cpython-312-x86_64-linux-gnu.so" | awk '{print $1}')
  export PYTHONPATH=$MOK_ADAPTER_PATH:$REPO/python:$MOK_ROOT
  export MOK_SGLANG_PREFILL=1
  export MOK_SGLANG_LAYERS=all
  # The 90K prefix is page-aligned to 89984 cached tokens, so the measured
  # 100K request has 10016 uncached tokens. 10048 is only the per-rank chunk
  # ceiling; it is not reduced by DSA before the MoE.
  export MOK_SGLANG_PREFILL_TOKENS=$MOK_TARGET_M
  export MOK_SGLANG_FWD_COMM_SMS=${GLM52_MOK_FWD_COMM_SMS:-32}
  export MOK_SGLANG_MINIBATCH_SIZE=${GLM52_MOK_MINIBATCH_SIZE:-2560}
  export MOK_SGLANG_MACROBATCH_SIZE=${GLM52_MOK_MACROBATCH_SIZE:-20480}
  export MOK_SGLANG_SCHEDULE_CAPACITY_MULTIPLIER=${GLM52_MOK_CAPACITY_MULTIPLIER:-1.0}
  export MOK_SGLANG_FORWARD_ONLY_WORKSPACE=${GLM52_MOK_FORWARD_ONLY_WORKSPACE:-1}
  export MOK_SGLANG_REUSE_PADDING_STAGING=${GLM52_MOK_REUSE_PADDING_STAGING:-0}
  export MOK_SGLANG_MIN_EP_UTILIZATION=${GLM52_MOK_MIN_EP_UTILIZATION:-0.0}
  export MOK_SGLANG_DIRECT_SYMMETRIC_INPUT=$MOK_DIRECT_SYMMETRIC_INPUT
  export MOK_SGLANG_STREAM_ORDERED_TRANSITIONS=${GLM52_MOK_STREAM_ORDERED_TRANSITIONS:-0}
  export MOK_SGLANG_PRUNE_DUMMY_ROUTES=$MOK_PRUNE_DUMMY_ROUTES
  export MOK_SGLANG_CONCISE_RUNTIME_EVIDENCE=$MOK_CONCISE_RUNTIME_EVIDENCE
  export MOK_SGLANG_USE_SCHEDULER_EP_PLAN=$MOK_USE_SCHEDULER_EP_PLAN
  if [[ "$MOK_CANDIDATE" != stable ]]; then
    export MOK_SGLANG_CANDIDATE=$MOK_CANDIDATE
  else
    unset MOK_SGLANG_CANDIDATE || true
  fi
  export MOK_SGLANG_MODEL_PATH=$MODEL
  export MOK_SGLANG_ARM_FILE=$MOK_ARM_FILE
  if [[ "$EXECUTOR_PREWARM" == 1 ]]; then
    export MOK_SGLANG_EVIDENCE_ARM_FILE=$MOK_EVIDENCE_ARM_FILE
  else
    unset MOK_SGLANG_EVIDENCE_ARM_FILE || true
  fi
  export MOK_SGLANG_VALIDATE_LAYER=$MOK_VALIDATE_LAYER
  export MOK_SGLANG_LAYER_AB_PROBE_WARMUP=${GLM52_MOK_LAYER_AB_PROBE_WARMUP:-2}
  export MOK_SGLANG_LAYER_AB_PROBE_ITERS=$LAYER_AB_PROBE_ITERS
  export MOK_SGLANG_NVTX=$MOK_NVTX
  rm -f "$MOK_ARM_FILE"
  rm -f "$MOK_EVIDENCE_ARM_FILE"
else
  export MOK_SGLANG_PREFILL=0
fi

set -a
source "$ENV_FILE"
set +a
export SGLANG_GLM52_ENV_FILE=$ENV_FILE
export SGLANG_GLM52_OPT_HIT_FILE=$OUT/hits.json
rm -f "$SGLANG_GLM52_OPT_HIT_FILE"
if [[ "$RECORD_EXPERT_DISTRIBUTION" == 1 ]]; then
  export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=$EXPERT_RECORD_DIR
fi
if [[ -n "$STATIC_EXPERT_MAP" ]]; then
  cp "$STATIC_EXPERT_MAP" "$OUT/$(basename "$STATIC_EXPERT_MAP")"
fi

if container_exec bash -lc \
    "lsof -nP -iTCP:${PORT} -sTCP:LISTEN -t 2>/dev/null | grep -q ."; then
  echo "[ERR] port $PORT is already in use inside $RUNTIME_CONTAINER" >&2
  exit 2
fi

cp "$ENV_FILE" "$OUT/$(basename "$ENV_FILE")"
cp "$0" "$OUT/run_glm52_100k_attention_backend_host.sh"
{
  echo "mok_candidate=$MOK_CANDIDATE"
  echo "mok_adapter_path=$MOK_ADAPTER_PATH"
  echo "reuse_padding_staging=${MOK_SGLANG_REUSE_PADDING_STAGING:-unset}"
  echo "min_ep_utilization=${MOK_SGLANG_MIN_EP_UTILIZATION:-unset}"
  echo "direct_symmetric_input=${MOK_SGLANG_DIRECT_SYMMETRIC_INPUT:-unset}"
  echo "stream_ordered_transitions=${MOK_SGLANG_STREAM_ORDERED_TRANSITIONS:-unset}"
  echo "prune_dummy_routes=${MOK_SGLANG_PRUNE_DUMMY_ROUTES:-unset}"
  echo "concise_runtime_evidence=${MOK_SGLANG_CONCISE_RUNTIME_EVIDENCE:-unset}"
  echo "scheduler_ep_plan_reuse=${MOK_SGLANG_USE_SCHEDULER_EP_PLAN:-unset}"
  echo "fwd_comm_sms=${MOK_SGLANG_FWD_COMM_SMS:-unset}"
  echo "minibatch_size=${MOK_SGLANG_MINIBATCH_SIZE:-unset}"
  echo "macrobatch_size=${MOK_SGLANG_MACROBATCH_SIZE:-unset}"
  echo "schedule_capacity_multiplier=${MOK_SGLANG_SCHEDULE_CAPACITY_MULTIPLIER:-unset}"
  echo "forward_only_workspace=${MOK_SGLANG_FORWARD_ONLY_WORKSPACE:-unset}"
  echo "validate_layer=${MOK_SGLANG_VALIDATE_LAYER:-unset}"
  echo "layer_ab_probe_warmup=${MOK_SGLANG_LAYER_AB_PROBE_WARMUP:-unset}"
  echo "layer_ab_probe_iters=${MOK_SGLANG_LAYER_AB_PROBE_ITERS:-unset}"
  echo "mok_nvtx=${MOK_SGLANG_NVTX:-unset}"
  echo "nsys_report_base=${NSYS_REPORT_BASE:-unset}"
  echo "server_seed=$SERVER_SEED"
  echo "deepep_config_json=$DEEPEP_CONFIG_JSON"
  echo "executor_prewarm=$EXECUTOR_PREWARM"
  echo "runtime_container=$RUNTIME_CONTAINER"
  echo "container_id=$CONTAINER_ID"
  echo "container_image_id=$CONTAINER_IMAGE_ID"
  echo "container_image_ref=$CONTAINER_IMAGE_REF"
  echo "container_user=$CONTAINER_USER"
  echo "container_home=$CONTAINER_HOME"
  echo "sglang_commit=$ACTUAL_SGLANG_COMMIT"
  echo "expected_sglang_commit=$EXPECTED_SGLANG_COMMIT"
  echo "runtime_source_contract_result=$RUNTIME_SOURCE_CONTRACT_RESULT"
  echo "mok_runtime_source_contract_result=${MOK_RUNTIME_SOURCE_CONTRACT_RESULT:-disabled}"
  echo "mok_extension_sha256=${MOK_EXTENSION_SHA256:-disabled}"
  sha256sum "$RUNNER" "$CORRECTNESS_PROBE" "$COMPARE" "$ENV_FILE" \
    "$RUNTIME_SOURCE_CONTRACT" "$RUNTIME_SOURCE_VERIFY" \
    "$MOK_RUNTIME_SOURCE_CONTRACT" "$CONTAINER_RUNTIME_WRAPPER"
  if [[ -n "$EXACT_X1_RUNNER" ]]; then
    sha256sum "$EXACT_X1_RUNNER"
  fi
  if [[ "$MOK_PREFILL" == 1 ]]; then
    sha256sum "$MOK_ADAPTER_PATH/sitecustomize.py"
    if [[ "$MOK_CANDIDATE" != stable ]]; then
      sha256sum "$MOK_ADAPTER_PATH/../mok_sglang_prefill_patch."*.py
    else
      sha256sum "$MOK_ADAPTER_PATH/mok_sglang_prefill_patch.py"
    fi
  fi
} > "$OUT/mok_arm_manifest.txt"
git status --short > "$OUT/git_status.txt"

SERVER_ENV_FILE=$OUT/runtime_server.env
CLIENT_ENV_FILE=$OUT/runtime_client.env
{
  printf 'HOME=%s\n' "$CONTAINER_HOME"
  printf 'PYTHONPATH=%s\n' "$PYTHONPATH"
  printf 'SGLANG_GLM52_ROUTER_STATIC_PLACEMENT_FUSION=%s\n' \
    "$SGLANG_GLM52_ROUTER_STATIC_PLACEMENT_FUSION"
  printf 'SGLANG_GLM52_ENV_FILE=%s\n' "$SGLANG_GLM52_ENV_FILE"
  printf 'SGLANG_GLM52_OPT_HIT_FILE=%s\n' "$SGLANG_GLM52_OPT_HIT_FILE"
  printf 'GLM52_DEEPEP_SMS=%s\n' "$DEEPEP_SMS"
  printf 'GLM52_PROCESS_CPUSET=%s\n' "$PROCESS_CPUSET"
  if [[ -n "$SET_CPU_AFFINITY" ]]; then
    printf 'SGLANG_SET_CPU_AFFINITY=%s\n' "$SET_CPU_AFFINITY"
  fi
  if [[ -n "$NSYS_CAPTURE_TRIGGER" ]]; then
    printf 'SGLANG_GLM52_NSYS_GATE=1\n'
    printf 'SGLANG_GLM52_NSYS_TRIGGER=%s\n' "$NSYS_CAPTURE_TRIGGER"
    printf 'SGLANG_GLM52_NSYS_SECONDS=%s\n' \
      "${GLM52_NSYS_GATE_SECONDS:-8}"
  fi
  grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "$ENV_FILE"
  if [[ "$RECORD_EXPERT_DISTRIBUTION" == 1 ]]; then
    printf 'SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=%s\n' \
      "$SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR"
  fi
  printf 'MOK_SGLANG_PREFILL=%s\n' "$MOK_SGLANG_PREFILL"
  if [[ "$MOK_PREFILL" == 1 ]]; then
    for key in MOK_SGLANG_LAYERS MOK_SGLANG_PREFILL_TOKENS \
      MOK_SGLANG_FWD_COMM_SMS MOK_SGLANG_MINIBATCH_SIZE \
      MOK_SGLANG_MACROBATCH_SIZE MOK_SGLANG_SCHEDULE_CAPACITY_MULTIPLIER \
      MOK_SGLANG_FORWARD_ONLY_WORKSPACE MOK_SGLANG_REUSE_PADDING_STAGING \
      MOK_SGLANG_MIN_EP_UTILIZATION MOK_SGLANG_DIRECT_SYMMETRIC_INPUT \
      MOK_SGLANG_STREAM_ORDERED_TRANSITIONS MOK_SGLANG_PRUNE_DUMMY_ROUTES \
      MOK_SGLANG_CONCISE_RUNTIME_EVIDENCE MOK_SGLANG_USE_SCHEDULER_EP_PLAN \
      MOK_SGLANG_MODEL_PATH MOK_SGLANG_ARM_FILE MOK_SGLANG_VALIDATE_LAYER \
      MOK_SGLANG_LAYER_AB_PROBE_WARMUP MOK_SGLANG_LAYER_AB_PROBE_ITERS \
      MOK_SGLANG_NVTX; do
      printf '%s=%s\n' "$key" "${!key}"
    done
    if [[ "$MOK_CANDIDATE" != stable ]]; then
      printf 'MOK_SGLANG_CANDIDATE=%s\n' "$MOK_SGLANG_CANDIDATE"
    fi
    if [[ "$EXECUTOR_PREWARM" == 1 ]]; then
      printf 'MOK_SGLANG_EVIDENCE_ARM_FILE=%s\n' \
        "$MOK_SGLANG_EVIDENCE_ARM_FILE"
    fi
  fi
} > "$SERVER_ENV_FILE"
{
  printf 'HOME=%s\n' "$CONTAINER_HOME"
  printf 'PYTHONPATH=%s\n' "$REPO/python"
  printf 'MOK_SGLANG_PREFILL=0\n'
  printf 'GLM52_CLIENT_CPUSET=%s\n' "$PROCESS_CPUSET"
} > "$CLIENT_ENV_FILE"
client_prefix=()
if [[ -n "$PROCESS_CPUSET" ]]; then
  client_prefix=(taskset -c "$PROCESS_CPUSET")
fi
runtime_exec() {
  docker exec --user "$CONTAINER_USER" --workdir "$REPO" \
    --env-file "$CLIENT_ENV_FILE" "$RUNTIME_CONTAINER" \
    "${client_prefix[@]}" "$@"
}

if [[ -n "$PROCESS_CPUSET" ]]; then
  runtime_exec "$VENV/bin/python" -c '
import json
import os
import sys

expected = {int(value) for value in sys.argv[1].split(",")}
observed = set(os.sched_getaffinity(0))
payload = {
    "status": "PASS" if observed == expected else "FAIL",
    "expected_affinity": sorted(expected),
    "observed_affinity": sorted(observed),
}
print(json.dumps(payload, indent=2, sort_keys=True))
if payload["status"] != "PASS":
    raise SystemExit(2)
' "$PROCESS_CPUSET" > "$OUT/client_process_affinity.json"
fi

launch_args=(
  --model-path "$MODEL"
  --served-model-name GLM-5.2-FP8
  --trust-remote-code
  --json-model-override-args '{"qk_rope_head_dim":64,"qk_nope_head_dim":192}'
  --quantization fp8
  --kv-cache-dtype fp8_e4m3
  --speculative-draft-model-quantization unquant
  --context-length 100032
  --mem-fraction-static "$MEM_FRACTION_STATIC"
  --max-running-requests "$MAX_RUNNING_REQUESTS"
  --random-seed "$SERVER_SEED"
  --chunked-prefill-size "$GLOBAL_CHUNKED_PREFILL_SIZE"
  --max-prefill-tokens 80384
  --schedule-conservativeness 1.0
  --watchdog-timeout 3600
  --dist-timeout 3600
  --decode-log-interval 10
  --tp-size 8
  --dp-size "$DP_SIZE"
  --ep-size 8
  --attn-cp-size "$ATTN_CP_SIZE"
  --enable-prefill-cp
  --cp-strategy zigzag
  --moe-dense-tp-size 1
  --enable-dp-attention
  --enable-dp-lm-head
  --attention-backend dsa
  --dsa-prefill-backend "$DSA_PREFILL_BACKEND"
  --dsa-decode-backend "$DSA_DECODE_BACKEND"
  --dsa-topk-backend sgl-kernel
  --moe-a2a-backend deepep
  --deepep-mode "$([[ "$RECORD_EXPERT_DISTRIBUTION" == 1 ]] && echo normal || echo auto)"
  --deepep-config "$DEEPEP_CONFIG_JSON"
  --cuda-graph-backend-decode disabled
  --reasoning-parser glm45
  --tool-call-parser glm47
  --allow-auto-truncate
  --enable-metrics
  --host 127.0.0.1
  --port "$PORT"
)
launch_args+=(--disable-overlap-schedule)
launch_args+=(--skip-server-warmup)
if [[ "$ENFORCE_SHARED_EXPERT_FUSION" == 1 ]]; then
  launch_args+=(--enforce-shared-experts-fusion)
fi
if [[ "$RECORD_EXPERT_DISTRIBUTION" == 1 ]]; then
  launch_args+=(
    --expert-distribution-recorder-mode stat
    --expert-distribution-recorder-buffer-size 256
  )
fi
if [[ -n "$STATIC_EXPERT_MAP" ]]; then
  launch_args+=(
    --init-expert-location "$STATIC_EXPERT_MAP"
    --ep-dispatch-algorithm static
  )
fi

echo "[INFO] OUT=$OUT"
RUNTIME_PID_FILE=$OUT/runtime.pid
rm -f "$RUNTIME_PID_FILE"
runtime_prefix=()
if [[ -n "$PROCESS_CPUSET" ]]; then
  runtime_prefix=(taskset -c "$PROCESS_CPUSET")
fi
if [[ -n "$NSYS_REPORT_BASE" ]]; then
  docker exec --user "$CONTAINER_USER" --workdir "$REPO" \
    --env-file "$SERVER_ENV_FILE" "$RUNTIME_CONTAINER" \
    "${runtime_prefix[@]}" bash "$CONTAINER_RUNTIME_WRAPPER" "$RUNTIME_PID_FILE" "$SERVER_LOG" \
    "$NSYS_BIN" profile \
    --force-overwrite=true \
    -o "$NSYS_REPORT_BASE" \
    --trace=cuda,nvtx \
    --sample=none \
    --cpuctxsw=none \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop-shutdown \
    --kill=none \
    "$VENV/bin/python" -m sglang.launch_server "${launch_args[@]}" &
else
  docker exec --user "$CONTAINER_USER" --workdir "$REPO" \
    --env-file "$SERVER_ENV_FILE" "$RUNTIME_CONTAINER" \
    "${runtime_prefix[@]}" bash "$CONTAINER_RUNTIME_WRAPPER" "$RUNTIME_PID_FILE" "$SERVER_LOG" \
    "$VENV/bin/python" -m sglang.launch_server "${launch_args[@]}" &
fi
server_supervisor_pid=$!
echo "$server_supervisor_pid" > "$OUT/docker_exec.pid"
for _ in $(seq 1 50); do
  [[ -s "$RUNTIME_PID_FILE" ]] && break
  if ! kill -0 "$server_supervisor_pid" 2>/dev/null; then
    echo "[ERR] container runtime exited before recording its PID" >&2
    [[ -f "$SERVER_LOG" ]] && tail -160 "$SERVER_LOG" >&2
    exit 2
  fi
  sleep 0.1
done
[[ -s "$RUNTIME_PID_FILE" ]] || {
  echo "[ERR] container runtime PID was not recorded: $RUNTIME_PID_FILE" >&2
  exit 2
}
runtime_pid=$(<"$RUNTIME_PID_FILE")
[[ "$runtime_pid" =~ ^[0-9]+$ ]] || {
  echo "[ERR] invalid container runtime PID: $runtime_pid" >&2
  exit 2
}
server_pid=
if [[ "$MOK_PREFILL" == 1 ]]; then
  # The server has inherited the opt-in adapter environment. Keep helper and
  # client Python processes on the ordinary SGLang path so their startup and
  # logs cannot be polluted by sitecustomize.
  export PYTHONPATH=$REPO/python
  export MOK_SGLANG_PREFILL=0
fi

stop_pid_group() {
  local pid=$1 pgid
  if [[ -n "$pid" ]] && container_exec kill -0 "$pid" 2>/dev/null; then
    pgid=$(container_exec ps -o pgid= -p "$pid" | tr -d ' ')
    [[ "$pgid" =~ ^[0-9]+$ && "$pgid" -gt 1 ]] \
      && container_exec kill -TERM -- -"$pgid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      container_exec kill -0 "$pid" 2>/dev/null || return 0
      sleep 1
    done
    container_exec kill -KILL -- -"$pgid" 2>/dev/null || true
  fi
}
stop_server() {
  # Under nsys the profiler can exit after cudaProfilerStop while its launch-
  # server child remains alive.  Stop the resolved listener group first, then
  # the wrapper/runtime group if it is distinct or still present.
  stop_pid_group "${server_pid:-}"
  stop_pid_group "${runtime_pid:-}"
}
cleanup() {
  stop_server
  wait "$server_supervisor_pid" 2>/dev/null || true
}
trap cleanup EXIT

ready=0
for i in $(seq 1 180); do
  code=$(runtime_exec curl -s -o /dev/null -w '%{http_code}' --connect-timeout 2 \
    "http://127.0.0.1:$PORT/v1/models" 2>/dev/null || true)
  fired=$(grep -c "The server is fired up and ready to roll" "$SERVER_LOG" 2>/dev/null || true)
  flush_code=000
  if [[ "$fired" -ge 1 ]]; then
    flush_code=$(runtime_exec curl -s -o /dev/null -w '%{http_code}' --connect-timeout 2 \
      -X POST "http://127.0.0.1:$PORT/flush_cache" 2>/dev/null || true)
  fi
  echo "[WAIT] ready=$i code=${code:-000} fired=$fired flush=${flush_code:-000}"
  if [[ "$code" == 200 && "$fired" -ge 1 && "$flush_code" == 200 ]]; then
    server_pid=$(container_exec bash -lc \
      "lsof -nP -iTCP:${PORT} -sTCP:LISTEN -t 2>/dev/null | head -1")
    [[ -n "$server_pid" ]] || {
      echo "[ERR] cannot resolve container server PID on port $PORT" >&2
      exit 2
    }
    echo "$server_pid" > "$OUT/server.pid"
    if [[ -n "$NSYS_REPORT_BASE" ]]; then
      container_exec pgrep -n -x nsys > "$OUT/nsys.pid" 2>/dev/null || true
    fi
    ready=1
    break
  fi
  if ! kill -0 "$server_supervisor_pid" 2>/dev/null; then
    echo "[ERR] server exited before readiness" >&2
    tail -160 "$SERVER_LOG" >&2
    exit 2
  fi
  sleep 10
done
[[ "$ready" -eq 1 ]] || { echo "[ERR] readiness timeout" >&2; exit 2; }

if [[ -n "$PROCESS_CPUSET" ]]; then
  docker exec -i --user "$CONTAINER_USER" -e HOME="$CONTAINER_HOME" \
    --workdir "$REPO" "$RUNTIME_CONTAINER" "$VENV/bin/python" \
    - "$runtime_pid" "$PROCESS_CPUSET" > "$OUT/process_tree_affinity.json" <<'PY'
import json
import os
import re
import sys
from pathlib import Path


def parse_cpu_list(value):
    result = set()
    for item in value.split(","):
        if "-" in item:
            first, last = map(int, item.split("-", 1))
            result.update(range(first, last + 1))
        else:
            result.add(int(item))
    return result


root_pid = int(sys.argv[1])
expected = parse_cpu_list(sys.argv[2])
processes = {}
for status_path in Path("/proc").glob("[0-9]*/status"):
    try:
        text = status_path.read_text()
        pid = int(re.search(r"^Pid:\s+(\d+)$", text, re.MULTILINE).group(1))
        ppid = int(re.search(r"^PPid:\s+(\d+)$", text, re.MULTILINE).group(1))
        name = re.search(r"^Name:\s+(.+)$", text, re.MULTILINE).group(1)
        processes[pid] = {"ppid": ppid, "name": name}
    except (FileNotFoundError, AttributeError, PermissionError, ProcessLookupError):
        continue

descendants = {root_pid}
changed = True
while changed:
    changed = False
    for pid, payload in processes.items():
        if pid not in descendants and payload["ppid"] in descendants:
            descendants.add(pid)
            changed = True

observed = []
violations = []
for pid in sorted(descendants):
    try:
        affinity = set(os.sched_getaffinity(pid))
    except (PermissionError, ProcessLookupError):
        continue
    entry = {
        "pid": pid,
        "ppid": processes.get(pid, {}).get("ppid"),
        "name": processes.get(pid, {}).get("name"),
        "affinity": sorted(affinity),
    }
    observed.append(entry)
    if not affinity or not affinity.issubset(expected):
        violations.append(entry)

result = {
    "status": "PASS" if observed and not violations else "FAIL",
    "root_pid": root_pid,
    "expected_affinity": sorted(expected),
    "observed_processes": observed,
    "violations": violations,
}
print(json.dumps(result, indent=2, sort_keys=True))
if result["status"] != "PASS":
    raise SystemExit(2)
PY
fi

MAX_TOTAL_TOKENS=$(grep -oE 'max_total_num_tokens=[0-9]+' "$SERVER_LOG" \
  | tail -1 | cut -d= -f2)
if [[ -z "$MAX_TOTAL_TOKENS" || "$MAX_TOTAL_TOKENS" -lt 131072 ]]; then
  echo "[ERR] KV capacity is insufficient for an untruncated 100K request: " \
       "max_total_num_tokens=${MAX_TOTAL_TOKENS:-missing}" >&2
  exit 2
fi
printf '%s\n' "$MAX_TOTAL_TOKENS" > "$OUT/max_total_num_tokens.txt"

if [[ -n "$REFERENCE_LOG" ]]; then
  runtime_exec "$VENV/bin/python" "$COMPARE" "$REFERENCE_LOG" "$SERVER_LOG" \
    > "$OUT/server_args.diff"
  cat "$OUT/server_args.diff"
else
  echo "comparison_deferred_to_alternating_screen" > "$OUT/server_args.diff"
fi

if [[ "$CORRECTNESS_PROBE_ONLY" == 0 ]]; then
run_index=0
for x in "$@"; do
  run_index=$((run_index + 1))
  point_runner=$RUNNER
  if [[ "$x" == 1 && -n "$EXACT_X1_RUNNER" ]]; then
    point_runner=$EXACT_X1_RUNNER
  fi
  runner_args=(
    "$VENV/bin/python" "$point_runner"
    --x "$x" --arm "$RUNNER_ARM" \
    --server-log "$SERVER_LOG" \
    --result-root "$RESULT_ROOT"
  )
  if [[ "$EXECUTOR_PREWARM" == 1 ]]; then
    runner_args+=(--executor-prewarm)
  fi
  # Optimized HIT keys are intentionally logged once per worker process.
  # After the first measured point, validate those immutable process-lifetime
  # lines while keeping metrics and prefix-cache evidence scoped to this run.
  if (( run_index > 1 )) && [[ "$RUNNER_ARM" != baseline ]]; then
    runner_args+=(--allow-existing-hit-evidence)
  fi
  if [[ "$MOK_PREFILL" == 1 ]]; then
    runner_args+=(
      --mok-arm-file "$MOK_ARM_FILE"
      --mok-target-m "$MOK_TARGET_M"
    )
    if [[ "$EXECUTOR_PREWARM" == 1 ]]; then
      runner_args+=(--mok-evidence-arm-file "$MOK_EVIDENCE_ARM_FILE")
    fi
  fi
  if [[ "$RECORD_EXPERT_DISTRIBUTION" == 1 ]]; then
    runner_args+=(
      --record-expert-distribution
      --expert-record-dir "$EXPERT_RECORD_DIR"
    )
  fi
  if [[ -n "$NSYS_CAPTURE_TRIGGER" && "$run_index" -eq 1 ]]; then
    runner_args+=(--capture-trigger "$NSYS_CAPTURE_TRIGGER")
  fi
  runtime_exec "${runner_args[@]}" | tee "$OUT/x${x}_r${run_index}.log"
done
fi

if [[ -n "$CORRECTNESS_OUTPUT" ]]; then
  if [[ "$MOK_PREFILL" == 1 ]]; then
    touch "$MOK_ARM_FILE"
  fi
  probe_server_offset=$(wc -c < "$SERVER_LOG")
  runtime_exec "$VENV/bin/python" "$CORRECTNESS_PROBE" \
    --output "$CORRECTNESS_OUTPUT" \
    --label "$CORRECTNESS_LABEL" \
    | tee "$OUT/correctness_probe.log"
  if [[ "$CORRECTNESS_PROBE_ONLY" == 1 \
        && "$MOK_PREFILL" == 1 \
        && "$MOK_CANDIDATE" != stable ]]; then
    tail -c "+$((probe_server_offset + 1))" "$SERVER_LOG" \
      > "$OUT/correctness_server_delta.log"
    candidate_probe_verify_args=(
      "$VENV/bin/python" "$CANDIDATE_VERIFY"
      "$OUT/correctness_server_delta.log"
      --candidate "$MOK_CANDIDATE" --runtime-only
    )
    if (( ${MOK_SGLANG_VALIDATE_LAYER:--1} >= 0 )); then
      candidate_probe_verify_args+=(
        --require-validate-layer "$MOK_SGLANG_VALIDATE_LAYER"
      )
    fi
    if (( LAYER_AB_PROBE_ITERS > 0 )); then
      candidate_probe_verify_args+=(--require-layer-ab)
    fi
    runtime_exec "${candidate_probe_verify_args[@]}" \
      | tee "$OUT/correctness_candidate_evidence_check.log"
  fi
fi

if [[ "$MOK_PREFILL" == 1 && "$MOK_CANDIDATE" != stable ]]; then
  candidate_full_verify_args=(
    "$VENV/bin/python" "$CANDIDATE_VERIFY" "$SERVER_LOG"
    --candidate "$MOK_CANDIDATE"
  )
  if [[ "$CORRECTNESS_PROBE_ONLY" == 1 ]] \
      && (( ${MOK_SGLANG_VALIDATE_LAYER:--1} >= 0 )); then
    candidate_full_verify_args+=(
      --require-validate-layer "$MOK_SGLANG_VALIDATE_LAYER"
    )
  fi
  if (( LAYER_AB_PROBE_ITERS > 0 )); then
    candidate_full_verify_args+=(--require-layer-ab)
  fi
  if [[ "$EXECUTOR_PREWARM" == 1 ]]; then
    candidate_full_verify_args+=(--require-prewarm-summary)
  fi
  runtime_exec "${candidate_full_verify_args[@]}" \
    | tee "$OUT/candidate_evidence_check.log"
fi

stop_server
for i in $(seq 1 30); do
  if ! container_exec kill -0 "$server_pid" 2>/dev/null; then
    break
  fi
  sleep 1
done
if [[ -n "$NSYS_REPORT_BASE" ]]; then
  set +e
  wait "$server_supervisor_pid"
  nsys_status=$?
  set -e
  [[ -s "$NSYS_REPORT_BASE.nsys-rep" ]] || {
    echo "[ERR] nsys report was not produced (status=$nsys_status): " \
         "$NSYS_REPORT_BASE.nsys-rep" >&2
    exit 2
  }
  # The launcher terminates the process group after the workload, which can
  # make nsys report SIGTERM even though stop-shutdown flushed a valid report.
  # Treat the report artifact as authoritative and retain the status for audit.
  echo "$nsys_status" > "$OUT/nsys.exit_status"
fi
trap - EXIT
echo "[DONE] $OUT"
