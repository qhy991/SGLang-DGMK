#!/usr/bin/env bash
set -uo pipefail

# This script is intentionally a single serialized four-GPU campaign. It must
# be invoked by with_all_gpus_lock.sh; never run it directly.

HARNESS=/home/qinhaiyan/glm52-goal-runs/24-tp_allreduce_reachability/kernel-harness
SGLANG=/home/qinhaiyan/glm52-goal-runs/24-tp_allreduce_reachability/sglang
LOCK_ROOT=/home/qinhaiyan/glm52-goal-runs/locks

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 /tmp/tp_allreduce_reachability_<run-id>" >&2
  exit 64
fi

OUT_ROOT="$(realpath -m "$1")"
case "$OUT_ROOT" in
  /tmp/tp_allreduce_reachability_*) ;;
  *)
    echo "output must be a dedicated /tmp/tp_allreduce_reachability_* path" >&2
    exit 64
    ;;
esac
if [[ -e "$OUT_ROOT" ]]; then
  echo "refusing to mix evidence in an existing path: $OUT_ROOT" >&2
  exit 64
fi
if [[ "${CUDA_VISIBLE_DEVICES:-}" != "0,1,2,3" ]]; then
  echo "CUDA_VISIBLE_DEVICES must be exactly 0,1,2,3 from the all-GPU wrapper" >&2
  exit 75
fi

for pair in 9:gpu0.lock 10:gpu1.lock 11:gpu2.lock 12:gpu3.lock; do
  fd="${pair%%:*}"
  lock_name="${pair#*:}"
  actual="$(readlink "/proc/$$/fd/$fd" 2>/dev/null || true)"
  if [[ "$actual" != "$LOCK_ROOT/$lock_name" ]]; then
    echo "required inherited lock fd $fd is absent: $actual" >&2
    exit 75
  fi
done

for repo in "$HARNESS" "$SGLANG"; do
  if [[ -n "$(git -C "$repo" status --porcelain=v1)" ]]; then
    echo "source worktree must be clean before measurement: $repo" >&2
    git -C "$repo" status --short >&2
    exit 3
  fi
done

mkdir -p \
  "$OUT_ROOT/environment" \
  "$OUT_ROOT/reachability" \
  "$OUT_ROOT/semantics" \
  "$OUT_ROOT/baseline" \
  "$OUT_ROOT/paired" \
  "$OUT_ROOT/backend_scout" \
  "$OUT_ROOT/profile" \
  "$OUT_ROOT/producer_abi"

STATUS="$OUT_ROOT/status.tsv"
printf 'requirement\tstep\texit_code\tstarted_utc\tfinished_utc\tlog\n' >"$STATUS"
REQUIRED_FAILED=0
STEP_TIMEOUT_SECONDS=1800

run_step() {
  local requirement="$1"
  local name="$2"
  shift 2
  local log="$OUT_ROOT/${name}.log"
  local started finished rc command_rc tee_rc
  local -a pipe_status
  started="$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"
  {
    printf 'started_utc=%s\n' "$started"
    printf 'command='
    printf ' %q' "$@"
    printf '\n'
    printf 'timeout_seconds=%s\n' "$STEP_TIMEOUT_SECONDS"
    timeout --signal=TERM --kill-after=30s "${STEP_TIMEOUT_SECONDS}s" "$@"
  } 2>&1 | tee "$log"
  pipe_status=("${PIPESTATUS[@]}")
  command_rc="${pipe_status[0]}"
  tee_rc="${pipe_status[1]}"
  rc="$command_rc"
  if [[ "$rc" -eq 0 && "$tee_rc" -ne 0 ]]; then
    rc="$tee_rc"
  fi
  finished="$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$requirement" "$name" "$rc" "$started" "$finished" "$log" >>"$STATUS"
  if [[ "$requirement" == "required" && "$rc" -ne 0 ]]; then
    REQUIRED_FAILED=1
  fi
  return 0
}

require_phase() {
  local phase="$1"
  if [[ "$REQUIRED_FAILED" -ne 0 ]]; then
    echo "required $phase phase failed; stopping before later GPU work" >&2
    exit 1
  fi
}

export SGLANG_ROOT="$SGLANG"
export KERNEL_HARNESS_PYTHON="$HARNESS/.venv/bin/python"
export PYTHONPATH="$SGLANG/python:$HARNESS:${PYTHONPATH:-}"
export SGLANG_GLM52_OPT=0
unset SGLANG_ALL_REDUCE_TRACE

run_step required environment/lock_receipt \
  bash -c '
    set -euo pipefail
    printf "CUDA_VISIBLE_DEVICES=%s\n" "${CUDA_VISIBLE_DEVICES:-}"
    [[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3" ]]
    for pair in 9:gpu0.lock 10:gpu1.lock 11:gpu2.lock 12:gpu3.lock; do
      fd="${pair%%:*}"
      lock_name="${pair#*:}"
      actual="$(readlink "/proc/$$/fd/$fd")"
      printf "fd=%s expected=%s/%s actual=%s\n" \
        "$fd" "$1" "$lock_name" "$actual"
      [[ "$actual" == "$1/$lock_name" ]]
    done
  ' _ "$LOCK_ROOT"
run_step required environment/source_identity \
  bash -c 'git -C "$1" status --short; git -C "$1" rev-parse HEAD; git -C "$2" status --short; git -C "$2" rev-parse HEAD' \
  _ "$HARNESS" "$SGLANG"
run_step required environment/check_env \
  "$HARNESS/.venv/bin/python" "$HARNESS/testbench/bin/check_env.py"
run_step required environment/nvidia_smi \
  nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu \
  --format=csv
run_step required environment/compute_processes_before \
  nvidia-smi \
  --query-compute-apps=timestamp,gpu_uuid,pid,process_name,used_gpu_memory \
  --format=csv
run_step required environment/topology nvidia-smi topo -m
run_step required environment/p2p_capability \
  bash -c '
    set -euo pipefail
    for capability in r w n; do
      printf "capability=%s\n" "$capability"
      nvidia-smi topo -p2p "$capability"
    done
  '
run_step required environment/nvlink_status nvidia-smi nvlink --status
run_step required environment/nvlink_throughput_before nvidia-smi nvlink --getthroughput d
require_phase environment

trace_run() {
  local short_name="$1"
  local task="$2"
  local mode="$3"
  local stream="$4"
  run_step required "reachability/$short_name" \
    env "SGLANG_ALL_REDUCE_TRACE=$OUT_ROOT/reachability/$short_name.{rank}.{pid}.jsonl" \
    "$HARNESS/serving_native/run.sh" "$task" \
    --execution-mode "$mode" --stream "$stream" --warmup 0 --repeat 1 \
    --output "$OUT_ROOT/reachability/$short_name.result.json"
}

trace_run m16 tp4_allreduce_decode_m16 cuda_graph nondefault
trace_run m32 tp4_allreduce_decode_m32 cuda_graph nondefault
trace_run prefill tp4_allreduce_prefill eager nondefault
unset SGLANG_ALL_REDUCE_TRACE
require_phase reachability

for task in \
  tp4_allreduce_decode_m16 \
  tp4_allreduce_decode_m32 \
  tp4_allreduce_prefill; do
  for setting in "eager default" "eager nondefault" "cuda_graph nondefault"; do
    read -r mode stream <<<"$setting"
    run_step required "semantics/${task}.${mode}.${stream}" \
      "$HARNESS/serving_native/run.sh" "$task" \
      --execution-mode "$mode" --stream "$stream" --warmup 3 --repeat 10 \
      --output "$OUT_ROOT/semantics/${task}.${mode}.${stream}.json"
  done
done
require_phase semantics

task_settings() {
  case "$1" in
    tp4_allreduce_decode_m16)
      SHORT=m16
      MODE=cuda_graph
      STREAM=nondefault
      ;;
    tp4_allreduce_decode_m32)
      SHORT=m32
      MODE=cuda_graph
      STREAM=nondefault
      ;;
    tp4_allreduce_prefill)
      SHORT=prefill
      MODE=eager
      STREAM=nondefault
      ;;
    *) return 64 ;;
  esac
}

for task in \
  tp4_allreduce_decode_m16 \
  tp4_allreduce_decode_m32 \
  tp4_allreduce_prefill; do
  task_settings "$task"
  for run in 1 2 3; do
    run_step required "baseline/${SHORT}_run${run}" \
      "$HARNESS/serving_native/run.sh" "$task" \
      --execution-mode "$MODE" --stream "$STREAM" --warmup 10 --repeat 100 \
      --output "$OUT_ROOT/baseline/${SHORT}_run${run}.json"
  done

  run_step required "paired/${SHORT}_reference_control" \
    "$HARNESS/serving_native/run.sh" "$task" \
    --candidate "$HARNESS/serving_native/candidates/reference.py" \
    --execution-mode "$MODE" --stream "$STREAM" --warmup 10 --repeat 100 \
    --output "$OUT_ROOT/paired/${SHORT}_reference_control.json"

  # Both ABI variants are attempted. Exact alias/poststate validation rejects
  # the one that does not match the runtime-selected production reference.
  run_step attempt "paired/${SHORT}_c10d_inplace" \
    "$HARNESS/serving_native/run.sh" "$task" \
    --candidate "$HARNESS/serving_native/candidates/allreduce_torch.py" \
    --execution-mode "$MODE" --stream "$STREAM" --warmup 10 --repeat 100 \
    --output "$OUT_ROOT/paired/${SHORT}_c10d_inplace.json"
  run_step attempt "paired/${SHORT}_c10d_outplace" \
    "$HARNESS/serving_native/run.sh" "$task" \
    --candidate "$HARNESS/serving_native/candidates/allreduce_torch_outplace.py" \
    --execution-mode "$MODE" --stream "$STREAM" --warmup 10 --repeat 100 \
    --output "$OUT_ROOT/paired/${SHORT}_c10d_outplace.json"
done
require_phase baseline_and_paired_control

# This upstream sweep is performance-only scouting. Its provider timings do not
# replace the exact production-ABI gate above.
run_step attempt backend_scout/custom_allreduce \
  env \
  _IS_BENCH_MULTIGPU_SGLANG_JIT_KERNEL=1 \
  "_IS_BENCH_MULTIGPU_SGLANG_JIT_KERNEL_PID=$$" \
  "$HARNESS/.venv/bin/python" -m torch.distributed.run \
  --standalone --nproc-per-node=4 \
  "$SGLANG/test/registered/jit/benchmark/bench_custom_all_reduce.py"

for task in linear_attn_o_decode_m16 linear_attn_o_decode_m32; do
  run_step required "producer_abi/${task}" \
    "$HARNESS/serving_native/run.sh" "$task" \
    --candidate "$HARNESS/serving_native/candidates/reference.py" \
    --warmup 10 --repeat 100 \
    --output "$OUT_ROOT/producer_abi/${task}.json"
done
require_phase producer_abi

profile_run() {
  local short_name="$1"
  local task="$2"
  local mode="$3"
  local stream="$4"
  run_step required "profile/${short_name}_nsys" \
    nsys profile \
    --trace=cuda,nvtx,nccl,osrt \
    --cuda-graph-trace=node \
    --sample=none \
    --cpuctxsw=none \
    --force-overwrite=true \
    --output="$OUT_ROOT/profile/${short_name}" \
    "$HARNESS/serving_native/run.sh" "$task" \
    --execution-mode "$mode" --stream "$stream" --warmup 3 --repeat 20 \
    --output "$OUT_ROOT/profile/${short_name}.result.json"
  if [[ ! -s "$OUT_ROOT/profile/${short_name}.nsys-rep" ]]; then
    printf 'missing expected Nsight Systems report: %s\n' \
      "$OUT_ROOT/profile/${short_name}.nsys-rep" | \
      tee "$OUT_ROOT/profile/${short_name}_report_check.log"
    printf 'required\tprofile/%s_report_check\t1\t%s\t%s\t%s\n' \
      "$short_name" \
      "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" \
      "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" \
      "$OUT_ROOT/profile/${short_name}_report_check.log" >>"$STATUS"
    REQUIRED_FAILED=1
  fi
}

check_candidate_report() {
  local short_name="$1"
  if [[ -s "$OUT_ROOT/profile/${short_name}_c10d.nsys-rep" ]]; then
    return
  fi
  printf 'missing expected Nsight Systems report: %s\n' \
    "$OUT_ROOT/profile/${short_name}_c10d.nsys-rep" | \
    tee "$OUT_ROOT/profile/${short_name}_c10d_report_check.log"
  printf 'required\tprofile/%s_c10d_report_check\t1\t%s\t%s\t%s\n' \
    "$short_name" \
    "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" \
    "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" \
    "$OUT_ROOT/profile/${short_name}_c10d_report_check.log" >>"$STATUS"
  REQUIRED_FAILED=1
}

profile_run m16 tp4_allreduce_decode_m16 cuda_graph nondefault
profile_run m32 tp4_allreduce_decode_m32 cuda_graph nondefault
profile_run prefill tp4_allreduce_prefill eager nondefault
require_phase stock_profile

profile_candidate_run() {
  local short_name="$1"
  local task="$2"
  local mode="$3"
  local stream="$4"
  local candidate result_path
  if [[ -s "$OUT_ROOT/paired/${short_name}_c10d_outplace.json" ]]; then
    candidate="$HARNESS/serving_native/candidates/allreduce_torch_outplace.py"
    result_path="$OUT_ROOT/paired/${short_name}_c10d_outplace.json"
  elif [[ -s "$OUT_ROOT/paired/${short_name}_c10d_inplace.json" ]]; then
    candidate="$HARNESS/serving_native/candidates/allreduce_torch.py"
    result_path="$OUT_ROOT/paired/${short_name}_c10d_inplace.json"
  else
    run_step attempt "profile/${short_name}_c10d_skipped" \
      bash -c 'echo "no ABI-compatible c10d result was persisted; profiler delta is inapplicable"; exit 2'
    return
  fi
  run_step required "profile/${short_name}_c10d_nsys" \
    nsys profile \
    --trace=cuda,nvtx,nccl,osrt \
    --cuda-graph-trace=node \
    --sample=none \
    --cpuctxsw=none \
    --force-overwrite=true \
    --output="$OUT_ROOT/profile/${short_name}_c10d" \
    "$HARNESS/serving_native/run.sh" "$task" \
    --candidate "$candidate" \
    --execution-mode "$mode" --stream "$stream" --warmup 3 --repeat 20 \
    --output "$OUT_ROOT/profile/${short_name}_c10d.result.json"
  check_candidate_report "$short_name"
  printf '%s\t%s\t%s\n' "$short_name" "$candidate" "$result_path" \
    >>"$OUT_ROOT/profile/c10d_profile_selection.tsv"
}

profile_candidate_run m16 tp4_allreduce_decode_m16 cuda_graph nondefault
profile_candidate_run m32 tp4_allreduce_decode_m32 cuda_graph nondefault
profile_candidate_run prefill tp4_allreduce_prefill eager nondefault
require_phase candidate_profile

run_step required environment/nvlink_throughput_after nvidia-smi nvlink --getthroughput d
run_step required environment/nvidia_smi_after \
  nvidia-smi --query-gpu=index,uuid,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu \
  --format=csv
run_step required environment/compute_processes_after \
  nvidia-smi \
  --query-compute-apps=timestamp,gpu_uuid,pid,process_name,used_gpu_memory \
  --format=csv
run_step required environment/source_identity_after \
  bash -c 'for repo in "$1" "$2"; do status="$(git -C "$repo" status --porcelain=v1)"; printf "%s\n%s\n" "$repo" "$status"; [[ -z "$status" ]] || exit 1; git -C "$repo" rev-parse HEAD; done' \
  _ "$HARNESS" "$SGLANG"

if [[ "$REQUIRED_FAILED" -ne 0 ]]; then
  echo "campaign completed with at least one required step failure; inspect $STATUS" >&2
  exit 1
fi
echo "campaign completed successfully: $OUT_ROOT"
