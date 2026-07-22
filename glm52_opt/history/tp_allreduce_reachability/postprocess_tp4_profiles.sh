#!/usr/bin/env bash
set -uo pipefail

# CPU-only postprocessing for reports captured by run_locked_tp4_campaign.sh.
# This script must run after the all-GPU lock is released; nsys stats reads the
# immutable report and never initializes CUDA.

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 /tmp/tp_allreduce_reachability_<run-id>" >&2
  exit 64
fi

OUT_ROOT="$(realpath -e "$1")"
case "$OUT_ROOT" in
  /tmp/tp_allreduce_reachability_*) ;;
  *)
    echo "input must be a completed /tmp/tp_allreduce_reachability_* path" >&2
    exit 64
    ;;
esac

PROFILE="$OUT_ROOT/profile"
STATUS="$PROFILE/stats_status.tsv"
STATUS_TMP="$PROFILE/.stats_status.tsv.tmp.$$"
trap 'rm -f -- "$STATUS_TMP"' EXIT
if [[ -d "$STATUS" ]] || ! printf 'report\texit_code\tlog\n' >"$STATUS_TMP"; then
  echo "cannot initialize postprocess status ledger: $STATUS" >&2
  exit 1
fi
FAILED=0

profile_range() {
  case "$1" in
    m16)
      printf '%s\n' \
        'serving_native/tp4_allreduce_decode_m16/cuda_graph/nondefault/reference'
      ;;
    m32)
      printf '%s\n' \
        'serving_native/tp4_allreduce_decode_m32/cuda_graph/nondefault/reference'
      ;;
    prefill)
      printf '%s\n' \
        'serving_native/tp4_allreduce_prefill/eager/nondefault/reference'
      ;;
    m16_c10d)
      printf '%s\n' \
        'serving_native/tp4_allreduce_decode_m16/cuda_graph/nondefault/paired'
      ;;
    m32_c10d)
      printf '%s\n' \
        'serving_native/tp4_allreduce_decode_m32/cuda_graph/nondefault/paired'
      ;;
    prefill_c10d)
      printf '%s\n' \
        'serving_native/tp4_allreduce_prefill/eager/nondefault/paired'
      ;;
    *) return 64 ;;
  esac
}

run_one() {
  local name="$1"
  local range report sqlite log rc
  range="$(profile_range "$name")" || return 64
  report="$PROFILE/$name.nsys-rep"
  sqlite="$PROFILE/$name.sqlite"
  log="$PROFILE/$name.postprocess.log"
  rc=0

  if [[ ! -s "$report" ]]; then
    printf 'missing report: %s\n' "$report" >"$log"
    rc=1
  else
    (
      nsys stats --quiet --force-export=true --force-overwrite=true \
        --sqlite "$sqlite" \
        --report cuda_gpu_kern_sum,cuda_api_sum,cuda_kern_exec_sum,nvtx_pushpop_sum,nvtx_kern_sum,nvtx_gpu_proj_sum,osrt_sum \
        --format csv --output - "$report" \
        >"$PROFILE/$name.stats.log" || exit

      nsys stats --quiet --force-export=false --sqlite "$sqlite" \
        --report cuda_api_trace --format csv --output - "$report" \
        >"$PROFILE/$name.lifecycle_cuda_api_trace.csv" || exit

      nsys stats --quiet --force-export=false --sqlite "$sqlite" \
        --report nvtx_pushpop_trace --format csv --output - "$report" \
        >"$PROFILE/$name.lifecycle_nvtx_pushpop_trace.csv" || exit

      bounds="$(
        python3 - \
          "$PROFILE/$name.lifecycle_nvtx_pushpop_trace.csv" "$range" <<'PY'
import csv
import sys

path, expected = sys.argv[1:]
with open(path, newline="", encoding="utf-8") as handle:
    rows = [
        row
        for row in csv.DictReader(handle)
        if row.get("Name") in {expected, ":" + expected}
    ]
if len(rows) != 1:
    raise SystemExit(f"expected exactly one measured NVTX range, got {rows!r}")
try:
    start = int(rows[0]["Start (ns)"])
    end = int(rows[0]["End (ns)"])
except (KeyError, TypeError, ValueError) as exc:
    raise SystemExit(f"invalid measured NVTX bounds: {rows[0]!r}: {exc}")
if start <= 0 or end <= start:
    raise SystemExit(f"invalid measured NVTX interval: {start}/{end}")
print(f"{start}/{end}")
PY
      )" || exit
      IFS=/ read -r start_ns end_ns <<<"$bounds"
      printf 'range\tstart_ns\tend_ns\n%s\t%s\t%s\n' \
        "$range" "$start_ns" "$end_ns" \
        >"$PROFILE/$name.measured_window.tsv" || exit

      nsys stats --quiet --force-export=false --sqlite "$sqlite" \
        --filter-time "$bounds" --report cuda_api_trace \
        --format csv --output - "$report" \
        >"$PROFILE/$name.measured_cuda_api_trace.csv" || exit

      nsys stats --quiet --force-export=false --sqlite "$sqlite" \
        --filter-time "$bounds" --report cuda_kern_exec_trace:base \
        --format csv --output - "$report" \
        >"$PROFILE/$name.measured_cuda_kern_exec_trace.csv" || exit

      nsys stats --quiet --force-export=false --sqlite "$sqlite" \
        --filter-time "$bounds" --report cuda_gpu_trace:base \
        --format csv --output - "$report" \
        >"$PROFILE/$name.measured_cuda_gpu_trace.csv" || exit

      nsys stats --quiet --force-export=false --sqlite "$sqlite" \
        --filter-time "$bounds" --report nvtx_pushpop_trace \
        --format csv --output - "$report" \
        >"$PROFILE/$name.measured_nvtx_pushpop_trace.csv" || exit

      nsys stats --quiet --force-export=false --sqlite "$sqlite" \
        --filter-time "$bounds" --report nvtx_gpu_proj_trace \
        --format csv --output - "$report" \
        >"$PROFILE/$name.measured_nvtx_gpu_proj_trace.csv" || exit
    ) 2>"$log" || rc=$?
  fi

  if [[ "$rc" -eq 0 ]]; then
    for required in \
      "$sqlite" \
      "$PROFILE/$name.stats.log" \
      "$PROFILE/$name.lifecycle_cuda_api_trace.csv" \
      "$PROFILE/$name.lifecycle_nvtx_pushpop_trace.csv" \
      "$PROFILE/$name.measured_window.tsv" \
      "$PROFILE/$name.measured_cuda_api_trace.csv" \
      "$PROFILE/$name.measured_cuda_kern_exec_trace.csv" \
      "$PROFILE/$name.measured_cuda_gpu_trace.csv" \
      "$PROFILE/$name.measured_nvtx_pushpop_trace.csv"; do
      if [[ ! -s "$required" ]]; then
        printf 'missing or empty postprocessed artifact: %s\n' "$required" \
          >>"$log"
        rc=1
      fi
    done
  fi

  if ! printf '%s\t%s\t%s\n' "$name" "$rc" "$log" >>"$STATUS_TMP"; then
    echo "cannot append postprocess status ledger: $STATUS" >&2
    exit 1
  fi
  if [[ "$rc" -ne 0 ]]; then
    FAILED=1
  fi
}

for name in m16 m32 prefill m16_c10d m32_c10d prefill_c10d; do
  run_one "$name"
done

if ! mv -fT -- "$STATUS_TMP" "$STATUS"; then
  echo "cannot publish postprocess status ledger: $STATUS" >&2
  exit 1
fi
trap - EXIT

if [[ "$FAILED" -ne 0 ]]; then
  echo "one or more Nsight Systems reports failed postprocessing" >&2
  exit 1
fi
echo "Nsight Systems postprocessing completed: $PROFILE"
