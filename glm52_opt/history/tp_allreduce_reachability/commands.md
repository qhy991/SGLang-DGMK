# TP AllReduce command ledger

All CUDA initialization, rank launch, benchmarking, serving, and profiling for
this goal is serialized by the four-GPU lock. Exit 75 means busy; continue
CPU-only work and retry later. Four-rank artifacts are TP4 diagnostics and can
never be cited as TP8 production acceptance.

The committed campaign refuses to run without the inherited GPU 0-3 lock file
descriptors and refuses to write into either source worktree. Use a fresh `/tmp`
destination so benchmark provenance records clean source SHAs; copy immutable
artifacts into this history directory only after every measurement completes.

```bash
/home/qinhaiyan/glm52-goal-runs/with_all_gpus_lock.sh \
  bash /home/qinhaiyan/glm52-goal-runs/24-tp_allreduce_reachability/sglang/glm52_opt/history/tp_allreduce_reachability/run_locked_tp4_campaign.sh \
  /tmp/tp_allreduce_reachability_20260722T120000Z
```

The single lock acquisition runs, in order:

1. inherited-lock receipt, repository/stack checks, pre-run compute-process
   snapshot, four B200 identities, topology, P2P read/write/NVLink capability,
   and NVLink state;
2. short M16, M32, and prefill coordinator traces;
3. eager/default, eager/nondefault, and graph/nondefault exact semantic checks;
4. three uncontended rank-max baselines per shape;
5. paired reference controls and in-place/out-of-place c10d/NCCL attempts;
6. the upstream SGLang backend sweep as performance-only scouting;
7. single-GPU production-ABI O-projection measurements while retaining the lock;
8. full-lifecycle Nsight Systems captures for all three stock shapes and every
   ABI-compatible c10d attempt; and
9. final NVLink counters, clocks, power, compute-process snapshot, and per-step
   exit status.

`SGLANG_ALL_REDUCE_TRACE` is import-time gated and appears only in the short
reachability runs. The benchmark marks those timings ineligible. Python can see
eager dispatch or CUDA Graph capture, but not replay; the Nsight reports use the
runner's stable measured-only NVTX range and kernel names to establish replay.
Only rank 0 owns that NVTX range. CPU-group barriers make it a wall-clock bracket
around all four ranks' measured work, but it is not a process-scoped NVTX
enclosure for ranks 1-3. Process-tree CUDA traces and timestamps are therefore
required when attributing the other ranks.

After the GPU lock is released, generate text profiler tables without CUDA:

```bash
set -euo pipefail
OUT=/tmp/tp_allreduce_reachability_20260722T120000Z
STATS_STATUS="$OUT/profile/stats_status.tsv"
printf 'report\texit_code\tlog\n' >"$STATS_STATUS"

for name in m16 m32 prefill; do
  test -s "$OUT/profile/$name.nsys-rep"
done

for name in m16 m32 prefill m16_c10d m32_c10d prefill_c10d; do
  REPORT="$OUT/profile/$name.nsys-rep"
  LOG="$OUT/profile/$name.stats.log"
  [[ -s "$REPORT" ]] || continue
  set +e
  nsys stats --force-export=true \
    --report cuda_gpu_kern_sum,cuda_api_sum,cuda_kern_exec_sum,nvtx_pushpop_sum,nvtx_kern_sum,nvtx_gpu_proj_sum \
    --format csv --output - "$REPORT" >"$LOG" 2>&1
  RC=$?
  set -e
  printf '%s\t%s\t%s\n' "$name" "$RC" "$LOG" >>"$STATS_STATUS"
  test "$RC" -eq 0
  test -s "$LOG"
done

python3 \
  /home/qinhaiyan/glm52-goal-runs/24-tp_allreduce_reachability/sglang/glm52_opt/history/tp_allreduce_reachability/analyze_tp4_campaign.py \
  "$OUT" --output "$OUT/summary.json" \
  >"$OUT/analyzer.stdout.json"
cmp "$OUT/summary.json" "$OUT/analyzer.stdout.json"
```

`nsys stats --force-export=true` may also create SQLite intermediates. They are
included below so the archive is an exact copy of the analyzed campaign rather
than an undocumented subset. Archive only after the analyzer exits zero. The
destination must not already exist; the manifest covers every copied file except
the manifest itself and is verified before any evidence commit:

```bash
set -euo pipefail
OUT=/tmp/tp_allreduce_reachability_20260722T120000Z
SGLANG=/home/qinhaiyan/glm52-goal-runs/24-tp_allreduce_reachability/sglang
CAMPAIGN_ID="$(basename "$OUT")"
ARCHIVE_PARENT="$SGLANG/glm52_opt/history/tp_allreduce_reachability/runtime"
ARCHIVE="$ARCHIVE_PARENT/$CAMPAIGN_ID"

[[ "$CAMPAIGN_ID" == tp_allreduce_reachability_* ]]
test -s "$OUT/summary.json"
test ! -e "$ARCHIVE"
mkdir -p "$ARCHIVE_PARENT"
mkdir "$ARCHIVE"
cp -a -- "$OUT/." "$ARCHIVE/"
(
  cd "$ARCHIVE"
  find . -type f ! -name SHA256SUMS -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 -r sha256sum
) >"$ARCHIVE/SHA256SUMS"
(
  cd "$ARCHIVE"
  sha256sum --check SHA256SUMS
)
git -C "$SGLANG" status --short -- "$ARCHIVE"
```

## Preserved external TP8 production gate

This host has four GPUs and
`/mnt/OS-oKqEXySb/models/GLM-5.2-NVFP4` is empty. The following gate is therefore
blocked here and must run under the external host's exclusive eight-GPU
scheduler. Never change its TP/DP/EP sizes to four and never treat the local
campaign as satisfying it.

Terminal A starts the stock fallback with short-lived tracing. The DP-attention
flag is essential: without it, `--tp-size 8 --dp-size 8` describes 64 workers,
not the shared eight-rank TP8/DP8 topology. The EP/A2A flags must match the
saved deployment manifest; the command below freezes the intended GLM lane to
EP8 DeepEP auto mode.

```bash
set -euo pipefail
OUT=/tmp/glm52_tp8_allreduce_acceptance_stock
MODEL=/mnt/OS-oKqEXySb/models/GLM-5.2-NVFP4
SGLANG=/path/to/the/validated/sglang
PY=/path/to/the/validated/venv/bin/python
test "$(nvidia-smi -L | wc -l)" -ge 8
test -s "$MODEL/config.json"
mkdir -p "$OUT/trace"
printf '%s\n' "$$" >"$OUT/server.pid"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
SGLANG_GLM52_OPT=0 \
SGLANG_ALL_REDUCE_TRACE="$OUT/trace/tp8.{rank}.{pid}.jsonl" \
PYTHONPATH="$SGLANG/python" \
exec "$PY" -m sglang.launch_server \
  --model-path "$MODEL" \
  --tp-size 8 --dp-size 8 --ep-size 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep --deepep-mode auto \
  --host 127.0.0.1 --port 30000
```

After readiness, Terminal B drives separately named prefill, decode-M16, and
decode-M32 hypotheses three times. These request settings are workload
generators, not proof of local tensor M; the trace must show exact BF16
`[4096,6144]`, `[16,6144]`, and `[32,6144]` hits before matching TP8
microbenchmarks may be added.

```bash
set -euo pipefail
OUT=/tmp/glm52_tp8_allreduce_acceptance_stock
MODEL=/mnt/OS-oKqEXySb/models/GLM-5.2-NVFP4
SGLANG=/path/to/the/validated/sglang
PY=/path/to/the/validated/venv/bin/python
export PYTHONPATH="$SGLANG/python"
for run in 1 2 3; do
  "$PY" -m sglang.benchmark.serving \
    --backend sglang --base-url http://127.0.0.1:30000 \
    --model "$MODEL" --tokenizer "$MODEL" --dataset-name random \
    --num-prompts 1 --random-input-len 32768 --random-output-len 1 \
    --random-range-ratio 0 --max-concurrency 1 \
    --output-file "$OUT/prefill_run${run}.jsonl"
  "$PY" -m sglang.benchmark.serving \
    --backend sglang --base-url http://127.0.0.1:30000 \
    --model "$MODEL" --tokenizer "$MODEL" --dataset-name random \
    --num-prompts 128 --random-input-len 1024 --random-output-len 64 \
    --random-range-ratio 0 --max-concurrency 128 \
    --output-file "$OUT/decode_m16_run${run}.jsonl"
  "$PY" -m sglang.benchmark.serving \
    --backend sglang --base-url http://127.0.0.1:30000 \
    --model "$MODEL" --tokenizer "$MODEL" --dataset-name random \
    --num-prompts 256 --random-input-len 1024 --random-output-len 64 \
    --random-range-ratio 0 --max-concurrency 256 \
    --output-file "$OUT/decode_m32_run${run}.jsonl"
done
kill -INT "$(cat "$OUT/server.pid")"
```

Wait for Terminal A to exit so atexit flushes every rank's trace. Acceptance
then requires: eight trace files with no hook failures; exact caller/shape/
stride/dtype/bytes/stream/graph/alias evidence; Nsight replay kernel mapping;
three uncontended stock baselines; an explicitly named TP8 harness added only
for trace-proven ABIs; paired candidate correctness and p50 gain of at least
3% per enabled bucket; no enabled regression; and three identical candidate
server runs improving the containing-region and end-to-end metrics. Until every
item passes, no `tp8_allreduce_*` workload is created and stock remains active.
