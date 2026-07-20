# Run Log — GLM-5.2 o_proj decode 35% HBM campaign

All measurements on the single pinned idle **B200 GPU id 1**. This log records the
environment, the dirty-tree baseline, the byte-for-byte seed provenance, and every
authoritative command executed.

## Environment (captured 2026-07-18)

| Item | Value |
|------|-------|
| Target GPU | NVIDIA B200, id **1** (`REMOTE_GPU_ID=1`, `CUDA_VISIBLE_DEVICES=1`) |
| Compute capability | sm_100 (10, 0) |
| GPU 1 idle at start | util 0%, mem 0 MiB used / 183359 MiB, no compute processes |
| Driver | 610.43.02 |
| CUDA (nvcc) | release 13.2, V13.2.78 |
| torch | 2.11.0+cu130 (cuda_runtime 13.0) |
| triton | 3.6.0 |
| deep_gemm | 0.1.4 |
| cupti | available (authoritative timing = `cupti-cold-l2-device-kernel-median`) |
| L2 cache size | 132,644,864 bytes (~126.5 MiB) — flushed before each timed iteration |
| KDA_HARNESS_ROOT | `/home/qinhaiyan/Kernel-Harness` |
| CLAUDE_PROJECT_DIR | `.../glm52_harness_o_proj_decode_hbm35-decode-hbm35-20260718/llm/glm52_harness_o_proj_decode_hbm35` |
| Harness HEAD | `7d79e5ecbb52f2d943ed6644b8fa9f06a3c52c2d` |
| Campaign task branch | `kda/glm52_harness_o_proj_decode_hbm35-decode-hbm35-20260718` |
| Review base branch | `kda-base/glm52_harness_o_proj_decode_hbm35-decode-hbm35-20260718` |

## Dirty-tree baseline (AC-1)

`git -C "$KDA_HARNESS_ROOT" status --short` at campaign start (first check, session open):

```
 M testbench/tasks/glm52/index_k_proj_decode/candidate.py
 M testbench/tasks/glm52/o_proj_decode/candidate.py
```

**Concurrent-campaign note (honesty for AC-1):** A short time later the shared Harness
tree showed a **third** modified path:

```
 M testbench/tasks/glm52/index_k_proj_decode/candidate.py
 M testbench/tasks/glm52/o_proj_decode/candidate.py
 M testbench/tasks/glm52/o_proj_prefill/candidate.py    <-- appeared after start
```

`o_proj_prefill/candidate.py` is **not attributable to this campaign**. The task prompt
explicitly warns of a *concurrent prefill campaign* ("Use a B200 GPU distinct from any
concurrent prefill campaign"), and `git log` shows an `o_proj_prefill` campaign. This
campaign:
- Never writes to any file under `$KDA_HARNESS_ROOT` (it only *reads* the seed and copies
  it out to the worktree at `candidate/candidate.py`).
- Never touches `index_k_proj_decode/candidate.py` or `o_proj_prefill/candidate.py`.
- Evaluates exclusively through `run.sh --candidate $CLAUDE_PROJECT_DIR/candidate/candidate.py`.

AC-1's intent — *this campaign modifies no Harness file* — therefore holds. The extra
dirty path is external and will be re-checked at finalization to confirm no new edits are
attributable here.

## Seed provenance (AC-1)

Byte-for-byte copy of the dirty Harness seed into the campaign worktree:

```
cp -p $KDA_HARNESS_ROOT/testbench/tasks/glm52/o_proj_decode/candidate.py \
      $CLAUDE_PROJECT_DIR/candidate/candidate.py
```

| File | SHA-256 |
|------|---------|
| Harness seed `testbench/tasks/glm52/o_proj_decode/candidate.py` | `d98f2710b59a0cc5c3ba197495c9a4405235958e01807cf615c2dfe1011a5b7b` |
| Campaign copy `candidate/candidate.py` (initial) | `d98f2710b59a0cc5c3ba197495c9a4405235958e01807cf615c2dfe1011a5b7b` |

`cmp -s` reports the files IDENTICAL; the two SHA-256 values are equal. Seed = Triton
split-K for M=16 + DeepGEMM `fp8_gemm_nt` reference fallback for M=32.

## Command ledger

| # | Purpose | Command | Notes |
|---|---------|---------|-------|
| 0 | GPU idle check | `nvidia-smi -i 1 ...` | 0% util, 0 MiB, no procs |
| 1 | Seed SHA + copy | `sha256sum`, `cp -p`, `cmp -s` | byte-identical, SHAs match |

(Subsequent authoritative `run.sh --candidate` invocations are appended below as they run.)

## Authoritative gate runs

All via: `cd $KDA_HARNESS_ROOT/testbench/tasks/glm52/o_proj_decode && REMOTE_GPU_ID=1 CUDA_VISIBLE_DEVICES=1 ./run.sh --candidate $CLAUDE_PROJECT_DIR/candidate/candidate.py`
(default protocol warmup=3 repeat=10 iterations=30; CUPTI cold-L2 device-kernel median).

| Run | candidate | run_id | M=16 us / HBM% | M=32 us / HBM% | exit |
|-----|-----------|--------|----------------|----------------|------|
| Baseline (seed) | `d98f2710…` | `20260718T045740Z-66a2fa` | 45.50 / 27.79% | 53.26 / 23.85% | 0 |
| Direction A gate 1 | `74f922a7…` | `20260718T055605Z-92aef9` | 33.03 / 38.29% | 33.82 / 37.56% | 0 |
| Direction A gate 2 | `74f922a7…` | `20260718T060108Z-4b1dae` | 32.99 / 38.33% | 33.87 / 37.50% | 0 |
| Direction A gate 3 | `74f922a7…` | `20260718T060115Z-f2a7a9` | 33.02 / 38.29% | 33.86 / 37.51% | 0 |

**Per-shape triple-gate median (AC-4):** M=16 = 33.024us (38.29% HBM, ≤36.13 ✓, margin 3.11us);
M=32 = 33.864us (37.51% HBM, ≤36.29 ✓, margin 2.43us). GPU 1 re-verified idle (0%, 0 MiB)
before each gate batch. `runs/` output is gitignored in the Harness tree (no tracked edits).

Candidate artifacts: `candidate/candidate.py` (SHA `74f922a754b9c4ae9391852da05db00e0e1e7ec43688e6dfcc454c37ffe753e8`),
`candidate/scale_pack.cu` (SHA `de77825355a8e2685f188840b869cfd158e64b81c941b88f67fc04c8b72e6664`).

## Final artifact (post compliance-review hardening)

After the independent Codex compliance audit (see `docs/compliance_review.md`) the candidate
was hardened with `C10_CUDA_KERNEL_LAUNCH_CHECK()` and a corrected byte-accounting docstring
(no behavior change). Final SHAs:
- `candidate/candidate.py`: `92ab42a2ea43d1b1c18f851ff12b8e4f44d1a27acc34b8969bf828133283ffa6`
- `candidate/scale_pack.cu`: `c8106d4227a351c9341f74b5b9acf9f1e3ebc9c7da340de6ce1dcc1d78bef821`

Final triple-gate (default protocol, GPU 1 idle 0%/0 MiB), run IDs `20260718T065002Z-915fcd`,
`20260718T065010Z-c77425`, `20260718T065018Z-df3b8d`:

| Shape | gate1 | gate2 | gate3 | median | HBM% | ceiling | verdict |
|---|--:|--:|--:|--:|--:|--:|---|
| M=16 | 32.960 | 33.024 | 33.040 | **33.024us** | 38.29% | 36.13 | PASS (margin 3.11us) |
| M=32 | 33.808 | 33.824 | 33.872 | **33.824us** | 37.56% | 36.29 | PASS (margin 2.47us) |

All three: CORRECT, 2/2 WIN, calc_diff 0 (pre & post timing), exit 0.

## Round 1 — Directions B/C gates + A1 re-confirm

**GPU-idle policy note:** During Round 1 a concurrent KerSor/sol-execbench MoE campaign
(3× `eval_driver.py`, ~63 GB, 90% util) transiently occupied B200 GPU 1. Per policy, all
authoritative measurements below were taken only after polling GPU 1 back to idle (0%/0 MiB,
no compute processes); an initial Direction-B probe taken as it ramped was discarded and
re-gated clean (numbers matched within noise).

Direction B (`variants/directionB/candidate.py`), idle GPU: M=16 69.41us/18.22%,
M=32 73.92us/17.18%, calc_diff 3.2e-9, CORRECT/REGRESS (exit 1). Log `artifacts/directionB/gate_clean.log`.

Direction C (`variants/directionC/candidate.py` + `scale_gemm.cu`), idle GPU: M=16 3961us/0.32%,
M=32 5146us/0.25%, calc_diff 3.2e-9, CORRECT/REGRESS (exit 1). Log `artifacts/directionC/gate1.log`.

A1 re-confirm after build-failure warning edit — final candidate SHA
`054cf91a7cf5beaa886f5c6c34ed60a36afd7a0f6db30048aa3d64196c258551` (scale_pack.cu unchanged
`c8106d4227a3…`). Run IDs `20260718T083015Z-f580e8`, `…538a24`, `…0a1bda`, GPU 1 idle:

| Shape | gate1 | gate2 | gate3 | median | HBM% | verdict |
|---|--:|--:|--:|--:|--:|---|
| M=16 | 33.040 | 33.040 | 33.032 | **33.040us** | 38.27% | PASS (≤36.13, margin 3.09us) |
| M=32 | 33.856 | 33.824 | 33.835 | **33.835us** | 37.54% | PASS (≤36.29, margin 2.45us) |

All CORRECT, 2/2 WIN, calc_diff 0 pre & post. A1 remains the selected winner.
