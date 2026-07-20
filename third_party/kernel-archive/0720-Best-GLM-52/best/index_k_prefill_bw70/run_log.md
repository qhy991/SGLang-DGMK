# Run Log — glm52 index_k_prefill 70% HBM campaign

## Environment (Round 0, 2026-07-18)

| Item | Value |
|---|---|
| Target GPU | NVIDIA B200, physical id **2** (pinned) |
| GPU pin mechanism | `CUDA_VISIBLE_DEVICES=2` (harness default `--device cuda:0` → physical GPU 2); `REMOTE_GPU_ID=2` exported |
| GPU 2 idle at start | utilization 0%, memory 0 MiB, no compute processes |
| GPU 2 clocks | sm current 870 MHz (idle), max 1965 MHz; clocks NOT locked (per DEC-3) |
| Driver | 610.43.02 |
| CUDA runtime | 13.0 |
| PyTorch | 2.11.0+cu130 |
| DeepGEMM | 0.1.4 |
| Python | 3.12.13 (`$KDA_HARNESS_ROOT/.venv`) |
| Kernel-Harness commit | 7d79e5e |
| KDA_HARNESS_ROOT | /home/qinhaiyan/Kernel-Harness |
| CLAUDE_PROJECT_DIR | /home/qinhaiyan/KDA-Pilot-Exp-worktrees/glm52_harness_index_k_prefill_bw70-index-k-prefill-bw70-20260718/llm/glm52_harness_index_k_prefill_bw70 |

Per DEC-3: GPU clocks are NOT locked or changed; observed clocks recorded above; the
normal Harness CUPTI cold-L2 timing protocol is used.

## Candidate Seed (byte-identical, AC-6)

| Item | Value |
|---|---|
| Source (clean harness seed) | `$KDA_HARNESS_ROOT/testbench/tasks/glm52/index_k_prefill/candidate.py` |
| Source SHA256 | `5ccce9d2c3ac33b48a7e35d419b5195fd7f78963d0a1403709a7ab2165b18bc3` |
| Destination | `candidate/candidate.py` |
| Destination SHA256 | `5ccce9d2c3ac33b48a7e35d419b5195fd7f78963d0a1403709a7ab2165b18bc3` |
| Byte-identical | YES (`cmp` IDENTICAL) |

### Harness dirty-tree state at campaign start (must remain unmodified)
`git status --porcelain` in `$KDA_HARNESS_ROOT` shows these pre-existing modified
candidates from OTHER campaigns (not touched by this campaign):
- `testbench/tasks/glm52/index_k_proj_decode/candidate.py` (M)
- `testbench/tasks/glm52/o_proj_decode/candidate.py` (M)
- `testbench/tasks/glm52/o_proj_prefill/candidate.py` (M)

`testbench/tasks/glm52/index_k_prefill/candidate.py` is **clean** (our seed source).
This campaign edits ONLY the task-local `candidate/candidate.py`.

## Gate command
```bash
CUDA_VISIBLE_DEVICES=2 REMOTE_GPU_ID=2 \
  bash "$KDA_HARNESS_ROOT/testbench/tasks/glm52/index_k_prefill/run.sh" \
  --candidate "$CLAUDE_PROJECT_DIR/candidate/candidate.py"
```

## Authoritative measurement preflight checks (AC-7, Round 1)

Per AC-7, GPU 2 idleness is verified **immediately before each acceptance-critical
measurement**. The Round 0 numbers were re-run in Round 1 with a per-command preflight
snapshot captured right before launch (util, memory, active compute-process count,
clocks); the re-run numbers match the recorded no-go within noise.

| Measurement (in order) | UTC timestamp | pin | GPU2 util | GPU2 mem | compute procs on GPU2 | sm/max clock | power |
|---|---|---|---:|---:|---:|---|---:|
| GEMM-only floor (mnk, pdl, num_sms=128) | 2026-07-18T09:48:27Z | CVD=2, RGID=2 | 0% | 0 MiB | 0 | 120/1965 MHz | 145.4 W |
| Authoritative gate run 1 (`candidate/candidate.py`) | 2026-07-18T09:48:31Z | CVD=2, RGID=2 | 1% | 0 MiB | 0 | 1965/1965 MHz | 216.7 W |
| Authoritative gate run 2 (`candidate/candidate.py`) | 2026-07-18T09:48:42Z | CVD=2, RGID=2 | 0% | 0 MiB | 0 | 1965/1965 MHz | 219.7 W |
| Authoritative gate run 3 (`candidate/candidate.py`) | 2026-07-18T09:48:53Z | CVD=2, RGID=2 | 0% | 0 MiB | 0 | 1965/1965 MHz | 216.6 W |

GPU 2 was idle before every command (0-1% util, 0 MiB used, 0 compute processes). Clocks
are recorded, not locked (DEC-3): the pre-floor snapshot caught the card at its 120 MHz
idle clock; by the gate runs it had ramped to the 1965 MHz application clock (expected;
the harness warms up before its timed window).

Re-run results (Round 1), all correct (calc_diff=0 pre+post), gate exit 0 (3 wins):

| Measurement | Round 1 re-run | Round 0 recorded | match |
|---|---|---|---|
| GEMM-only floor (mnk,pdl,num_sms=128) | 80.26 us / 67.4% | 80.19 us / 67.5% | within noise |
| M=1024 median (3 gates) | 84.07 us / 64.4% | 84.09 us / 64.3% | within noise |
| M=2048 median (3 gates) | 84.13 us / 64.3% | 84.11 us / 64.3% | within noise |
| M=4096 median (3 gates) | 83.94 us / 64.4% | 83.98 us / 64.4% | within noise |

The re-run reproduces the recorded no-go: the best candidate is ~84.0 us / ~64.4% HBM and
the DeepGEMM GEMM-only floor is ~80.3 us / ~67.4% HBM, both above the 77.29 us / 70%
target. `candidate/candidate.py` was unchanged; no correctness or performance regression.

## Baseline (A0) — reproduced live on GPU 2

| label | candidate us | reference us | HBM util | achieved GB/s | calc_diff | post-timing | gate verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| M=1024 | 99.864 | 99.88 | 54.17% | 4333.9 | 0.0 | 0.0 | neutral |
| M=2048 | 100.369 | 100.001 | 53.90% | 4312.1 | 0.0 | 0.0 | regress (noise) |
| M=4096 | 100.098 | 100.057 | 54.05% | 4323.8 | 0.0 | 0.0 | neutral |

Gate exit 1 (CORRECT, performance_ok false). compute_util ~0.23, bound=memory,
ridge=562.5, AI=238.17. Achieved ~4.33 TB/s of 8.0 TB/s peak. Target needs 5.6 TB/s
(<=77.29us). This confirms `docs/prior_knowledge.md`.

## DeepGEMM knob exploration (AC-5 evidence, all on GPU 2, correctness calc_diff=0 pre+post)

| Attempt | Config (on `fp8_gemm_nt`) | M=1024 us / HBM | M=2048 us / HBM | M=4096 us / HBM | gate | vs baseline |
|---|---|---|---|---|---|---|
| A0 | default (seed) | 99.86 / 54.17% | 100.37 / 53.90% | 100.10 / 54.05% | exit 1 | — |
| A1 | `compiled_dims="nk"` | 98.78 / 54.77% | 99.11 / 54.58% | 98.93 / 54.69% | exit 0 (3 wins) | ~+1.3% |
| A2 | `compiled_dims="mnk"` | 98.75 / 54.78% | 99.03 / 54.63% | 98.80 / 54.76% | exit 0 (3 wins) | ~+1.3% |
| A3 | `mnk` + `set_pdl(True)` | 97.58 / 55.44% | 97.84 / 55.29% | 97.50 / 55.49% | exit 0 (3 wins) | ~+2.5% |
| A4 | `mnk` + `set_num_sms(132)` | 97.78 / 55.33% | 97.50 / 55.49% | 97.64 / 55.41% | exit 0 (3 wins) | ~+2.5% |

Notes:
- `compiled_dims="mnk"` is legitimate here: the physical GEMM is ALWAYS S=65536,N=128,K=6144
  (the nominal M label never changes tensor shapes), so baking M is real specialization of
  the single GEMM, not a fake per-label dispatch. All labels run the identical baked kernel.
- `num_sms` default is 148 (full B200). `set_num_sms(132)` is within noise of default — the
  large M-tile grid already saturates the SMs, so `num_sms` is not a useful lever (matches
  prior-campaign evidence). `set_tc_util` default 100 (max), irrelevant for a memory-bound op.
- `set_pdl(True)` gives a small, reproducible gain (overlaps the internal scale-transform
  launch with the main GEMM). Best correct candidate so far: **mnk + pdl ≈ 97.5us / 55.4%**.
- **Verdict on knobs:** the full knob stack yields only ~2.5% (100→97.5us). The hard target
  needs ~29% (<=77.29us / >=70% HBM). DeepGEMM knobs are conclusively insufficient; NCU
  characterization (next) is required to name the binding bound before any custom-kernel
  decision (AC-5, DEC-1).

## Attempt Ledger

| Attempt | Candidate | Config | Result (per-label us / HBM%) | Correctness | Decision |
|---|---|---|---|---|---|
| A0 | seed (reference call) | `fp8_gemm_nt` default | ~100.1us / ~54.0% | PASS (0.0) | reference/baseline |
| A1 | `variants/candidate_nk.py` | `compiled_dims="nk"` | ~98.9us / ~54.7% | PASS (0.0) | superseded by A3 |
| A2 | `variants/candidate_mnk.py` | `compiled_dims="mnk"` | ~98.9us / ~54.7% | PASS (0.0) | superseded by A3 |
| A3 | `variants/candidate_mnk_pdl.py` | `mnk` + `set_pdl(True)` | ~97.5us / ~55.4% | PASS (0.0) | superseded by A5 |
| A4 | `variants/candidate_mnk_sms132.py` | `mnk` + `num_sms(132)` | ~97.6us / ~55.4% | PASS (0.0) | within noise; rejected |
| A5 | `candidate/candidate.py` | `mnk` + `pdl` + per-call scale pre-pack (`get_mn_major_tma_aligned_packed_ue8m0_tensor`, `disable_ue8m0_cast=True`) | 93.06 / 93.50 / 93.05us · 58.14 / 57.86 / 58.14% | PASS (0.0 pre+post) | **BEST correct so far** (gate exit 0, 3 wins) |

## NCU characterization (task6) — the binding bound

Timing scope (verified in `testbench/harness/timing.py:132`): the timed value is the
device-kernel **span** `max(kernel.end) - min(kernel.start)` over ALL kernels launched
in one `run()` call. So every scale-preprocessing kernel is counted.

### Baseline/A3 path (`fp8_gemm_nt` with f32 scales, internal `disable_ue8m0_cast=False`), M=1024
NCU report: `docs/ncu/mnk_pdl_M1024.ncu-rep`. Six kernels per call:

| kernel | duration | DRAM %peak | note |
|---|---:|---:|---|
| `pack_fp32_into_ue8m0` | 9.76us | 16.9% | f32→ue8m0 pack (x_scale) |
| `elementwise_kernel_with_index` | 3.87us | ~0 | torch layout helper |
| `at::vectorized_elementwise_kernel` | 4.83us | ~0 | torch layout helper |
| `at::vectorized_gather_kernel` | 4.96us | ~0 | torch gather |
| `transpose_and_pack_fp32_into_ue8m0` | 5.57us | ~0 | mn-major pack (w) |
| **`sm100_fp8_fp4_gemm_1d1d_impl`** | **73.22us** | **75.0%** | the GEMM |

Scale-prep ≈ 29us; **GEMM alone = 73.22us ≈ 73.9% HBM (ABOVE the 70% target).** The
per-call f32→ue8m0 scale transform is the entire reason the total sits at ~55% HBM.

### A5 prepack path (`disable_ue8m0_cast=True` + helper), M=1024
NCU report: `docs/ncu/prepack_M1024.ncu-rep`. Four kernels (torch gather/index gone):

| kernel | duration | DRAM %peak |
|---|---:|---:|
| `pack_fp32_into_ue8m0` (x_scale, 12.58MB) | 10.08us | 16.4% |
| `at::elementwise` (my `ws.expand().contiguous()`) | 4.86us | ~0 |
| `transpose_and_pack_fp32_into_ue8m0` (w) | 5.02us | ~0 |
| `sm100_fp8_fp4_gemm_1d1d_impl` | 72.86us | 75.4% |

Scale-prep ≈ 20us (down from ~29us). Total span 92.83us.

### Custom-kernel budget analysis (Codex-reviewed, `docs/ncu/task7_codex_analysis.md`)
- Target span = 77.29us; GEMM ≈ 73.0us ⇒ **visible scale-prep budget ≈ 4.07us**.
- The x_scale pack runs at only **16% DRAM** (12.58MB in ~10us). The bandwidth floor for
  reading 12.58MB is ~1.6us; a well-optimized fused pack could plausibly hit ~2-3us.
- The w-side (~10us: `transpose_and_pack` + my `.contiguous()`) is pure launch/latency
  overhead on trivial data (one 128×128 block); a bespoke path can make it ~sub-µs.
- **Therefore the identified path to TARGET is a per-call fused ue8m0 pack kernel**
  (Triton/CUDA) producing DeepGEMM's exact mn-major TMA-aligned packed layout in ≈2-3us
  total, then `fp8_gemm_nt(..., disable_ue8m0_cast=True)`. Margin is thin (~76 vs 77.29us).
- **NO-GO bound (named):** if no legal per-call scale pack can bring visible scale-prep
  ≤4.07us (e.g. the pack is intrinsically DRAM-latency-bound above that on this shape),
  the candidate cannot reach ≤77.29us under the frozen harness; the bound is
  `T_total ≥ T_gemm(≈73us) + T_scale_pack`.

## Conditional escalation (task8/task9) — custom fused UE8M0 pack kernel

Since NCU proved the scale transform is the deficit, escalated (with the required NCU
evidence, per DEC-1) to a per-call fused Triton pack kernel that streams x_scale and
w_scale and writes DeepGEMM's exact mn-major TMA-aligned packed-UE8M0 int32 layout,
then `fp8_gemm_nt(..., disable_ue8m0_cast=True)`.

- Losslessness: the frozen scales are **exact powers of two** (verified), so the UE8M0
  exponent is exact: `packed[m,kb] = e0 | e1<<8 | e2<<16 | e3<<24`,
  `ej = ((f32_bits>>23)&0xFF)`. Output is **byte-identical** to
  `get_mn_major_tma_aligned_packed_ue8m0_tensor` (`torch.equal` True for x and w) and
  end-to-end `calc_diff=0` on all labels. Lossless repack, not re-quantization.
- Kernel tuning: v1 (1D grid, BLOCK_M=1024) → 28.5us pack (6% occ, gate 108us). v2
  (2D grid (M-blocks, K-groups), BLOCK_M=512) → 6.7us pack NCU, **gate 85us / 63.7%**.
  BLOCK_M sweep {256,512,1024,2048}: live number flat at ~85us (pack compute is not the
  bottleneck). Final config adds `set_num_sms(128)` (best GEMM knob, below).
- nsys timeline (warm): pack (~4us) and GEMM (~70us) run **back-to-back / slightly
  overlapped via PDL** (GEMM starts ~0.5us before pack ends) — there is NO large launch
  gap. The harness 85us vs warm 73us is the cold-L2 penalty.

### The binding wall: the DeepGEMM GEMM kernel itself (decisive)

Decomposed the cold-L2 CUPTI cost with scales PRE-PACKED (zero scale-prep in the timed
region) using the harness timer `timing.bench_gpu_time_with_cupti`:

| measurement | latency | HBM% |
|---|---:|---:|
| **GEMM-only (prepacked scales, disable_ue8m0_cast=True)** | **80.19–81.79us** | **66.4–67.5%** |
| pack-only (helper) | 13.06us | — |
| full (helper prepack + GEMM) | 93.66us | — |
| baseline (f32 internal transform) | 101.28us | — |

GEMM-only floor swept across every GEMM knob (all FAIL the 77.29us target):

| config | GEMM-only us | HBM% |
|---|---:|---:|
| compiled_dims=mnk, pdl=T, num_sms=148 | 81.47 | 66.4% |
| compiled_dims=mnk, pdl=T, **num_sms=128** | **80.19** | **67.5%** (best) |
| compiled_dims=mnk, pdl=T, num_sms=132 | 80.29 | 67.4% |
| compiled_dims=nk / "" | 81.7–83.0 | 65–66% |
| num_sms 114/96/64 | 83.8/88.2/99.5 | worse |

**Named no-go bound:** the DeepGEMM `fp8_gemm_nt` SM100 kernel for the exact physical
shape S=65536, N=128, K=6144 executes at **>=80.19us (<=67.5% HBM)** under the frozen
harness cold-L2 CUPTI protocol, *with scales already packed and zero scale-prep in the
timed window*, across all public GEMM knobs (compiled_dims in {mnk,nk,""}, pdl in {T,F},
num_sms in {64..148}). Since 80.19us > 77.29us, reaching >=70% HBM is **impossible** by
any scale-preprocessing optimization. The only remaining theoretical path is a custom
SM100 GEMM that beats vendor-tuned DeepGEMM's tcgen05/TMEM kernel on this narrow-N=128
memory-bound shape — a disproportionate, highest-risk effort (flagged by Codex and the
plan) that is out of proportion to the memory-bound-knob/scale-prep scope of this
campaign, and unlikely to beat DeepGEMM meaningfully at N=128.

### Three authoritative gate runs (task10) — best candidate, GPU 2, per-label medians

Final candidate = fused Triton pack + `compiled_dims="mnk"` + `set_pdl(True)` +
`set_num_sms(128)`:

| label | run1 | run2 | run3 | **median** | HBM% | target <=77.29us |
|---|---:|---:|---:|---:|---:|---|
| M=1024 | 83.98 | 84.09 | 84.12 | **84.09** | 64.3% | FAIL |
| M=2048 | 84.10 | 84.15 | 84.11 | **84.11** | 64.3% | FAIL |
| M=4096 | 83.98 | 84.05 | 83.87 | **83.98** | 64.4% | FAIL |

All correct (calc_diff=0 pre+post), gate exit 0 (3 wins, 0 regress), very stable
(±0.15us). **TARGET NOT MET.** Best correct candidate preserved as `candidate/candidate.py`.

| A6 | `candidate/candidate.py` | fused Triton UE8M0 pack + `mnk` + `pdl` + `num_sms=128` + `disable_ue8m0_cast=True` | 84.09 / 84.11 / 83.98us · ~64.4% | PASS (0.0 pre+post) | **BEST correct; 1.19x; TARGET NOT MET (GEMM-floor bound)** |

### Layout gotcha (recorded for reproducibility)
`disable_ue8m0_cast=True` uses DeepGEMM's 1D1D packed format where the WEIGHT scale uses
`gran_mn=1` (per-row), so `w_scale (1,48)` must be broadcast to `(128,48)` before packing
(`check_sf_layout` asserts `sf.size(-2) == ceil_div(mn, gran_mn)` → 128, and
`sf.size(-1) == ceil_div(k, gran_k*4)` → 12). Passing `(1,48)` fails the assertion. The
pack reproduces DeepGEMM's own f32→ue8m0 cast exactly (calc_diff=0), so it is lossless
repacking, not re-quantization.
