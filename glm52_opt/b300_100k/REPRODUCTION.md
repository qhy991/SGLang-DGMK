# N6 reproduction contract

N6 is a three-part treatment. Do not benchmark a partial configuration and
label it N6, and do not compare it with a result from another context length or
decode workload.

The archived performance authority is the frozen runtime used for the 2026-08
formal run. The router code on the current `main` is a default-off,
source-reviewed port; it has not yet reproduced the formal numbers on a new
healthy B300 host. Reproduction must therefore establish the native anchor
before it evaluates the candidate.

## Scope and non-goals

This contract covers saturated cached-prefill first-token latency on 8×B300.
It does not cover decode/TPOT, continuous online arrivals, arbitrary context
lengths, other EP sizes, CUDA Graph mode, or attention CP8. The archived cell
uses TP8/DP8/EP8 with `attn_cp_size=1`.

Do not combine this reproduction with a wheel, image, model, attention backend,
overlap, graph, or scheduler upgrade. Add research candidates only after the
N6 anchor is restored.

## Host health gate

Before loading the model:

1. Record GPU model/count, driver, CUDA, NVLink topology, clocks, ECC, and Xid.
2. Record CPU socket/core/SMT topology and test every frequency domain under a
   short load. The retired host had large CPU domains pinned near 500 MHz.
3. Check governor, turbo, RAPL, temperature, throttling, container IPC/shm,
   NUMA, and cpuset.
4. Run a short NCCL/DeepEP collective smoke.

Do not run promotion measurements on an unhealthy host. Affinity may prevent a
worker from escaping its parent mask, but it cannot repair a platform frequency
fault.

## Candidate settings

Set the default-off router gate:

```bash
export SGLANG_GLM52_ROUTER_STATIC_PLACEMENT_FUSION=1
```

Launch SGLang with static expert dispatch, the accepted map, and the same
DeepEP resource setting for normal dispatch and combine:

```text
--ep-dispatch-algorithm static
--init-expert-location glm52_opt/glm52_100k_x11_static_expert_map.json
--deepep-config {"normal_dispatch":{"num_sms":120},"normal_combine":{"num_sms":120}}
```

The baseline uses identity placement, native logical-to-physical postprocess,
and 136 SMs for both normal dispatch and combine. All other values come from
[`workload_contract.json`](workload_contract.json).

The accepted map SHA-256 must be:

```text
36d13233672288317fd69495d4cedb46844b8ae99033d184d84aff0c99c68f09
```

Record the repository HEAD and dirty state, image digest, CUDA/Python/Triton/
DeepEP/FlashMLA/DeepGEMM versions, model config/index/weight revision, tokenizer,
quantization artifacts, dataset/input hash, and complete server/client command.
The old archive does not fully freeze every weight shard/tokenizer/quantization
artifact, so a new run should close that provenance gap.

## Focused code checks

Run:

- `test/registered/jit/test_moe_fused_gate_pad_mask.py`;
- `test/registered/jit/test_moe_fused_gate_static_placement.py`.

The baseline must not print the N6 selection marker. The candidate must print:

```text
GLM-5.2 router static-placement fusion selected:
logical-to-physical + optional padded-row mask + int64 IDs
```

If the marker is absent, the request may have followed the existing fallback
path. A successful request without the marker is not an N6 router-fusion
measurement.

Before performance timing, compare the candidate against two baseline repeats
on identical inputs. Check exact generated tokens and the predeclared selected-
token logprob tolerance. Token IDs are categorical; do not use MAE or cosine on
token IDs.

## Required checks

1. Confirm 8×B300, TP8/DP8/EP8, DP-attention, and `attn_cp_size=1`. “CP=8”
   is not the attention topology used by the archived result.
2. Run the focused CUDA router tests before service timing.
3. Verify path markers show the static-placement fused router on every active
   MoE layer and the expected FlashMLA/DeepEP path.
4. Check 110/110 successful requests and exact generated tokens.
5. Restore the healthy baseline anchor: P50 ≤2000 ms, P90 ≤5000 ms, and
   throughput ≥438000 token/s.
6. Collect five fresh-server AB/BA pairs without a profiler, then an independent
   client-seed holdout.
7. Use Nsys only after the no-profiler gates pass. Nsys timings explain causes;
   they are not promotion measurements.

Also verify the frozen request details and path facts: 90,000 logical shared
prefix tokens, 89,984 actual cache hits, 10,000 new suffix tokens, output=1,
110 requests, concurrency=11, page size 64, global chunk 80,384, per-rank M up
to 10,048, FlashMLA-KV, DeepEP normal, memory fraction 0.82, and overlap/graph
disabled.

Preserve the raw order of every pair. Report median, spread, paired wins,
request completion, and control drift; do not keep only a manually transcribed
percentage. Correctness, primary performance, holdout, and profile are separate
runs and must not be merged into one claim.

## Promotion and stop rules

The archived five-pair screen requires at least 4/5 P50 wins, P50 median
improvement of at least 1%, P90 regression no worse than 2%, throughput
regression no worse than 1%, P50≤2000 ms, P90≤5000 ms, and 110/110 requests.
Then run the independent client-seed holdout.

Stop the candidate immediately on any of the following:

- token/logprob failure, NaN, illegal memory access, or communication hang;
- missing selection marker or unrecorded fallback;
- workload/path/config drift;
- unhealthy baseline or excessive control drift;
- early pairs make 4/5 wins mathematically impossible;
- P50 or throughput fails the predeclared gate, even if a secondary tail metric
  improves;
- profiler-instrumented timing is being treated as official performance.

After an E2E win, use matched Nsys to locate the changed critical-path interval
across all ranks. Use NCU only if Systems narrows the remaining hypothesis to a
representative kernel and a precise question about memory, registers,
occupancy, geometry, or stalls.

If the baseline anchor is unhealthy, stop and diagnose the host. A relative
candidate win on an unhealthy anchor does not replace N6.
