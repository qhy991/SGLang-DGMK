# Experimental MoK MXFP8 prefill adapter for GLM-5.2

This directory contains the opt-in adapter used to evaluate Cursor's
`mixture-of-kittens` (MoK) routed-expert kernel on the GLM-5.2 FP8 checkpoint.
It is an experiment harness, not a default SGLang execution path.

## Validated configuration

- 8 NVIDIA B300 GPUs (SM103), TP8/DP8/EP8 with DP attention
- GLM-5.2 block-FP8 checkpoint: 256 routed experts, Top-8, one shared expert
- hidden size 6144, routed intermediate size 2048, sparse layers 3 through 77
- MoK commit `0af9e80e67767af5cc2ecd1a64179abbe0fc41af`
- ThunderKittens commit `1c3920d993404dd49a6d4c7267ea11d583bd5c68`
- SGLang base commit `fc4b5d22f2dbf5e82e6a5012cda9765de272daa6`
- eight concurrent requests, 4096 input tokens/request, one output token

SGLang clamps chunked prefill to 1024 tokens under DP attention in this
configuration, so each 4K request executes four M=1024 MoK calls. The adapter
converts all 75 routed-expert layers from checkpoint block-FP8 to MoK MXFP8 at
load time. The shared expert remains BF16 because that is the current MoK API.

## Run

Build MoK for SM103 first, then provide the two external paths explicitly:

```bash
export MOK_ROOT=/path/to/mixture-of-kittens
export MODEL_PATH=/path/to/GLM-5.2-FP8
export MOK_SGLANG_LOG_PATH=/tmp/mok_glm52_prefill.log
./glm52_opt/mok_prefill/launch_prefill_server.sh
```

The launcher defaults to the validated values: all sparse layers, target
M=1024, 32 communication SMs, minibatch 2560, macrobatch 20480, and schedule
capacity multiplier 1.0. Set `MOK_SGLANG_PREFILL=0` with the same command to
launch the native block-FP8/DeepEP baseline.

After the service is healthy, run the deterministic smoke probe:

```bash
python3 glm52_opt/mok_prefill/deterministic_4k_probe.py \
  --output /tmp/deterministic_mok_output.json
```

The adapter is loaded via `sitecustomize.py` only when
`MOK_SGLANG_PREFILL=1`. No SGLang class is changed when the switch is off.

## What the adapter does

1. Intercepts FP8 MoE post-load before DeepGEMM requantization.
2. Dequantizes 128x128 checkpoint block-FP8 weights to BF16 and converts the
   routed experts to MoK's MXFP8 data/scale layout.
3. Loads the separate GLM shared expert as BF16 and compensates for the outer
   routed scaling factor.
4. Synchronizes the maximum local M across the EP group. Ranks with fewer
   tokens are zero-padded; dummy routes have zero weights and expert IDs are
   striped over all 256 experts to avoid capacity hot spots.
5. Calls MoK schedule + fused forward and slices each rank back to its original
   local token count. Non-enabled token counts fall back to native DeepEP.

## Measured 4K result

Ten post-warmup HTTP-to-first-token runs were collected for each path. The
input cache was disabled and all 32768 prompt tokens were processed on every
run. Raw numbers are in `results_4k_b8.json`.

| Metric | Native FP8 + DeepEP | MoK MXFP8, 75 layers | Change |
|---|---:|---:|---:|
| Batch latency mean | 1.24567 s | 1.01235 s | -18.73% |
| Batch latency median | 1.33710 s | 1.01290 s | -24.25% |
| Batch latency P90 | 1.35001 s | 1.01660 s | -24.70% |
| Input throughput mean | 26575.12 tok/s | 32375.15 tok/s | +21.83% |
| Input throughput median | 24511.05 tok/s | 32356.46 tok/s | +32.01% |

Eight fixed 4096-token requests produced identical greedy first-token IDs on
the native and MoK services (8/8 exact match). This is a smoke check, not a
replacement for logits, long-generation, or task-level quality evaluation.

## Known limitations

- The prototype retains native FP8 weights for fallback and adds MoK weights.
  SGLang reported 205.03 GB/card versus 109.48 GB/card for native loading.
- The shape synchronization is an extra collective and currently also runs
  before fallback on non-target shapes. Production integration should gate on
  prefill forward mode and fuse this value with SGLang's existing MLP sync.
- The adapter is specialized for the validated GLM-5.2 expert layout and uses
  private SGLang attributes. It intentionally fails fast on incompatible
  layouts.
- Decode local M was 8-16 in the measured PD service, below MoK's minimum 512.
  Decode must continue to use the native DeepEP path.
- A PD deployment should let the prefill worker keep only the MoK weight format
  and let the decode worker keep native block-FP8 weights.
