# N6 reproduction contract

N6 is a three-part treatment. Do not benchmark a partial configuration and
label it N6, and do not compare it with a result from another context length or
decode workload.

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

If the baseline anchor is unhealthy, stop and diagnose the host. A relative
candidate win on an unhealthy anchor does not replace N6.
