# GLM-5.2 TP AllReduce static reachability

Status: **CPU-only source map; runtime reachability is not yet proven.**

Analysis date: 2026-07-22

Source identities:

- SGLang isolated worktree: `/home/qinhaiyan/glm52-goal-runs/24-tp_allreduce_reachability/sglang`
- SGLang base commit at analysis start: `f93f8867b4bc124c9809c9110ec7361ed11b6b4a`
- SGLang landed trace/campaign-audit implementation head before later
  evidence-only hardening: `0b51106d8138e950add18b5ec9cf6915ab9d321e`
- Kernel-Harness isolated worktree: `/home/qinhaiyan/glm52-goal-runs/24-tp_allreduce_reachability/kernel-harness`
- Kernel-Harness base commit at analysis start: `bcd005409e65786af82c86f621507ebef12b2766`
- Kernel-Harness landed pre-campaign head, including the exact TP4
  diagnostic runner: `799765caad984ac2a010f762adaa873a7374018d`
- Production target named by the goal: NVIDIA B200/SM100.

This file records only conclusions supported by files in the two isolated
repositories, plus the separately labeled public architecture config below.
The tracing, exact graph-workload, and campaign-audit implementation is landed
at the heads recorded above. Subsequent analyzer/report hardening is
evidence-only; the locked campaign records the actual final branch heads and
rejects source drift before or during measurement. A public config fact is not
used to assert a selected runtime backend without a hit counter/trace.

## Deployment lanes and process groups

The checked-in B200 FP8 recipes establish two distinct serving layouts:

- Low latency is plain TP8 with EAGLE and no DP-attention or MoE A2A flag
  ([recipe lines 283-296](../../../docs_new/src/snippets/configs/zai-org/glm-5.2.jsx#L283)).
- Balanced is TP8/DP8 with DP attention and DeepEP
  ([recipe lines 299-319](../../../docs_new/src/snippets/configs/zai-org/glm-5.2.jsx#L299)).
- High throughput is also TP8/DP8 with DP attention and DeepEP
  ([recipe lines 322-335](../../../docs_new/src/snippets/configs/zai-org/glm-5.2.jsx#L322)).

For flags omitted by the low-latency recipe, repository defaults are DP=1 and
MoE-DP=1
([server_args.py lines 898-932](../../../python/sglang/srt/server_args.py#L898)),
EP=1 and MoE A2A=`none`
([server_args.py lines 1886-1910](../../../python/sglang/srt/server_args.py#L1886)).

An A2A backend spanning the model-parallel group forces `ep_size=tp_size`
([overrides.py lines 2047-2051](../../../python/sglang/srt/arg_groups/overrides.py#L2047)).
For the balanced/high-throughput recipe, source arithmetic therefore gives:

- `attn_tp_size = tp_size / attn_cp_size / attn_dp_size = 8 / 1 / 8 = 1`
  ([parallel_state.py lines 2243-2245](../../../python/sglang/srt/distributed/parallel_state.py#L2243)).
- `moe_ep_size=8`; because it equals TP size, the MoE EP group aliases the TP
  group ([parallel_state.py lines 2348-2353](../../../python/sglang/srt/distributed/parallel_state.py#L2348)).
- With repository defaults `moe_dp_size=1`, `moe_tp_size=8/8/1=1`
  ([server_args.py lines 926-932](../../../python/sglang/srt/server_args.py#L926),
  [parallel_state.py lines 2313-2315](../../../python/sglang/srt/distributed/parallel_state.py#L2313)).

The full-world TP group is created with the default optional communicators
([parallel_state.py lines 2177-2199](../../../python/sglang/srt/distributed/parallel_state.py#L2177)).
A separate attention-TP group explicitly disables custom AR, MSCCL++, and torch
symmetric-memory AR ([parallel_state.py lines 2279-2310](../../../python/sglang/srt/distributed/parallel_state.py#L2279)).
A separate MoE-TP group explicitly disables PyNCCL and custom AR
([parallel_state.py lines 2372-2398](../../../python/sglang/srt/distributed/parallel_state.py#L2372)).

The model entry class `GlmMoeDsaForCausalLM` inherits the DeepSeek-V2 serving
implementation ([glm4_moe.py lines 1466-1468](../../../python/sglang/srt/models/glm4_moe.py#L1466)).
The repository cookbook states 78 transformer layers, 256 routed experts, and
eight active experts per token
([GLM-5.2.mdx line 69](../../../docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx#L69)).

### Public architecture facts, not runtime reachability

The public NVIDIA GLM-5.2 NVFP4 config, read on 2026-07-22, reports these
architecture fields: [nvidia/GLM-5.2-NVFP4 `config.json`](https://huggingface.co/nvidia/GLM-5.2-NVFP4/blob/main/config.json).

| Field | Public config value |
|---|---:|
| `architectures[0]` | `GlmMoeDsaForCausalLM` |
| `hidden_size` | 6144 |
| `intermediate_size` | 12288 |
| `moe_intermediate_size` | 2048 |
| `first_k_dense_replace` | 3 |
| `moe_layer_freq` | 1 |
| `n_routed_experts` | 256 |
| `n_shared_experts` | 1 |
| `num_experts_per_tok` | 8 |
| `num_hidden_layers` | 78 |

Applied to SGLang's source predicate, that public config describes layers 0-2
as dense and layers 3-77 as sparse
([deepseek_v2.py lines 2197-2202](../../../python/sglang/srt/models/deepseek_v2.py#L2197)).
This is useful architecture prior, but it is not proof that a particular FP8 or
NVFP4 deployment loaded the same revision, launch flags, padding mode, graph
mode, communicator, or caller. The running checkpoint's config fields must still
be logged before attaching per-layer call counts to runtime evidence.

## Material callers

All public wrappers reach `GroupCoordinator.all_reduce` through one of these
group accessors:

- generic TP: [communication_op.py lines 18-20](../../../python/sglang/srt/distributed/communication_op.py#L18)
- attention TP: [communication_op.py lines 65-67](../../../python/sglang/srt/distributed/communication_op.py#L65)
- MoE TP: [communication_op.py lines 77-79](../../../python/sglang/srt/distributed/communication_op.py#L77)
- MoE EP: [communication_op.py lines 82-84](../../../python/sglang/srt/distributed/communication_op.py#L82)

### Vocab-parallel input embedding

`DeepseekV2Model` constructs `VocabParallelEmbedding` through
`get_embedding_tp_kwargs` and invokes it at the start of the model forward
([deepseek_v2.py lines 2385-2392](../../../python/sglang/srt/models/deepseek_v2.py#L2385),
[lines 2518-2525](../../../python/sglang/srt/models/deepseek_v2.py#L2518)).
Unless `SGLANG_ENABLE_EMBED_REPLICATION` is enabled, plain TP shards the vocab
table across the full TP group; DP attention instead shards it across the
attention-TP group
([vocab_parallel_embedding.py lines 163-182](../../../python/sglang/srt/layers/vocab_parallel_embedding.py#L163)).
The repository default for embedding replication is false
([environ.py line 1002](../../../python/sglang/srt/environ.py#L1002)), and the
checked-in B200 recipe declares no environment overrides.
After masked local lookup, the embedding calls attention-TP or generic TP
AllReduce on `[token_rows, hidden_size]`
([vocab_parallel_embedding.py lines 501-535](../../../python/sglang/srt/layers/vocab_parallel_embedding.py#L501)).

For GLM-5.2 DSA, attention-input-scattered mode is disallowed by the `not
is_dsa` predicate
([communicator.py lines 260-278](../../../python/sglang/srt/layers/communicator.py#L260)).
Therefore the plain TP8 recipe has one code-mandatory generic Group AllReduce per
target-model embedding forward unless embedding replication is explicitly
enabled. Its immediate consumer is the first decoder layer's input
RMSNorm/attention. In balanced TP8/DP8, `use_attn_tp_group=True` and derived
attention TP is one, so the embedding constructor sets `self.tp_size=1` and the
AllReduce is bypassed
([vocab_parallel_embedding.py lines 224-253](../../../python/sglang/srt/layers/vocab_parallel_embedding.py#L224)).

### DP gather before a FULL dense MLP

The decoder calls `LayerCommunicator.prepare_mlp`
([deepseek_v2.py lines 2247-2249](../../../python/sglang/srt/models/deepseek_v2.py#L2247)).
When attention output must transition from `TP_ATTN_FULL` to `FULL`, the
communicator chooses `_gather_hidden_states_and_residual`
([communicator.py lines 958-969](../../../python/sglang/srt/layers/communicator.py#L958)).
With DP attention it performs RMSNorm before or after `dp_gather_replicate` /
`dp_gather_partial` according to the attention-TP layout
([communicator.py lines 1068-1103](../../../python/sglang/srt/layers/communicator.py#L1068)).

The SUM_LEN path zeroes a global buffer, copies the local rank's rows, and calls
TP AllReduce ([dp_attention.py lines 379-413](../../../python/sglang/srt/layers/dp_attention.py#L379)).
The exact floating-point statement is:

```python
global_tokens[:] = tensor_model_parallel_all_reduce(global_tokens)
```

Consequences:

- If the returned tensor has different storage, the slice assignment copies the
  full result back into `global_tokens`.
- If returned storage aliases `global_tokens`, the same slice assignment remains
  in the Python ABI but it is a self-assignment at the storage level.
- The coordinator's registered "outplace" wrapper is not sufficient to infer
  storage aliasing. In particular, custom AR V2 push explicitly uses the input as
  output ([custom_all_reduce_push.cuh lines 205-223](../../../python/sglang/jit_kernel/csrc/distributed/custom_all_reduce_push.cuh#L205));
  pull aliases input except graph-captured one-shot pull, which allocates an
  `empty_like` output
  ([custom_all_reduce_pull.cuh lines 140-155](../../../python/sglang/jit_kernel/csrc/distributed/custom_all_reduce_pull.cuh#L140)).
  Record both Python object identity and `data_ptr`/storage alias at runtime.
- The following consumer is post-attention RMSNorm when norm is after the gather,
  otherwise the dense MLP gate/up projection consumes the replicated full buffer.

The MAX_LEN path is not AllReduce: it uses AllGather, with an attention-TP
reduce-scatter first only when attention TP exceeds one
([dp_attention.py lines 416-433](../../../python/sglang/srt/layers/dp_attention.py#L416)).
The dispatcher selects AllGather for MAX_LEN and AllReduce for the remaining
default case ([dp_attention.py lines 507-543](../../../python/sglang/srt/layers/dp_attention.py#L507)).
The optional gatherv route is another non-AllReduce path
([dp_attention.py lines 436-504](../../../python/sglang/srt/layers/dp_attention.py#L436)).

### Dense MLP row-parallel down projection

`DeepseekV2MLP.down_proj` is a reduction-enabled `RowParallelLinear`
([deepseek_v2.py lines 239-274](../../../python/sglang/srt/models/deepseek_v2.py#L239))
and is invoked after activation at
[deepseek_v2.py line 428](../../../python/sglang/srt/models/deepseek_v2.py#L428).
`RowParallelLinear.forward` calls generic TP AllReduce unless an explicit skip,
fusion, or reduce-scatter flag is active
([linear.py lines 1555-1603](../../../python/sglang/srt/layers/linear.py#L1555)).
The following consumer is layer postprocessing/residual handling and then the
next decoder layer.

MAX_LEN DP decode can replace this AllReduce with reduce-scatter:
`should_use_reduce_scatter` returns true for the relevant FULL-to-scattered
postprocess function and MAX_LEN padding
([communicator.py lines 755-770](../../../python/sglang/srt/layers/communicator.py#L755));
the row-parallel layer observes the published skip flag
([moe/utils.py lines 419-427](../../../python/sglang/srt/layers/moe/utils.py#L419)).
This is a code-reachable exclusion, not yet proof that the production decode
trace selected MAX_LEN.

### Attention output before post-attention RMSNorm

The attention O projection is deliberately created with `reduce_results=False`
([deepseek_v2.py lines 2083-2105](../../../python/sglang/srt/models/deepseek_v2.py#L2083));
its `RowParallelLinear` definition preserves that flag
([deepseek_v2.py lines 1695-1705](../../../python/sglang/srt/models/deepseek_v2.py#L1695)).
The reduction is deferred to `LayerCommunicator`:

- generic attention-group AllReduce followed by RMSNorm:
  [communicator.py lines 1104-1131](../../../python/sglang/srt/layers/communicator.py#L1104)
- direct attention-group variant for a dense-TP layout:
  [communicator.py lines 1032-1048](../../../python/sglang/srt/layers/communicator.py#L1032)
- full-TP scattered-residual variant:
  [communicator.py lines 1155-1168](../../../python/sglang/srt/layers/communicator.py#L1155)

The immediate consumer is RMSNorm; its output enters the dense MLP or MoE.

For the plain-TP B200 lane, GLM-5.2 is in the auto-enable allowlist for
FlashInfer AllReduce+RMSNorm fusion when TP>1, DP attention is off, and A2A is
`none` ([overrides.py lines 1461-1507](../../../python/sglang/srt/arg_groups/overrides.py#L1461)).
A successful fused call bypasses `GroupCoordinator.all_reduce`; fallback code
must be distinguished at runtime
([layernorm.py lines 159-213](../../../python/sglang/srt/layers/layernorm.py#L159)).

### Standard, non-A2A MoE output

When `_enable_a2a_moe` is false, the normal and dual-stream MoE paths call TP
AllReduce after routed/shared expert combination:

- [deepseek_v2.py lines 1011-1018](../../../python/sglang/srt/models/deepseek_v2.py#L1011)
- [deepseek_v2.py lines 1133-1140](../../../python/sglang/srt/models/deepseek_v2.py#L1133)

The following consumer is residual/postprocessing and the next layer. Fusion,
reduce-scatter, reduce-scatterv, or an A2A backend can suppress this call
([moe/utils.py lines 430-464](../../../python/sglang/srt/layers/moe/utils.py#L430)).
The decoder publishes the fusion/reduce-scatter flags around the MLP call and
postprocesses only when fusion is absent
([deepseek_v2.py lines 2251-2296](../../../python/sglang/srt/models/deepseek_v2.py#L2251)).
If it marks a tensor for deferred next-layer fusion but the fused implementation
is unavailable at `prepare_attn`, the communicator falls back to MoE-TP Group
AllReduce followed by input RMSNorm
([communicator.py lines 547-581](../../../python/sglang/srt/layers/communicator.py#L547)).
That fallback is a separate runtime hit site even though it reduces the same MLP
output; the following consumer is the next layer's attention.

### DeepEP sparse MoE exclusion

An enabled A2A backend routes `DeepseekV2MoE.forward` to `forward_deepep`
([deepseek_v2.py lines 860-918](../../../python/sglang/srt/models/deepseek_v2.py#L860)).
That function returns the dispatch/expert/combine result without a
`GroupCoordinator.all_reduce`
([deepseek_v2.py lines 1396-1424](../../../python/sglang/srt/models/deepseek_v2.py#L1396)).
This is an A2A region and must not be relabeled as TP AllReduce.

## Proven non-AllReduce sites

- DSA indexer scattered-to-attention-full uses attention-TP AllGather
  ([dsa_indexer.py lines 2514-2528](../../../python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L2514)).
- DSA prefill CP key movement uses `cp_all_gather_rerange_output`
  ([dsa_indexer.py lines 639-653](../../../python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L639),
  [dsa_indexer.py line 2230](../../../python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L2230)).
- DSA CP layer communication is AllGather/reduce-scatter
  ([communicator_dsa_cp.py lines 68-92](../../../python/sglang/srt/layers/communicator_dsa_cp.py#L68)).
- DP MAX_LEN is AllGather/reduce-scatter, and opt-in SUM_LEN gatherv is
  AllGatherv; neither reaches `GroupCoordinator.all_reduce`.
- DeepEP sparse MoE is A2A dispatch/combine.

These exclusions are name-preserving: an optimization of any one of them cannot
be presented as a TP AllReduce optimization.

## Exact BF16 byte calculations

For contiguous BF16 `[M, 6144]`, bytes are `M * 6144 * 2`.

### Fixed direct TP4 diagnostic workloads

These workloads call `GroupCoordinator.all_reduce(inputs["local"])` directly
([Kernel-Harness workloads.py lines 238-339](../../../../kernel-harness/serving_native/workloads.py#L238),
[tp_allreduce_runner.py](../../../../kernel-harness/serving_native/tp_allreduce_runner.py)).

| Workload | Reduced rows | Exact bytes | Binary size | Label |
|---|---:|---:|---:|---|
| `tp4_allreduce_decode_m16` | 16 | 196,608 | 192 KiB | TP4 diagnostic only |
| `tp4_allreduce_decode_m32` | 32 | 393,216 | 384 KiB | TP4 diagnostic only |
| `tp4_allreduce_prefill` | 8192 | 100,663,296 | 96 MiB | TP4 diagnostic only |

They reproduce the public coordinator callable, contiguous BF16 layout, direct
return, and TP4 topology. That local `[M,6144]` shape matches the plain-TP
vocab-parallel embedding and other unfused plain-TP callers, subject to runtime
proof of the phase bucket. The runner now exposes eager default/nondefault and
CUDA-Graph nondefault modes (graph/default fails closed), including coordinator graph capture
([tp_allreduce_runner.py](../../../../kernel-harness/serving_native/tp_allreduce_runner.py)).
It still does not reproduce the DP gather's zero-fill/local-copy/full-buffer-copy
consumer region, and its direct shape remains a standalone diagnostic rather
than production-caller proof
([Kernel-Harness README lines 75-80](../../../../kernel-harness/serving_native/README.md#L75)).

### DP global-buffer messages implied by repository workload constants

The serving-native workload registry fixes decode local M=16/32, production
prefill local M=4096 at DP8, and TP4 prefill local M=8192
([workloads.py lines 30-42](../../../../kernel-harness/serving_native/workloads.py#L30),
[workloads.py lines 238-267](../../../../kernel-harness/serving_native/workloads.py#L238)).
If runtime selects the DP global-buffer AllReduce caller described above, its
message has `global_rows = local_rows * dp_size` for equal local buckets:

| Lane | Local M | DP | Global rows | Exact bytes | Binary size | Evidence class |
|---|---:|---:|---:|---:|---:|---|
| TP4/DP4 decode | 16 | 4 | 64 | 786,432 | 768 KiB | derived code/workload shape; runtime unproven |
| TP4/DP4 decode | 32 | 4 | 128 | 1,572,864 | 1.5 MiB | derived code/workload shape; runtime unproven |
| TP4/DP4 prefill | 8192 | 4 | 32768 | 402,653,184 | 384 MiB | derived code/workload shape; runtime unproven |
| TP8/DP8 decode | 16 | 8 | 128 | 1,572,864 | 1.5 MiB | production topology; external TP8 runtime unproven |
| TP8/DP8 decode | 32 | 8 | 256 | 3,145,728 | 3 MiB | production topology; external TP8 runtime unproven |
| TP8/DP8 prefill | 4096 | 8 | 32768 | 402,653,184 | 384 MiB | production topology; external TP8 runtime unproven |

Thus the fixed direct workload sizes are not interchangeable with the balanced
DP global-buffer ABI. Plain TP without DP retains local `[M,6144]` messages when
the unfused Group call is reached.

## Backend dispatch, aliasing, graph, and stream contracts

The user-facing coordinator decides in-place versus out-of-place before entering
a registered custom op
([parallel_state.py lines 168-186](../../../python/sglang/srt/distributed/parallel_state.py#L168)).
The exact priority is
[parallel_state.py lines 585-751](../../../python/sglang/srt/distributed/parallel_state.py#L585):

1. world-size-one bypass and non-CUDA platform paths;
2. early in-place PyNCCL when NCCL symmetric memory is enabled and MSCCL++ did
   not claim the tensor;
3. custom AR (`ca`);
4. quick AR (`qr`);
5. PyMSCCL++;
6. torch symmetric memory;
7. out-of-place PyNCCL for piecewise CUDA Graph;
8. registered in-place fallback.

Concrete implementation symbols are selected at
[parallel_state.py lines 828-885](../../../python/sglang/srt/distributed/parallel_state.py#L828):

| Route | Concrete symbol | Output alias contract | Principal source predicates |
|---|---|---|---|
| early symmetric-memory NCCL | `pynccl_comm.all_reduce` | input aliases output | PyNCCL exists; symmetric-memory allocator enabled; MSCCL++ did not claim input |
| custom | `ca_comm.custom_all_reduce` | V2 eager push/pull and two-shot storage-alias input; graph-captured one-shot pull is distinct; measure runtime pointer | enabled; MSCCL++ did not claim input; `should_custom_ar` |
| quick | `qr_comm.quick_all_reduce` | distinct tensor | enabled and `should_quick_allreduce` |
| MSCCL++ | `pymscclpp_comm.all_reduce` | aliases input despite outplace wrapper | enabled and tuned eligible configuration |
| torch symmetric memory | `torch_symm_mem_comm.all_reduce` | normally distinct; in-place fallback passes `out=input` | explicitly enabled, BF16/device/alignment/size eligible |
| PyNCCL piecewise | `pynccl_comm.outplace_all_reduce` | distinct tensor | piecewise graph and PyNCCL exists after prior routes decline |
| in-place PyNCCL | `pynccl_comm.all_reduce` | aliases input | PyNCCL currently enabled |
| c10d | `torch.distributed.all_reduce` | aliases input | final eager fallback |

Defaults are custom enabled, MSCCL++ disabled, and torch symmetric-memory AR
disabled ([parallel_state.py lines 1914-1931](../../../python/sglang/srt/distributed/parallel_state.py#L1914));
server arguments set them before group creation
([model_runner.py lines 1251-1253](../../../python/sglang/srt/model_executor/model_runner.py#L1251)).

On CUDA, custom AR V2 is the default implementation when its topology preflight
passes ([custom_all_reduce.py lines 344-369](../../../python/sglang/srt/distributed/device_communicators/custom_all_reduce.py#L344)).
Preflight requires a single-node supported world size, full NVLink for world
sizes above two, and P2P access
([custom_all_reduce_utils.py lines 415-485](../../../python/sglang/srt/distributed/device_communicators/custom_all_reduce_utils.py#L415)).
Per-message eligibility requires a 16-byte-multiple, weak-contiguous payload no
larger than the communicator buffer
([custom_all_reduce_v2.py lines 134-144](../../../python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py#L134)).
Custom V2's DLPack return can be a different Python tensor object while sharing
input storage, so dispatch-classification as a registered outplace custom op is
not an alias guarantee.
The default maximum is 16 MiB; B200 thresholds are 2 MiB for TP4 and 720 KiB for
TP8 before two-shot pull
([custom_all_reduce_v2.py lines 38-75](../../../python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py#L38),
[lines 194-210](../../../python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py#L194)).
Consequently, by source thresholds only:

- direct TP4 192/384 KiB decode selects one-shot push;
- direct TP4 96 MiB prefill is too large and falls back;
- derived TP4 768 KiB/1.5 MiB DP decode remains one-shot push;
- derived TP8 1.5/3 MiB DP decode selects two-shot pull;
- derived 384 MiB DP prefill falls back.

These are dispatch predictions, not measured backend identity.

Quick AR is ROCm-only at construction and therefore unreachable on B200
([quick_all_reduce.py lines 27-38](../../../python/sglang/srt/distributed/device_communicators/quick_all_reduce.py#L27)).
PyMSCCL++ supports world sizes 8/16/32 and FP32/FP16/BF16 and rejects piecewise
graph/compile/capture-stream phases
([pymscclpp.py lines 22-24](../../../python/sglang/srt/distributed/device_communicators/pymscclpp.py#L22),
[lines 328-377](../../../python/sglang/srt/distributed/device_communicators/pymscclpp.py#L328)).
Torch symmetric memory supports BF16 and exact per-SM/world size ceilings; on
SM100 those ceilings are 64 MiB at TP4 and 128 MiB at TP8
([all_reduce_utils.py lines 3-15](../../../python/sglang/srt/distributed/device_communicators/all_reduce_utils.py#L3),
[torch_symm_mem.py lines 112-170](../../../python/sglang/srt/distributed/device_communicators/torch_symm_mem.py#L112)).
PyNCCL uses the current device stream for both in-place and out-of-place calls
([pynccl.py lines 123-179](../../../python/sglang/srt/distributed/device_communicators/pynccl.py#L123)).

Graph capture switches to a capture stream only after that stream waits for the
previous current stream, registers custom-AR capture state, and temporarily
enables PyNCCL/MSCCL++
([parallel_state.py lines 523-583](../../../python/sglang/srt/distributed/parallel_state.py#L523)).
Therefore backend, stream, capture state, and alias behavior are one combined
ABI; an eager c10d comparison is not automatically the production reference.

## Static lane disposition

### Balanced/high-throughput TP8/DP8/DeepEP

- Sparse-layer MoE regions use DeepEP A2A and are excluded from Group AllReduce.
- A dense-layer attention-to-MLP transition can reach DP global-buffer AllReduce
  in SUM_LEN mode.
- MAX_LEN switches that transition to AllGather and can switch the dense MLP
  output to reduce-scatter, potentially eliminating both suspected Group calls.
- The exact dense-layer count, padding mode per phase/bucket, CUDA Graph mode,
  and resulting call frequency remain runtime blockers.

### Low-latency plain TP8

- Vocab-parallel embedding contributes one local `[M,6144]` Group AllReduce per
  target-model forward unless embedding replication is explicitly enabled.
- Attention O-projection reduction and non-A2A MoE/dense-MLP reduction are code
  reachable with local `[M,6144]` BF16 messages.
- FlashInfer fused AllReduce+RMSNorm can bypass the generic Group call.
- Runtime must prove which calls survive fusion, including the terminal layer
  and EAGLE/MTP worker.

### Four-card lane

- The repository contains exact TP4 direct coordinator workloads for M16, M32,
  and M8192.
- They support eager default/nondefault and graph nondefault-stream diagnostics,
  but remain
  independent TP4 direct-call tests. They cannot prove TP8 topology, thresholds,
  kernels, caller reachability, or production acceptance.

## Missing runtime evidence and blockers

The following must remain explicitly unresolved until a wrapper-locked trace:

1. Loaded model identity and config fields: architecture, hidden size,
   `num_hidden_layers`, `first_k_dense_replace`, `moe_layer_freq`, and routed
   expert count.
2. Exact launch flags after argument resolution, including TP/DP/EP, A2A,
   custom-AR/MSCCL++/torch-symmetric-memory/symmetric-allocator flags, FlashInfer
   fusion, DP padding/gatherv/reduce-scatterv, and graph settings.
3. For every hit: caller tag, group ranks, dtype, shape/stride, bytes, graph
   capture/replay state, current stream, branch-observed selected backend,
   inferred algorithm plus inference source, input/output pointers, alias
   result, and following kernel/consumer.
4. Physical topology and full-NVLink/P2P result.
5. Kernel-Harness import resolution. The runner prefers the isolated sibling
   SGLang worktree and verifies the imported package remains below it
   ([runner.py lines 31-49](../../../../kernel-harness/serving_native/runner.py#L31),
   [lines 107-115](../../../../kernel-harness/serving_native/runner.py#L107));
   saved runtime evidence must retain the resolved path and SHA.
6. Production CUDA Graph replay. The direct coordinator harness can capture and
   replay its callable, but a Python/NVTX event during graph capture does not
   prove replay at a real model caller; use backend kernel names plus the code
   mapping or a capture-safe device hit counter.
7. Production TP8 acceptance. This host exposes only four physical GPUs under
   the required lock wrapper, so TP8/DP8/EP8 remains an external validation gate.

The implemented diagnostic is import-time env-gated and does no file I/O or
CUDA synchronization in the collective path. It buffers one JSON row per unique
bounded caller chain, rank/group, tensor ABI, graph state, entry stream, and
selected backend; caller-chain and stream identity are part of the pre-dedup
key so generic wrappers and stream transitions do not collapse. The fallback
backend is explicitly labeled as a pre-dispatch mirror rather than an observed
inner custom-op dispatch. Python is not re-entered during graph replay, so the
tracer does not claim replay proof; stable backend kernel names in Nsight Systems
must provide that evidence. Its process-lifetime dedup state also makes it a
short diagnostic, not an indefinitely enabled production-server facility.
Explicit flush is performed only at the runner's quiescent post-measurement
point because serialization and file I/O hold the recorder lock.

The exact authorized future commands are maintained separately in
[commands.md](commands.md).
