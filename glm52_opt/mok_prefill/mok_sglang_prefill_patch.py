"""Experimental MoK MXFP8 prefill adapter for SGLang GLM-5.2.

This module is intentionally loaded through ``sitecustomize`` and is gated by
``MOK_SGLANG_PREFILL=1``.  It leaves the normal DeepEP/FP8 path untouched for
all shapes except the explicitly enabled prefill token counts.

The adapter targets the GLM-5.2 layout used on B300-M2:

* 256 routed experts, EP8 (32 routed experts per rank)
* one per-rank fused shared-expert slot
* block-FP8 checkpoint weights with 128x128 FP32 scales
* Top-8 routed experts plus the appended shared expert

At model post-load time, selected sparse layers are converted once from the
checkpoint block-FP8 representation to MoK's MXFP8 weight/scaling layout.  At
runtime the adapter inverts SGLang's per-rank shared-slot expert-ID remapping,
then calls the MoK schedule + fused forward path.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist


_INSTALLED = False
_ORIG_FP8_POST_LOAD = None
_ORIG_DEEPEP_FORWARD = None
_ORIG_DSV2_FORWARD_DEEPEP = None
_WEIGHT_MAP = None


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


def _enabled_layers() -> set[int] | None:
    spec = os.environ.get("MOK_SGLANG_LAYERS", "all").strip().lower()
    if spec in ("", "all", "*"):
        return None
    result: set[int] = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            lo, hi = (int(x) for x in item.split("-", 1))
            result.update(range(lo, hi + 1))
        else:
            result.add(int(item))
    return result


def _enabled_tokens() -> set[int]:
    spec = os.environ.get("MOK_SGLANG_PREFILL_TOKENS", "1024")
    return {int(x.strip()) for x in spec.split(",") if x.strip()}


_LAYERS = _enabled_layers()
_TOKENS = _enabled_tokens()
_VALIDATE_LAYER = _env_int("MOK_SGLANG_VALIDATE_LAYER", -1)
_ARM_FILE = os.environ.get("MOK_SGLANG_ARM_FILE")
_LAST_ENABLED_LAYER = 77 if _LAYERS is None else max(_LAYERS)


def _layer_enabled(layer_id: int) -> bool:
    return _LAYERS is None or layer_id in _LAYERS


def _runtime_armed() -> bool:
    return _ARM_FILE is None or Path(_ARM_FILE).exists()


def _synchronized_target_m(forward_batch, local_m: int, device: torch.device) -> int:
    """Return the max local token count across the EP group for this forward."""

    cached = getattr(forward_batch, "_mok_synchronized_target_m", None)
    if cached is not None:
        return int(cached)

    from sglang.srt.distributed import get_moe_ep_group

    target = torch.tensor([local_m], dtype=torch.int64, device=device)
    dist.all_reduce(
        target,
        op=dist.ReduceOp.MAX,
        group=get_moe_ep_group().device_group,
    )
    value = int(target.item())
    forward_batch._mok_synchronized_target_m = value
    return value


def _clear_synchronized_target_m(forward_batch, layer_id: int) -> None:
    if layer_id == _LAST_ENABLED_LAYER and hasattr(
        forward_batch, "_mok_synchronized_target_m"
    ):
        del forward_batch._mok_synchronized_target_m


def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else -1


def _log(message: str, *, all_ranks: bool = False) -> None:
    rank = _rank()
    if all_ranks or rank in (-1, 0):
        print(f"[mok-prefill rank={rank}] {message}", flush=True)


def _prepare_mok_weights(layer) -> None:
    """Convert one loaded SGLang FP8 MoE layer to MoK's forward layout."""

    if getattr(layer, "_mok_prefill_prepared", False):
        return
    layer_id = int(layer.layer_id)
    if not _layer_enabled(layer_id):
        return

    from mok import ops
    from sglang.srt.layers.quantization.fp8_utils import block_quant_dequant

    num_fused_shared = int(layer.num_fused_shared_experts)
    if num_fused_shared not in (0, 1):
        raise RuntimeError(
            "MoK GLM-5.2 adapter supports zero or one fused shared expert; "
            f"got {num_fused_shared}"
        )
    num_local_routed = int(layer._num_local_routed)
    num_local_total = int(layer.num_local_experts)
    if num_local_total != num_local_routed + num_fused_shared:
        raise RuntimeError(
            f"unexpected local expert layout: routed={num_local_routed}, "
            f"fused_shared={num_fused_shared}, total={num_local_total}"
        )
    if tuple(layer.quant_method.weight_block_size) != (128, 128):
        raise RuntimeError(
            f"MoK adapter requires 128x128 block FP8, got "
            f"{layer.quant_method.weight_block_size}"
        )
    if layer.w13_weight_scale_inv.dtype != torch.float32:
        raise RuntimeError(
            "MoK conversion must run before SGLang's DeepGEMM UE8M0 "
            f"requantization; got w13 scale dtype={layer.w13_weight_scale_inv.dtype}"
        )

    t0 = time.perf_counter()
    torch.cuda.synchronize(layer.w13_weight.device)
    before = torch.cuda.memory_allocated(layer.w13_weight.device)

    intermediate = int(layer.intermediate_size_per_partition)
    hidden = int(layer.hidden_size)
    expected_w13 = (num_local_total, 2 * intermediate, hidden)
    expected_w2 = (num_local_total, hidden, intermediate)
    if tuple(layer.w13_weight.shape) != expected_w13:
        raise RuntimeError(
            f"unexpected w13 shape {tuple(layer.w13_weight.shape)}, "
            f"expected {expected_w13}"
        )
    if tuple(layer.w2_weight.shape) != expected_w2:
        raise RuntimeError(
            f"unexpected w2 shape {tuple(layer.w2_weight.shape)}, expected {expected_w2}"
        )

    # w13 follows the conventional [gate (w1), up (w3)] concatenation.
    w13_bf16 = block_quant_dequant(
        layer.w13_weight.data,
        layer.w13_weight_scale_inv.data,
        [128, 128],
        torch.bfloat16,
    )
    gate_bf16 = w13_bf16[:num_local_routed, :intermediate, :].contiguous()
    up_bf16 = w13_bf16[:num_local_routed, intermediate:, :].contiguous()
    if num_fused_shared:
        shared_gate = w13_bf16[num_local_routed, :intermediate, :].contiguous()
        shared_up = w13_bf16[num_local_routed, intermediate:, :].contiguous()
    else:
        shared_gate = shared_up = None

    gate_fp8, gate_scale, _, _ = ops.mxfp8_quantize(gate_bf16, True, False)
    up_fp8, up_scale, _, _ = ops.mxfp8_quantize(up_bf16, True, False)
    del gate_bf16, up_bf16, w13_bf16

    w2_bf16 = block_quant_dequant(
        layer.w2_weight.data,
        layer.w2_weight_scale_inv.data,
        [128, 128],
        torch.bfloat16,
    )
    down_bf16 = w2_bf16[:num_local_routed].contiguous()
    down_fp8, down_scale, _, _ = ops.mxfp8_quantize(down_bf16, True, False)

    # If SGLang applies routed_scaling_factor after the expert kernel, MoK's
    # shared branch must carry its inverse so that the outer multiply leaves
    # shared output unchanged.  Some runners fuse that scale into top-k weights
    # instead, in which case the outer scale is one.
    rsf = float(layer.moe_runner_config.routed_scaling_factor or 1.0)
    outer_scale = (
        1.0
        if bool(layer.should_fuse_routed_scaling_factor_in_topk)
        else rsf
    )
    if num_fused_shared:
        shared_down = (
            w2_bf16[num_local_routed] / outer_scale
        ).to(torch.bfloat16).contiguous()
    else:
        shared_gate, shared_up, shared_down = _load_shared_checkpoint_weights(
            layer_id, layer.w13_weight.device, outer_scale
        )
    del down_bf16, w2_bf16

    layer._mok_routed_gate = (gate_fp8, gate_scale)
    layer._mok_routed_up = (up_fp8, up_scale)
    layer._mok_routed_down = (down_fp8, down_scale)
    layer._mok_shared_gate = shared_gate
    layer._mok_shared_up = shared_up
    layer._mok_shared_down = shared_down
    layer._mok_outer_scale = outer_scale
    layer._mok_prefill_prepared = True
    layer._mok_prefill_hit_logged = False
    layer._mok_prefill_validated = False
    layer._mok_prefill_shape_logged = set()
    layer._mok_prefill_enter_logged = False
    layer._mok_prefill_outer_logged = False

    torch.cuda.synchronize(layer.w13_weight.device)
    elapsed = time.perf_counter() - t0
    after = torch.cuda.memory_allocated(layer.w13_weight.device)
    _log(
        f"prepared layer={layer_id} routed={num_local_routed} H={hidden} "
        f"I={intermediate} fused_shared={num_fused_shared} "
        f"outer_scale={outer_scale:.6g} seconds={elapsed:.3f} "
        f"persistent_delta_gib={(after - before) / 2**30:.3f}"
    )


def _load_shared_checkpoint_weights(
    layer_id: int, device: torch.device, outer_scale: float
):
    """Load the separate GLM shared expert when SGLang fusion is disabled."""

    global _WEIGHT_MAP
    from safetensors import safe_open
    from sglang.srt.layers.quantization.fp8_utils import block_quant_dequant

    model_path_env = os.environ.get("MOK_SGLANG_MODEL_PATH")
    if not model_path_env:
        raise RuntimeError("MOK_SGLANG_MODEL_PATH must name the FP8 checkpoint")
    model_path = Path(model_path_env)
    if _WEIGHT_MAP is None:
        with (model_path / "model.safetensors.index.json").open() as handle:
            _WEIGHT_MAP = json.load(handle)["weight_map"]

    outputs = {}
    for projection in ("gate_proj", "up_proj", "down_proj"):
        base = f"model.layers.{layer_id}.mlp.shared_experts.{projection}"
        weight_key = f"{base}.weight"
        scale_key = f"{base}.weight_scale_inv"
        with safe_open(
            str(model_path / _WEIGHT_MAP[weight_key]), framework="pt", device="cpu"
        ) as handle:
            weight = handle.get_tensor(weight_key).to(device)
        with safe_open(
            str(model_path / _WEIGHT_MAP[scale_key]), framework="pt", device="cpu"
        ) as handle:
            scale = handle.get_tensor(scale_key).to(device)
        value = block_quant_dequant(
            weight, scale, [128, 128], torch.bfloat16
        ).contiguous()
        if projection == "down_proj":
            value = (value / outer_scale).to(torch.bfloat16).contiguous()
        outputs[projection] = value
        del weight, scale
    return outputs["gate_proj"], outputs["up_proj"], outputs["down_proj"]


def _patched_fp8_post_load(self, layer) -> None:
    # Import locally so sitecustomize remains harmless in non-SGLang processes.
    from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE

    if isinstance(layer, DeepEPMoE) and _layer_enabled(int(layer.layer_id)):
        _prepare_mok_weights(layer)
    _ORIG_FP8_POST_LOAD(self, layer)


def _mok_config():
    from mok import functional

    return functional.MoKConfig(
        fwd_num_comm_sms=_env_int("MOK_SGLANG_FWD_COMM_SMS", 32),
        bwd_num_comm_sms=28,
        minibatch_size=_env_int("MOK_SGLANG_MINIBATCH_SIZE", 2560),
        macrobatch_size=_env_int("MOK_SGLANG_MACROBATCH_SIZE", 20480),
        schedule_capacity_multiplier=float(
            os.environ.get("MOK_SGLANG_SCHEDULE_CAPACITY_MULTIPLIER", "1.0")
        ),
        all_gather_top_experts_chunk_bytes=_env_int(
            "MOK_SGLANG_ROUTE_CHUNK_BYTES", 2048
        ),
    )


_CONFIG = None


def _run_mok(layer, hidden_states: torch.Tensor, topk_output):
    global _CONFIG

    from mok import functional
    from sglang.srt.distributed import get_moe_ep_group
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    if not TopKOutputChecker.format_is_standard(topk_output):
        raise RuntimeError(f"MoK requires standard TopKOutput, got {topk_output.format}")

    x = hidden_states.contiguous()
    if x.dtype != torch.bfloat16:
        raise RuntimeError(f"MoK requires BF16 hidden states, got {x.dtype}")
    local_m = int(x.shape[0])
    target_m = int(getattr(layer, "_mok_target_m", local_m))
    if target_m < local_m:
        raise RuntimeError(
            f"MoK synchronized target M={target_m} is smaller than local M={local_m}"
        )

    num_fused_shared = int(layer.num_fused_shared_experts)
    routed_topk = int(layer.top_k) - num_fused_shared
    if num_fused_shared not in (0, 1) or routed_topk != 8:
        raise RuntimeError(
            f"unexpected GLM top-k layout: top_k={layer.top_k}, "
            f"fused_shared={num_fused_shared}"
        )

    routed_ids = topk_output.topk_ids[:, :routed_topk]
    if num_fused_shared:
        # SGLang maps contiguous routed IDs into [32 routed + 1 shared] rank
        # blocks: p = e + floor(e/32).  Invert it for MoK's 0..255 ID space.
        local_stride = int(layer._num_local_routed) + num_fused_shared
        routed_ids = routed_ids - torch.div(
            routed_ids, local_stride, rounding_mode="floor"
        )
    routed_ids = routed_ids.to(torch.int64).contiguous()
    routed_weights = (
        topk_output.topk_weights[:, :routed_topk].to(torch.float32).contiguous()
    )
    group = get_moe_ep_group().device_group

    # SGLang DP-attention keeps the EP ranks in the same MLP iteration but
    # permits different local token counts, including idle M=0 ranks.  MoK
    # requires identical shapes on every EP rank.  Pad each local batch to the
    # synchronized maximum with zero activations and zero routing weights;
    # dummy routes perform no mathematical work and are sliced off afterward.
    if target_m > local_m:
        padded_x = x.new_zeros((target_m, x.shape[1]))
        padded_ids = routed_ids.new_zeros((target_m, routed_topk))
        padded_weights = routed_weights.new_zeros((target_m, routed_topk))
        if local_m:
            padded_x[:local_m].copy_(x)
            padded_ids[:local_m].copy_(routed_ids)
            padded_weights[:local_m].copy_(routed_weights)
        # Zero router weights make dummy tokens mathematically inert, but their
        # IDs still consume schedule capacity.  Stripe them across all routed
        # experts instead of hot-spotting expert zero.
        num_global_routed = int(layer._num_local_routed) * dist.get_world_size(
            group=group
        )
        dummy_ids = torch.arange(
            (target_m - local_m) * routed_topk,
            dtype=torch.int64,
            device=routed_ids.device,
        ).remainder_(num_global_routed)
        padded_ids[local_m:].copy_(dummy_ids.view(-1, routed_topk))
        x = padded_x
        routed_ids = padded_ids
        routed_weights = padded_weights

    if _CONFIG is None:
        _CONFIG = _mok_config()
    workspace = functional.get_workspace(
        _CONFIG,
        group,
        device=x.device,
        num_local_tokens=x.shape[0],
        hidden_size=x.shape[1],
        topk=routed_topk,
    )
    schedule = functional.build_schedule(
        workspace,
        _CONFIG,
        routed_ids,
        num_local_experts=int(layer._num_local_routed),
    )
    output, _context = functional.forward(
        _CONFIG,
        workspace,
        schedule,
        x,
        routed_weights,
        layer._mok_shared_gate,
        layer._mok_shared_up,
        layer._mok_shared_down,
        layer._mok_routed_gate,
        layer._mok_routed_up,
        layer._mok_routed_down,
    )
    return output[:local_m]


def _patched_deepep_forward(self, hidden_states: torch.Tensor, topk_output):
    layer_id = int(self.layer_id)
    m = int(hidden_states.shape[0])
    target_m = int(getattr(self, "_mok_target_m", m))
    if (
        getattr(self, "_mok_prefill_prepared", False)
        and _layer_enabled(layer_id)
        and m not in (1, 64)
        and m not in self._mok_prefill_shape_logged
    ):
        _log(
            f"SHAPE layer={layer_id} M={m} "
            f"topk_shape={tuple(topk_output.topk_ids.shape)}",
            all_ranks=True,
        )
        self._mok_prefill_shape_logged.add(m)
    if (
        not getattr(self, "_mok_prefill_prepared", False)
        or not _layer_enabled(layer_id)
        or target_m not in _TOKENS
        or not _runtime_armed()
    ):
        return _ORIG_DEEPEP_FORWARD(self, hidden_states, topk_output)

    if not self._mok_prefill_enter_logged:
        _log(
            f"ENTER layer={layer_id} local_M={m} target_M={target_m}",
            all_ranks=True,
        )
        self._mok_prefill_enter_logged = True

    if not self._mok_prefill_hit_logged:
        _log(
            f"HIT layer={layer_id} local_M={m} target_M={target_m} "
            f"topk_shape={tuple(topk_output.topk_ids.shape)} "
            f"comm_sms={_env_int('MOK_SGLANG_FWD_COMM_SMS', 32)}"
        )
        self._mok_prefill_hit_logged = True

    if layer_id == _VALIDATE_LAYER and not self._mok_prefill_validated:
        # Both branches run on all ranks in the same order.  The native expert
        # call returns routed-only here because GLM keeps its shared expert
        # outside DeepEPMoE, whereas MoK returns routed+shared.  Add the native
        # shared branch with the same pre-outer scaling before comparing.
        native_routed = _ORIG_DEEPEP_FORWARD(
            self, hidden_states.clone(), topk_output
        )
        shared_module = getattr(self, "_mok_outer_shared_experts", None)
        if shared_module is None:
            raise RuntimeError("missing outer shared-expert module for validation")
        if m:
            native_shared = shared_module(hidden_states.clone())
            native_output = native_routed + native_shared / float(
                self._mok_outer_scale
            )
        else:
            native_output = native_routed
        mok_output = _run_mok(self, hidden_states, topk_output)
        if native_output.numel() == 0:
            validation_message = (
                f"VALIDATE_EMPTY layer={layer_id} local_M=0 target_M={target_m}"
            )
        else:
            diff = (mok_output.float() - native_output.float()).abs()
            denom = native_output.float().abs().mean().clamp_min(1e-12)
            rel = diff.mean() / denom
            cosine = torch.nn.functional.cosine_similarity(
                mok_output.float().reshape(1, -1),
                native_output.float().reshape(1, -1),
            )[0]
            validation_message = (
                f"VALIDATE layer={layer_id} local_M={m} target_M={target_m} "
                f"mae={diff.mean().item():.6e} max={diff.max().item():.6e} "
                f"rel={rel.item():.6e} cosine={cosine.item():.8f} "
                f"expert_id_min={topk_output.topk_ids.min().item()} "
                f"expert_id_max={topk_output.topk_ids.max().item()}"
            )
        _log(validation_message, all_ranks=True)
        self._mok_prefill_validated = True
        return mok_output

    output = _run_mok(self, hidden_states, topk_output)
    if not getattr(self, "_mok_prefill_return_logged", False):
        _log(
            f"RETURN layer={layer_id} local_M={m} target_M={target_m}",
            all_ranks=True,
        )
        self._mok_prefill_return_logged = True
    return output


def _patched_dsv2_forward_deepep(
    self,
    hidden_states: torch.Tensor,
    forward_batch,
    input_ids_global=None,
):
    """Let MoK own the shared branch when SGLang keeps it separate.

    The original method uses ``self.num_fused_shared_experts`` only to decide
    whether it should launch/add the separate shared MLP.  Temporarily setting
    it to one skips that branch while leaving the TopK object unchanged (Top-8
    routed IDs), so the patched ``DeepEPMoE.forward`` can produce routed+shared
    in one MoK call.  The value is restored before returning.
    """

    experts = getattr(self, "experts", None)
    layer_id = int(getattr(self, "layer_id", -1))
    m = int(hidden_states.shape[0])
    global_tokens = getattr(forward_batch, "global_num_tokens_cpu", None)
    target_m = (
        _synchronized_target_m(forward_batch, m, hidden_states.device)
        if experts is not None
        and getattr(experts, "_mok_prefill_prepared", False)
        and _layer_enabled(layer_id)
        and _runtime_armed()
        else m
    )
    use_mok = (
        experts is not None
        and getattr(experts, "_mok_prefill_prepared", False)
        and _layer_enabled(layer_id)
        and target_m in _TOKENS
        and _runtime_armed()
    )
    if not use_mok or int(self.num_fused_shared_experts) != 0:
        try:
            return _ORIG_DSV2_FORWARD_DEEPEP(
                self,
                hidden_states,
                forward_batch,
                input_ids_global=input_ids_global,
            )
        finally:
            _clear_synchronized_target_m(forward_batch, layer_id)

    # Expose the separate shared module to the inner adapter's optional
    # one-shot correctness comparison.  This is only a module reference.
    if not experts._mok_prefill_outer_logged:
        _log(
            f"OUTER layer={layer_id} local_M={m} target_M={target_m} "
            f"global_tokens={global_tokens}",
            all_ranks=True,
        )
        experts._mok_prefill_outer_logged = True
    experts._mok_outer_shared_experts = self.shared_experts
    experts._mok_target_m = target_m
    self.num_fused_shared_experts = 1
    try:
        return _ORIG_DSV2_FORWARD_DEEPEP(
            self, hidden_states, forward_batch, input_ids_global=input_ids_global
        )
    finally:
        self.num_fused_shared_experts = 0
        del experts._mok_target_m
        _clear_synchronized_target_m(forward_batch, layer_id)


def install() -> None:
    global _INSTALLED, _ORIG_FP8_POST_LOAD, _ORIG_DEEPEP_FORWARD
    global _ORIG_DSV2_FORWARD_DEEPEP
    if _INSTALLED:
        return
    if os.environ.get("MOK_SGLANG_PREFILL", "0") != "1":
        return

    from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE
    from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE

    _ORIG_FP8_POST_LOAD = Fp8MoEMethod.process_weights_after_loading
    _ORIG_DEEPEP_FORWARD = DeepEPMoE.forward
    _ORIG_DSV2_FORWARD_DEEPEP = DeepseekV2MoE.forward_deepep
    Fp8MoEMethod.process_weights_after_loading = _patched_fp8_post_load
    DeepEPMoE.forward = _patched_deepep_forward
    DeepseekV2MoE.forward_deepep = _patched_dsv2_forward_deepep
    _INSTALLED = True
    _log(
        f"installed layers={'all' if _LAYERS is None else sorted(_LAYERS)} "
        f"tokens={sorted(_TOKENS)} validate_layer={_VALIDATE_LAYER}"
    )


install()
