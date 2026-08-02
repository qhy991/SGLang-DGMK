#!/usr/bin/env python3
"""Validate a prebuilt GLM-5.2 FlashMLA stack against installed stock.

The probe intentionally uses a non-promotional KV page count.  It therefore
checks both the r2a+c2 math and the dynamic-page ABI that production serving
needs, while retaining the exact FP8 KV layout, sparse top-k and scheduler ABI.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--so", type=Path, required=True)
    parser.add_argument("--m", type=int, choices=(16, 32), required=True)
    parser.add_argument("--pages", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--nonzero-storage-offsets",
        action="store_true",
        help=(
            "Place every non-empty launch tensor in an aligned leading-dimension "
            "view to exercise the serving graph-buffer ABI."
        ),
    )
    return parser.parse_args()


def load_extension(path: Path):
    path = path.expanduser().resolve()
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load extension {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def compare(torch, reference, candidate, name: str) -> dict[str, object]:
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise AssertionError(f"{name}: shape or dtype mismatch")
    ref_f = reference.float()
    cand_f = candidate.float()
    ref_bad = ~torch.isfinite(ref_f)
    cand_bad = ~torch.isfinite(cand_f)
    if not torch.equal(ref_bad, cand_bad):
        raise AssertionError(f"{name}: anomaly positions differ")
    finite = ~ref_bad
    max_abs = (
        float((cand_f[finite] - ref_f[finite]).abs().max().item())
        if finite.any()
        else 0.0
    )
    if finite.any():
        torch.testing.assert_close(
            cand_f[finite], ref_f[finite], rtol=2e-2, atol=2e-2
        )
    return {
        "exact": bool(torch.equal(reference, candidate)),
        "max_abs": max_abs,
        "finite_elements": int(finite.sum().item()),
        "anomaly_elements": int(ref_bad.sum().item()),
        "anomaly_positions_match": True,
    }


def main() -> int:
    args = parse_args()
    so_path = args.so.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not so_path.is_file():
        raise FileNotFoundError(so_path)
    if output_path.exists():
        raise RuntimeError(f"refusing to overwrite evidence: {output_path}")
    if args.pages <= 0:
        raise ValueError("--pages must be positive")

    import torch
    from sgl_kernel.flash_mla import flash_mla_with_kvcache, get_mla_metadata
    from sglang.srt.layers.attention.dsa.quant_k_cache import quantize_k_cache

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    extension = load_extension(so_path)
    m = args.m
    page_size = 64
    topk = 2048
    usable_tokens = args.pages * page_size - page_size
    context = usable_tokens // m
    if context < topk:
        raise ValueError(
            f"{args.pages} pages provide only {context} tokens/request; need {topk}"
        )

    def generator(seed: int):
        out = torch.Generator(device=device)
        out.manual_seed(seed)
        return out

    def aligned_offset_view(tensor, alignment: int = 16):
        """Copy a contiguous tensor into an aligned, nonzero-offset view."""

        if tensor.ndim == 0 or tensor.shape[0] == 0:
            raise ValueError("offset-view fixture requires a non-empty leading dimension")
        row_bytes = tensor.stride(0) * tensor.element_size()
        leading_rows = next(
            rows for rows in range(1, alignment + 1) if rows * row_bytes % alignment == 0
        )
        base_shape = (tensor.shape[0] + leading_rows, *tensor.shape[1:])
        base = torch.empty(base_shape, dtype=tensor.dtype, device=tensor.device)
        view = base.narrow(0, leading_rows, tensor.shape[0])
        view.copy_(tensor)
        if view.storage_offset() <= 0 or not view.is_contiguous():
            raise AssertionError("failed to construct a contiguous nonzero-offset view")
        if view.data_ptr() % alignment:
            raise AssertionError(
                f"offset view does not preserve {alignment}-byte alignment"
            )
        return view

    q = (
        torch.randn(
            (m, 1, 64, 576),
            dtype=torch.bfloat16,
            device=device,
            generator=generator(202608021),
        )
        * 0.05
    )
    logical_kv = (
        torch.randn(
            (args.pages, page_size, 1, 576),
            dtype=torch.bfloat16,
            device=device,
            generator=generator(202608022),
        )
        * 0.05
    )
    kv = quantize_k_cache(logical_kv)
    del logical_kv

    positions = (torch.arange(topk, device=device, dtype=torch.int64) * 4051) % context
    sequence_bases = page_size + torch.arange(m, device=device) * context
    indices = (sequence_bases[:, None] + positions[None, :]).to(torch.int32)
    indices = indices.unsqueeze(1).contiguous()
    cache_seqlens = torch.full((m,), topk, dtype=torch.int32, device=device)
    metadata, num_splits = get_mla_metadata(
        cache_seqlens=cache_seqlens,
        num_q_tokens_per_head_k=64,
        num_heads_k=1,
        num_heads_q=64,
        is_fp8_kvcache=True,
        topk=topk,
    )
    block_table = torch.empty((m, 0), dtype=torch.int32, device=device)

    if args.nonzero_storage_offsets:
        q = aligned_offset_view(q)
        kv = aligned_offset_view(kv)
        indices = aligned_offset_view(indices)
        metadata = aligned_offset_view(metadata, alignment=32)
        num_splits = aligned_offset_view(num_splits)
        if metadata.data_ptr() % 32:
            raise AssertionError("scheduler metadata view must remain 32-byte aligned")

    out = torch.empty((m, 1, 64, 512), dtype=torch.bfloat16, device=device)
    lse_base = torch.empty((m, 1, 64), dtype=torch.float32, device=device)
    lse = lse_base.transpose(1, 2)
    lse_accum = torch.empty((m + 148, 1, 64), dtype=torch.float32, device=device)
    o_accum = torch.empty(
        (m + 148, 1, 64, 512), dtype=torch.float32, device=device
    )
    if args.nonzero_storage_offsets:
        out = aligned_offset_view(out)
        lse_base = aligned_offset_view(lse_base)
        lse = lse_base.transpose(1, 2)
        lse_accum = aligned_offset_view(lse_accum)
        o_accum = aligned_offset_view(o_accum)

    def stock():
        return flash_mla_with_kvcache(
            q=q,
            k_cache=kv,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            head_dim_v=512,
            tile_scheduler_metadata=metadata,
            num_splits=num_splits,
            softmax_scale=0.0625,
            causal=False,
            is_fp8_kvcache=True,
            indices=indices,
        )

    def candidate():
        extension.launch(
            q,
            kv,
            indices,
            metadata,
            num_splits,
            out,
            lse_base,
            lse_accum,
            o_accum,
        )
        return out, lse

    for _ in range(3):
        stock()
        candidate()
    torch.cuda.synchronize(device)

    cases: list[dict[str, object]] = []

    def run_case(name: str) -> None:
        out.fill_(float("nan"))
        lse_base.fill_(float("nan"))
        q_before = q.clone()
        indices_before = indices.clone()
        reference_out, reference_lse = stock()
        candidate_out, candidate_lse = candidate()
        torch.cuda.synchronize(device)
        cases.append(
            {
                "name": name,
                "output": compare(torch, reference_out, candidate_out, "output"),
                "lse": compare(torch, reference_lse, candidate_lse, "lse"),
                "poison_overwritten": bool(
                    not torch.isnan(candidate_out).any()
                    and not torch.isnan(candidate_lse).any()
                ),
                "q_immutable": bool(torch.equal(q, q_before)),
                "indices_immutable": bool(torch.equal(indices, indices_before)),
            }
        )

    run_case("random_affine_indices")
    indices.copy_(indices.flip(-1))
    run_case("reverse_indices")
    indices.copy_(indices[..., :1].expand_as(indices))
    run_case("duplicate_indices")
    indices[..., 2::3] = -1
    run_case("duplicate_with_minus_one")

    all_passed = all(
        bool(case["output"]["exact"])
        and bool(case["lse"]["exact"])
        and bool(case["poison_overwritten"])
        and bool(case["q_immutable"])
        and bool(case["indices_immutable"])
        for case in cases
    )
    evidence = {
        "schema_version": 1,
        "stage": "dynamic_page_prebuilt_correctness",
        "nonzero_storage_offsets": bool(args.nonzero_storage_offsets),
        "all_cases_passed": all_passed,
        "m": m,
        "pages": args.pages,
        "context_capacity_per_request": context,
        "topk": topk,
        "q": {
            "shape": list(q.shape),
            "stride": list(q.stride()),
            "dtype": str(q.dtype),
            "storage_offset": int(q.storage_offset()),
        },
        "kv": {
            "shape": list(kv.shape),
            "stride": list(kv.stride()),
            "dtype": str(kv.dtype),
            "storage_offset": int(kv.storage_offset()),
        },
        "metadata": {
            "shape": list(metadata.shape),
            "stride": list(metadata.stride()),
            "storage_offset": int(metadata.storage_offset()),
        },
        "launch_storage_offsets": {
            "q": int(q.storage_offset()),
            "kv": int(kv.storage_offset()),
            "indices": int(indices.storage_offset()),
            "metadata": int(metadata.storage_offset()),
            "num_splits": int(num_splits.storage_offset()),
            "out": int(out.storage_offset()),
            "lse_base": int(lse_base.storage_offset()),
            "lse_accum": int(lse_accum.storage_offset()),
            "o_accum": int(o_accum.storage_offset()),
        },
        "num_splits": num_splits.detach().cpu().tolist(),
        "extension": {
            "path": str(so_path),
            "size_bytes": so_path.stat().st_size,
            "sha256": hashlib.sha256(so_path.read_bytes()).hexdigest(),
        },
        "launch_count": (
            int(extension.launch_count())
            if hasattr(extension, "launch_count")
            else None
        ),
        "cases": cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0 if all_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
