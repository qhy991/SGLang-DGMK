"""DSA attention optimized paths."""

from __future__ import annotations

from sglang.srt.layers.glm52_opt.archive_loader import load_run_fn


def run_dsa_decode(archive_ref: str, inputs: dict):
    return load_run_fn(archive_ref)(inputs)
