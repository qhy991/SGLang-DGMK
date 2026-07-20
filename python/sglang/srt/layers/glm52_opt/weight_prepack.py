"""Runtime weight-scale helpers (no checkpoint mutation)."""

from __future__ import annotations

import torch


def maybe_expand_weight_scale_per_row(
    weight_scale: torch.Tensor, out_features: int
) -> torch.Tensor:
    if weight_scale.ndim == 2 and weight_scale.shape[0] * 128 == out_features:
        return weight_scale
    return weight_scale
