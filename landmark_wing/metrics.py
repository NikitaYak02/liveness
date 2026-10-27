"""Normalized mean error over visible internal landmarks."""

from __future__ import annotations

import torch

from .scheme import NORM_LEFT, NORM_RIGHT


def interocular_distance(landmarks: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(
        landmarks[..., NORM_LEFT, :] - landmarks[..., NORM_RIGHT, :],
        dim=-1,
    )


def nme(
    prediction: torch.Tensor,
    target: torch.Tensor,
    visibility: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-sample NME over visible points; shape ``[B]``."""
    if prediction.shape != target.shape:
        raise ValueError(f"shape mismatch: {prediction.shape} vs {target.shape}")
    denom = interocular_distance(target).clamp_min(1e-6)
    point_err = torch.linalg.norm(prediction - target, dim=-1)
    if visibility is None:
        mean_err = point_err.mean(dim=-1)
    else:
        weights = visibility
        mean_err = (point_err * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)
    return mean_err / denom
