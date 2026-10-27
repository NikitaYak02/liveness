"""Normalized mean error for facial landmarks."""

from __future__ import annotations

import torch

from .data import NORM_INDICES


def interocular_distance(landmarks: torch.Tensor, left: int, right: int) -> torch.Tensor:
    """``landmarks`` is ``[..., N, 2]``; returns ``[...]`` distances."""
    return torch.linalg.norm(landmarks[..., left, :] - landmarks[..., right, :], dim=-1)


def nme(
    prediction: torch.Tensor,
    target: torch.Tensor,
    left: int | None = None,
    right: int | None = None,
) -> torch.Tensor:
    """Per-sample NME; returns shape ``[B]``.

    If eye indices are omitted they are taken from ``NORM_INDICES`` for the
    landmark count.
    """
    if prediction.shape != target.shape:
        raise ValueError(f"shape mismatch: {prediction.shape} vs {target.shape}")
    num_landmarks = prediction.shape[-2]
    if left is None or right is None:
        if num_landmarks not in NORM_INDICES:
            raise ValueError(f"no default norm indices for N={num_landmarks}")
        left, right = NORM_INDICES[num_landmarks]
    denom = interocular_distance(target, left, right).clamp_min(1e-6)
    point_error = torch.linalg.norm(prediction - target, dim=-1).mean(dim=-1)
    return point_error / denom


def mean_nme(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(nme(prediction, target).mean().item())
