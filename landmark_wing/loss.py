"""Wing loss on decoded coords + MSE on Gaussian heatmaps."""

from __future__ import annotations

import math

import torch
from torch import nn

from .heatmaps import make_gaussian_heatmaps, soft_argmax
from .scheme import CHANNEL_OFFSET, NUM_LANDMARKS


class WingLoss(nn.Module):
    """Piecewise Wing loss (Feng et al., CVPR 2018)."""

    def __init__(self, width: float = 10.0, epsilon: float = 2.0) -> None:
        super().__init__()
        if width <= 0:
            raise ValueError(f"width must be positive, got {width}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        self.width = float(width)
        self.epsilon = float(epsilon)
        self.constant = self.width - self.width * math.log(1.0 + self.width / self.epsilon)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        visibility: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = (prediction - target).abs()
        wing = self.width * torch.log1p(residual / self.epsilon)
        linear = residual - self.constant
        per_coord = torch.where(residual < self.width, wing, linear)
        if visibility is None:
            return per_coord.mean()
        weights = visibility.unsqueeze(-1).expand_as(per_coord)
        denom = weights.sum().clamp_min(1.0)
        return (per_coord * weights).sum() / denom


class LandmarkHeatmapLoss(nn.Module):
    """NNT-like training signal: heatmap MSE + Wing on soft-argmax points.

    Approximates ``cost.struct_kp`` for PyTorch: Gaussian targets with
    ``tau``-like sigma and coordinate Wing loss on visible landmarks.
    """

    def __init__(
        self,
        wing_w: float = 10.0,
        wing_eps: float = 2.0,
        heatmap_weight: float = 1.0,
        coord_weight: float = 1.0,
        sigma: float = 2.0,
    ) -> None:
        super().__init__()
        self.wing = WingLoss(width=wing_w, epsilon=wing_eps)
        self.heatmap_weight = float(heatmap_weight)
        self.coord_weight = float(coord_weight)
        self.sigma = float(sigma)

    def forward(
        self,
        pred_heatmaps: torch.Tensor,
        landmarks: torch.Tensor,
        visibility: torch.Tensor,
    ) -> torch.Tensor:
        size = pred_heatmaps.shape[-1]
        target = make_gaussian_heatmaps(landmarks, visibility, size=size, sigma=self.sigma)
        # Supervise only point channels; channel 0 stays free.
        pred_pts = pred_heatmaps[:, CHANNEL_OFFSET : CHANNEL_OFFSET + NUM_LANDMARKS]
        tgt_pts = target[:, CHANNEL_OFFSET : CHANNEL_OFFSET + NUM_LANDMARKS]
        vis = visibility.view(-1, NUM_LANDMARKS, 1, 1)
        heatmap_err = ((pred_pts - tgt_pts) ** 2) * vis
        heatmap_loss = heatmap_err.sum() / vis.sum().clamp_min(1.0)

        coords = soft_argmax(pred_heatmaps)
        coord_loss = self.wing(coords, landmarks, visibility)
        return self.heatmap_weight * heatmap_loss + self.coord_weight * coord_loss
