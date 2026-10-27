"""Gaussian heatmap targets and soft-argmax decoding."""

from __future__ import annotations

import torch

from .scheme import CHANNEL_OFFSET, HEATMAP_CHANNELS, NUM_LANDMARKS


def make_gaussian_heatmaps(
    landmarks: torch.Tensor,
    visibility: torch.Tensor,
    size: int,
    sigma: float = 2.0,
) -> torch.Tensor:
    """Build ``[B, 31, H, W]`` targets; channel 0 is zeros, points in 1..30."""
    if landmarks.ndim != 3 or landmarks.shape[-1] != 2:
        raise ValueError(f"expected [B, N, 2], got {tuple(landmarks.shape)}")
    batch, num, _ = landmarks.shape
    if num != NUM_LANDMARKS:
        raise ValueError(f"expected {NUM_LANDMARKS} landmarks, got {num}")

    device = landmarks.device
    dtype = landmarks.dtype
    yy = torch.arange(size, device=device, dtype=dtype).view(1, 1, size, 1)
    xx = torch.arange(size, device=device, dtype=dtype).view(1, 1, 1, size)

    xs = landmarks[:, :, 0].view(batch, num, 1, 1)
    ys = landmarks[:, :, 1].view(batch, num, 1, 1)
    heat = torch.exp(-((xx - xs) ** 2 + (yy - ys) ** 2) / (2.0 * sigma * sigma))
    heat = heat * visibility.view(batch, num, 1, 1)

    out = torch.zeros(batch, HEATMAP_CHANNELS, size, size, device=device, dtype=dtype)
    out[:, CHANNEL_OFFSET : CHANNEL_OFFSET + num] = heat
    return out


def soft_argmax(heatmaps: torch.Tensor, channel_offset: int = CHANNEL_OFFSET) -> torch.Tensor:
    """Decode point channels to ``[B, N, 2]`` crop-pixel coordinates.

    Uses spatial L1 normalization of non-negative responses (not softmax over
    the full map), so a sparse Gaussian peak does not collapse to the center.
    """
    points = heatmaps[:, channel_offset : channel_offset + NUM_LANDMARKS]
    batch, num, height, width = points.shape
    positive = points.clamp_min(0)
    mass = positive.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    prob = (positive / mass).view(batch, num, -1)
    xs = torch.arange(width, device=heatmaps.device, dtype=heatmaps.dtype)
    ys = torch.arange(height, device=heatmaps.device, dtype=heatmaps.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid_x = grid_x.reshape(1, 1, -1)
    grid_y = grid_y.reshape(1, 1, -1)
    x = (prob * grid_x).sum(dim=-1)
    y = (prob * grid_y).sum(dim=-1)
    return torch.stack([x, y], dim=-1)


def hard_argmax(heatmaps: torch.Tensor, channel_offset: int = CHANNEL_OFFSET) -> torch.Tensor:
    points = heatmaps[:, channel_offset : channel_offset + NUM_LANDMARKS]
    batch, num, height, width = points.shape
    flat = points.view(batch, num, -1)
    idx = flat.argmax(dim=-1)
    x = (idx % width).to(heatmaps.dtype)
    y = torch.div(idx, width, rounding_mode="floor").to(heatmaps.dtype)
    return torch.stack([x, y], dim=-1)
