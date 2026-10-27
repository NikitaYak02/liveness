"""Unit tests that do not require public landmark datasets on disk."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import torch

from landmark_wing.data import image_to_tensor
from landmark_wing.heatmaps import make_gaussian_heatmaps, soft_argmax
from landmark_wing.loss import LandmarkHeatmapLoss, WingLoss
from landmark_wing.mapping import map_landmarks
from landmark_wing.model import HRNetFace
from landmark_wing.scheme import FLIP_INDICES, IMAGE_SIZE, NUM_LANDMARKS
from landmark_wing.vis import make_grid, save_epoch_grid


def test_wing_loss_small_residual_uses_log_branch() -> None:
    loss_fn = WingLoss(width=10.0, epsilon=2.0)
    residual = 1.0
    expected = 10.0 * math.log1p(residual / 2.0)
    prediction = torch.tensor([[[0.0, 0.0]]], dtype=torch.float32)
    target = torch.tensor([[[residual, 0.0]]], dtype=torch.float32)
    value = float(loss_fn(prediction, target).item())
    assert abs(value - 0.5 * expected) < 1e-5


def test_wing_loss_large_residual_uses_linear_branch() -> None:
    loss_fn = WingLoss(width=10.0, epsilon=2.0)
    residual = 20.0
    expected = residual - loss_fn.constant
    prediction = torch.tensor([[[0.0, 0.0]]], dtype=torch.float32)
    target = torch.tensor([[[residual, 0.0]]], dtype=torch.float32)
    value = float(loss_fn(prediction, target).item())
    assert abs(value - 0.5 * expected) < 1e-5


def test_flip_indices_are_permutation() -> None:
    assert FLIP_INDICES.shape == (NUM_LANDMARKS,)
    assert set(FLIP_INDICES.tolist()) == set(range(NUM_LANDMARKS))
    assert np.array_equal(FLIP_INDICES[FLIP_INDICES], np.arange(NUM_LANDMARKS))


def test_hrnet_forward_and_backward() -> None:
    model = HRNetFace()
    images = torch.randn(2, 1, IMAGE_SIZE, IMAGE_SIZE)
    targets = torch.rand(2, NUM_LANDMARKS, 2) * float(IMAGE_SIZE)
    visibility = torch.ones(2, NUM_LANDMARKS)
    heatmaps = model(images)
    assert heatmaps.shape == (2, NUM_LANDMARKS + 1, IMAGE_SIZE, IMAGE_SIZE)
    loss = LandmarkHeatmapLoss()(heatmaps, targets, visibility)
    loss.backward()
    assert model.head_out.weight.grad is not None


def test_soft_argmax_near_peak() -> None:
    landmarks = torch.tensor([[[40.0, 60.0]] + [[10.0, 10.0]] * (NUM_LANDMARKS - 1)])
    visibility = torch.ones(1, NUM_LANDMARKS)
    heat = make_gaussian_heatmaps(landmarks, visibility, size=IMAGE_SIZE, sigma=1.5)
    decoded = soft_argmax(heat)
    assert torch.allclose(decoded[0, 0], landmarks[0, 0], atol=1.5)


def test_map_300w_has_no_pupils() -> None:
    # Synthetic Multi-PIE-like 68 points on a grid.
    pts = np.stack([np.linspace(10, 110, 68), np.linspace(20, 100, 68)], axis=1).astype(np.float32)
    mapped = map_landmarks(pts, "300w")
    assert "eye_left_outer" in mapped
    assert "pupil_left" not in mapped
    assert "pupil_right" not in mapped
    assert "chin" in mapped


def test_save_epoch_grid_writes_png() -> None:
    model = HRNetFace()
    gray = np.full((IMAGE_SIZE, IMAGE_SIZE), 120, dtype=np.uint8)
    images = torch.stack([image_to_tensor(gray), image_to_tensor(gray)], dim=0)
    targets = torch.rand(2, NUM_LANDMARKS, 2) * float(IMAGE_SIZE)
    visibility = torch.ones(2, NUM_LANDMARKS)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "epoch_001.png"
        save_epoch_grid(model, images, targets, out, visibility=visibility, cols=2)
        assert out.is_file()
        assert out.stat().st_size > 0


def test_make_grid_shape() -> None:
    tiles = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(5)]
    grid = make_grid(tiles, cols=3)
    assert grid.shape == (32, 48, 3)
