"""Unit tests that do not require public landmark datasets on disk."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import torch

from landmark_wing.data import (
    FLIP_INDICES_68,
    FLIP_INDICES_98,
    FLIP_INDICES_106,
    flip_indices_for,
    image_to_tensor,
)
from landmark_wing.loss import WingLoss
from landmark_wing.metrics import nme
from landmark_wing.model import MobileNetV2Landmarks
from landmark_wing.vis import make_grid, save_epoch_grid


def test_wing_loss_small_residual_uses_log_branch() -> None:
    loss_fn = WingLoss(width=10.0, epsilon=2.0)
    residual = 1.0
    expected = 10.0 * math.log1p(residual / 2.0)
    prediction = torch.tensor([[[0.0, 0.0]]], dtype=torch.float32)
    target = torch.tensor([[[residual, 0.0]]], dtype=torch.float32)
    # Mean over all coordinates: one non-zero and one zero residual.
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


def test_flip_indices_are_permutations() -> None:
    for count, indices in (
        (68, FLIP_INDICES_68),
        (98, FLIP_INDICES_98),
        (106, FLIP_INDICES_106),
    ):
        assert indices.shape == (count,)
        assert set(indices.tolist()) == set(range(count))
        twice = indices[indices]
        assert np.array_equal(twice, np.arange(count))
        assert np.array_equal(flip_indices_for(count), indices)


def test_model_backward_on_random_batch() -> None:
    model = MobileNetV2Landmarks(num_landmarks=68, pretrained=False)
    images = torch.randn(2, 3, 224, 224)
    targets = torch.rand(2, 68, 2) * 224.0
    preds = model(images)
    assert preds.shape == (2, 68, 2)
    loss = WingLoss()(preds, targets)
    loss.backward()
    assert model.head.weight.grad is not None


def test_nme_zero_when_equal() -> None:
    base = torch.zeros(1, 68, 2)
    base[0, 36] = torch.tensor([10.0, 10.0])
    base[0, 45] = torch.tensor([30.0, 10.0])
    value = float(nme(base, base).item())
    assert value == 0.0


def test_save_epoch_grid_writes_png() -> None:
    model = MobileNetV2Landmarks(num_landmarks=68, pretrained=False)
    rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    rgb[:, :] = (40, 80, 120)
    images = torch.stack([image_to_tensor(rgb), image_to_tensor(rgb)], dim=0)
    targets = torch.rand(2, 68, 2) * 224.0
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "epoch_001.png"
        save_epoch_grid(model, images, targets, out, cols=2)
        assert out.is_file()
        assert out.stat().st_size > 0


def test_make_grid_shape() -> None:
    tiles = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(5)]
    grid = make_grid(tiles, cols=3)
    assert grid.shape == (32, 48, 3)
