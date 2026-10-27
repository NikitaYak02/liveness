"""Prediction overlays for landmark training."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from .data import denormalize_image
from .heatmaps import soft_argmax


def draw_landmarks(
    image_rgb: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray | None = None,
    visibility: np.ndarray | None = None,
    pred_color: tuple[int, int, int] = (255, 64, 64),
    gt_color: tuple[int, int, int] = (64, 220, 64),
    radius: int = 1,
) -> np.ndarray:
    canvas = image_rgb.copy()
    if target is not None:
        for index, (x, y) in enumerate(target):
            if visibility is not None and visibility[index] < 0.5:
                continue
            cv2.circle(canvas, (int(round(x)), int(round(y))), radius, gt_color, -1, lineType=cv2.LINE_AA)
    for index, (x, y) in enumerate(prediction):
        if visibility is not None and visibility[index] < 0.5:
            continue
        cv2.circle(canvas, (int(round(x)), int(round(y))), radius, pred_color, -1, lineType=cv2.LINE_AA)
    return canvas


def make_grid(images: list[np.ndarray], cols: int = 4) -> np.ndarray:
    if not images:
        raise ValueError("no images to grid")
    height, width = images[0].shape[:2]
    cols = max(1, min(cols, len(images)))
    rows = (len(images) + cols - 1) // cols
    canvas = np.zeros((rows * height, cols * width, 3), dtype=np.uint8)
    for index, image in enumerate(images):
        row, col = divmod(index, cols)
        canvas[row * height : (row + 1) * height, col * width : (col + 1) * width] = image
    return canvas


@torch.inference_mode()
def save_epoch_grid(
    model: torch.nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    out_path: Path,
    visibility: torch.Tensor | None = None,
    cols: int = 4,
) -> Path:
    model.eval()
    device = next(model.parameters()).device
    images = images.to(device)
    targets = targets.to(device)
    heatmaps = model(images)
    preds = soft_argmax(heatmaps)
    tiles: list[np.ndarray] = []
    for index in range(images.shape[0]):
        rgb = denormalize_image(images[index])
        vis = None if visibility is None else visibility[index].detach().cpu().numpy()
        tile = draw_landmarks(
            rgb,
            preds[index].detach().cpu().numpy(),
            targets[index].detach().cpu().numpy(),
            visibility=vis,
        )
        tiles.append(tile)
    grid = make_grid(tiles, cols=cols)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(out_path), bgr):
        raise RuntimeError(f"failed to write visualization: {out_path}")
    return out_path
