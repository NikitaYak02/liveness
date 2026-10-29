"""Choosing which input photo drives each target pose."""

from __future__ import annotations

import math
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Pitch errors are less damaging than yaw errors for side-of-face content.
PITCH_WEIGHT = 0.5


def list_images(path: str | Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    images = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise FileNotFoundError(f"no images in {path}")
    return images


def pose_distance(a: dict, b: dict) -> float:
    return math.hypot(a["yaw"] - b["yaw"], PITCH_WEIGHT * (a["pitch"] - b["pitch"]))


def _known(source_poses: dict[str, dict | None]) -> dict[str, dict]:
    return {k: v for k, v in source_poses.items() if v is not None}


def most_frontal(images: list[str], source_poses: dict[str, dict | None]) -> str:
    """Image closest to (0, 0); the first image when no pose is known."""
    known = _known(source_poses)
    candidates = [img for img in images if img in known]
    if not candidates:
        return images[0]
    return min(candidates, key=lambda img: pose_distance(known[img], {"yaw": 0.0, "pitch": 0.0}))


def nearest_source(target: dict, images: list[str], source_poses: dict[str, dict | None]) -> str:
    known = _known(source_poses)
    candidates = [img for img in images if img in known]
    if not candidates:
        return images[0]
    return min(candidates, key=lambda img: pose_distance(known[img], target))
