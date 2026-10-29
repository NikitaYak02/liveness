"""Shared helpers for adapters. Runs inside each method's own environment,
so it may only rely on the standard library, numpy and OpenCV."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

THIRD_PARTY = Path(os.environ.get("PS_THIRD_PARTY", Path(__file__).resolve().parent.parent / "third_party"))


def repo_dir(name: str) -> Path:
    path = THIRD_PARTY / name
    if not path.is_dir():
        raise SystemExit(f"{path} not found; run envs/install_<method>.sh first")
    return path


def enter_repo(name: str, *subdirs: str) -> Path:
    """chdir into a third-party repo and make it importable (most of them use relative paths)."""
    path = repo_dir(name)
    os.chdir(path)
    for sub in reversed(("",) + subdirs):
        p = str(path / sub) if sub else str(path)
        if p not in sys.path:
            sys.path.insert(0, p)
    return path


def job_args(extra=None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--job", required=True)
    p.add_argument("--out", required=True)
    if extra:
        extra(p)
    return p.parse_args()


def load_job(path: str) -> dict:
    return json.loads(Path(path).read_text())


def frames_dir(out: str) -> Path:
    d = Path(out) / "frames"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_frame(out: str, index: int, image_bgr: np.ndarray) -> None:
    cv2.imwrite(str(frames_dir(out) / f"{index:04d}.png"), image_bgr)


def save_meta(out: str, meta: dict) -> None:
    (Path(out) / "meta.json").write_text(json.dumps(meta, indent=1))


def read_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"cannot read {path}")
    return img


def log(*args) -> None:
    print(*args, flush=True)


def sign(name: str, default: float) -> float:
    """Axis sign of an adapter, overridable as PS_<NAME>=-1 when compare.py reports an inverted axis."""
    return float(os.environ.get(f"PS_{name.upper()}", default))


def face_crop(img: np.ndarray, box: list[float] | None, size: int, scale: float = 2.0,
              shift_y: float = -0.1) -> np.ndarray:
    """Square crop centred on the 68-landmark box, `scale` times its larger side.

    The landmark box spans brows to chin, so scale 2 roughly covers the whole
    head with hair; shift_y moves the centre up (fraction of the box size).
    Without a box the image is centre-cropped.
    """
    h, w = img.shape[:2]
    if box is None:
        side = min(h, w)
        cx, cy = w / 2, h / 2
    else:
        x0, y0, x1, y1 = box
        side = scale * max(x1 - x0, y1 - y0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 + shift_y * (y1 - y0)
    half = side / 2
    M = np.array([[size / side, 0, -(cx - half) * size / side],
                  [0, size / side, -(cy - half) * size / side]], np.float32)
    return cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_AREA if side > size else cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def source_box(job: dict, path: str) -> list[float] | None:
    pose = job["source_poses"].get(path)
    return pose.get("box") if pose else None


def pose_delta(job: dict, path: str, frame: dict) -> tuple[float, float]:
    """Requested change (yaw, pitch) relative to the source photo's 3DDFA-V2 pose."""
    ref = job["source_poses"].get(path) or {"yaw": 0.0, "pitch": 0.0}
    return frame["yaw"] - ref["yaw"], frame["pitch"] - ref["pitch"]
