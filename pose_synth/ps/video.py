"""Frame labelling, per-method videos and the side-by-side comparison grid."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

TILE = 384
FONT = cv2.FONT_HERSHEY_SIMPLEX


def frame_path(frames_dir: Path, index: int) -> Path:
    return frames_dir / f"{index:04d}.png"


def fit_tile(image: np.ndarray | None, size: int = TILE) -> np.ndarray:
    """Letterbox an image into a size x size BGR tile; black when missing."""
    tile = np.zeros((size, size, 3), np.uint8)
    if image is None:
        return tile
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    h, w = image.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    tile[y0:y0 + nh, x0:x0 + nw] = resized
    return tile


def _text(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.5, color=(255, 255, 255)) -> None:
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, 1, cv2.LINE_AA)


def _fmt_angle(value) -> str:
    return "--" if value is None else f"{value:+.0f}"


def label_tile(tile: np.ndarray, title: str, requested: dict, measured: dict | None) -> np.ndarray:
    out = tile.copy()
    _text(out, title, (8, 20), 0.6, (80, 220, 255))
    _text(out, f"req  yaw {_fmt_angle(requested['yaw'])} pitch {_fmt_angle(requested['pitch'])}", (8, out.shape[0] - 30))
    m_yaw = measured.get("yaw") if measured else None
    m_pitch = measured.get("pitch") if measured else None
    _text(out, f"meas yaw {_fmt_angle(m_yaw)} pitch {_fmt_angle(m_pitch)}", (8, out.shape[0] - 10), 0.5, (160, 255, 160))
    return out


def no_result_tile(title: str, reason: str, size: int = TILE) -> np.ndarray:
    tile = np.full((size, size, 3), 40, np.uint8)
    _text(tile, title, (8, 20), 0.6, (80, 220, 255))
    _text(tile, "NO RESULT", (size // 2 - 70, size // 2), 0.8, (90, 90, 255))
    for i, line in enumerate(_wrap(reason, 44)[:6]):
        _text(tile, line, (8, size // 2 + 30 + 18 * i), 0.42, (200, 200, 200))
    return tile


def _wrap(text: str, width: int) -> list[str]:
    words, lines, line = text.split(), [], ""
    for word in words:
        if len(line) + len(word) + 1 > width and line:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        lines.append(line)
    return lines


def grid_shape(n: int) -> tuple[int, int]:
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def compose_grid(tiles: list[np.ndarray]) -> np.ndarray:
    rows, cols = grid_shape(len(tiles))
    size = tiles[0].shape[0]
    canvas = np.zeros((rows * size, cols * size, 3), np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        canvas[r * size:(r + 1) * size, c * size:(c + 1) * size] = tile
    return canvas


class VideoSink:
    def __init__(self, path: Path, fps: float):
        self.path = Path(path)
        self.fps = fps
        self._writer = None

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            h, w = frame.shape[:2]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            if not self._writer.isOpened():
                raise RuntimeError(f"cannot open video writer for {self.path}")
        self._writer.write(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


def write_method_video(
    method: str,
    frames_dir: Path,
    records: list[dict],
    video_path: Path,
    fps: float,
    size: int = 512,
) -> int:
    """Labelled per-method video; returns how many frames were present."""
    sink = VideoSink(video_path, fps)
    present = 0
    try:
        for rec in records:
            img = cv2.imread(str(frame_path(frames_dir, rec["index"])))
            present += img is not None
            sink.write(label_tile(fit_tile(img, size), method, rec["requested"], rec.get("measured")))
    finally:
        sink.close()
    return present


def write_compare_video(
    methods: list[str],
    out_dir: Path,
    records_by_method: dict[str, list[dict] | None],
    errors: dict[str, str],
    requested: list[dict],
    video_path: Path,
    fps: float,
) -> None:
    sink = VideoSink(video_path, fps)
    try:
        for i, req in enumerate(requested):
            tiles = []
            for method in methods:
                records = records_by_method.get(method)
                if records is None:
                    tiles.append(no_result_tile(method, errors.get(method, "not run")))
                    continue
                rec = records[i]
                img = cv2.imread(str(frame_path(out_dir / method / "frames", rec["index"])))
                tiles.append(label_tile(fit_tile(img), method, req, rec.get("measured")))
            sink.write(compose_grid(tiles))
    finally:
        sink.close()
