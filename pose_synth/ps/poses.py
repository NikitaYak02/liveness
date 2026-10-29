"""Target head-pose trajectories.

Angles are absolute head pose in degrees, in the convention of 3DDFA-V2
``calc_pose``: yaw is rotation around the vertical axis, pitch around the
horizontal one, (0, 0) is a frontal face. Every adapter maps these targets to
its own camera or keypoint controls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MODES = ("sweep", "yaw", "pitch", "grid")


@dataclass(frozen=True)
class Pose:
    yaw: float
    pitch: float

    def as_dict(self) -> dict:
        return {"yaw": round(float(self.yaw), 3), "pitch": round(float(self.pitch), 3)}


def _ramp(start: float, stop: float, step: float) -> list[float]:
    """Values from start to stop (inclusive) with |step|, start excluded."""
    if step <= 0:
        raise ValueError("step must be positive")
    if start == stop:
        return []
    n = int(np.ceil(abs(stop - start) / step))
    return [float(v) for v in np.linspace(start, stop, n + 1)[1:]]


def _axis_sweep(lo: float, hi: float, step: float) -> list[float]:
    """0 -> lo -> hi -> 0, so the video starts and ends frontal."""
    values = [0.0]
    for target in (lo, hi, 0.0):
        values += _ramp(values[-1], target, step)
    return values


def build_trajectory(
    mode: str,
    yaw_range: tuple[float, float] = (-75.0, 75.0),
    pitch_range: tuple[float, float] = (-30.0, 30.0),
    step: float = 5.0,
) -> list[Pose]:
    yaw_lo, yaw_hi = sorted(yaw_range)
    pitch_lo, pitch_hi = sorted(pitch_range)

    if mode == "yaw":
        return [Pose(y, 0.0) for y in _axis_sweep(yaw_lo, yaw_hi, step)]
    if mode == "pitch":
        return [Pose(0.0, p) for p in _axis_sweep(pitch_lo, pitch_hi, step)]
    if mode == "sweep":
        yaw = [Pose(y, 0.0) for y in _axis_sweep(yaw_lo, yaw_hi, step)]
        pitch = [Pose(0.0, p) for p in _axis_sweep(pitch_lo, pitch_hi, step)]
        return yaw + pitch[1:]
    if mode == "grid":
        yaws = np.arange(yaw_lo, yaw_hi + 1e-6, step)
        pitches = np.arange(pitch_lo, pitch_hi + 1e-6, step)
        poses = []
        for row, pitch in enumerate(pitches):
            # Serpentine order keeps consecutive frames close for video playback.
            row_yaws = yaws if row % 2 == 0 else yaws[::-1]
            poses += [Pose(float(y), float(pitch)) for y in row_yaws]
        return poses
    raise ValueError(f"unknown trajectory mode {mode!r}, expected one of {MODES}")
