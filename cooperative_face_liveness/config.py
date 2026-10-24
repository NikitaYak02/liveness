from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class CooperativeScenarioConfig:
    """Hyper-parameters reconstructed from the paper."""

    target_face_heights: tuple[float, float, float] = (0.500, 0.625, 0.750)
    crop_size: int = 256
    crop_margin: float = 0.10
    clip_fraction: float = 0.20
    num_flow_updates: int = 3
    scan_stride: int = 1
    frame_pool_quantile: float = 0.10
    flow_resolutions: tuple[int, ...] = (192, 224, 256, 288, 320)
    perspective_jitter: float = 0.08
    eye_left_indices: tuple[int, ...] = field(default_factory=lambda: (36, 37, 38, 39, 40, 41))
    eye_right_indices: tuple[int, ...] = field(default_factory=lambda: (42, 43, 44, 45, 46, 47))
