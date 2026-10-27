"""HRNet face-keypoint training on the internal 30-point scheme."""

from .loss import LandmarkHeatmapLoss, WingLoss
from .model import HRNetFace
from .scheme import KP_NAMES, NUM_LANDMARKS

__all__ = [
    "HRNetFace",
    "KP_NAMES",
    "LandmarkHeatmapLoss",
    "NUM_LANDMARKS",
    "WingLoss",
]
