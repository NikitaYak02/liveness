"""Compact facial landmark regression with Wing loss."""

from .loss import WingLoss
from .model import MobileNetV2Landmarks

__all__ = ["MobileNetV2Landmarks", "WingLoss"]
