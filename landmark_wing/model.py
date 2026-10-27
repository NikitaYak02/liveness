"""MobileNetV2 regressor for facial landmarks."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


class MobileNetV2Landmarks(nn.Module):
    """ImageNet-pretrained MobileNetV2 with a linear landmark head.

    Output shape is ``[B, num_landmarks, 2]`` in crop-pixel coordinates.
    """

    def __init__(self, num_landmarks: int, pretrained: bool = True) -> None:
        super().__init__()
        if num_landmarks <= 0:
            raise ValueError(f"num_landmarks must be positive, got {num_landmarks}")
        self.num_landmarks = int(num_landmarks)
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v2(weights=weights)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.head = nn.Linear(in_features, self.num_landmarks * 2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        coords = self.head(features)
        return coords.view(-1, self.num_landmarks, 2)
