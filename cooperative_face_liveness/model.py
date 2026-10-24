from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def _adapt_conv1_weight(weight: torch.Tensor, in_channels: int) -> torch.Tensor:
    if in_channels == 3:
        return weight
    channel_mean = weight.mean(dim=1, keepdim=True)
    return channel_mean.repeat(1, in_channels, 1, 1)


class ResNet18Encoder(nn.Module):
    def __init__(self, in_channels: int, pretrained: bool) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)

        if in_channels != 3:
            original_conv = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                in_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False,
            )
            with torch.no_grad():
                backbone.conv1.weight.copy_(_adapt_conv1_weight(original_conv.weight, in_channels))

        self.backbone = backbone
        self.out_dim = backbone.fc.in_features
        self.backbone.fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class DualResNet18Liveness(nn.Module):
    """Dual-backbone architecture from the paper.

    Flow backbone:
        1-channel clipped optical-flow magnitude between f1 and f3.
    RGB backbone:
        3-channel preprocessed middle frame f2.
    """

    def __init__(
        self,
        pretrained_rgb: bool = True,
        pretrained_flow: bool = True,
        fusion_hidden_dim: int = 256,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.flow_backbone = ResNet18Encoder(in_channels=1, pretrained=pretrained_flow)
        self.rgb_backbone = ResNet18Encoder(in_channels=3, pretrained=pretrained_rgb)
        fusion_dim = self.flow_backbone.out_dim + self.rgb_backbone.out_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def forward(
        self,
        flow_magnitude: torch.Tensor,
        rgb_frame: torch.Tensor,
        return_embeddings: bool = False,
    ):
        flow_features = self.flow_backbone(flow_magnitude)
        rgb_features = self.rgb_backbone(rgb_frame)
        fused = torch.cat([flow_features, rgb_features], dim=1)
        logits = self.classifier(fused)
        if return_embeddings:
            return logits, {"flow": flow_features, "rgb": rgb_features, "fused": fused}
        return logits
