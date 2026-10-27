"""Lightweight HRNet port of face_kp_train_tools/HRNet_src.json."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .scheme import HEATMAP_CHANNELS, NUM_LANDMARKS


class SymReLU(nn.Module):
    """NNT SymReLU with a=b=1: clamp activations to ``[-1, 1]``."""

    def __init__(self, a: float = 1.0, b: float = 1.0) -> None:
        super().__init__()
        self.a = float(a)
        self.b = float(b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, -self.a, self.b)


def conv_bn_act(
    in_ch: int,
    out_ch: int,
    kernel: int = 3,
    stride: int = 1,
    padding: int | None = None,
    use_act: bool = True,
) -> nn.Sequential:
    if padding is None:
        padding = kernel // 2
    layers: list[nn.Module] = [
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch, momentum=0.1),
    ]
    if use_act:
        layers.append(SymReLU(1.0, 1.0))
    return nn.Sequential(*layers)


def upsample(x: torch.Tensor, scale: int) -> torch.Tensor:
    if scale == 1:
        return x
    return F.interpolate(x, scale_factor=scale, mode="nearest")


class HRNetFace(nn.Module):
    """Multi-resolution HRNet-style face keypoint net from the NNT JSON.

    Input: ``[B, 1, 128, 128]`` grayscale.
    Output: ``[B, 31, 128, 128]`` heatmaps (channel 0 unused; points in 1..30).
    """

    def __init__(self, num_landmarks: int = NUM_LANDMARKS, in_channels: int = 1) -> None:
        super().__init__()
        if num_landmarks != NUM_LANDMARKS:
            raise ValueError(
                f"this architecture is fixed to the internal {NUM_LANDMARKS}-point scheme, "
                f"got {num_landmarks}"
            )
        self.num_landmarks = num_landmarks
        self.heatmap_channels = num_landmarks + 1

        self.stem = nn.Sequential(
            conv_bn_act(in_channels, 32, stride=2),
            conv_bn_act(32, 32, stride=1),
        )

        # Stage 2
        self.s2_b1 = nn.Sequential(
            conv_bn_act(32, 32),
            conv_bn_act(32, 32),
            conv_bn_act(32, 32),
            conv_bn_act(32, 32),
        )
        self.s2_b2 = nn.Sequential(
            conv_bn_act(32, 48, stride=2),
            conv_bn_act(48, 48),
            conv_bn_act(48, 48),
        )
        self.s2_f_b1 = conv_bn_act(32 + 48, 32, kernel=1, padding=0)
        self.s2_f_to_b2 = conv_bn_act(32, 48, stride=2)
        self.s2_f_b2 = conv_bn_act(48 + 48, 48, kernel=1, padding=0)

        # Stage 3
        self.s3_b1 = conv_bn_act(32, 32)
        self.s3_b2 = conv_bn_act(48, 48)
        self.s3_b3 = nn.Sequential(
            conv_bn_act(48, 64, stride=2),
            conv_bn_act(64, 64),
        )
        self.s3_f_b1 = conv_bn_act(32 + 48 + 64, 32, kernel=1, padding=0)
        self.s3_f_to_b2_from_b1 = conv_bn_act(32, 48, stride=2)
        self.s3_f_b2 = conv_bn_act(48 + 48 + 64, 48, kernel=1, padding=0)
        self.s3_f_to_b3_from_b1 = conv_bn_act(32, 64, stride=4)
        self.s3_f_to_b3_from_b2 = conv_bn_act(48, 64, stride=2)
        self.s3_f_b3 = conv_bn_act(64 + 64 + 64, 64, kernel=1, padding=0)

        # Stage 4
        self.s4_b1 = conv_bn_act(32, 32)
        self.s4_b2 = conv_bn_act(48, 48)
        self.s4_b3 = conv_bn_act(64, 64)
        self.s4_b4 = nn.Sequential(
            conv_bn_act(64, 128, stride=2),
            conv_bn_act(128, 128),
        )
        self.s4_f_b1 = conv_bn_act(32 + 48 + 64 + 128, 32, kernel=1, padding=0)
        self.s4_to_b2_from_b1 = conv_bn_act(32, 48, stride=2)
        self.s4_f_b2 = conv_bn_act(48 + 48 + 64 + 128, 48, kernel=1, padding=0)
        self.s4_to_b3_from_b1 = conv_bn_act(32, 64, stride=4)
        self.s4_to_b3_from_b2 = conv_bn_act(48, 64, stride=2)
        self.s4_f_b3 = conv_bn_act(64 + 64 + 64 + 128, 64, kernel=1, padding=0)
        self.s4_to_b4_from_b1 = conv_bn_act(32, 128, stride=8)
        self.s4_to_b4_from_b2 = conv_bn_act(48, 128, stride=4)
        self.s4_to_b4_from_b3 = conv_bn_act(64, 128, stride=2)
        self.s4_f_b4 = conv_bn_act(128 + 128 + 128 + 128, 128, kernel=1, padding=0)

        # Head
        self.head_reduce = conv_bn_act(32 + 48 + 64 + 128, 48, kernel=1, padding=0)
        self.head_refine1 = conv_bn_act(48, 48)
        self.head_refine2 = conv_bn_act(48, 48)
        self.head_out = nn.Conv2d(48, self.heatmap_channels, kernel_size=1, bias=True)

        # Alias kept for older imports / checkpoints naming in docs.
        self.num_out = self.heatmap_channels

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        stem = self.stem(images)

        b1 = self.s2_b1(stem)
        b2 = self.s2_b2(stem)
        f1 = self.s2_f_b1(torch.cat([b1, upsample(b2, 2)], dim=1))
        f2 = self.s2_f_b2(torch.cat([b2, self.s2_f_to_b2(b1)], dim=1))

        b1 = self.s3_b1(f1)
        b2 = self.s3_b2(f2)
        b3 = self.s3_b3(f2)
        f1 = self.s3_f_b1(torch.cat([b1, upsample(b2, 2), upsample(b3, 4)], dim=1))
        f2 = self.s3_f_b2(
            torch.cat([b2, self.s3_f_to_b2_from_b1(b1), upsample(b3, 2)], dim=1)
        )
        f3 = self.s3_f_b3(
            torch.cat([b3, self.s3_f_to_b3_from_b2(b2), self.s3_f_to_b3_from_b1(b1)], dim=1)
        )

        b1 = self.s4_b1(f1)
        b2 = self.s4_b2(f2)
        b3 = self.s4_b3(f3)
        b4 = self.s4_b4(f3)
        f1 = self.s4_f_b1(
            torch.cat([b1, upsample(b2, 2), upsample(b3, 4), upsample(b4, 8)], dim=1)
        )
        f2 = self.s4_f_b2(
            torch.cat(
                [b2, self.s4_to_b2_from_b1(b1), upsample(b3, 2), upsample(b4, 4)],
                dim=1,
            )
        )
        f3 = self.s4_f_b3(
            torch.cat(
                [b3, self.s4_to_b3_from_b2(b2), self.s4_to_b3_from_b1(b1), upsample(b4, 2)],
                dim=1,
            )
        )
        f4 = self.s4_f_b4(
            torch.cat(
                [
                    b4,
                    self.s4_to_b4_from_b3(b3),
                    self.s4_to_b4_from_b2(b2),
                    self.s4_to_b4_from_b1(b1),
                ],
                dim=1,
            )
        )

        merged = torch.cat(
            [f1, upsample(f2, 2), upsample(f3, 4), upsample(f4, 8)],
            dim=1,
        )
        x = self.head_reduce(merged)
        x = self.head_refine1(x)
        x = upsample(x, 2)
        x = self.head_refine2(x)
        return self.head_out(x)


# Backward-compatible name used by early landmark_wing docs/tests.
MobileNetV2Landmarks = HRNetFace
