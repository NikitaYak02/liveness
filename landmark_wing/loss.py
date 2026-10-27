"""Wing loss for facial landmark regression (Feng et al., CVPR 2018)."""

from __future__ import annotations

import math

import torch
from torch import nn


class WingLoss(nn.Module):
    """Piecewise loss that emphasizes small localization errors.

    For absolute residual ``|x|``:

    - ``w * ln(1 + |x| / eps)`` when ``|x| < w``
    - ``|x| - C`` otherwise, where ``C = w - w * ln(1 + w / eps)``
    """

    def __init__(self, width: float = 10.0, epsilon: float = 2.0) -> None:
        super().__init__()
        if width <= 0:
            raise ValueError(f"width must be positive, got {width}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        self.width = float(width)
        self.epsilon = float(epsilon)
        self.constant = self.width - self.width * math.log(1.0 + self.width / self.epsilon)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        residual = (prediction - target).abs()
        wing = self.width * torch.log1p(residual / self.epsilon)
        linear = residual - self.constant
        return torch.where(residual < self.width, wing, linear).mean()
