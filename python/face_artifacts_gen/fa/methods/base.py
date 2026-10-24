"""Base interface for artifact generators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from ..landmarks import FaceLandmarks


@dataclass
class GenerationResult:
    image_bgr: np.ndarray
    change_mask: np.ndarray
    prompt: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class ArtifactGenerator(ABC):
    """generate(img, face, artifact, mask, rng) -> GenerationResult."""

    name: str = 'base'

    def __init__(self, device: str = 'cuda', **kwargs: Any):
        self.device = device
        self.kwargs = kwargs

    @abstractmethod
    def generate(
        self,
        img_bgr: np.ndarray,
        face: FaceLandmarks,
        artifact: str,
        mask: np.ndarray,
        rng: np.random.Generator,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        raise NotImplementedError

    def close(self) -> None:
        """Optional cleanup of GPU resources."""
        return None
