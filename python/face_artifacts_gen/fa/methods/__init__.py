"""Artifact generation backends."""

from __future__ import annotations

from typing import Any, Dict, Type

from .base import ArtifactGenerator, GenerationResult
from .flux_fill import FluxFillGenerator
from .instruct_edit import InstructEditGenerator
from .procedural import ProceduralGenerator
from .sdxl_cn import SdxlControlNetGenerator
from .stylegan import StyleGANGenerator

GENERATORS: Dict[str, Type[ArtifactGenerator]] = {
    'flux_fill': FluxFillGenerator,
    'sdxl_cn': SdxlControlNetGenerator,
    'instruct_edit': InstructEditGenerator,
    'stylegan': StyleGANGenerator,
    'procedural': ProceduralGenerator,
}


def create_generator(method: str, **kwargs: Any) -> ArtifactGenerator:
    if method not in GENERATORS:
        raise ValueError(f'unknown method: {method!r}; choose from {sorted(GENERATORS)}')
    return GENERATORS[method](**kwargs)


__all__ = [
    'ArtifactGenerator',
    'GenerationResult',
    'GENERATORS',
    'create_generator',
]
