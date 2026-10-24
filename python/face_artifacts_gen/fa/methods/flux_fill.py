"""Method 1: FLUX.1 Fill [dev] masked inpainting."""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from ..constants import ARTIFACT_PROMPTS
from ..diffusion_common import (
    filter_call_kwargs,
    load_flux_fill,
    resize_for_inference,
    resolve_torch_dtype,
    restore_original_resolution,
)
from ..landmarks import FaceLandmarks
from .base import ArtifactGenerator, GenerationResult

DEFAULT_MODEL = 'black-forest-labs/FLUX.1-Fill-dev'


class FluxFillGenerator(ArtifactGenerator):
    name = 'flux_fill'

    def __init__(
        self,
        device: str = 'cuda',
        model_id: str = DEFAULT_MODEL,
        steps: int = 28,
        guidance: float = 30.0,
        inference_size: int = 1024,
        resize_mode: str = 'fit',
        dtype: str = 'bfloat16',
        cpu_offload: bool = True,
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        self.model_id = model_id
        self.steps = steps
        self.guidance = guidance
        self.inference_size = inference_size
        self.resize_mode = resize_mode
        self.dtype_name = dtype
        self.cpu_offload = cpu_offload
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                'flux_fill requires: pip install torch diffusers transformers accelerate'
            ) from exc

        dtype = resolve_torch_dtype(self.dtype_name, self.device)
        logging.info('Loading FLUX Fill: %s (%s)', self.model_id, dtype)
        self._pipe = load_flux_fill(self.model_id, dtype)
        if self.cpu_offload and hasattr(self._pipe, 'enable_model_cpu_offload'):
            self._pipe.enable_model_cpu_offload()
        else:
            self._pipe = self._pipe.to(self.device)

    def _generator(self, seed: Optional[int]):
        if seed is None:
            return None
        import torch

        try:
            return torch.Generator(device='cpu').manual_seed(int(seed))
        except Exception:
            return None

    def generate(
        self,
        img_bgr: np.ndarray,
        face: FaceLandmarks,
        artifact: str,
        mask: np.ndarray,
        rng: np.random.Generator,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        from PIL import Image

        self._load()
        prompts = ARTIFACT_PROMPTS.get(artifact, ARTIFACT_PROMPTS['beard'])
        prompt = prompts['prompt']

        work_img, work_mask, meta = resize_for_inference(
            img_bgr,
            mask,
            self.inference_size,
            self.resize_mode,
            size_multiple=16,
        )
        pil_img = Image.fromarray(cv2.cvtColor(work_img, cv2.COLOR_BGR2RGB))
        pil_mask = Image.fromarray(work_mask).convert('L')

        call_kwargs = {
            'prompt': prompt,
            'image': pil_img,
            'mask_image': pil_mask,
            'height': work_img.shape[0],
            'width': work_img.shape[1],
            'num_inference_steps': self.steps,
            'guidance_scale': self.guidance,
            'generator': self._generator(seed if seed is not None else int(rng.integers(0, 2**31 - 1))),
            'max_sequence_length': 512,
        }
        filtered = filter_call_kwargs(self._pipe, call_kwargs)
        result = self._pipe(**filtered).images[0]
        inpainted = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        out = restore_original_resolution(inpainted, img_bgr, mask, meta)
        return GenerationResult(image_bgr=out, change_mask=mask, prompt=prompt)
