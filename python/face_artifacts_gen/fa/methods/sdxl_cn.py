"""Method 2: SDXL Inpainting + ControlNet (canny / depth)."""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from ..constants import ARTIFACT_PROMPTS
from ..diffusion_common import (
    filter_call_kwargs,
    resize_for_inference,
    resolve_torch_dtype,
    restore_original_resolution,
)
from ..landmarks import FaceLandmarks
from .base import ArtifactGenerator, GenerationResult

DEFAULT_INPAINT = 'diffusers/stable-diffusion-xl-1.0-inpainting-0.1'
DEFAULT_CONTROLNET = 'xinsir/controlnet-union-sdxl-1.0'


def _canny_conditioning(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)


def _depth_proxy(img_bgr: np.ndarray) -> np.ndarray:
    """Lightweight depth proxy (blurred luminance) — no MiDaS dependency."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    depth = cv2.GaussianBlur(gray, (0, 0), sigmaX=3)
    return cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)


class SdxlControlNetGenerator(ArtifactGenerator):
    name = 'sdxl_cn'

    def __init__(
        self,
        device: str = 'cuda',
        model_id: str = DEFAULT_INPAINT,
        controlnet_id: str = DEFAULT_CONTROLNET,
        steps: int = 30,
        guidance: float = 8.0,
        strength: float = 0.95,
        controlnet_scale: float = 0.6,
        inference_size: int = 1024,
        resize_mode: str = 'fit',
        dtype: str = 'float16',
        cpu_offload: bool = False,
        control_mode: str = 'canny',
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        self.model_id = model_id
        self.controlnet_id = controlnet_id
        self.steps = steps
        self.guidance = guidance
        self.strength = strength
        self.controlnet_scale = controlnet_scale
        self.inference_size = inference_size
        self.resize_mode = resize_mode
        self.dtype_name = dtype
        self.cpu_offload = cpu_offload
        self.control_mode = control_mode
        self._pipe = None
        self._use_controlnet = True

    def _load(self):
        if self._pipe is not None:
            return
        try:
            from diffusers import (
                ControlNetModel,
                StableDiffusionXLControlNetInpaintPipeline,
                StableDiffusionXLInpaintPipeline,
            )
        except ImportError as exc:
            raise RuntimeError(
                'sdxl_cn requires: pip install torch diffusers transformers accelerate'
            ) from exc

        dtype = resolve_torch_dtype(self.dtype_name, self.device)
        logging.info('Loading SDXL inpaint: %s', self.model_id)

        try:
            controlnet = ControlNetModel.from_pretrained(self.controlnet_id, torch_dtype=dtype)
            self._pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
                self.model_id,
                controlnet=controlnet,
                torch_dtype=dtype,
            )
            self._use_controlnet = True
            logging.info('ControlNet loaded: %s', self.controlnet_id)
        except Exception as exc:
            logging.warning(
                'ControlNet inpaint unavailable (%s); falling back to SDXL inpaint only',
                exc,
            )
            self._pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
            )
            self._use_controlnet = False

        if self.cpu_offload and hasattr(self._pipe, 'enable_model_cpu_offload'):
            self._pipe.enable_model_cpu_offload()
        else:
            self._pipe = self._pipe.to(self.device)

    def _generator(self, seed: Optional[int]):
        if seed is None:
            return None
        import torch

        try:
            return torch.Generator(device=self.device).manual_seed(int(seed))
        except Exception:
            return torch.Generator(device='cpu').manual_seed(int(seed))

    def _control_image(self, img_bgr: np.ndarray):
        from PIL import Image

        rgb = _depth_proxy(img_bgr) if self.control_mode == 'depth' else _canny_conditioning(img_bgr)
        return Image.fromarray(rgb)

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
        negative = prompts['negative']

        work_img, work_mask, meta = resize_for_inference(
            img_bgr,
            mask,
            self.inference_size,
            self.resize_mode,
            size_multiple=8,
        )
        pil_img = Image.fromarray(cv2.cvtColor(work_img, cv2.COLOR_BGR2RGB))
        pil_mask = Image.fromarray(work_mask).convert('L')
        use_seed = seed if seed is not None else int(rng.integers(0, 2**31 - 1))

        call_kwargs = {
            'prompt': prompt,
            'negative_prompt': negative,
            'image': pil_img,
            'mask_image': pil_mask,
            'height': work_img.shape[0],
            'width': work_img.shape[1],
            'num_inference_steps': self.steps,
            'guidance_scale': self.guidance,
            'strength': self.strength,
            'generator': self._generator(use_seed),
        }
        if self._use_controlnet:
            call_kwargs['control_image'] = self._control_image(work_img)
            call_kwargs['controlnet_conditioning_scale'] = self.controlnet_scale

        filtered = filter_call_kwargs(self._pipe, call_kwargs)
        try:
            result = self._pipe(**filtered).images[0]
        except TypeError:
            filtered.pop('control_image', None)
            filtered.pop('controlnet_conditioning_scale', None)
            result = self._pipe(**filtered).images[0]

        inpainted = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        out = restore_original_resolution(inpainted, img_bgr, mask, meta)
        return GenerationResult(
            image_bgr=out,
            change_mask=mask,
            prompt=prompt,
            extra={'controlnet': self._use_controlnet, 'control_mode': self.control_mode},
        )
