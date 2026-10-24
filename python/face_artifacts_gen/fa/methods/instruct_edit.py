"""Method 3: Instruction-based editing (Qwen-Image-Edit / FLUX Kontext) + diff paste-back."""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from ..constants import ARTIFACT_PROMPTS
from ..diffusion_common import filter_call_kwargs, resolve_torch_dtype
from ..io_utils import diff_mask, paste_back
from ..landmarks import FaceLandmarks
from .base import ArtifactGenerator, GenerationResult

# Prefer Apache-2.0 Qwen; FLUX Kontext as optional alternative.
DEFAULT_BACKEND = 'qwen'  # qwen | flux_kontext | procedural_fallback
QWEN_MODEL = 'Qwen/Qwen-Image-Edit'
FLUX_KONTEXT_MODEL = 'black-forest-labs/FLUX.1-Kontext-dev'


class InstructEditGenerator(ArtifactGenerator):
    name = 'instruct_edit'

    def __init__(
        self,
        device: str = 'cuda',
        backend: str = DEFAULT_BACKEND,
        model_id: Optional[str] = None,
        steps: int = 28,
        guidance: float = 4.0,
        dtype: str = 'bfloat16',
        cpu_offload: bool = True,
        diff_threshold: float = 18.0,
        paste_feather: int = 15,
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        self.backend = backend
        self.model_id = model_id or (FLUX_KONTEXT_MODEL if backend == 'flux_kontext' else QWEN_MODEL)
        self.steps = steps
        self.guidance = guidance
        self.dtype_name = dtype
        self.cpu_offload = cpu_offload
        self.diff_threshold = diff_threshold
        self.paste_feather = paste_feather
        self._pipe = None
        self._fallback = None

    def _load(self):
        if self._pipe is not None or self.backend == 'procedural_fallback':
            return
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                'instruct_edit requires: pip install torch diffusers transformers accelerate'
            ) from exc

        dtype = resolve_torch_dtype(self.dtype_name, self.device)
        logging.info('Loading instruct-edit backend=%s model=%s', self.backend, self.model_id)

        try:
            if self.backend == 'flux_kontext':
                self._pipe = self._load_flux_kontext(dtype)
            else:
                self._pipe = self._load_qwen(dtype)
        except Exception as exc:
            logging.warning(
                'Failed to load instruct model (%s); will use procedural paste-back fallback on generate',
                exc,
            )
            self.backend = 'procedural_fallback'

        if self._pipe is not None:
            if self.cpu_offload and hasattr(self._pipe, 'enable_model_cpu_offload'):
                self._pipe.enable_model_cpu_offload()
            else:
                self._pipe = self._pipe.to(self.device)

    def _load_qwen(self, dtype):
        """Try Qwen-Image-Edit style pipelines available in recent diffusers."""
        try:
            from diffusers import QwenImageEditPipeline

            return QwenImageEditPipeline.from_pretrained(self.model_id, torch_dtype=dtype)
        except Exception:
            pass
        # Generic fallback: some hubs expose via DiffusionPipeline.
        from diffusers import DiffusionPipeline

        return DiffusionPipeline.from_pretrained(self.model_id, torch_dtype=dtype)

    def _load_flux_kontext(self, dtype):
        try:
            from diffusers import FluxKontextPipeline

            return FluxKontextPipeline.from_pretrained(self.model_id, torch_dtype=dtype)
        except Exception:
            from diffusers import DiffusionPipeline

            return DiffusionPipeline.from_pretrained(self.model_id, torch_dtype=dtype)

    def _generator(self, seed: Optional[int]):
        if seed is None:
            return None
        import torch

        return torch.Generator(device='cpu').manual_seed(int(seed))

    def _run_pipe(self, img_bgr: np.ndarray, prompt: str, seed: Optional[int]) -> np.ndarray:
        from PIL import Image

        pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        call_kwargs = {
            'image': pil,
            'prompt': prompt,
            'num_inference_steps': self.steps,
            'guidance_scale': self.guidance,
            'generator': self._generator(seed),
        }
        filtered = filter_call_kwargs(self._pipe, call_kwargs)
        if 'prompt' not in filtered:
            # Some edit pipelines expose `instruction` instead of `prompt`.
            filtered = filter_call_kwargs(self._pipe, {**call_kwargs, 'instruction': prompt})

        out = self._pipe(**filtered)
        if hasattr(out, 'images'):
            edited = out.images[0]
        elif isinstance(out, (list, tuple)):
            edited = out[0]
        else:
            edited = out
        return cv2.cvtColor(np.array(edited), cv2.COLOR_RGB2BGR)

    def generate(
        self,
        img_bgr: np.ndarray,
        face: FaceLandmarks,
        artifact: str,
        mask: np.ndarray,
        rng: np.random.Generator,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        prompts = ARTIFACT_PROMPTS.get(artifact, ARTIFACT_PROMPTS['beard'])
        prompt = prompts.get('instruct') or prompts['prompt']
        use_seed = seed if seed is not None else int(rng.integers(0, 2**31 - 1))

        self._load()

        if self.backend == 'procedural_fallback' or self._pipe is None:
            # Soft tint inside the provided mask so the pipeline still produces output
            # without heavy model weights (useful for CPU smoke tests).
            edited = img_bgr.copy()
            tint = np.zeros_like(edited)
            if artifact == 'beard':
                tint[:] = (40, 60, 80)
            elif artifact == 'glasses_glare':
                tint[:] = (200, 220, 240)
            elif artifact == 'bangs':
                tint[:] = (30, 40, 50)
            else:
                tint[:] = (90, 90, 120)
            alpha = (mask.astype(np.float32) / 255.0)[..., None] * 0.45
            edited = np.clip(edited.astype(np.float32) * (1 - alpha) + tint * alpha, 0, 255).astype(np.uint8)
            change = mask
            out = paste_back(img_bgr, edited, change, feather=self.paste_feather)
            return GenerationResult(
                image_bgr=out,
                change_mask=change,
                prompt=prompt,
                extra={'backend': 'procedural_fallback'},
            )

        edited = self._run_pipe(img_bgr, prompt, use_seed)
        # Restrict changes to region of interest: intersect model diff with provided mask.
        dmask = diff_mask(img_bgr, edited, threshold=self.diff_threshold)
        if mask is not None:
            # Dilate provided mask slightly so instruct edits near edges are kept.
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            region = cv2.dilate((mask > 16).astype(np.uint8) * 255, k)
            change = cv2.bitwise_and(dmask, region)
            if int(change.sum()) < 100:
                # Model changed little inside ROI — fall back to full ROI blend of edit.
                change = region
        else:
            change = dmask

        out = paste_back(img_bgr, edited, change, feather=self.paste_feather)
        return GenerationResult(
            image_bgr=out,
            change_mask=change,
            prompt=prompt,
            extra={'backend': self.backend, 'model_id': self.model_id},
        )
