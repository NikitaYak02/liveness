"""Shared diffusion helpers (resize / dtype / pipeline loading).

Adapted from face_kp_train_tools/close_eyes_in_dataset.py for reuse
without importing the full eye-closing CLI.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


def round_to_multiple(value: int, multiple: int = 8) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def inference_size(
    orig_w: int,
    orig_h: int,
    target: int,
    mode: str = 'fit',
    size_multiple: int = 8,
) -> Tuple[int, int]:
    if mode == 'none':
        return (
            round_to_multiple(orig_w, size_multiple),
            round_to_multiple(orig_h, size_multiple),
        )
    if mode == 'square':
        side = round_to_multiple(target, size_multiple)
        return side, side
    scale = target / max(orig_w, orig_h)
    return (
        round_to_multiple(int(round(orig_w * scale)), size_multiple),
        round_to_multiple(int(round(orig_h * scale)), size_multiple),
    )


def _resize_image(img_bgr: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    orig_h, orig_w = img_bgr.shape[:2]
    interp = cv2.INTER_LINEAR if (new_w > orig_w or new_h > orig_h) else cv2.INTER_AREA
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=interp)


@dataclass
class ResizeMeta:
    orig_wh: Tuple[int, int]
    mode: str
    content_wh: Optional[Tuple[int, int]] = None
    pad_xy: Optional[Tuple[int, int]] = None


def resize_for_inference(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    target: int,
    mode: str = 'fit',
    size_multiple: int = 8,
) -> Tuple[np.ndarray, np.ndarray, ResizeMeta]:
    h, w = img_bgr.shape[:2]
    new_w, new_h = inference_size(w, h, target, mode, size_multiple)

    if mode == 'square':
        scale = min(new_w / w, new_h / h)
        cw, ch = int(round(w * scale)), int(round(h * scale))
        resized = _resize_image(img_bgr, cw, ch)
        mask_r = cv2.resize(mask, (cw, ch), interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        mcanvas = np.zeros((new_h, new_w), dtype=np.uint8)
        pad_x = (new_w - cw) // 2
        pad_y = (new_h - ch) // 2
        canvas[pad_y : pad_y + ch, pad_x : pad_x + cw] = resized
        mcanvas[pad_y : pad_y + ch, pad_x : pad_x + cw] = mask_r
        meta = ResizeMeta(orig_wh=(w, h), mode=mode, content_wh=(cw, ch), pad_xy=(pad_x, pad_y))
        return canvas, mcanvas, meta

    work = _resize_image(img_bgr, new_w, new_h)
    work_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return work, work_mask, ResizeMeta(orig_wh=(w, h), mode=mode)


def restore_original_resolution(
    inpainted_bgr: np.ndarray,
    original_bgr: np.ndarray,
    mask: np.ndarray,
    meta: ResizeMeta,
) -> np.ndarray:
    ow, oh = meta.orig_wh
    if meta.mode == 'square' and meta.content_wh and meta.pad_xy:
        cw, ch = meta.content_wh
        pad_x, pad_y = meta.pad_xy
        crop = inpainted_bgr[pad_y : pad_y + ch, pad_x : pad_x + cw]
        restored = cv2.resize(crop, (ow, oh), interpolation=cv2.INTER_LINEAR)
    else:
        restored = cv2.resize(inpainted_bgr, (ow, oh), interpolation=cv2.INTER_LINEAR)

    # Keep outside-mask pixels identical to original.
    alpha = (mask.astype(np.float32) / 255.0)[..., None] if mask.ndim == 2 else mask.astype(np.float32) / 255.0
    if alpha.shape[:2] != original_bgr.shape[:2]:
        alpha = cv2.resize(alpha, (ow, oh), interpolation=cv2.INTER_LINEAR)
        if alpha.ndim == 2:
            alpha = alpha[..., None]
    out = original_bgr.astype(np.float32) * (1.0 - alpha) + restored.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def resolve_torch_dtype(name: str, device: str):
    import torch

    name = (name or 'float16').lower()
    mapping = {
        'float16': torch.float16,
        'fp16': torch.float16,
        'bfloat16': torch.bfloat16,
        'bf16': torch.bfloat16,
        'float32': torch.float32,
        'fp32': torch.float32,
    }
    dtype = mapping.get(name, torch.float16)
    if device == 'cpu' and dtype in (torch.float16, torch.bfloat16):
        logging.warning('CPU does not support %s well; falling back to float32', name)
        return torch.float32
    if device == 'mps' and dtype == torch.bfloat16:
        logging.warning('MPS: bfloat16 may be unsupported; using float16')
        return torch.float16
    return dtype


def filter_call_kwargs(pipe, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        params = inspect.signature(pipe.__call__).parameters
    except (TypeError, ValueError):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def load_flux_fill(model_id: str, dtype):
    from diffusers import FluxFillPipeline

    return FluxFillPipeline.from_pretrained(model_id, torch_dtype=dtype)


def load_sdxl_inpaint(model_id: str, dtype):
    from diffusers import StableDiffusionXLInpaintPipeline

    return StableDiffusionXLInpaintPipeline.from_pretrained(model_id, torch_dtype=dtype)
