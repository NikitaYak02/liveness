"""Method 4: StyleGAN2 latent editing (e4e / InterFaceGAN-style) with FFHQ align paste-back.

Prototype strategy:
- If full e4e + StyleGAN2 + direction vectors are installed, use them.
- Otherwise fall back to a lightweight aligned-crop color/texture edit inside the
  artifact mask, then inverse-warp back. This keeps the I/O contract identical
  for smoke tests without custom CUDA ops.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from ..io_utils import paste_back
from ..landmarks import FaceLandmarks
from .base import ArtifactGenerator, GenerationResult

# Approximate FFHQ 5-point template (1024x1024), used for alignment.
FFHQ_TEMPLATE_5 = np.array(
    [
        [384.0, 460.0],  # left eye
        [640.0, 460.0],  # right eye
        [512.0, 600.0],  # nose
        [400.0, 760.0],  # left mouth
        [624.0, 760.0],  # right mouth
    ],
    dtype=np.float32,
)


def _five_points(face: FaceLandmarks) -> Optional[np.ndarray]:
    left = face.get('kp_pupil_left') or face.get('kp_eye_left_outer')
    right = face.get('kp_pupil_right') or face.get('kp_eye_right_outer')
    nose = face.get('kp_nose_tip') or face.get('kp_nose_bridge')
    ml = face.get('kp_mouth_left')
    mr = face.get('kp_mouth_right')
    if not all([left, right, nose, ml, mr]):
        # Try dense 5-pt from insightface kps.
        if face.dense is not None and len(face.dense) >= 5 and len(face.dense) < 106:
            return np.asarray(face.dense[:5], dtype=np.float32)
        if face.dense is not None and len(face.dense) >= 106:
            # Approximate from 2d106.
            return np.array(
                [
                    face.dense[74],  # left eye center-ish
                    face.dense[83],  # right eye
                    face.dense[54],  # nose tip
                    face.dense[88],  # mouth left
                    face.dense[92],  # mouth right
                ],
                dtype=np.float32,
            )
        return None
    return np.array([left, right, nose, ml, mr], dtype=np.float32)


def align_ffhq(
    img_bgr: np.ndarray,
    pts5: np.ndarray,
    out_size: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """Similarity transform to FFHQ template. Returns (aligned, M 2x3)."""
    dst = FFHQ_TEMPLATE_5 * (out_size / 1024.0)
    M = cv2.estimateAffinePartial2D(pts5, dst, method=cv2.LMEDS)[0]
    if M is None:
        M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    aligned = cv2.warpAffine(
        img_bgr,
        M,
        (out_size, out_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return aligned, M.astype(np.float32)


def invert_affine(M: np.ndarray) -> np.ndarray:
    A = np.vstack([M, [0.0, 0.0, 1.0]]).astype(np.float32)
    inv = np.linalg.inv(A)
    return inv[:2]


def warp_back(
    edited_aligned: np.ndarray,
    original_bgr: np.ndarray,
    M: np.ndarray,
    mask_aligned: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Warp edited crop + mask back to original resolution."""
    h, w = original_bgr.shape[:2]
    Minv = invert_affine(M)
    edited_full = cv2.warpAffine(
        edited_aligned,
        Minv,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    mask_full = cv2.warpAffine(
        mask_aligned,
        Minv,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return edited_full, mask_full


class StyleGANGenerator(ArtifactGenerator):
    name = 'stylegan'

    def __init__(
        self,
        device: str = 'cuda',
        out_size: int = 1024,
        direction_scale: float = 1.0,
        weights_dir: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        self.out_size = out_size
        self.direction_scale = direction_scale
        self.weights_dir = weights_dir
        self._sg_ready = False
        self._tried_load = False

    def _try_load_stylegan(self):
        if self._tried_load:
            return
        self._tried_load = True
        # Optional heavy stack — kept as a soft dependency.
        try:
            # Placeholder for real e4e/StyleGAN2 integration.
            # Users can drop a `fa_stylegan_runtime.py` providing invert()+edit().
            from . import stylegan_runtime  # type: ignore

            self._runtime = stylegan_runtime
            self._sg_ready = True
            logging.info('StyleGAN runtime loaded')
        except Exception as exc:
            logging.info(
                'StyleGAN runtime not available (%s); using aligned-crop procedural edit',
                exc,
            )
            self._sg_ready = False

    def _latent_edit(self, aligned: np.ndarray, artifact: str, seed: int) -> np.ndarray:
        self._try_load_stylegan()
        if self._sg_ready:
            return self._runtime.edit(aligned, artifact, seed, self.direction_scale, self.device)

        # Lightweight stand-in: modify color/texture inside a central face band.
        out = aligned.copy().astype(np.float32)
        h, w = out.shape[:2]
        if artifact == 'beard':
            y0, y1 = int(h * 0.55), int(h * 0.95)
            x0, x1 = int(w * 0.2), int(w * 0.8)
            region = out[y0:y1, x0:x1]
            region = region * np.array([0.75, 0.8, 0.85], dtype=np.float32)
            noise = np.random.default_rng(seed).normal(0, 6, region.shape)
            out[y0:y1, x0:x1] = region + noise
        elif artifact == 'glasses_glare':
            for cx_frac in (0.38, 0.62):
                cx, cy = int(w * cx_frac), int(h * 0.45)
                cv2.ellipse(out, (cx, cy), (int(w * 0.1), int(h * 0.06)), 0, 0, 360, (220, 230, 240), 3)
                cv2.ellipse(out, (cx - 10, cy - 8), (18, 10), -20, 0, 360, (255, 255, 255), -1)
            # Soft white glare alpha.
            glare = np.zeros_like(out)
            cv2.ellipse(glare, (int(w * 0.38), int(h * 0.43)), (20, 12), -25, 0, 360, (255, 255, 255), -1)
            cv2.ellipse(glare, (int(w * 0.62), int(h * 0.43)), (20, 12), -25, 0, 360, (255, 255, 255), -1)
            out = out * 0.85 + glare * 0.15 + out * 0.0
        elif artifact == 'bangs':
            y0, y1 = int(h * 0.05), int(h * 0.38)
            x0, x1 = int(w * 0.2), int(w * 0.8)
            region = out[y0:y1, x0:x1]
            hair = region * np.array([0.55, 0.55, 0.6], dtype=np.float32)
            # Vertical strand-like noise.
            strand = np.random.default_rng(seed).normal(0, 10, (y1 - y0, 1, 3))
            strand = np.repeat(strand, x1 - x0, axis=1)
            out[y0:y1, x0:x1] = hair + strand * 0.3
        else:  # occluder
            cx, cy = int(w * 0.55), int(h * 0.55)
            cv2.ellipse(out, (cx, cy), (int(w * 0.18), int(h * 0.14)), 25, 0, 360, (90, 100, 140), -1)
        return np.clip(out, 0, 255).astype(np.uint8)

    def generate(
        self,
        img_bgr: np.ndarray,
        face: FaceLandmarks,
        artifact: str,
        mask: np.ndarray,
        rng: np.random.Generator,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        use_seed = seed if seed is not None else int(rng.integers(0, 2**31 - 1))
        pts5 = _five_points(face)
        if pts5 is None:
            # No alignment possible — edit in place inside mask.
            edited = self._latent_edit(
                cv2.resize(img_bgr, (self.out_size, self.out_size)),
                artifact,
                use_seed,
            )
            edited = cv2.resize(edited, (img_bgr.shape[1], img_bgr.shape[0]))
            out = paste_back(img_bgr, edited, mask, feather=21)
            return GenerationResult(
                image_bgr=out,
                change_mask=mask,
                prompt=f'stylegan:{artifact}',
                extra={'aligned': False, 'sg_ready': self._sg_ready},
            )

        aligned, M = align_ffhq(img_bgr, pts5, self.out_size)
        # Warp the ROI mask into aligned space for paste-back restriction.
        mask_aligned = cv2.warpAffine(
            mask,
            M,
            (self.out_size, self.out_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        if int((mask_aligned > 16).sum()) < 50:
            # Ensure some central face mask so edit is visible.
            mask_aligned = np.zeros((self.out_size, self.out_size), dtype=np.uint8)
            cv2.ellipse(
                mask_aligned,
                (self.out_size // 2, int(self.out_size * 0.55)),
                (int(self.out_size * 0.28), int(self.out_size * 0.38)),
                0,
                0,
                360,
                255,
                -1,
            )

        edited_aligned = self._latent_edit(aligned, artifact, use_seed)
        # Keep outside-mask pixels of aligned crop unchanged.
        alpha = (mask_aligned.astype(np.float32) / 255.0)[..., None]
        edited_aligned = (
            aligned.astype(np.float32) * (1 - alpha) + edited_aligned.astype(np.float32) * alpha
        ).astype(np.uint8)

        edited_full, mask_full = warp_back(edited_aligned, img_bgr, M, mask_aligned)
        # Intersect with original mask to avoid leaking alignment border.
        change = cv2.bitwise_and(mask_full, ((mask > 16).astype(np.uint8) * 255))
        if int(change.sum()) < 100:
            change = mask_full
        out = paste_back(img_bgr, edited_full, change, feather=21)
        return GenerationResult(
            image_bgr=out,
            change_mask=change,
            prompt=f'stylegan:{artifact}',
            extra={'aligned': True, 'sg_ready': self._sg_ready},
        )
