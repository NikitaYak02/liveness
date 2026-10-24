"""Method 5: Procedural glasses (tint/refraction/glare) + NatOcc-style occluder pasting.

No neural network required. Works on CPU. Uses optional PNG assets from
python/face_artifacts_gen/assets/{glasses,occluders,env_maps}.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..io_utils import feather_mask, paste_back
from ..landmarks import FaceLandmarks
from .base import ArtifactGenerator, GenerationResult

ASSETS_ROOT = Path(__file__).resolve().parents[2] / 'assets'


def _list_pngs(folder: Path) -> List[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() == '.png')


def _eye_geometry(face: FaceLandmarks) -> Optional[dict]:
    lo = face.get('kp_eye_left_outer')
    li = face.get('kp_eye_left_inner')
    ro = face.get('kp_eye_right_outer')
    ri = face.get('kp_eye_right_inner')
    lp = face.get('kp_pupil_left')
    rp = face.get('kp_pupil_right')

    if lo and li:
        lcx = (lo[0] + li[0]) / 2
        lcy = (lo[1] + li[1]) / 2
        lw = max(abs(lo[0] - li[0]), 1.0)
        lang = np.degrees(np.arctan2(li[1] - lo[1], li[0] - lo[0]))
    elif lp:
        lcx, lcy, lw, lang = lp[0], lp[1], 20.0, 0.0
    else:
        return None

    if ro and ri:
        rcx = (ro[0] + ri[0]) / 2
        rcy = (ro[1] + ri[1]) / 2
        rw = max(abs(ro[0] - ri[0]), 1.0)
        rang = np.degrees(np.arctan2(ri[1] - ro[1], ri[0] - ro[0]))
    elif rp:
        rcx, rcy, rw, rang = rp[0], rp[1], 20.0, 0.0
    else:
        return None

    return {
        'left': (lcx, lcy, lw, lang),
        'right': (rcx, rcy, rw, rang),
        'angle': float((lang + rang) / 2),
        'bridge': ((lcx + rcx) / 2, (lcy + rcy) / 2),
    }


def _draw_frame(
    img: np.ndarray,
    geo: dict,
    frame_color: Tuple[int, int, int],
    thickness: int,
) -> np.ndarray:
    out = img.copy()
    for key in ('left', 'right'):
        cx, cy, ew, ang = geo[key]
        rx, ry = int(ew * 0.75), int(ew * 0.45)
        cv2.ellipse(
            out,
            (int(cx), int(cy)),
            (rx, ry),
            ang,
            0,
            360,
            frame_color,
            thickness,
            lineType=cv2.LINE_AA,
        )
    # Bridge.
    bx, by = geo['bridge']
    cv2.line(
        out,
        (int(geo['left'][0] + geo['left'][2] * 0.35), int(geo['left'][1])),
        (int(geo['right'][0] - geo['right'][2] * 0.35), int(geo['right'][1])),
        frame_color,
        max(1, thickness - 1),
        lineType=cv2.LINE_AA,
    )
    # Temples.
    for key, direction in (('left', -1), ('right', 1)):
        cx, cy, ew, ang = geo[key]
        length = ew * 1.6
        rad = np.radians(ang)
        end = (cx + direction * length * np.cos(rad), cy + direction * length * 0.15)
        cv2.line(
            out,
            (int(cx + direction * ew * 0.7), int(cy)),
            (int(end[0]), int(end[1])),
            frame_color,
            max(1, thickness - 1),
            lineType=cv2.LINE_AA,
        )
    return out


def _lens_mask(shape: Tuple[int, int], geo: dict, pad: float = 0.95) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for key in ('left', 'right'):
        cx, cy, ew, ang = geo[key]
        rx, ry = int(ew * 0.7 * pad), int(ew * 0.42 * pad)
        cv2.ellipse(mask, (int(cx), int(cy)), (max(1, rx), max(1, ry)), ang, 0, 360, 255, -1)
    return mask


def _apply_lens_tint(img: np.ndarray, lens: np.ndarray, tint_bgr: Tuple[int, int, int], alpha: float) -> np.ndarray:
    out = img.astype(np.float32)
    a = (lens.astype(np.float32) / 255.0 * alpha)[..., None]
    tint = np.array(tint_bgr, dtype=np.float32)
    out = out * (1 - a) + (out * 0.55 + tint * 0.45) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_refraction(img: np.ndarray, lens: np.ndarray, strength: float = 1.5) -> np.ndarray:
    """Simple radial-ish pixel shift inside lenses (cheap refraction look)."""
    h, w = img.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    # Shift toward lens centroids approximated via distance transform.
    dist = cv2.distanceTransform((lens > 0).astype(np.uint8), cv2.DIST_L2, 3)
    gx = cv2.Sobel(dist, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(dist, cv2.CV_32F, 0, 1, ksize=3)
    map_x = xs + gx * strength * 0.05
    map_y = ys + gy * strength * 0.05
    warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    a = (lens.astype(np.float32) / 255.0)[..., None]
    return np.clip(img.astype(np.float32) * (1 - a) + warped.astype(np.float32) * a, 0, 255).astype(np.uint8)


def _apply_glare(
    img: np.ndarray,
    geo: dict,
    rng: np.random.Generator,
    env_bgr: Optional[np.ndarray] = None,
) -> np.ndarray:
    out = img.astype(np.float32)
    h, w = img.shape[:2]
    glare = np.zeros_like(out)

    for key in ('left', 'right'):
        cx, cy, ew, ang = geo[key]
        # Primary specular ellipse.
        ax = int(ew * rng.uniform(0.25, 0.45))
        ay = int(ew * rng.uniform(0.12, 0.25))
        offset_x = ew * rng.uniform(-0.25, 0.15)
        offset_y = ew * rng.uniform(-0.3, -0.05)
        center = (int(cx + offset_x), int(cy + offset_y))
        cv2.ellipse(
            glare,
            center,
            (max(1, ax), max(1, ay)),
            ang + rng.uniform(-30, 10),
            0,
            360,
            (255, 255, 255),
            -1,
            lineType=cv2.LINE_AA,
        )
        # Secondary smaller streak.
        if rng.random() < 0.7:
            cv2.ellipse(
                glare,
                (int(cx + ew * 0.2), int(cy - ew * 0.1)),
                (max(1, int(ew * 0.15)), max(1, int(ew * 0.05))),
                ang - 40,
                0,
                360,
                (240, 245, 255),
                -1,
                lineType=cv2.LINE_AA,
            )

    glare = cv2.GaussianBlur(glare, (0, 0), sigmaX=2.5)
    intensity = float(rng.uniform(0.35, 0.7))

    if env_bgr is not None:
        env = cv2.resize(env_bgr, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        # Window/screen-like reflection modulated by glare mask.
        galpha = (glare.max(axis=2, keepdims=True) / 255.0) * intensity * 0.6
        out = out * (1 - galpha) + env * galpha + glare * (intensity * 0.4)
    else:
        galpha = (glare / 255.0) * intensity
        out = out * (1 - galpha * 0.85) + glare * 0.85 * intensity + out * 0

    return np.clip(out, 0, 255).astype(np.uint8)


def apply_glasses(
    img_bgr: np.ndarray,
    face: FaceLandmarks,
    rng: np.random.Generator,
    assets_root: Path = ASSETS_ROOT,
) -> Tuple[np.ndarray, np.ndarray]:
    geo = _eye_geometry(face)
    if geo is None:
        logging.warning('glasses: missing eye landmarks')
        return img_bgr, np.zeros(img_bgr.shape[:2], dtype=np.uint8)

    frame_colors = [
        (20, 20, 20),
        (40, 40, 50),
        (30, 60, 90),
        (10, 10, 10),
        (80, 80, 90),
    ]
    tints = [
        (180, 200, 210),
        (160, 180, 200),
        (140, 160, 140),
        (120, 140, 180),
        (200, 200, 210),
    ]
    frame_color = frame_colors[int(rng.integers(0, len(frame_colors)))]
    tint = tints[int(rng.integers(0, len(tints)))]
    thickness = int(rng.integers(2, 5))
    tint_alpha = float(rng.uniform(0.25, 0.55))

    out = _draw_frame(img_bgr, geo, frame_color, thickness)
    lens = _lens_mask(img_bgr.shape[:2], geo)
    out = _apply_lens_tint(out, lens, tint, tint_alpha)
    out = _apply_refraction(out, lens, strength=float(rng.uniform(1.0, 2.5)))

    env = None
    envs = _list_pngs(assets_root / 'env_maps')
    if envs:
        env = cv2.imread(str(envs[int(rng.integers(0, len(envs)))]), cv2.IMREAD_COLOR)
    out = _apply_glare(out, geo, rng, env_bgr=env)

    # Change mask = lenses + frame neighborhood.
    change = lens.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    change = cv2.dilate(change, k)
    # Also mark frame by diff.
    diff = cv2.absdiff(img_bgr, out)
    diff_m = (cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) > 8).astype(np.uint8) * 255
    change = cv2.bitwise_or(change, diff_m)
    return out, change


def _color_transfer_lab(src: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Match mean/std of src to ref inside mask (LAB)."""
    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    m = mask > 16
    if m.sum() < 10:
        return src
    out = src_lab.copy()
    for c in range(3):
        s = src_lab[:, :, c][m]
        r = ref_lab[:, :, c][m]
        s_mean, s_std = float(s.mean()), float(s.std() + 1e-6)
        r_mean, r_std = float(r.mean()), float(r.std() + 1e-6)
        adj = (src_lab[:, :, c] - s_mean) * (r_std / s_std) + r_mean
        out[:, :, c] = np.where(m, adj, src_lab[:, :, c])
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)


def _synthetic_occluder(h: int, w: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a simple hand/object-like RGBA-ish patch."""
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    skin = (
        int(rng.integers(60, 120)),
        int(rng.integers(90, 160)),
        int(rng.integers(140, 210)),
    )
    # Palm.
    cv2.ellipse(
        canvas,
        (w // 2, int(h * 0.55)),
        (int(w * 0.28), int(h * 0.32)),
        float(rng.uniform(-20, 20)),
        0,
        360,
        skin,
        -1,
    )
    cv2.ellipse(
        mask,
        (w // 2, int(h * 0.55)),
        (int(w * 0.28), int(h * 0.32)),
        0,
        0,
        360,
        255,
        -1,
    )
    # Fingers.
    for i in range(4):
        fx = int(w * (0.3 + i * 0.12))
        fy0 = int(h * 0.45)
        fy1 = int(h * rng.uniform(0.05, 0.25))
        cv2.line(canvas, (fx, fy0), (fx + int(rng.integers(-8, 8)), fy1), skin, int(w * 0.08))
        cv2.line(mask, (fx, fy0), (fx + int(rng.integers(-8, 8)), fy1), 255, int(w * 0.08))
    # Soft shading.
    shade = cv2.GaussianBlur(canvas, (0, 0), sigmaX=3)
    canvas = cv2.addWeighted(canvas, 0.7, shade, 0.3, 0)
    return canvas, mask


def apply_occluder(
    img_bgr: np.ndarray,
    face: FaceLandmarks,
    mask_hint: np.ndarray,
    rng: np.random.Generator,
    assets_root: Path = ASSETS_ROOT,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img_bgr.shape[:2]
    assets = _list_pngs(assets_root / 'occluders')

    if face.rect:
        fx, fy, fw, fh = face.rect
    else:
        fx, fy, fw, fh = w * 0.25, h * 0.2, w * 0.5, h * 0.6

    # Place occluder roughly over the hint mask centroid, else random on face.
    ys, xs = np.where(mask_hint > 16)
    if len(xs) > 0:
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
    else:
        cx = fx + fw * float(rng.uniform(0.3, 0.7))
        cy = fy + fh * float(rng.uniform(0.3, 0.7))

    target_w = int(fw * float(rng.uniform(0.35, 0.7)))
    target_h = int(fh * float(rng.uniform(0.3, 0.65)))
    target_w = max(16, target_w)
    target_h = max(16, target_h)

    if assets:
        rgba = cv2.imread(str(assets[int(rng.integers(0, len(assets)))]), cv2.IMREAD_UNCHANGED)
        if rgba is not None and rgba.shape[2] == 4:
            patch = rgba[:, :, :3]
            alpha = rgba[:, :, 3]
        elif rgba is not None:
            patch = rgba
            gray = cv2.cvtColor(rgba, cv2.COLOR_BGR2GRAY)
            alpha = ((gray < 250) & (gray > 5)).astype(np.uint8) * 255
        else:
            patch, alpha = _synthetic_occluder(target_h, target_w, rng)
    else:
        patch, alpha = _synthetic_occluder(target_h, target_w, rng)

    patch = cv2.resize(patch, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    alpha = cv2.resize(alpha, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # Color transfer toward local face region.
    x0 = int(np.clip(cx - target_w / 2, 0, w - target_w))
    y0 = int(np.clip(cy - target_h / 2, 0, h - target_h))
    ref = img_bgr[y0 : y0 + target_h, x0 : x0 + target_w]
    if ref.shape[:2] == patch.shape[:2]:
        patch = _color_transfer_lab(patch, ref, alpha)

    # Optional rotation.
    angle = float(rng.uniform(-35, 35))
    M = cv2.getRotationMatrix2D((target_w / 2, target_h / 2), angle, 1.0)
    patch = cv2.warpAffine(patch, M, (target_w, target_h), borderMode=cv2.BORDER_CONSTANT)
    alpha = cv2.warpAffine(alpha, M, (target_w, target_h), borderMode=cv2.BORDER_CONSTANT)

    out = img_bgr.copy()
    change = np.zeros((h, w), dtype=np.uint8)
    a = (alpha.astype(np.float32) / 255.0)[..., None]
    roi = out[y0 : y0 + target_h, x0 : x0 + target_w].astype(np.float32)
    blended = roi * (1 - a) + patch.astype(np.float32) * a
    out[y0 : y0 + target_h, x0 : x0 + target_w] = np.clip(blended, 0, 255).astype(np.uint8)
    change[y0 : y0 + target_h, x0 : x0 + target_w] = np.maximum(
        change[y0 : y0 + target_h, x0 : x0 + target_w],
        alpha,
    )
    # Soft shadow under occluder.
    shadow = cv2.GaussianBlur(change, (0, 0), sigmaX=5)
    shadow_shift = np.roll(np.roll(shadow, 4, axis=0), 3, axis=1)
    sa = (shadow_shift.astype(np.float32) / 255.0 * 0.25)[..., None]
    out = np.clip(out.astype(np.float32) * (1 - sa), 0, 255).astype(np.uint8)
    return out, change


def apply_beard_procedural(
    img_bgr: np.ndarray,
    face: FaceLandmarks,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stipple / noise hair texture inside beard mask (CPU baseline)."""
    out = img_bgr.astype(np.float32)
    a = feather_mask(mask, 21)[..., None]
    # Darken + add oriented noise.
    hair_color = np.array(
        [
            float(rng.uniform(20, 60)),
            float(rng.uniform(30, 70)),
            float(rng.uniform(40, 90)),
        ],
        dtype=np.float32,
    )
    noise = rng.normal(0, 12, out.shape).astype(np.float32)
    # Stretch noise vertically a bit for hair-like streaks.
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=0.8, sigmaY=2.5)
    textured = out * 0.55 + hair_color * 0.45 + noise
    out = out * (1 - a * 0.85) + textured * (a * 0.85)
    return np.clip(out, 0, 255).astype(np.uint8), mask


def apply_bangs_procedural(
    img_bgr: np.ndarray,
    face: FaceLandmarks,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    out = img_bgr.astype(np.float32)
    a = feather_mask(mask, 25)[..., None]
    hair = np.array(
        [
            float(rng.uniform(15, 50)),
            float(rng.uniform(20, 55)),
            float(rng.uniform(25, 70)),
        ],
        dtype=np.float32,
    )
    h, w = img_bgr.shape[:2]
    # Vertical strands.
    strand = rng.normal(0, 1, (h, 1, 3)).astype(np.float32)
    strand = np.repeat(strand, w, axis=1)
    strand = cv2.GaussianBlur(strand, (0, 0), sigmaX=1.5, sigmaY=0.3) * 18
    textured = out * 0.35 + hair * 0.65 + strand
    out = out * (1 - a * 0.9) + textured * (a * 0.9)
    return np.clip(out, 0, 255).astype(np.uint8), mask


class ProceduralGenerator(ArtifactGenerator):
    name = 'procedural'

    def __init__(self, device: str = 'cpu', assets_root: Optional[str] = None, **kwargs):
        super().__init__(device=device, **kwargs)
        self.assets_root = Path(assets_root) if assets_root else ASSETS_ROOT

    def generate(
        self,
        img_bgr: np.ndarray,
        face: FaceLandmarks,
        artifact: str,
        mask: np.ndarray,
        rng: np.random.Generator,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        if artifact == 'glasses_glare':
            out, change = apply_glasses(img_bgr, face, rng, self.assets_root)
        elif artifact == 'occluder':
            out, change = apply_occluder(img_bgr, face, mask, rng, self.assets_root)
        elif artifact == 'beard':
            out, change = apply_beard_procedural(img_bgr, face, mask, rng)
        elif artifact == 'bangs':
            out, change = apply_bangs_procedural(img_bgr, face, mask, rng)
        else:
            raise ValueError(f'procedural: unsupported artifact {artifact!r}')

        # Ensure outside original ROI (when meaningful) stays stable for beard/bangs.
        if artifact in ('beard', 'bangs') and mask is not None:
            out = paste_back(img_bgr, out, mask, feather=15)
            change = mask
        return GenerationResult(
            image_bgr=out,
            change_mask=change,
            prompt=f'procedural:{artifact}',
            extra={'assets_root': str(self.assets_root)},
        )
