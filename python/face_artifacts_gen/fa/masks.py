"""Region mask builders for beard, bangs, glasses, and occluders."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from .io_utils import feather_mask
from .landmarks import FaceLandmarks


def _clamp_pt(pt: Tuple[float, float], w: int, h: int) -> Tuple[int, int]:
    return (
        int(np.clip(round(pt[0]), 0, w - 1)),
        int(np.clip(round(pt[1]), 0, h - 1)),
    )


def _face_scale(face: FaceLandmarks, img_shape: Tuple[int, int]) -> float:
    if face.rect:
        return float(max(face.rect[2], face.rect[3]))
    pts = list(face.points.values())
    if len(pts) >= 2:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return float(max(max(xs) - min(xs), max(ys) - min(ys), 1.0))
    h, w = img_shape
    return float(min(h, w) * 0.4)


def _jitter(rng: np.random.Generator, scale: float, amount: float = 0.05) -> float:
    return float(1.0 + rng.uniform(-amount, amount)) * scale


def build_beard_mask(
    shape: Tuple[int, int],
    face: FaceLandmarks,
    rng: np.random.Generator,
    mouth_cutout: bool = True,
    feather: int = 21,
) -> np.ndarray:
    """Polygon over lower face (cheeks → jaw → chin), optional mouth cutout."""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    scale = _face_scale(face, shape)

    # Prefer dense jaw contour from InsightFace 2d106.
    poly: list = []
    if face.dense is not None and len(face.dense) >= 33:
        jaw = face.dense[0:33]
        # Expand slightly outward / downward.
        expand = scale * rng.uniform(0.02, 0.08)
        cx = float(np.mean(jaw[:, 0]))
        cy = float(np.mean(jaw[:, 1]))
        for x, y in jaw:
            dx, dy = x - cx, y - cy
            poly.append((x + dx * 0.05, y + dy * 0.05 + expand * 0.3))
        # Close with upper cheek / nose-base line.
        nose = face.get('kp_nose_tip') or face.get('kp_nose_bridge')
        if nose is None and len(face.dense) > 54:
            nose = tuple(map(float, face.dense[54]))
        if nose:
            left_cheek = face.get('kp_cheek_left') or tuple(map(float, face.dense[2]))
            right_cheek = face.get('kp_cheek_right') or tuple(map(float, face.dense[30]))
            # Insert upper boundary going left→right above mouth.
            upper_y = nose[1] + scale * rng.uniform(0.02, 0.12)
            poly = (
                [(left_cheek[0], upper_y)]
                + [(p[0], max(p[1], upper_y)) for p in poly]
                + [(right_cheek[0], upper_y)]
            )
    else:
        # Sparse keypoints fallback.
        chin = face.get('kp_chin')
        nose = face.get('kp_nose_tip') or face.get('kp_nose_bridge')
        ml = face.get('kp_mouth_left')
        mr = face.get('kp_mouth_right')
        cl = face.get('kp_cheek_left')
        cr = face.get('kp_cheek_right')
        if chin is None and face.rect:
            x, y, fw, fh = face.rect
            chin = (x + fw * 0.5, y + fh * 0.95)
        if nose is None and face.rect:
            x, y, fw, fh = face.rect
            nose = (x + fw * 0.5, y + fh * 0.55)
        if not (chin and nose):
            return mask
        if ml is None:
            ml = (nose[0] - scale * 0.18, nose[1] + scale * 0.12)
        if mr is None:
            mr = (nose[0] + scale * 0.18, nose[1] + scale * 0.12)
        if cl is None:
            cl = (nose[0] - scale * 0.35, nose[1] + scale * 0.05)
        if cr is None:
            cr = (nose[0] + scale * 0.35, nose[1] + scale * 0.05)
        drop = scale * rng.uniform(0.05, 0.18)
        poly = [
            cl,
            (ml[0], nose[1] + scale * 0.05),
            (mr[0], nose[1] + scale * 0.05),
            cr,
            (cr[0], chin[1] + drop),
            (chin[0], chin[1] + drop * 1.2),
            (cl[0], chin[1] + drop),
        ]

    pts = np.array([_clamp_pt(p, w, h) for p in poly], dtype=np.int32)
    if len(pts) >= 3:
        cv2.fillPoly(mask, [pts], 255)

    if mouth_cutout:
        ul = face.get('kp_upper_lip_top')
        ll = face.get('kp_under_lip_bottom')
        ml = face.get('kp_mouth_left')
        mr = face.get('kp_mouth_right')
        if ul and ll and ml and mr:
            mw = max(abs(mr[0] - ml[0]), 1.0)
            rx = int(mw * rng.uniform(0.55, 0.75))
            ry = int(max(abs(ll[1] - ul[1]), mw * 0.25) * rng.uniform(0.7, 1.1))
            cx = int((ml[0] + mr[0]) / 2)
            cy = int((ul[1] + ll[1]) / 2)
            cv2.ellipse(mask, (cx, cy), (max(1, rx), max(1, ry)), 0, 0, 360, 0, -1)

    # Soften edges for blending / inpaint.
    alpha = feather_mask(mask, feather)
    return (alpha * 255).astype(np.uint8)


def build_bangs_mask(
    shape: Tuple[int, int],
    face: FaceLandmarks,
    rng: np.random.Generator,
    feather: int = 25,
) -> np.ndarray:
    """Horizontal band from eyebrows up to forehead / hairline."""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    scale = _face_scale(face, shape)

    brow_pts = [
        face.get('kp_eyebrow_left_outer'),
        face.get('kp_eyebrow_left_center'),
        face.get('kp_eyebrow_left_inner'),
        face.get('kp_eyebrow_right_inner'),
        face.get('kp_eyebrow_right_center'),
        face.get('kp_eyebrow_right_outer'),
    ]
    brow_pts = [p for p in brow_pts if p is not None]

    forehead = face.get('kp_forehead')
    if face.dense is not None and len(face.dense) >= 51 and len(brow_pts) < 3:
        # InsightFace eyebrow arcs.
        left = face.dense[33:42]
        right = face.dense[42:51]
        brow_pts = [tuple(map(float, p)) for p in list(left) + list(right)]

    if not brow_pts and face.rect:
        x, y, fw, fh = face.rect
        brow_y = y + fh * 0.28
        brow_pts = [(x + fw * 0.15, brow_y), (x + fw * 0.85, brow_y)]
        forehead = forehead or (x + fw * 0.5, y + fh * 0.05)

    if not brow_pts:
        return mask

    brow_y = float(np.mean([p[1] for p in brow_pts]))
    left_x = min(p[0] for p in brow_pts) - scale * rng.uniform(0.02, 0.08)
    right_x = max(p[0] for p in brow_pts) + scale * rng.uniform(0.02, 0.08)

    if forehead is None:
        forehead = (
            (left_x + right_x) / 2,
            brow_y - scale * rng.uniform(0.25, 0.4),
        )

    top_y = min(forehead[1], brow_y) - scale * rng.uniform(0.05, 0.18)
    # Bangs often hang slightly below brows.
    bottom_y = brow_y + scale * rng.uniform(0.02, 0.12)

    # Irregular bottom edge via a few control points.
    n_ctrl = int(rng.integers(4, 8))
    xs = np.linspace(left_x, right_x, n_ctrl)
    bottom = []
    for x in xs:
        jitter = scale * rng.uniform(-0.04, 0.08)
        bottom.append((float(x), float(bottom_y + jitter)))
    top = [(float(x), float(top_y + scale * rng.uniform(-0.02, 0.03))) for x in xs]

    poly = top + list(reversed(bottom))
    pts = np.array([_clamp_pt(p, w, h) for p in poly], dtype=np.int32)
    if len(pts) >= 3:
        cv2.fillPoly(mask, [pts], 255)

    alpha = feather_mask(mask, feather)
    return (alpha * 255).astype(np.uint8)


def build_glasses_mask(
    shape: Tuple[int, int],
    face: FaceLandmarks,
    rng: np.random.Generator,
    feather: int = 11,
) -> np.ndarray:
    """Two ellipses around eyes plus a bridge band."""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    scale = _face_scale(face, shape)

    def eye_pair(side: str):
        outer = face.get(f'kp_eye_{side}_outer')
        inner = face.get(f'kp_eye_{side}_inner')
        pupil = face.get(f'kp_pupil_{side}')
        return outer, inner, pupil

    lenses = []
    for side in ('left', 'right'):
        outer, inner, pupil = eye_pair(side)
        if outer and inner:
            cx = (outer[0] + inner[0]) / 2
            cy = (outer[1] + inner[1]) / 2
            eye_w = max(abs(outer[0] - inner[0]), 1.0)
            angle = float(np.degrees(np.arctan2(inner[1] - outer[1], inner[0] - outer[0])))
        elif pupil:
            cx, cy = pupil
            eye_w = scale * 0.12
            angle = 0.0
        else:
            continue
        pad = rng.uniform(1.15, 1.55)
        rx = eye_w * pad * 0.7
        ry = eye_w * pad * rng.uniform(0.45, 0.65)
        lenses.append(((cx, cy), (rx, ry), angle))

    if not lenses and face.rect:
        x, y, fw, fh = face.rect
        for frac in (0.35, 0.65):
            lenses.append(
                (
                    (x + fw * frac, y + fh * 0.38),
                    (fw * 0.14, fh * 0.08),
                    0.0,
                )
            )

    for center, axes, angle in lenses:
        cx, cy = _clamp_pt(center, w, h)
        ax = max(1, int(round(axes[0])))
        ay = max(1, int(round(axes[1])))
        cv2.ellipse(mask, (cx, cy), (ax, ay), angle, 0, 360, 255, -1)

    # Bridge between lenses.
    if len(lenses) >= 2:
        c0, a0, _ = lenses[0]
        c1, a1, _ = lenses[1]
        bx = int((c0[0] + c1[0]) / 2)
        by = int((c0[1] + c1[1]) / 2)
        bw = max(2, int(abs(c1[0] - c0[0]) * 0.25))
        bh = max(2, int(min(a0[1], a1[1]) * 0.6))
        cv2.ellipse(mask, (bx, by), (bw, bh), 0, 0, 360, 255, -1)

    # Optional temples extending sideways a bit.
    if lenses and rng.random() < 0.7:
        for center, axes, angle in lenses:
            length = scale * rng.uniform(0.12, 0.22)
            rad = np.radians(angle)
            # Outward from face center.
            face_cx = face.rect[0] + face.rect[2] / 2 if face.rect else np.mean([c[0] for c, _, _ in lenses])
            direction = -1.0 if center[0] < face_cx else 1.0
            end = (
                center[0] + direction * length * np.cos(rad),
                center[1] + direction * length * np.sin(rad) * 0.3,
            )
            thickness = max(1, int(axes[1] * 0.35))
            cv2.line(
                mask,
                _clamp_pt(center, w, h),
                _clamp_pt(end, w, h),
                255,
                thickness,
                lineType=cv2.LINE_AA,
            )

    alpha = feather_mask(mask, feather)
    return (alpha * 255).astype(np.uint8)


def build_occluder_mask(
    shape: Tuple[int, int],
    face: FaceLandmarks,
    rng: np.random.Generator,
    feather: int = 17,
) -> np.ndarray:
    """Random soft blob covering part of the face (hand/object placeholder region)."""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    scale = _face_scale(face, shape)

    if face.rect:
        x, y, fw, fh = face.rect
        cx = x + fw * rng.uniform(0.2, 0.8)
        cy = y + fh * rng.uniform(0.25, 0.85)
    else:
        cx = w * rng.uniform(0.3, 0.7)
        cy = h * rng.uniform(0.3, 0.7)

    rx = scale * rng.uniform(0.18, 0.4)
    ry = scale * rng.uniform(0.15, 0.35)
    angle = float(rng.uniform(-40, 40))
    cv2.ellipse(
        mask,
        _clamp_pt((cx, cy), w, h),
        (max(1, int(rx)), max(1, int(ry))),
        angle,
        0,
        360,
        255,
        -1,
    )
    # Add a second overlapping blob for irregularity.
    if rng.random() < 0.6:
        cx2 = cx + scale * rng.uniform(-0.15, 0.15)
        cy2 = cy + scale * rng.uniform(-0.15, 0.15)
        cv2.ellipse(
            mask,
            _clamp_pt((cx2, cy2), w, h),
            (max(1, int(rx * 0.7)), max(1, int(ry * 0.7))),
            angle + rng.uniform(-20, 20),
            0,
            360,
            255,
            -1,
        )

    alpha = feather_mask(mask, feather)
    return (alpha * 255).astype(np.uint8)


def build_artifact_mask(
    artifact: str,
    shape: Tuple[int, int],
    face: FaceLandmarks,
    rng: np.random.Generator,
) -> np.ndarray:
    if artifact == 'beard':
        return build_beard_mask(shape, face, rng)
    if artifact == 'bangs':
        return build_bangs_mask(shape, face, rng)
    if artifact == 'glasses_glare':
        return build_glasses_mask(shape, face, rng)
    if artifact == 'occluder':
        return build_occluder_mask(shape, face, rng)
    raise ValueError(f'unknown artifact: {artifact!r}')
