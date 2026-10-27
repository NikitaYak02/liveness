"""Map public landmark layouts onto the internal 30-point scheme.

Logic mirrors face_kp_train_tools/convert_wflw_lapa_keypoints.py.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .scheme import KP_NAMES

Point = Tuple[float, float]

DIRECT_POINTS = {
    "wflw": {
        "eye_right_outer": 60,
        "eyelid_right_upper_outer": 61,
        "eyelid_right_upper_inner": 63,
        "eye_right_inner": 64,
        "eyelid_right_lower_inner": 65,
        "eyelid_right_lower_outer": 67,
        "pupil_right": 96,
        "eye_left_inner": 68,
        "eyelid_left_upper_inner": 69,
        "eyelid_left_upper_outer": 71,
        "eye_left_outer": 72,
        "eyelid_left_lower_outer": 73,
        "eyelid_left_lower_inner": 75,
        "pupil_left": 97,
        "nose_tip": 54,
        "mouth_right": 76,
        "upper_lip_top": 79,
        "mouth_left": 82,
        "under_lip_bottom": 85,
    },
    "lapa": {
        "eye_right_outer": 66,
        "eyelid_right_upper_outer": 67,
        "eyelid_right_upper_inner": 69,
        "eye_right_inner": 70,
        "eyelid_right_lower_inner": 71,
        "eyelid_right_lower_outer": 73,
        "pupil_right": 104,
        "eye_left_inner": 75,
        "eyelid_left_upper_inner": 76,
        "eyelid_left_upper_outer": 78,
        "eye_left_outer": 79,
        "eyelid_left_lower_outer": 80,
        "eyelid_left_lower_inner": 82,
        "pupil_left": 105,
        "nose_tip": 54,
        "mouth_right": 84,
        "upper_lip_top": 87,
        "mouth_left": 90,
        "under_lip_bottom": 93,
    },
    "300w": {
        "eye_right_outer": 36,
        "eyelid_right_upper_outer": 37,
        "eyelid_right_upper_inner": 38,
        "eye_right_inner": 39,
        "eyelid_right_lower_inner": 40,
        "eyelid_right_lower_outer": 41,
        "eye_left_inner": 42,
        "eyelid_left_upper_inner": 43,
        "eyelid_left_upper_outer": 44,
        "eye_left_outer": 45,
        "eyelid_left_lower_outer": 46,
        "eyelid_left_lower_inner": 47,
        "eyebrow_right_outer": 17,
        "eyebrow_right_center": 19,
        "eyebrow_right_inner": 21,
        "eyebrow_left_inner": 22,
        "eyebrow_left_center": 24,
        "eyebrow_left_outer": 26,
        "nose_tip": 30,
        "mouth_right": 48,
        "upper_lip_top": 51,
        "mouth_left": 54,
        "under_lip_bottom": 57,
    },
}

BROW_INDICES = {
    "right": tuple(range(33, 42)),
    "left": tuple(range(42, 51)),
}

LIVENESS_ANCHORS = {
    "wflw": {"cheek_right": 6, "cheek_left": 26, "chin": 16, "nose_bridge": 51},
    "lapa": {"cheek_right": 6, "cheek_left": 26, "chin": 16, "nose_bridge": 51},
    "300w": {"cheek_right": 3, "cheek_left": 13, "chin": 8, "nose_bridge": 27},
}

FOREHEAD_BROW_SCALE = 1.5
CHIN_INDEX = 16
NOSE_BRIDGE_INDEX = 51


def _point_at(landmarks: np.ndarray, index: int) -> Optional[Point]:
    if index < 0 or index >= len(landmarks):
        return None
    x, y = float(landmarks[index, 0]), float(landmarks[index, 1])
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    return (x, y)


def _unit(vec: np.ndarray) -> Optional[np.ndarray]:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return None
    return vec / norm


def face_axes(landmarks: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if landmarks.shape[0] <= max(CHIN_INDEX, NOSE_BRIDGE_INDEX):
        # 300W uses different chin / bridge indices; fall back via anchors later.
        chin_i = 8 if landmarks.shape[0] == 68 else CHIN_INDEX
        bridge_i = 27 if landmarks.shape[0] == 68 else NOSE_BRIDGE_INDEX
        if landmarks.shape[0] <= max(chin_i, bridge_i):
            return None
        up = _unit(landmarks[bridge_i] - landmarks[chin_i])
    else:
        up = _unit(landmarks[NOSE_BRIDGE_INDEX] - landmarks[CHIN_INDEX])
    if up is None:
        return None
    subject_left = np.array([-up[1], up[0]], dtype=np.float64)
    return up, subject_left


def select_eyebrow(
    points: np.ndarray,
    up: np.ndarray,
    outward: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rel = points - points.mean(axis=0)
    lateral = rel @ outward
    height = rel @ up
    outer_i = int(np.argmax(lateral))
    inner_i = int(np.argmin(lateral))
    mask = np.ones(len(points), dtype=bool)
    mask[outer_i] = False
    if inner_i != outer_i:
        mask[inner_i] = False
    if mask.any():
        candidates = np.flatnonzero(mask)
        center_i = int(candidates[np.argmax(height[candidates])])
    else:
        center_i = outer_i
    return points[outer_i], points[center_i], points[inner_i]


def forehead_point(
    bridge: Sequence[float],
    chin: Sequence[float],
    brow_right: Sequence[float],
    brow_left: Sequence[float],
) -> Optional[Point]:
    bridge_v = np.asarray(bridge, dtype=np.float64)
    up = _unit(bridge_v - np.asarray(chin, dtype=np.float64))
    if up is None:
        return None
    brow_mid = 0.5 * (
        np.asarray(brow_right, dtype=np.float64) + np.asarray(brow_left, dtype=np.float64)
    )
    height = float(np.dot(brow_mid - bridge_v, up))
    if height < 1.0:
        return None
    pt = bridge_v + up * (FOREHEAD_BROW_SCALE * height)
    if not np.isfinite(pt).all():
        return None
    return (float(pt[0]), float(pt[1]))


def map_landmarks(landmarks: np.ndarray, dataset: str) -> Dict[str, Point]:
    """Named internal points present for this sample (missing names omitted)."""
    dataset = "300w" if dataset == "300w" else dataset.lower()
    if dataset not in DIRECT_POINTS:
        raise ValueError(f"unknown dataset {dataset!r}")
    wanted = set(KP_NAMES)
    out: Dict[str, Point] = {}
    for name, index in DIRECT_POINTS[dataset].items():
        if name not in wanted:
            continue
        pt = _point_at(landmarks, index)
        if pt is not None:
            out[name] = pt

    if dataset in ("wflw", "lapa"):
        axes = face_axes(landmarks)
        if axes is not None:
            up, subject_left = axes
            outward = {"right": -subject_left, "left": subject_left}
            for side, indices in BROW_INDICES.items():
                if max(indices) >= len(landmarks):
                    continue
                brow = landmarks[list(indices)]
                if not np.isfinite(brow).all():
                    continue
                outer, center, inner = select_eyebrow(brow, up, outward[side])
                out[f"eyebrow_{side}_outer"] = (float(outer[0]), float(outer[1]))
                out[f"eyebrow_{side}_center"] = (float(center[0]), float(center[1]))
                out[f"eyebrow_{side}_inner"] = (float(inner[0]), float(inner[1]))

    for name, index in LIVENESS_ANCHORS[dataset].items():
        if name not in wanted:
            continue
        pt = _point_at(landmarks, index)
        if pt is not None:
            out[name] = pt

    if "forehead" in wanted:
        if dataset == "300w":
            brow_right = _point_at(landmarks, DIRECT_POINTS["300w"]["eyebrow_right_center"])
            brow_left = _point_at(landmarks, DIRECT_POINTS["300w"]["eyebrow_left_center"])
        else:
            brow_right = out.get("eyebrow_right_center")
            brow_left = out.get("eyebrow_left_center")
        chin = _point_at(landmarks, LIVENESS_ANCHORS[dataset]["chin"])
        bridge = _point_at(landmarks, LIVENESS_ANCHORS[dataset]["nose_bridge"])
        if None not in (brow_right, brow_left, chin, bridge):
            pt = forehead_point(bridge, chin, brow_right, brow_left)  # type: ignore[arg-type]
            if pt is not None:
                out["forehead"] = pt

    return {name: out[name] for name in KP_NAMES if name in out}


def mapped_to_arrays(mapped: Dict[str, Point]) -> tuple[np.ndarray, np.ndarray]:
    """Return ``[30, 2]`` coords and ``[30]`` visibility mask (1 = present)."""
    coords = np.zeros((len(KP_NAMES), 2), dtype=np.float32)
    visible = np.zeros(len(KP_NAMES), dtype=np.float32)
    for index, name in enumerate(KP_NAMES):
        pt = mapped.get(name)
        if pt is None:
            continue
        coords[index] = (float(pt[0]), float(pt[1]))
        visible[index] = 1.0
    return coords, visible
