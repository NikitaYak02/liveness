"""Markup update: mark keypoints under change_mask as occluded / missing."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .face_kp_common import (
    KP_NAMES,
    MISSING_PT,
    convert_old_markup_to_new,
    is_new_face_markup,
    iter_markup_faces,
    read_new_keypoint,
    read_point,
    set_missing,
)

from .constants import ARTIFACT_KP_GROUPS, DEFAULT_OCCLUDED_POLICY


def _point_under_mask(pt: Tuple[float, float], mask: np.ndarray, thresh: int = 32) -> bool:
    h, w = mask.shape[:2]
    x, y = int(round(pt[0])), int(round(pt[1]))
    if x < 0 or y < 0 or x >= w or y >= h:
        return False
    return int(mask[y, x]) >= thresh


def _collect_occluded_names(
    points: Dict[str, Tuple[float, float]],
    mask: np.ndarray,
    artifact: str,
) -> List[str]:
    names = list(ARTIFACT_KP_GROUPS.get(artifact) or ())
    if not names:
        # Dynamic: any known kp under mask.
        names = [n for n in KP_NAMES if n in points]
    occluded = []
    for name in names:
        pt = points.get(name)
        if pt is not None and _point_under_mask(pt, mask):
            occluded.append(name)
    # Also check any other present points under dense occluder.
    if artifact == 'occluder':
        for name, pt in points.items():
            if name not in occluded and _point_under_mask(pt, mask):
                occluded.append(name)
    return occluded


def _points_from_props(props: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
    out = {}
    for name in KP_NAMES:
        pt = read_point(props, name)
        if pt is not None:
            out[name] = pt
    return out


def _points_from_face(face: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
    out = {}
    kps = face.get('keypoints') or {}
    for name in KP_NAMES:
        short = name[3:] if name.startswith('kp_') else name
        pt = read_new_keypoint(kps, short)
        if pt is not None:
            out[name] = pt
    return out


def update_markup(
    data: Optional[Dict[str, Any]],
    change_mask: np.ndarray,
    artifact: str,
    policy: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Return updated markup copy.

    policy:
      - keep: keep coordinates, set occluded=True flag on points / face meta
      - missing: set coordinates to MISSING_PT (old format) or drop/null (new)
    """
    if data is None:
        return None
    policy = policy or DEFAULT_OCCLUDED_POLICY.get(artifact, 'keep')
    out = copy.deepcopy(data)

    if is_new_face_markup(out) or 'main_face' in out:
        if not is_new_face_markup(out):
            out = convert_old_markup_to_new(out)
        for face_key, face in list(iter_markup_faces(out)):
            pts = _points_from_face(face)
            occluded = _collect_occluded_names(pts, change_mask, artifact)
            kps = face.setdefault('keypoints', {})
            meta = face.setdefault('meta', {})
            meta.setdefault('occluded_keypoints', [])
            for name in occluded:
                short = name[3:] if name.startswith('kp_') else name
                meta['occluded_keypoints'] = sorted(
                    set(meta.get('occluded_keypoints') or []) | {short}
                )
                if policy == 'missing':
                    kps[short] = [[MISSING_PT, MISSING_PT]]
                else:
                    # Keep coords; annotate if stored as dict.
                    val = kps.get(short)
                    if isinstance(val, dict):
                        val = dict(val)
                        val['occluded'] = True
                        kps[short] = val
            face['keypoints'] = kps
            face['meta'] = meta
            out[face_key] = face
        out.setdefault('meta', {})
        out['meta']['artifact'] = artifact
        out['meta']['occluded_policy'] = policy
        return out

    # Old props format.
    props = dict(out.get('props') or {})
    pts = _points_from_props(props)
    occluded = _collect_occluded_names(pts, change_mask, artifact)
    props.setdefault('occluded_keypoints', [])
    for name in occluded:
        short = name[3:] if name.startswith('kp_') else name
        props['occluded_keypoints'] = sorted(set(props['occluded_keypoints']) | {short})
        if policy == 'missing':
            set_missing(props, name)
    props['artifact'] = artifact
    props['occluded_policy'] = policy
    out['props'] = props
    return out
