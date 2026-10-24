"""Landmark extraction from markup or InsightFace 2d106 fallback."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .face_kp_common import (
    KP_MAP_INSIGHTFACE,
    KP_NAMES,
    face_rect,
    is_missing,
    iter_markup_faces,
    landmarks_to_points,
    read_point,
    read_new_keypoint,
)


@dataclass
class FaceLandmarks:
    """Normalized face landmark set used by mask builders and generators."""

    points: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    # Optional dense contour from InsightFace 2d106 (Nx2).
    dense: Optional[np.ndarray] = None
    rect: Optional[Tuple[float, float, float, float]] = None  # x, y, w, h
    source: str = 'unknown'
    face_index: int = 0

    def get(self, name: str) -> Optional[Tuple[float, float]]:
        return self.points.get(name)

    def require(self, *names: str) -> Optional[List[Tuple[float, float]]]:
        pts = []
        for n in names:
            p = self.points.get(n)
            if p is None:
                return None
            pts.append(p)
        return pts


def _points_from_old_props(props: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
    out: Dict[str, Tuple[float, float]] = {}
    for name in KP_NAMES:
        pt = read_point(props, name)
        if pt is not None:
            out[name] = pt
    # Also accept short names without kp_ prefix stored oddly.
    return out


def _points_from_new_face(face: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
    out: Dict[str, Tuple[float, float]] = {}
    kps = face.get('keypoints') or {}
    for name in KP_NAMES:
        short = name[3:] if name.startswith('kp_') else name
        pt = read_new_keypoint(kps, short)
        if pt is not None:
            out[name] = pt
    return out


def landmarks_from_markup(data: Dict[str, Any]) -> List[FaceLandmarks]:
    """Extract FaceLandmarks for every face in markup (old or new format)."""
    faces: List[FaceLandmarks] = []
    if not data:
        return faces

    # New structured format.
    try:
        indexed = list(iter_markup_faces(data))
    except Exception:
        indexed = []

    if indexed:
        for idx, (_key, face) in enumerate(indexed):
            pts = _points_from_new_face(face)
            rect = None
            quad = face.get('face_quad') or face.get('rect') or face.get('quad')
            if isinstance(quad, list) and len(quad) >= 4:
                xs = [float(p[0]) for p in quad]
                ys = [float(p[1]) for p in quad]
                x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
                rect = (x0, y0, x1 - x0, y1 - y0)
            faces.append(
                FaceLandmarks(points=pts, rect=rect, source='markup_new', face_index=idx)
            )
        return faces

    # Old props format.
    props = data.get('props') or {}
    pts = _points_from_old_props(props)
    rect = face_rect(props)
    if pts or rect:
        faces.append(FaceLandmarks(points=pts, rect=rect, source='markup_old', face_index=0))
    return faces


class InsightFaceLandmarker:
    """Lazy InsightFace 2d106 landmark detector."""

    def __init__(self, ctx_id: int = 0, det_size: Tuple[int, int] = (640, 640)):
        self.ctx_id = ctx_id
        self.det_size = det_size
        self._app = None

    def _load(self):
        if self._app is not None:
            return
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise RuntimeError(
                'InsightFace fallback requires: pip install insightface onnxruntime-gpu '
                '(or onnxruntime)'
            ) from exc
        app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'landmark_2d_106'])
        app.prepare(ctx_id=self.ctx_id, det_size=self.det_size)
        self._app = app

    def detect(self, img_bgr: np.ndarray) -> List[FaceLandmarks]:
        self._load()
        faces_raw = self._app.get(img_bgr)
        h, w = img_bgr.shape[:2]
        result: List[FaceLandmarks] = []
        for i, face in enumerate(faces_raw):
            dense = None
            if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
                dense = np.asarray(face.landmark_2d_106, dtype=np.float32)
            elif hasattr(face, 'kps') and face.kps is not None:
                dense = np.asarray(face.kps, dtype=np.float32)

            pts: Dict[str, Tuple[float, float]] = {}
            if dense is not None and dense.shape[0] >= 106:
                pts = landmarks_to_points(dense, w, h, KP_MAP_INSIGHTFACE)
                # Supplement with derived points useful for masks.
                pts.setdefault('kp_chin', tuple(map(float, dense[16])))
                pts.setdefault('kp_forehead', (
                    float((dense[33][0] + dense[42][0]) / 2),
                    float(min(dense[33][1], dense[42][1]) - abs(dense[16][1] - dense[51][1]) * 0.35),
                ))
                pts.setdefault('kp_cheek_left', tuple(map(float, dense[2])))
                pts.setdefault('kp_cheek_right', tuple(map(float, dense[30])))
                pts.setdefault('kp_nose_bridge', tuple(map(float, dense[51])))
            elif dense is not None and dense.shape[0] >= 5:
                # 5-point fallback: left_eye, right_eye, nose, left_mouth, right_mouth
                pts['kp_pupil_left'] = tuple(map(float, dense[0]))
                pts['kp_pupil_right'] = tuple(map(float, dense[1]))
                pts['kp_nose_tip'] = tuple(map(float, dense[2]))
                pts['kp_mouth_left'] = tuple(map(float, dense[3]))
                pts['kp_mouth_right'] = tuple(map(float, dense[4]))

            rect = None
            if hasattr(face, 'bbox') and face.bbox is not None:
                x1, y1, x2, y2 = map(float, face.bbox)
                rect = (x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1))

            result.append(
                FaceLandmarks(
                    points=pts,
                    dense=dense,
                    rect=rect,
                    source='insightface',
                    face_index=i,
                )
            )
        # Largest face first.
        result.sort(key=lambda f: (f.rect[2] * f.rect[3]) if f.rect else 0.0, reverse=True)
        return result


def resolve_face_landmarks(
    img_bgr: np.ndarray,
    markup_data: Optional[Dict[str, Any]],
    landmarker: Optional[InsightFaceLandmarker] = None,
    prefer_markup: bool = True,
) -> Optional[FaceLandmarks]:
    """
    Return the primary face landmarks.
    Prefer markup when available and complete enough; otherwise InsightFace.
    """
    if prefer_markup and markup_data:
        faces = landmarks_from_markup(markup_data)
        if faces and faces[0].points:
            # If jaw/forehead missing but we need dense contour, enrich via InsightFace.
            primary = faces[0]
            need_dense = primary.dense is None and (
                'kp_chin' not in primary.points or 'kp_forehead' not in primary.points
            )
            if need_dense and landmarker is not None:
                try:
                    if_faces = landmarker.detect(img_bgr)
                    if if_faces:
                        primary.dense = if_faces[0].dense
                        for k, v in if_faces[0].points.items():
                            primary.points.setdefault(k, v)
                        if primary.rect is None:
                            primary.rect = if_faces[0].rect
                except Exception as exc:
                    logging.debug('InsightFace enrich failed: %s', exc)
            return primary

    if landmarker is None:
        landmarker = InsightFaceLandmarker()
    try:
        if_faces = landmarker.detect(img_bgr)
    except Exception as exc:
        logging.warning('InsightFace detection failed: %s', exc)
        return None
    return if_faces[0] if if_faces else None
