"""Label synthesised frames with face landmarks for detector-augmentation datasets.

Detectors:
  face_alignment — pretrained 2D FAN (68 pts / 300W), mapped to the internal scheme
  insightface    — buffalo_l 2d106, mapped via the LaPa layout (has pupils)
  hrnet          — internal landmark_wing checkpoint (--keypoints PATH)

Default scheme ``base11`` keeps eyes / mouth / nose / pupils (FAN estimates
pupils as eye centres when the 68-pt layout has none).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

_LIVENESS_ROOT = Path(__file__).resolve().parents[2]
if str(_LIVENESS_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIVENESS_ROOT))

from landmark_wing.mapping import map_landmarks, mapped_to_arrays  # noqa: E402
from landmark_wing.scheme import BASE_11_NAMES, KP_NAMES, NUM_BASE_11, NUM_LANDMARKS  # noqa: E402

SCHEMES = ("base11", "all30")
DETECTORS = ("face_alignment", "insightface", "hrnet")


def scheme_names(scheme: str) -> list[str]:
    if scheme == "base11":
        return list(BASE_11_NAMES)
    if scheme == "all30":
        return list(KP_NAMES)
    raise ValueError(f"unknown keypoint scheme {scheme!r}, expected one of {SCHEMES}")


def _select(points: np.ndarray, scheme: str) -> np.ndarray:
    return points[:NUM_BASE_11] if scheme == "base11" else points


def _fill_pupils_from_68(raw68: np.ndarray, named: dict[str, tuple[float, float]]) -> None:
    """300W/FAN has no pupils; use the mean of each eye's 6 contour points."""
    if raw68.shape[0] < 48:
        return
    if "pupil_right" not in named:
        named["pupil_right"] = tuple(map(float, raw68[36:42].mean(axis=0)))
    if "pupil_left" not in named:
        named["pupil_left"] = tuple(map(float, raw68[42:48].mean(axis=0)))


def named_to_payload(named: dict, scheme: str, bbox: np.ndarray) -> dict:
    names = scheme_names(scheme)
    # Rebuild full 30-array then slice so missing names stay NaN rather than shifting.
    coords = np.full((NUM_LANDMARKS, 2), np.nan, dtype=np.float32)
    for i, name in enumerate(KP_NAMES):
        if name in named:
            coords[i] = named[name]
    points = _select(coords, scheme)
    visible = {name: bool(np.isfinite(points[i]).all()) for i, name in enumerate(names)}
    return {
        "bbox_xyxy": [float(x) for x in bbox],
        "landmarks": [[None if not np.isfinite(v) else float(v) for v in pt] for pt in points],
        "named_landmarks": {
            name: (None if not visible[name] else points[i].tolist())
            for i, name in enumerate(names)
        },
        "visibility": visible,
    }


class KeypointDetector(Protocol):
    name: str

    def label(self, image_bgr: np.ndarray, bbox_hint: np.ndarray | None, scheme: str) -> dict | None:
        ...


class FaceAlignmentDetector:
    """2D FAN via `face-alignment` (68 landmarks)."""

    name = "face_alignment"

    def __init__(self, device: str = "cuda") -> None:
        from cooperative_face_liveness.landmarks import FaceAlignmentLandmarkDetector
        self._det = FaceAlignmentLandmarkDetector(device=device)

    def label(self, image_bgr: np.ndarray, bbox_hint: np.ndarray | None, scheme: str) -> dict | None:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        obs = self._det.detect(rgb)
        if obs is None:
            return None
        named = map_landmarks(obs.landmarks, "300w")
        _fill_pupils_from_68(obs.landmarks, named)
        return named_to_payload(named, scheme, obs.bbox_xyxy)


class InsightFaceDetector:
    """InsightFace buffalo_l 2d106 landmarks."""

    name = "insightface"

    def __init__(self, device: str = "cuda") -> None:
        from insightface.app import FaceAnalysis
        ctx = 0 if device.startswith("cuda") else -1
        self._app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "landmark_2d_106"])
        self._app.prepare(ctx_id=ctx, det_size=(640, 640))

    def label(self, image_bgr: np.ndarray, bbox_hint: np.ndarray | None, scheme: str) -> dict | None:
        faces = self._app.get(image_bgr)
        if not faces:
            return None
        face = max(faces, key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))
        dense = getattr(face, "landmark_2d_106", None)
        if dense is None:
            return None
        dense = np.asarray(dense, dtype=np.float32)
        named = map_landmarks(dense, "lapa")
        bbox = np.asarray(face.bbox, dtype=np.float32)
        return named_to_payload(named, scheme, bbox)


class HRNetDetector:
    """Internal landmark_wing HRNet checkpoint."""

    name = "hrnet"

    def __init__(self, checkpoint: Path, device: str = "cuda") -> None:
        from landmark_wing.infer import load_model
        self._model, self._meta = load_model(Path(checkpoint), device)
        self._device = device
        self.checkpoint = str(Path(checkpoint).resolve())

    def label(self, image_bgr: np.ndarray, bbox_hint: np.ndarray | None, scheme: str) -> dict | None:
        from landmark_wing.infer import predict_landmarks
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        bbox = bbox_hint if bbox_hint is not None else np.array(
            [0.0, 0.0, float(w - 1), float(h - 1)], dtype=np.float32)
        points30, _, _ = predict_landmarks(
            self._model, gray, bbox, image_size=int(self._meta["image_size"]), device=self._device,
        )
        named = {name: (float(points30[i, 0]), float(points30[i, 1])) for i, name in enumerate(KP_NAMES)}
        return named_to_payload(named, scheme, bbox)


def build_detector(kind: str, device: str | None = None, checkpoint: Path | None = None) -> KeypointDetector:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "keypoint labeling needs torch in the orchestrator python "
            "(Docker: rebuild pose_synth:cu124 or use third_party/.venv_3ddfa via envs/docker_orch_python.sh)"
        ) from exc
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if kind == "face_alignment":
        return FaceAlignmentDetector(device=device)
    if kind == "insightface":
        return InsightFaceDetector(device=device)
    if kind == "hrnet":
        if checkpoint is None:
            raise SystemExit("--kp-detector hrnet requires --keypoints CHECKPOINT")
        return HRNetDetector(Path(checkpoint), device=device)
    raise SystemExit(f"unknown --kp-detector {kind!r}, expected one of {DETECTORS}")


def _bbox_from_pose(pose: dict | None, width: int, height: int) -> np.ndarray | None:
    box = (pose or {}).get("box") if isinstance(pose, dict) else None
    if box and len(box) == 4:
        return np.asarray(box, dtype=np.float32)
    return None


def label_method_dir(
    method_dir: Path,
    detector: KeypointDetector,
    scheme: str = "base11",
) -> Path:
    frames_dir = method_dir / "frames"
    poses_path = method_dir / "poses.json"
    poses = {r["index"]: r for r in json.loads(poses_path.read_text())} if poses_path.is_file() else {}
    records = []
    missed = 0
    for path in sorted(frames_dir.glob("*.png")):
        index = int(path.stem)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        h, w = image.shape[:2]
        measured = (poses.get(index) or {}).get("measured")
        labeled = detector.label(image, _bbox_from_pose(measured, w, h), scheme)
        if labeled is None:
            missed += 1
            continue
        records.append({
            "index": index,
            "path": str(path.relative_to(method_dir)),
            "requested": (poses.get(index) or {}).get("requested") or {},
            "measured": measured,
            **labeled,
        })

    names = scheme_names(scheme)
    payload = {
        "method": method_dir.name,
        "detector": detector.name,
        "scheme": scheme,
        "kp_names": names,
        "num_landmarks": len(names),
        "frames_labeled": len(records),
        "frames_missed": missed,
        "checkpoint": getattr(detector, "checkpoint", None),
        "frames": records,
    }
    out = method_dir / "keypoints.json"
    out.write_text(json.dumps(payload, indent=1))
    return out


def _source_landmarks_by_path(source_keypoints_path: Path) -> dict[str, dict]:
    payload = json.loads(source_keypoints_path.read_text())
    return {rec["path"]: rec for rec in payload.get("images", [])}


def label_liveportrait_warped(
    method_dir: Path,
    source_keypoints_path: Path,
    scheme: str = "base11",
) -> Path:
    """Propagate source detector landmarks through LivePortrait sparse kp motion (TPS)."""
    from ps.lp_warp_points import (
        scale_points_resize_to_limit,
        warp_landmarks_fullframe,
    )

    warp_path = method_dir / "warp_state.json"
    if not warp_path.is_file():
        raise FileNotFoundError(f"missing {warp_path}; re-run liveportrait adapter")
    if not source_keypoints_path.is_file():
        raise FileNotFoundError(f"missing source keypoints {source_keypoints_path}")

    warp = json.loads(warp_path.read_text())
    by_source = _source_landmarks_by_path(source_keypoints_path)
    names = scheme_names(scheme)
    poses_path = method_dir / "poses.json"
    poses = {r["index"]: r for r in json.loads(poses_path.read_text())} if poses_path.is_file() else {}
    records = []
    missed = 0
    for frame in warp.get("frames", []):
        index = int(frame["index"])
        src_rec = by_source.get(frame["source"])
        if src_rec is None:
            missed += 1
            continue
        raw = src_rec.get("landmarks") or []
        pts = np.asarray(
            [[np.nan, np.nan] if (p is None or p[0] is None or p[1] is None) else [p[0], p[1]]
             for p in raw],
            dtype=np.float64,
        )
        if pts.shape[0] != len(names):
            # Allow all30 source → base11 slice
            if scheme == "base11" and pts.shape[0] >= NUM_BASE_11:
                pts = pts[:NUM_BASE_11]
            else:
                missed += 1
                continue
        orig_hw = tuple(frame["orig_hw"])
        pts_resized = scale_points_resize_to_limit(pts, orig_hw, max_dim=1280)
        # Points that were NaN stay NaN after scale (scale copies finite only via multiply)
        valid = np.isfinite(pts).all(axis=1)
        pts_resized = np.where(valid[:, None], pts_resized, np.nan)
        warped = warp_landmarks_fullframe(
            pts_resized,
            x_s=np.asarray(frame["x_s"], dtype=np.float64),
            x_d=np.asarray(frame["x_d"], dtype=np.float64),
            m_c2o=np.asarray(frame["M_c2o"], dtype=np.float64),
            crop_size=float(frame.get("crop_dsize") or 512),
            valid_mask=valid,
        )
        # Output frames are paste_back onto resized canvas; if the written PNG was
        # further resized nowhere, landmarks stay in img_hw coordinates.
        img_hw = frame.get("img_hw") or list(orig_hw)
        finite = np.isfinite(warped).all(axis=1)
        if finite.any():
            ys, xs = warped[finite, 1], warped[finite, 0]
            bbox = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
        else:
            bbox = [0.0, 0.0, float(img_hw[1]), float(img_hw[0])]
            missed += 1
            continue
        landmarks = [[None if not np.isfinite(v) else float(v) for v in pt] for pt in warped]
        visibility = {name: bool(np.isfinite(warped[i]).all()) for i, name in enumerate(names)}
        records.append({
            "index": index,
            "path": f"frames/{index:04d}.png",
            "requested": (poses.get(index) or {}).get("requested") or {},
            "measured": (poses.get(index) or {}).get("measured"),
            "bbox_xyxy": bbox,
            "landmarks": landmarks,
            "named_landmarks": {
                name: (None if not visibility[name] else warped[i].tolist())
                for i, name in enumerate(names)
            },
            "visibility": visibility,
            "source": frame["source"],
            "img_hw": img_hw,
        })

    payload = {
        "method": method_dir.name,
        "detector": "liveportrait_warp",
        "scheme": scheme,
        "kp_names": names,
        "num_landmarks": len(names),
        "frames_labeled": len(records),
        "frames_missed": missed,
        "checkpoint": None,
        "source_keypoints": str(source_keypoints_path),
        "frames": records,
    }
    out = method_dir / "keypoints.json"
    out.write_text(json.dumps(payload, indent=1))
    return out


def label_sources(
    images: list[str],
    source_poses: dict[str, dict | None],
    out_path: Path,
    detector: KeypointDetector,
    scheme: str = "base11",
) -> Path:
    records = []
    for path in images:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            continue
        h, w = image.shape[:2]
        labeled = detector.label(image, _bbox_from_pose(source_poses.get(path), w, h), scheme)
        if labeled is None:
            continue
        records.append({"path": path, "pose": source_poses.get(path), **labeled})
    names = scheme_names(scheme)
    payload = {
        "detector": detector.name,
        "scheme": scheme,
        "kp_names": names,
        "num_landmarks": len(names),
        "checkpoint": getattr(detector, "checkpoint", None),
        "images": records,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=1))
    return out_path


def export_training_pairs(method_dir: Path, out_jsonl: Path | None = None) -> Path:
    payload = json.loads((method_dir / "keypoints.json").read_text())
    out_jsonl = out_jsonl or (method_dir / "keypoints_train.jsonl")
    with out_jsonl.open("w", encoding="utf-8") as fp:
        for frame in payload["frames"]:
            fp.write(json.dumps({
                "image": str((method_dir / frame["path"]).resolve()),
                "landmarks": frame["landmarks"],
                "kp_names": payload["kp_names"],
                "bbox_xyxy": frame["bbox_xyxy"],
                "yaw": (frame.get("requested") or {}).get("yaw"),
                "pitch": (frame.get("requested") or {}).get("pitch"),
                "scheme": payload["scheme"],
                "method": payload["method"],
                "detector": payload.get("detector"),
            }, ensure_ascii=False) + "\n")
    return out_jsonl


# Skeleton edges among the base-11 indices (from landmark_wing.scheme.SKELETON_EDGES).
_BASE11_EDGES = (
    (0, 9), (9, 2), (3, 10), (10, 1),
    (7, 4), (4, 6), (6, 5), (5, 7),
)


def _as_xy(point) -> tuple[int, int] | None:
    if point is None or len(point) < 2:
        return None
    if point[0] is None or point[1] is None:
        return None
    try:
        x, y = float(point[0]), float(point[1])
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    return int(round(x)), int(round(y))


def draw_keypoints(
    image_bgr: np.ndarray,
    landmarks: list,
    names: list[str] | None = None,
    draw_labels: bool = False,
) -> np.ndarray:
    """Overlay landmarks (and base-11 skeleton when applicable) on a BGR frame."""
    canvas = image_bgr.copy()
    points = [_as_xy(pt) for pt in landmarks]
    edges = _BASE11_EDGES if len(points) == NUM_BASE_11 else ()
    for a, b in edges:
        if a < len(points) and b < len(points) and points[a] and points[b]:
            cv2.line(canvas, points[a], points[b], (40, 200, 255), 1, cv2.LINE_AA)
    for i, pt in enumerate(points):
        if pt is None:
            continue
        cv2.circle(canvas, pt, 3, (0, 0, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, pt, 2, (80, 80, 255), -1, lineType=cv2.LINE_AA)
        if draw_labels and names and i < len(names):
            cv2.putText(canvas, str(i), (pt[0] + 4, pt[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def write_keypoints_video(method_dir: Path, video_path: Path | None = None, fps: float = 12.0) -> Path | None:
    """Write method_dir/keypoints.mp4 from keypoints.json + frames/."""
    from ps.video import VideoSink, fit_tile, label_tile

    kp_path = method_dir / "keypoints.json"
    if not kp_path.is_file():
        return None
    payload = json.loads(kp_path.read_text())
    by_index = {int(f["index"]): f for f in payload.get("frames", [])}
    poses_path = method_dir / "poses.json"
    poses = {r["index"]: r for r in json.loads(poses_path.read_text())} if poses_path.is_file() else {}
    video_path = video_path or (method_dir / "keypoints.mp4")
    frames_dir = method_dir / "frames"
    names = payload.get("kp_names") or scheme_names(payload.get("scheme", "base11"))
    sink = VideoSink(video_path, fps)
    wrote = 0
    try:
        for path in sorted(frames_dir.glob("*.png")):
            index = int(path.stem)
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            frame = by_index.get(index)
            if frame is not None:
                image = draw_keypoints(image, frame.get("landmarks") or [], names)
            rec = poses.get(index) or {"requested": {"yaw": None, "pitch": None}, "measured": None}
            tile = label_tile(fit_tile(image, 512), f"{method_dir.name} kp",
                              rec.get("requested") or {"yaw": None, "pitch": None},
                              rec.get("measured"))
            sink.write(tile)
            wrote += 1
    finally:
        sink.close()
    return video_path if wrote else None
