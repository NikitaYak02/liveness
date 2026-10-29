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
    import torch
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
