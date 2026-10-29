"""Label synthesised frames with the internal HRNet face-keypoint model.

Predicts the full 30-point scheme, then optionally keeps only the base-11 subset
(eyes / mouth / nose / pupils) for keypoint-detector augmentation datasets.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

# landmark_wing lives next to pose_synth under the liveness repo root.
_LIVENESS_ROOT = Path(__file__).resolve().parents[2]
if str(_LIVENESS_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIVENESS_ROOT))

from landmark_wing.infer import load_model, predict_landmarks  # noqa: E402
from landmark_wing.scheme import BASE_11_NAMES, KP_NAMES, NUM_BASE_11  # noqa: E402

SCHEMES = ("base11", "all30")


def scheme_names(scheme: str) -> list[str]:
    if scheme == "base11":
        return list(BASE_11_NAMES)
    if scheme == "all30":
        return list(KP_NAMES)
    raise ValueError(f"unknown keypoint scheme {scheme!r}, expected one of {SCHEMES}")


def _bbox_from_pose(pose: dict | None, width: int, height: int) -> np.ndarray:
    box = (pose or {}).get("box") if isinstance(pose, dict) else None
    if box and len(box) == 4:
        return np.asarray(box, dtype=np.float32)
    return np.array([0.0, 0.0, float(width - 1), float(height - 1)], dtype=np.float32)


def _select(points: np.ndarray, scheme: str) -> np.ndarray:
    if scheme == "base11":
        return points[:NUM_BASE_11]
    return points


def label_image(
    model,
    meta: dict,
    image_bgr: np.ndarray,
    bbox_xyxy: np.ndarray | None,
    scheme: str,
    device: str,
) -> dict:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    bbox = bbox_xyxy if bbox_xyxy is not None else _bbox_from_pose(None, w, h)
    points30, _, _ = predict_landmarks(
        model, gray, bbox, image_size=int(meta["image_size"]), device=device,
    )
    points = _select(points30, scheme)
    names = scheme_names(scheme)
    return {
        "bbox_xyxy": [float(x) for x in bbox],
        "landmarks": points.tolist(),
        "named_landmarks": {name: points[i].tolist() for i, name in enumerate(names)},
    }


def label_method_dir(
    method_dir: Path,
    checkpoint: Path,
    scheme: str = "base11",
    device: str | None = None,
) -> Path:
    """Write method_dir/keypoints.json for every present frame under frames/."""
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_model(Path(checkpoint), device)
    frames_dir = method_dir / "frames"
    poses_path = method_dir / "poses.json"
    poses = {r["index"]: r for r in json.loads(poses_path.read_text())} if poses_path.is_file() else {}

    records = []
    for path in sorted(frames_dir.glob("*.png")):
        index = int(path.stem)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        h, w = image.shape[:2]
        measured = (poses.get(index) or {}).get("measured")
        bbox = _bbox_from_pose(measured, w, h)
        labeled = label_image(model, meta, image, bbox, scheme, device)
        req = (poses.get(index) or {}).get("requested") or {}
        records.append({
            "index": index,
            "path": str(path.relative_to(method_dir)),
            "requested": req,
            "measured": measured,
            **labeled,
        })

    names = scheme_names(scheme)
    payload = {
        "method": method_dir.name,
        "scheme": scheme,
        "kp_names": names,
        "num_landmarks": len(names),
        "checkpoint": str(Path(checkpoint).resolve()),
        "frames": records,
    }
    out = method_dir / "keypoints.json"
    out.write_text(json.dumps(payload, indent=1))
    return out


def label_sources(
    images: list[str],
    source_poses: dict[str, dict | None],
    out_path: Path,
    checkpoint: Path,
    scheme: str = "base11",
    device: str | None = None,
) -> Path:
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_model(Path(checkpoint), device)
    names = scheme_names(scheme)
    records = []
    for path in images:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            continue
        h, w = image.shape[:2]
        bbox = _bbox_from_pose(source_poses.get(path), w, h)
        labeled = label_image(model, meta, image, bbox, scheme, device)
        records.append({"path": path, "pose": source_poses.get(path), **labeled})
    payload = {
        "scheme": scheme,
        "kp_names": names,
        "num_landmarks": len(names),
        "checkpoint": str(Path(checkpoint).resolve()),
        "images": records,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=1))
    return out_path


def export_training_pairs(method_dir: Path, out_jsonl: Path | None = None) -> Path:
    """Flatten keypoints.json into one JSONL line per frame for detector training."""
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
            }, ensure_ascii=False) + "\n")
    return out_jsonl
