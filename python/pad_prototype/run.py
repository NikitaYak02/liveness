from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np

from pad_prototype.hpe import facerecog2_ypr, prototype_pose

FEATURE_VERSION = 6
THRESHOLD = 1.80
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent / "models" / "face_landmarker.task"

NOSE, BRIDGE, FOREHEAD, CHIN = 1, 6, 10, 152
MOUTH_L, MOUTH_R = 61, 291
EYE_OUTER_A, EYE_INNER_A = 33, 133
EYE_OUTER_B, EYE_INNER_B = 263, 362
CHEEK_A, CHEEK_B = 117, 346
HOLDOUT = (CHEEK_A, CHEEK_B, FOREHEAD, CHIN, BRIDGE)
LIVE_CLASSES = {"real", "real_1"}
ATTACK_CLASSES = {
    "printed_flat",
    "dynamic_curved",
    "static_curved",
    "masks_with_nose",
    "masks_without_nose",
}
_TS = re.compile(r"(?P<day>\d+)\.(?P<month>\d+)\s+(?P<h>\d+)_(?P<m>\d+)_(?P<s>\d+)\.(?P<ms>\d+)")


def _stamp(name: str) -> float | None:
    match = _TS.search(name)
    if not match:
        return None
    milli = int(match.group("ms")[:3].ljust(3, "0"))
    return (
        int(match.group("day")) * 86400.0
        + int(match.group("h")) * 3600.0
        + int(match.group("m")) * 60.0
        + int(match.group("s"))
        + milli / 1000.0
    )


def _frames(folder: Path) -> list[Path]:
    listing = folder / "runlist.lst"
    if listing.exists():
        frames = []
        for line in listing.read_text(encoding="utf-8", errors="replace").splitlines():
            name = line.strip()
            if not name or name.endswith(".lst"):
                continue
            path = folder / name
            if path.exists():
                frames.append(path)
        if len(frames) >= 8:
            return frames
    frames = [path for path in folder.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]

    def sort_key(path: Path) -> tuple:
        stamp = _stamp(path.name)
        return (0, stamp, path.name) if stamp is not None else (1, 0.0, path.name)

    return sorted(frames, key=sort_key)


def discover_videos(root: Path) -> list[dict]:
    videos = []
    for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        attack_type = class_dir.name
        if attack_type not in LIVE_CLASSES and attack_type not in ATTACK_CLASSES:
            continue
        label = "live" if attack_type in LIVE_CLASSES else "attack"
        leaves = [path for path in class_dir.rglob("*") if path.is_dir()] or [class_dir]
        for folder in sorted(leaves):
            frames = _frames(folder)
            if len(frames) < 8:
                continue
            videos.append(
                {
                    "video_id": folder.relative_to(root).as_posix(),
                    "label": label,
                    "attack_type": attack_type,
                    "frames": frames,
                }
            )
    return videos


def _model() -> Path:
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 100_000:
        return MODEL_PATH
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


class FaceTracker:
    def __init__(self) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(_model())),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.35,
            min_face_presence_confidence=0.35,
            output_facial_transformation_matrixes=False,
            output_face_blendshapes=False,
        )
        self._detector = vision.FaceLandmarker.create_from_options(options)

    def detect(self, bgr: np.ndarray) -> np.ndarray | None:
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(image)
        if not result.face_landmarks:
            return None
        height, width = bgr.shape[:2]
        points = np.array([[lm.x * width, lm.y * height] for lm in result.face_landmarks[0]], dtype=np.float32)
        if points.shape != (478, 2) or not np.isfinite(points).all():
            return None
        return points


def _lr(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (a, b) if a[0] <= b[0] else (b, a)


def _eye_distance(points: np.ndarray) -> float:
    left = points[[EYE_OUTER_A, EYE_INNER_A]].mean(axis=0)
    right = points[[EYE_OUTER_B, EYE_INNER_B]].mean(axis=0)
    return float(np.linalg.norm(left - right))


def _pose_delta(first: dict, second: dict, pose: str) -> float:
    if pose == "facerecog2":
        if first["fr"] is None or second["fr"] is None:
            return 0.0
        return float(np.hypot(first["fr"][0] - second["fr"][0], first["fr"][1] - second["fr"][1]))
    if first["rotation"] is None or second["rotation"] is None:
        return 0.0
    delta = _angle_deg(first["rotation"], second["rotation"])
    # Переворот решения PnP около 180° не считаем движением.
    if delta > 75.0:
        return 0.0
    return delta


def _angle_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation_b @ rotation_a.T) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _parts(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    planar = np.vstack(
        [
            *_lr(points[EYE_OUTER_A], points[EYE_OUTER_B]),
            *_lr(points[EYE_INNER_A], points[EYE_INNER_B]),
            *_lr(points[MOUTH_L], points[MOUTH_R]),
        ]
    ).astype(np.float32)
    return planar, points[NOSE].astype(np.float32), points[list(HOLDOUT)].astype(np.float32)


def _errors(src: dict, dst: dict) -> tuple[float, float]:
    src_p, src_n, src_h = _parts(src["points"])
    dst_p, dst_n, dst_h = _parts(dst["points"])
    homography, _ = cv2.findHomography(src_p.reshape(-1, 1, 2), dst_p.reshape(-1, 1, 2), 0)
    if homography is None:
        return float("nan"), float("nan")
    nose = cv2.perspectiveTransform(src_n.reshape(1, 1, 2), homography).reshape(2)
    hold = cv2.perspectiveTransform(src_h.reshape(-1, 1, 2), homography).reshape(-1, 2)
    nose_error = float(np.linalg.norm(nose - dst_n))
    hold_error = float(np.linalg.norm(hold - dst_h, axis=1).mean())
    return nose_error, hold_error


def _median(values: list[float]) -> float:
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if array.size == 0:
        return float("nan")
    return float(np.median(array))


def _order(frames: list[Path]) -> list[Path]:
    stamps = [_stamp(path.name) for path in frames]
    if any(item is None for item in stamps):
        return frames
    order = np.argsort(np.asarray(stamps, dtype=np.float64))
    return [frames[index] for index in order]


def collect_faces(video: dict, tracker: FaceTracker) -> list[dict]:
    frames = _order(list(video["frames"]))
    stride = 1 if len(frames) <= 140 else 2 if len(frames) <= 280 else 3
    faces = []
    for path in frames[::stride]:
        bgr = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_2)
        if bgr is None or bgr.size == 0:
            continue
        points = tracker.detect(bgr)
        if points is None or _eye_distance(points) < 10:
            continue
        minimum, maximum = points.min(axis=0), points.max(axis=0)
        pose = prototype_pose(points, bgr.shape[:2])
        faces.append(
            {
                "eye": _eye_distance(points),
                "diag": float(np.hypot(*(maximum - minimum))),
                "rotation": None if pose is None else pose[0],
                "pnp": None if pose is None else pose[1],
                "fr": facerecog2_ypr(points),
                "points": points,
            }
        )
    return faces


def pair_specificity(faces: list[dict], pose: str) -> tuple[float, float]:
    step = max(1, len(faces) // 8)
    chosen = faces[::step][:10]
    nose_errors, specificities = [], []
    for i in range(len(chosen)):
        for j in range(i + 1, len(chosen)):
            first, second = chosen[i], chosen[j]
            scale = max(first["eye"], second["eye"]) / (min(first["eye"], second["eye"]) + 1e-6)
            delta = _pose_delta(first, second, pose)
            if delta < 3.0 and scale < 1.05:
                continue
            err, hold = _errors(first, second)
            diag = 0.5 * (first["diag"] + second["diag"]) + 1e-6
            if np.isfinite(err):
                nose_errors.append(err / diag)
            if np.isfinite(err) and np.isfinite(hold):
                specificities.append(err / (hold + 0.008 * diag))
    return _median(specificities), float(len(nose_errors))


def score_video(video: dict, tracker: FaceTracker, pose: str = "pnp") -> dict:
    faces = collect_faces(video, tracker)
    specificity, pairs = pair_specificity(faces, pose)
    n_faces = float(len(faces))
    if n_faces < 5 or pairs < 3 or not np.isfinite(specificity):
        label = "attack"
        text = "Атака. Мало лиц или мало пар с движением, объём носа не измерен."
    elif specificity >= THRESHOLD:
        label = "live"
        text = f"Живое. Ошибка носа в {specificity:.2f} раза больше ошибки контрольных точек (порог {THRESHOLD:.2f})."
    else:
        label = "attack"
        text = f"Атака. Отношение ошибки носа к контрольным точкам {specificity:.2f} ниже порога {THRESHOLD:.2f}."
    return {
        "video_id": video["video_id"],
        "attack_type": video["attack_type"],
        "true_label": video["label"],
        "predicted_label": label,
        "nose_specificity": None if not np.isfinite(specificity) else specificity,
        "n_faces": n_faces,
        "n_pose_pairs": pairs,
        "pose": pose,
        "explanation_ru": text,
    }


def _json_default(value):
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, np.integer):
        return int(value)
    raise TypeError(type(value))


def main() -> None:
    parser = argparse.ArgumentParser(description="Параллакс носа по кадрам android/")
    parser.add_argument("--data", type=Path, default=Path("android"))
    parser.add_argument("--out", type=Path, default=Path("python/pad_prototype/output"))
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument(
        "--pose",
        choices=("pnp", "facerecog2"),
        default="pnp",
        help="Чем отбирать пары кадров: угол solvePnP или hypot(Δyaw, Δpitch) как в facerecog2",
    )
    args = parser.parse_args()

    videos = discover_videos(args.data)
    if not videos:
        raise SystemExit(f"Нет клипов в {args.data}")

    cache_path = args.out / "decisions.json"
    cached = []
    if cache_path.exists() and not args.recompute:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("feature_version") == FEATURE_VERSION and payload.get("pose", "pnp") == args.pose:
            cached = payload.get("videos", [])
    by_id = {row["video_id"]: row for row in cached}
    pending = [video for video in videos if video["video_id"] not in by_id]
    print(f"Клипов {len(videos)}, в кэше {len(videos) - len(pending)}, считать {len(pending)}", flush=True)

    tracker = FaceTracker() if pending else None
    started = time.time()
    for index, video in enumerate(pending, start=1):
        t0 = time.time()
        row = score_video(video, tracker, args.pose)
        by_id[row["video_id"]] = row
        print(
            f"[{index}/{len(pending)}] {row['video_id']} {row['predicted_label']} "
            f"score={row['nose_specificity']} за {time.time() - t0:.1f}с",
            flush=True,
        )

    rows = [by_id[video["video_id"]] for video in videos]
    n_live = sum(row["true_label"] == "live" for row in rows)
    n_attack = len(rows) - n_live
    attack_as_live = sum(row["true_label"] == "attack" and row["predicted_label"] == "live" for row in rows)
    live_as_attack = sum(row["true_label"] == "live" and row["predicted_label"] != "live" for row in rows)
    summary = {
        "n_live": n_live,
        "n_attack": n_attack,
        "attack_as_live": attack_as_live,
        "live_as_attack": live_as_attack,
        "APCER": attack_as_live / n_attack if n_attack else None,
        "BPCER": live_as_attack / n_live if n_live else None,
        "feature_version": FEATURE_VERSION,
        "pose": args.pose,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {"summary": summary, "pose": args.pose, "videos": rows},
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    print(
        f"\nAPCER={summary['APCER']:.3f} ({attack_as_live}/{n_attack}) "
        f"BPCER={summary['BPCER']:.3f} ({live_as_attack}/{n_live})"
    )
    for row in rows:
        if row["true_label"] != row["predicted_label"]:
            print(f"  {row['video_id']}: {row['explanation_ru']}")
    print(f"Решения: {cache_path}")
    print(f"Время: {time.time() - started:.0f} с")


if __name__ == "__main__":
    main()
