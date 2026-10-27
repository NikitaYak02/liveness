"""APCER и BPCER текстурных признаков, каждый признак отдельно.

Порог и знак признака выбираются на обучающих фолдах по точке, где APCER ближе
всего к BPCER. На тесте каждый ролик ровно один раз. Это не подгонка порога
по тем же примерам, на которых считается ошибка.

android — один телефон, живые ролики и атаки (маски, печать, изогнутый экран).
axon — селфи против масок и cutout из AxonData/face-anti-spoofing-dataset.
Сенсор и сжатие там другие, поэтому числа axon не смешиваются с android.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np

from pad_prototype.run import FaceTracker, _order, discover_videos
from pad_prototype.texture_features import FEATURE_INFO, FEATURE_NAMES, _self_check, compute_features

FACE_SIZE = 192
PATCH_SIZE = 64
MAX_SIDE = 1280
N_FOLDS = 5
EYE_L = (33, 133)
EYE_R = (263, 362)
REGION_INDEX = {
    "forehead": 10,
    "cheek_l": 117,
    "cheek_r": 346,
    "nose": 1,
    "chin": 152,
}
AXON_ATTACKS = {
    "cutout_attacks": "cutout",
    "3d_paper_mask": "paper3d",
    "wrapped_3d_paper_mask": "wrapped",
    "latex_mask": "latex",
    "silicone_mask": "silicone",
    "textile 3d face mask attack sample": "textile",
}


def _limit_side(bgr: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    height, width = bgr.shape[:2]
    side = max(height, width)
    if side <= max_side:
        return bgr
    scale = max_side / side
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)


def _eye_centers(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = points[list(EYE_L)].mean(axis=0)
    right = points[list(EYE_R)].mean(axis=0)
    if left[0] > right[0]:
        left, right = right, left
    return left, right


def _align(bgr: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    left, right = _eye_centers(points)
    eye = float(np.linalg.norm(right - left))
    if eye < 40:
        return None
    angle = float(np.degrees(np.arctan2(right[1] - left[1], right[0] - left[0])))
    center = (left + right) / 2.0
    matrix = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), angle, 1.0)
    height, width = bgr.shape[:2]
    aligned = cv2.warpAffine(
        bgr,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    homo = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    moved = (matrix.astype(np.float32) @ homo.T).T
    return aligned, moved


def _square(bgr: np.ndarray, center: np.ndarray, side: float, size: int) -> np.ndarray | None:
    height, width = bgr.shape[:2]
    side_px = int(round(side))
    if side_px < 16:
        return None
    x0 = int(round(float(center[0]) - side_px / 2.0))
    y0 = int(round(float(center[1]) - side_px / 2.0))
    x1, y1 = x0 + side_px, y0 + side_px
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > width:
        x0 -= x1 - width
        x1 = width
    if y1 > height:
        y0 -= y1 - height
        y1 = height
    x0, y0 = max(0, x0), max(0, y0)
    crop = bgr[y0:y1, x0:x1]
    if crop.shape[0] < 16 or crop.shape[1] < 16:
        return None
    interpolation = cv2.INTER_AREA if crop.shape[0] > size and crop.shape[1] > size else cv2.INTER_LINEAR
    return cv2.resize(crop, (size, size), interpolation=interpolation)


def _crops(bgr: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], np.ndarray | None] | None:
    aligned = _align(bgr, points)
    if aligned is None:
        return None
    image, moved = aligned
    minimum, maximum = moved.min(axis=0), moved.max(axis=0)
    extent = float(max(maximum[0] - minimum[0], maximum[1] - minimum[1]))
    face = _square(image, (minimum + maximum) / 2.0, extent * 1.15, FACE_SIZE)
    if face is None:
        return None
    left, right = _eye_centers(moved)
    eye = float(np.linalg.norm(right - left))
    origin = moved.mean(axis=0)
    regions = []
    for index in REGION_INDEX.values():
        center = origin + (moved[index] - origin) * 0.8
        patch = _square(image, center, eye * 0.42, PATCH_SIZE)
        if patch is not None:
            regions.append(patch)
    context = _context_patch(image, (minimum + maximum) / 2.0, extent * 1.15)
    return face, regions, context


def _context_patch(image: np.ndarray, center: np.ndarray, side: float) -> np.ndarray | None:
    """Квадрат из самой широкой полосы кадра, которая не пересекает кроп лица."""
    height, width = image.shape[:2]
    half = side / 2.0
    face_x0 = float(center[0]) - half
    face_y0 = float(center[1]) - half
    face_x1 = float(center[0]) + half
    face_y1 = float(center[1]) + half
    bands = (
        (0, 0, max(0, int(np.floor(face_x0))), height),
        (min(width, int(np.ceil(face_x1))), 0, width, height),
        (0, 0, width, max(0, int(np.floor(face_y0)))),
        (0, min(height, int(np.ceil(face_y1))), width, height),
    )
    best = None
    best_area = 0
    for x0, y0, x1, y1 in bands:
        if x1 - x0 < 16 or y1 - y0 < 16:
            continue
        area = (x1 - x0) * (y1 - y0)
        if area > best_area:
            best_area = area
            best = image[y0:y1, x0:x1]
    if best is None:
        return None
    band_height, band_width = best.shape[:2]
    side_px = min(band_height, band_width)
    y = (band_height - side_px) // 2
    x = (band_width - side_px) // 2
    crop = best[y : y + side_px, x : x + side_px]
    interpolation = cv2.INTER_AREA if side_px > PATCH_SIZE else cv2.INTER_LINEAR
    return cv2.resize(crop, (PATCH_SIZE, PATCH_SIZE), interpolation=interpolation)


def _video_frames(path: Path, count: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        capture.release()
        return []
    start = max(0, int(total * 0.15))
    stop = max(start, int(total * 0.85) - 1)
    indexes = np.linspace(start, stop, count).astype(int)
    frames = []
    for index in indexes:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if ok and frame is not None and frame.size:
            frames.append(frame)
    capture.release()
    return frames


def _image_frames(paths: list[Path], count: int) -> list[np.ndarray]:
    if not paths:
        return []
    chosen = np.linspace(0, len(paths) - 1, min(count, len(paths))).astype(int)
    frames = []
    for index in chosen:
        frame = cv2.imread(str(paths[int(index)]), cv2.IMREAD_COLOR)
        if frame is not None and frame.size:
            frames.append(frame)
    return frames


def _android_samples(root: Path) -> list[dict]:
    samples = []
    for video in discover_videos(root):
        frames = _order(list(video["frames"]))
        samples.append(
            {
                "dataset": "android",
                "sample_id": video["video_id"],
                "label": video["label"],
                "attack_type": "live" if video["label"] == "live" else video["attack_type"],
                "kind": "images",
                "paths": frames,
                "n_frames": 6,
            }
        )
    return samples


def _axon_samples(root: Path) -> list[dict]:
    samples = []
    if not root.exists():
        return samples
    for folder in sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")):
        key = folder.name.strip().lower()
        if key == "selfies":
            label, attack_type = "live", "live"
        elif key in AXON_ATTACKS:
            label, attack_type = "attack", AXON_ATTACKS[key]
        else:
            continue
        videos = [
            path
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"} and ".cache" not in path.parts
        ]
        images = [
            path
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"} and ".cache" not in path.parts
        ]
        for path in sorted(videos):
            samples.append(
                {
                    "dataset": "axon",
                    "sample_id": path.relative_to(root).as_posix(),
                    "label": label,
                    "attack_type": attack_type,
                    "kind": "video",
                    "paths": [path],
                    "n_frames": 4,
                }
            )
        for path in sorted(images):
            samples.append(
                {
                    "dataset": "axon",
                    "sample_id": path.relative_to(root).as_posix(),
                    "label": label,
                    "attack_type": attack_type,
                    "kind": "images",
                    "paths": [path],
                    "n_frames": 1,
                }
            )
    return samples


def _score_sample(sample: dict, tracker: FaceTracker) -> dict[str, float] | None:
    if sample["kind"] == "video":
        frames = _video_frames(sample["paths"][0], sample["n_frames"])
    else:
        frames = _image_frames(sample["paths"], sample["n_frames"])
    rows = []
    for frame in frames:
        small = _limit_side(frame)
        points = tracker.detect(small)
        if points is None:
            continue
        crops = _crops(small, points)
        if crops is None:
            continue
        face, regions, context = crops
        if len(regions) < 3:
            continue
        rows.append(compute_features(face, regions, context))
    need = 1 if sample["n_frames"] == 1 else 2
    if len(rows) < need:
        return None
    stacked = {name: np.array([row[name] for row in rows], dtype=np.float64) for name in FEATURE_NAMES}
    return {name: float(np.nanmedian(values)) for name, values in stacked.items()}


def _auc(live: np.ndarray, attack: np.ndarray) -> float:
    if live.size == 0 or attack.size == 0:
        return float("nan")
    diff = attack[:, None] - live[None, :]
    return float((np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / diff.size)


def _rates(live: np.ndarray, attack: np.ndarray, threshold: float) -> tuple[float, float]:
    """APCER, BPCER при правиле «атака, если score >= threshold»."""
    apcer = float(np.mean(attack < threshold)) if attack.size else float("nan")
    bpcer = float(np.mean(live >= threshold)) if live.size else float("nan")
    return apcer, bpcer


def _eer_threshold(live: np.ndarray, attack: np.ndarray) -> float:
    values = np.unique(np.concatenate([live, attack]))
    if values.size == 1:
        return float(values[0])
    midpoints = (values[:-1] + values[1:]) / 2.0
    candidates = np.concatenate(([values[0] - 1.0], midpoints, [values[-1] + 1.0]))
    best_threshold = float(candidates[0])
    best_key = (2.0, 2.0)
    for threshold in candidates:
        apcer, bpcer = _rates(live, attack, float(threshold))
        key = (abs(apcer - bpcer), 0.5 * (apcer + bpcer))
        if key < best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def _folds(labels: list[str], attack_types: list[str], k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fold_of = np.zeros(len(labels), dtype=np.int32)
    groups: dict[str, list[int]] = {}
    for index, (label, attack_type) in enumerate(zip(labels, attack_types)):
        groups.setdefault(label if label == "live" else attack_type, []).append(index)
    for indexes in groups.values():
        order = np.array(indexes, dtype=np.int32)
        rng.shuffle(order)
        for position, index in enumerate(order):
            fold_of[index] = position % k
    return fold_of


def _evaluate(samples: list[dict], scores: list[dict[str, float]]) -> dict:
    labels = [sample["label"] for sample in samples]
    attack_types = [sample["attack_type"] for sample in samples]
    fold_of = _folds(labels, attack_types, N_FOLDS, seed=0)
    report = {}
    for name in FEATURE_NAMES:
        values = np.array([row[name] for row in scores], dtype=np.float64)
        valid = np.isfinite(values)
        live_mask = np.array([label == "live" for label in labels]) & valid
        attack_mask = np.array([label == "attack" for label in labels]) & valid
        live_scores = values[live_mask]
        attack_scores = values[attack_mask]
        raw_auc = _auc(live_scores, attack_scores)
        orientation = 1.0 if raw_auc >= 0.5 else -1.0
        auc = raw_auc if orientation > 0 else 1.0 - raw_auc

        predicted_attack = np.full(len(samples), False)
        used = np.zeros(len(samples), dtype=bool)
        for fold in range(N_FOLDS):
            test = (fold_of == fold) & valid
            train = (fold_of != fold) & valid
            train_live = values[train & live_mask]
            train_attack = values[train & ~live_mask & valid]
            if train_live.size == 0 or train_attack.size == 0 or not np.any(test):
                continue
            sign = 1.0 if _auc(train_live, train_attack) >= 0.5 else -1.0
            threshold = _eer_threshold(sign * train_live, sign * train_attack)
            predicted_attack[test] = (sign * values[test]) >= threshold
            used[test] = True

        def _fraction(mask: np.ndarray, as_live: bool) -> tuple[int, int, float | None]:
            chosen = mask & used
            total = int(chosen.sum())
            if total == 0:
                return 0, 0, None
            hits = chosen & (~predicted_attack if as_live else predicted_attack)
            count = int(hits.sum())
            return count, total, count / total

        # APCER: атака, принятая за живое. BPCER: живое, отвергнутое как атака.
        attack_as_live, n_attack, apcer = _fraction(attack_mask, as_live=True)
        live_as_attack, n_live, bpcer = _fraction(live_mask, as_live=False)
        by_type = {}
        for attack_type in sorted({item for item in attack_types if item != "live"}):
            type_mask = np.array([item == attack_type for item in attack_types]) & attack_mask
            missed, total, rate = _fraction(type_mask, as_live=True)
            by_type[attack_type] = {"attack_as_live": missed, "n": total, "APCER": rate}
        hter = None if apcer is None or bpcer is None else 0.5 * (apcer + bpcer)
        report[name] = {
            "title": FEATURE_INFO[name]["title"],
            "family": FEATURE_INFO[name]["family"],
            "AUC": auc,
            "sign": orientation,
            "APCER": apcer,
            "BPCER": bpcer,
            "HTER": hter,
            "attack_as_live": attack_as_live,
            "n_attack": n_attack,
            "live_as_attack": live_as_attack,
            "n_live": n_live,
            "by_type": by_type,
        }
    return report


def _json_default(value):
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(type(value))


def _print_report(dataset: str, report: dict) -> None:
    print(f"\n{dataset}")
    header = f"{'признак':<36} {'AUC':>6} {'APCER':>16} {'BPCER':>14} {'HTER':>6}"
    print(header)
    ordered = sorted(report, key=lambda name: (report[name]["HTER"] is None, report[name]["HTER"] or 1))
    for name in ordered:
        row = report[name]
        apcer = "—" if row["APCER"] is None else f"{row['APCER']:.3f} ({row['attack_as_live']}/{row['n_attack']})"
        bpcer = "—" if row["BPCER"] is None else f"{row['BPCER']:.3f} ({row['live_as_attack']}/{row['n_live']})"
        hter = "—" if row["HTER"] is None else f"{row['HTER']:.3f}"
        auc = "—" if row["AUC"] is None or not np.isfinite(row["AUC"]) else f"{row['AUC']:.3f}"
        print(f"{row['title']:<36} {auc:>6} {apcer:>16} {bpcer:>14} {hter:>6}")
    types = []
    for row in report.values():
        types = list(row["by_type"])
        break
    if not types:
        return
    print("APCER по типу атаки")
    print(f"{'признак':<36} " + " ".join(f"{item:>14}" for item in types))
    for name in ordered:
        cells = []
        for attack_type in types:
            item = report[name]["by_type"][attack_type]
            cells.append("—" if item["APCER"] is None else f"{item['APCER']:.2f} ({item['attack_as_live']}/{item['n']})")
        print(f"{report[name]['title']:<36} " + " ".join(f"{cell:>14}" for cell in cells))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="APCER/BPCER текстурных признаков маски и вставки")
    parser.add_argument("--android", type=Path, default=Path("android"))
    parser.add_argument(
        "--axon",
        type=Path,
        default=Path("data/hf/AxonData/face-anti-spoofing-dataset"),
    )
    parser.add_argument("--out", type=Path, default=Path("python/pad_prototype/output/texture_features.json"))
    parser.add_argument("--dataset", choices=("android", "axon", "both"), default="both")
    args = parser.parse_args()

    _self_check()
    groups = []
    if args.dataset in {"android", "both"}:
        groups.append(("android", _android_samples(args.android)))
    if args.dataset in {"axon", "both"}:
        groups.append(("axon", _axon_samples(args.axon)))
    if not any(samples for _, samples in groups):
        raise SystemExit("Нет клипов")

    tracker = FaceTracker()
    started = time.time()
    payload = {
        "protocol": (
            f"{N_FOLDS}-fold по клипам, стратификация по типу атаки. "
            "Знак и порог — EER на обучении, APCER/BPCER — на отложенных клипах, каждый клип один раз. "
            "Скор клипа — медиана кадров. Лицо выровнено по глазам, кроп 192, регионы 64."
        ),
        "features": FEATURE_INFO,
        "datasets": {},
    }
    for dataset, samples in groups:
        print(f"Считаю {dataset}: {len(samples)} клипов", flush=True)
        kept_samples = []
        kept_scores = []
        failed = 0
        for index, sample in enumerate(samples, start=1):
            t0 = time.time()
            try:
                scores = _score_sample(sample, tracker)
            except Exception as error:
                scores = None
                print(f"  ошибка {sample['sample_id']}: {error}", flush=True)
            if scores is None:
                failed += 1
                print(f"  [{index}/{len(samples)}] нет лица {sample['sample_id']}", flush=True)
                continue
            kept_samples.append(sample)
            kept_scores.append(scores)
            if index == 1 or index % 10 == 0 or index == len(samples):
                print(
                    f"  [{index}/{len(samples)}] {sample['sample_id']} за {time.time() - t0:.2f}с",
                    flush=True,
                )
        report = _evaluate(kept_samples, kept_scores)
        _print_report(dataset, report)
        payload["datasets"][dataset] = {
            "n_input": len(samples),
            "n_scored": len(kept_samples),
            "n_failed": failed,
            "samples": [
                {
                    "sample_id": sample["sample_id"],
                    "label": sample["label"],
                    "attack_type": sample["attack_type"],
                    "scores": scores,
                }
                for sample, scores in zip(kept_samples, kept_scores)
            ],
            "report": report,
        }
        print(f"  без лица: {failed}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(f"\nРезультат: {args.out}")
    print(f"Время: {time.time() - started:.0f} с")


if __name__ == "__main__":
    main()
