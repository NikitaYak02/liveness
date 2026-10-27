"""Стоп-кадры с сильным поворотом головы.

Скрипт обходит дерево видео, на выбранных кадрах оценивает yaw и pitch
и сохраняет исходный кадр, если поворот больше порога. Папки повторяют
вход: ролик ``класс/сессия/clip.mp4`` даёт кадры в ``класс/сессия/clip/``.

Запуск из корня репозитория:

    PYTHONPATH=python python -m pad_prototype.extract_pose_frames \\
        --input /path/videos --output /path/dataset --threshold 30
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from pad_prototype.hpe import (
    EYE_INNER_LEFT,
    EYE_INNER_RIGHT,
    EYE_OUTER_LEFT,
    EYE_OUTER_RIGHT,
    facerecog2_ypr,
    prototype_ypr,
)
from pad_prototype.run import FaceTracker

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg"}


def is_strong_turn(
    yaw: float,
    pitch: float,
    yaw_threshold: float,
    pitch_threshold: float,
    require: str,
) -> bool:
    yaw_hit = abs(yaw) > yaw_threshold
    pitch_hit = abs(pitch) > pitch_threshold
    if require == "both":
        return yaw_hit and pitch_hit
    if require == "yaw":
        return yaw_hit
    if require == "pitch":
        return pitch_hit
    return yaw_hit or pitch_hit


def frame_step(fps: float, sample_fps: float) -> int:
    if sample_fps <= 0 or fps <= 1e-3:
        return 1
    return max(1, int(round(fps / sample_fps)))


def discover_videos(root: Path, output: Path) -> list[Path]:
    output_root = output.resolve()
    if root.is_file():
        candidates = [root]
    else:
        candidates = [
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        ]
    videos = []
    for path in candidates:
        try:
            path.resolve().relative_to(output_root)
        except ValueError:
            videos.append(path)
    return sorted(videos)


def frames_dir(video: Path, root: Path, output: Path) -> Path:
    if root.is_file():
        relative = Path(video.stem)
    else:
        relative = video.relative_to(root).with_suffix("")
    return output / relative


def frame_name(index: int, yaw: float, pitch: float) -> str:
    return f"{index:06d}_yaw{yaw:+06.1f}_pitch{pitch:+06.1f}.jpg"


def _eye_distance(points: np.ndarray) -> float:
    left = points[[EYE_OUTER_LEFT, EYE_INNER_LEFT]].mean(axis=0)
    right = points[[EYE_OUTER_RIGHT, EYE_INNER_RIGHT]].mean(axis=0)
    return float(np.linalg.norm(left - right))


def _detection_view(bgr: np.ndarray, max_side: int) -> np.ndarray:
    height, width = bgr.shape[:2]
    longest = max(height, width)
    if max_side <= 0 or longest <= max_side:
        return bgr
    scale = max_side / float(longest)
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)


def _pose(points: np.ndarray, shape: tuple[int, int], method: str):
    if method == "facerecog2":
        return facerecog2_ypr(points)
    return prototype_ypr(points, shape)


def iter_frames(path: Path, sample_fps: float):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"не удалось открыть {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    step = frame_step(fps, sample_fps)
    index = 0
    try:
        while True:
            if index % step == 0:
                ok, frame = capture.read()
                if not ok or frame is None or frame.size == 0:
                    break
                timestamp = index / fps if fps > 1e-3 else 0.0
                yield index, timestamp, frame
            elif not capture.grab():
                break
            index += 1
    finally:
        capture.release()


def _video_label(video: Path, root: Path) -> str:
    if root.is_file():
        return video.name
    return video.relative_to(root).as_posix()


def _clear_images(folder: Path) -> None:
    if not folder.exists():
        return
    for path in folder.glob("*.jpg"):
        path.unlink()


def _remove_empty(folder: Path, stop: Path) -> None:
    current = folder
    while current != stop and current.exists():
        if any(current.iterdir()):
            return
        current.rmdir()
        current = current.parent


def extract_video(
    video: Path,
    root: Path,
    output: Path,
    tracker: FaceTracker,
    *,
    yaw_threshold: float,
    pitch_threshold: float,
    require: str,
    sample_fps: float,
    pose: str,
    max_side: int,
    jpeg_quality: int,
    min_eye: float,
    min_delta: float,
    max_per_video: int,
    manifest,
) -> dict[str, int]:
    destination = frames_dir(video, root, output)
    _clear_images(destination)
    checked = 0
    saved = 0
    last: tuple[float, float] | None = None
    for index, timestamp, frame in iter_frames(video, sample_fps):
        if max_per_video > 0 and saved >= max_per_video:
            break
        view = _detection_view(frame, max_side)
        points = tracker.detect(view)
        checked += 1
        if points is None or _eye_distance(points) < min_eye:
            continue
        angles = _pose(points, view.shape[:2], pose)
        if angles is None:
            continue
        yaw, pitch = float(angles[0]), float(angles[1])
        if not is_strong_turn(yaw, pitch, yaw_threshold, pitch_threshold, require):
            continue
        if last is not None and min_delta > 0:
            if abs(yaw - last[0]) < min_delta and abs(pitch - last[1]) < min_delta:
                continue
        name = frame_name(index, yaw, pitch)
        image_path = destination / name
        destination.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(
            str(image_path),
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
        )
        if not ok:
            raise RuntimeError(f"не удалось записать {image_path}")
        record = {
            "image": image_path.relative_to(output).as_posix(),
            "video": _video_label(video, root),
            "frame": int(index),
            "time_sec": round(float(timestamp), 3),
            "yaw": round(yaw, 2),
            "pitch": round(pitch, 2),
            "roll": round(float(angles[2]), 2),
        }
        manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
        manifest.flush()
        last = (yaw, pitch)
        saved += 1
    if saved == 0:
        _remove_empty(destination, output)
    return {"checked": checked, "saved": saved}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Сохранить стоп-кадры, где |yaw| или |pitch| больше порога",
    )
    parser.add_argument("--input", type=Path, required=True, help="Видео или папка с вложенными видео")
    parser.add_argument("--output", type=Path, required=True, help="Корень датасета")
    parser.add_argument(
        "--threshold",
        type=float,
        default=30.0,
        help="Порог в градусах для |yaw| и |pitch|, если отдельные пороги не заданы",
    )
    parser.add_argument("--yaw-threshold", type=float, default=None)
    parser.add_argument("--pitch-threshold", type=float, default=None)
    parser.add_argument(
        "--require",
        choices=("either", "both", "yaw", "pitch"),
        default="either",
        help="either: достаточно одного угла, both: оба, yaw/pitch: только этот угол",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=2.0,
        help="Сколько кадров в секунду проверять. 0 — каждый кадр",
    )
    parser.add_argument("--pose", choices=("pnp", "facerecog2"), default="pnp")
    parser.add_argument(
        "--max-side",
        type=int,
        default=960,
        help="Длинная сторона кадра для детекции и позы. На диск пишется исходный кадр",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--min-eye", type=float, default=12.0, help="Минимальная дистанция между глазами, px")
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="Не писать кадр, если и yaw, и pitch сдвинулись меньше чем на столько градусов",
    )
    parser.add_argument("--max-per-video", type=int, default=0, help="Лимит сохранённых кадров на ролик. 0 — без лимита")
    parser.add_argument(
        "--skip-done",
        action="store_true",
        help="Пропускать ролики, уже записанные в done.txt",
    )
    args = parser.parse_args()

    root = args.input
    output = args.output
    if not root.exists():
        raise SystemExit(f"нет пути {root}")
    yaw_threshold = args.threshold if args.yaw_threshold is None else args.yaw_threshold
    pitch_threshold = args.threshold if args.pitch_threshold is None else args.pitch_threshold
    videos = discover_videos(root, output)
    if not videos:
        raise SystemExit(f"видео не найдены в {root}")

    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.jsonl"
    done_path = output / "done.txt"
    done: set[str] = set()
    if args.skip_done and done_path.exists():
        done = {line.strip() for line in done_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    else:
        done_path.write_text("", encoding="utf-8")
    kept_lines: list[str] = []
    if args.skip_done and manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("video") in done:
                kept_lines.append(line)
    tracker: FaceTracker | None = None
    started = time.time()
    total_saved = 0
    total_checked = 0
    failed = 0
    skipped = 0

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for line in kept_lines:
            manifest.write(line + "\n")
        manifest.flush()
        for number, video in enumerate(videos, start=1):
            label = _video_label(video, root)
            if label in done:
                skipped += 1
                print(f"[{number}/{len(videos)}] {label} уже готово", flush=True)
                continue
            video_started = time.time()
            try:
                if tracker is None:
                    tracker = FaceTracker()
                stats = extract_video(
                    video,
                    root,
                    output,
                    tracker,
                    yaw_threshold=yaw_threshold,
                    pitch_threshold=pitch_threshold,
                    require=args.require,
                    sample_fps=args.sample_fps,
                    pose=args.pose,
                    max_side=args.max_side,
                    jpeg_quality=args.jpeg_quality,
                    min_eye=args.min_eye,
                    min_delta=args.min_delta,
                    max_per_video=args.max_per_video,
                    manifest=manifest,
                )
            except Exception as exc:
                failed += 1
                print(f"[{number}/{len(videos)}] {label} ошибка: {exc}", flush=True)
                continue
            total_saved += stats["saved"]
            total_checked += stats["checked"]
            with done_path.open("a", encoding="utf-8") as done_file:
                done_file.write(label + "\n")
            done.add(label)
            print(
                f"[{number}/{len(videos)}] {label} сохранено {stats['saved']} "
                f"из {stats['checked']} за {time.time() - video_started:.1f}с",
                flush=True,
            )

    print(
        f"готово: видео {len(videos)}, проверено кадров {total_checked}, "
        f"сохранено {total_saved}, пропущено {skipped}, ошибок {failed}, "
        f"{time.time() - started:.1f}с, манифест {manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
