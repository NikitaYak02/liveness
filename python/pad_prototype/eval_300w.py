"""Yaw, pitch, roll на 300W-LP.

Точки — MediaPipe Face Landmarker, публичная сеть, та же, что в прототипе.
Два определения углов: facerecog2 (нос и углы глаз) и solvePnP прототипа.
Эталон — pose из json DirectMHP (градусы). Порядок компонент и знаки
подбираются по корреляции и печатаются: в json это не подписано.
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import permutations
from pathlib import Path

import cv2
import numpy as np

from pad_prototype.hpe import angle_diff_deg, facerecog2_ypr, prototype_ypr
from pad_prototype.run import FaceTracker, _json_default

NAMES = ("yaw", "pitch", "roll")


def _index_images(root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        found.setdefault(path.name, path)
    return found


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    out = {}
    maes = []
    for index, name in enumerate(NAMES):
        diff = np.array(
            [angle_diff_deg(float(p), float(g)) for p, g in zip(pred[:, index], gt[:, index])],
            dtype=np.float64,
        )
        order = np.sort(np.abs(diff))
        n80 = max(1, int(0.8 * order.size))
        mae = float(order.mean())
        maes.append(mae)
        out[name] = {
            "mae": mae,
            "mae80": float(order[:n80].mean()),
            "corr": _corr(pred[:, index], gt[:, index]),
            "frac_abs_gt_90": float(np.mean(order > 90.0)),
        }
    out["mae_mean"] = float(np.mean(maes))
    return out


def _best_alignment(reference: np.ndarray, raw: np.ndarray) -> tuple[dict, np.ndarray]:
    """Оси эталона по facerecog2: перестановка с наибольшей |корреляцией|, знак — чтобы корреляция была положительной.

    Знак не меняет |корреляцию|, поэтому его нельзя выбирать перебором вместе с ней:
    иначе остаётся первая комбинация, и MAE смотрит не в ту сторону.
    """
    best = None
    best_gt = None
    best_score = -1.0
    for perm in permutations(range(3)):
        signed = []
        signs = []
        score = 0.0
        for index in range(3):
            column = raw[:, perm[index]]
            coefficient = _corr(reference[:, index], column)
            sign = 1.0 if coefficient >= 0 else -1.0
            signed.append(sign * column)
            signs.append(sign)
            score += abs(coefficient)
        score /= 3.0
        if score > best_score:
            best_score = score
            best_gt = np.column_stack(signed)
            best = {
                "json_index": {"yaw": perm[0], "pitch": perm[1], "roll": perm[2]},
                "signs": {"yaw": signs[0], "pitch": signs[1], "roll": signs[2]},
                "mean_abs_corr": score,
                "reference": "facerecog2",
            }
    return best, best_gt


def evaluate(images: Path, poses_path: Path, out: Path, limit: int, stride: int, report: str) -> dict:
    labels = json.loads(poses_path.read_text(encoding="utf-8"))
    if not isinstance(labels, dict):
        raise SystemExit("Ожидался json-словарь имя_файла -> {pose: [., ., .]}")
    names = sorted(labels)
    if stride > 1:
        names = names[::stride]
    if limit > 0:
        names = names[:limit]
    index = _index_images(images)
    tracker = FaceTracker()
    rows = []
    missing = 0
    started = time.time()
    for step, name in enumerate(names, start=1):
        path = index.get(name)
        if path is None:
            missing += 1
            continue
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None or bgr.size == 0:
            missing += 1
            continue
        points = tracker.detect(bgr)
        pose = labels[name].get("pose")
        if points is None or pose is None or len(pose) < 3:
            rows.append({"file": name, "ok": False, "gt": pose})
        else:
            rows.append(
                {
                    "file": name,
                    "ok": True,
                    "gt": [float(pose[0]), float(pose[1]), float(pose[2])],
                    "facerecog2": facerecog2_ypr(points),
                    "prototype": prototype_ypr(points, bgr.shape[:2]),
                }
            )
        if step % 200 == 0 or step == len(names):
            ok = sum(
                1
                for row in rows
                if row["ok"] and row.get("facerecog2") and row.get("prototype")
            )
            print(
                f"[{step}/{len(names)}] лиц с обоими HPE {ok}, нет файла/кадра {missing}, "
                f"{time.time() - started:.0f}с",
                flush=True,
            )

    usable = [
        row
        for row in rows
        if row["ok"] and row["facerecog2"] is not None and row["prototype"] is not None
    ]
    summary: dict = {
        "n_listed": len(names),
        "n_missing_image": missing,
        "n_detected": len(usable),
        "seconds": time.time() - started,
    }
    if usable:
        raw = np.asarray([row["gt"] for row in usable], dtype=np.float64)
        preds = {
            key: np.asarray([row[key] for row in usable], dtype=np.float64)
            for key in ("facerecog2", "prototype")
        }
        # Одна договорённость об осях эталона на оба метода: это свойство json, не оценки.
        alignment, gt = _best_alignment(preds["facerecog2"], raw)
        summary["gt_alignment"] = alignment
        for key, pred in preds.items():
            summary[key] = _metrics(pred, gt)
    out.mkdir(parents=True, exist_ok=True)
    path = out / report
    frames_path = path.with_name(path.stem + "_frames.jsonl")
    with frames_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=_json_default))
            handle.write("\n")
    path.write_text(
        json.dumps({"summary": summary, "frames": str(frames_path)}, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(f"Результат: {path}")
    align = summary.get("gt_alignment")
    if align:
        print(
            f"Эталон: json[{align['json_index']['yaw']},{align['json_index']['pitch']},"
            f"{align['json_index']['roll']}] * {align['signs']}, |corr|={align['mean_abs_corr']:.3f}"
        )
    for key in ("facerecog2", "prototype"):
        err = summary.get(key)
        if not err:
            continue
        print(
            f"{key}: MAE yaw/pitch/roll = {err['yaw']['mae']:.2f} / {err['pitch']['mae']:.2f} / "
            f"{err['roll']['mae']:.2f}  MAE80 = {err['yaw']['mae80']:.2f} / "
            f"{err['pitch']['mae80']:.2f} / {err['roll']['mae80']:.2f}  "
            f"corr = {err['yaw']['corr']:.2f} / {err['pitch']['corr']:.2f} / {err['roll']['corr']:.2f}"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Yaw/pitch/roll на 300W-LP")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("python/pad_prototype/output"))
    parser.add_argument("--limit", type=int, default=0, help="0 — все записи json")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--report", default="hpe_300w.json")
    args = parser.parse_args()
    evaluate(args.images, args.poses, args.out, args.limit, args.stride, args.report)


if __name__ == "__main__":
    main()
