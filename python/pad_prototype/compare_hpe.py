"""Сравнение двух HPE на папках живости и то, как каждый отбирает пары для параллакса носа.

Один проход MediaPipe Face Landmarker. На кадре считаются:
- facerecog2: yaw/pitch/roll по носу и четырём углам глаз;
- prototype: yaw/pitch/roll из solvePnP.

Детекция атаки — прежний nose_specificity, порог 1.80. Меняется только ворота
движения: геодезический угол solvePnP или hypot(Δyaw, Δpitch) facerecog2.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pad_prototype.hpe import angle_diff_deg
from pad_prototype.run import (
    THRESHOLD,
    FaceTracker,
    _json_default,
    collect_faces,
    discover_videos,
    pair_specificity,
)


def _finite_pairs(faces: list[dict]) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    pairs = []
    for face in faces:
        if face["pnp"] is None or face["fr"] is None:
            continue
        pairs.append((face["pnp"], face["fr"]))
    return pairs


def _mae(diffs: np.ndarray) -> float | None:
    if diffs.size == 0:
        return None
    return float(np.mean(np.abs(diffs)))


def _rates(rows: list[dict], key: str) -> dict:
    n_live = sum(row["true_label"] == "live" for row in rows)
    n_attack = len(rows) - n_live
    attack_as_live = sum(
        row["true_label"] == "attack" and row[key] == "live" for row in rows
    )
    live_as_attack = sum(
        row["true_label"] == "live" and row[key] != "live" for row in rows
    )
    return {
        "n_live": n_live,
        "n_attack": n_attack,
        "attack_as_live": attack_as_live,
        "live_as_attack": live_as_attack,
        "APCER": attack_as_live / n_attack if n_attack else None,
        "BPCER": live_as_attack / n_live if n_live else None,
    }


def _label(specificity: float, pairs: float, n_faces: float) -> str:
    if n_faces < 5 or pairs < 3 or not np.isfinite(specificity):
        return "attack"
    return "live" if specificity >= THRESHOLD else "attack"


def _span(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return float(max(values) - min(values))


def compare(data: Path, out: Path) -> dict:
    videos = discover_videos(data)
    if not videos:
        raise SystemExit(f"Нет клипов в {data}")
    tracker = FaceTracker()
    rows = []
    yaw_d, pitch_d, roll_d = [], [], []
    started = time.time()
    for index, video in enumerate(videos, start=1):
        t0 = time.time()
        faces = collect_faces(video, tracker)
        spec_pnp, pairs_pnp = pair_specificity(faces, "pnp")
        spec_fr, pairs_fr = pair_specificity(faces, "facerecog2")
        both = _finite_pairs(faces)
        for pnp, fr in both:
            yaw_d.append(angle_diff_deg(pnp[0], fr[0]))
            pitch_d.append(angle_diff_deg(pnp[1], fr[1]))
            roll_d.append(angle_diff_deg(pnp[2], fr[2]))
        pnp_yaw = [face["pnp"][0] for face in faces if face["pnp"] is not None]
        fr_yaw = [face["fr"][0] for face in faces if face["fr"] is not None]
        row = {
            "video_id": video["video_id"],
            "attack_type": video["attack_type"],
            "true_label": video["label"],
            "n_faces": len(faces),
            "n_both": len(both),
            "pnp_specificity": None if not np.isfinite(spec_pnp) else spec_pnp,
            "pnp_pairs": pairs_pnp,
            "pnp_label": _label(spec_pnp, pairs_pnp, len(faces)),
            "fr_specificity": None if not np.isfinite(spec_fr) else spec_fr,
            "fr_pairs": pairs_fr,
            "fr_label": _label(spec_fr, pairs_fr, len(faces)),
            "pnp_yaw_span": _span(pnp_yaw),
            "fr_yaw_span": _span(fr_yaw),
        }
        rows.append(row)
        print(
            f"[{index}/{len(videos)}] {row['video_id']} "
            f"pnp={row['pnp_label']} {row['pnp_specificity']} "
            f"fr={row['fr_label']} {row['fr_specificity']} "
            f"за {time.time() - t0:.1f}с",
            flush=True,
        )

    def _angle_stats(values: list[float]) -> dict:
        array = np.asarray(values, dtype=np.float64)
        if array.size == 0:
            return {"mae": None, "mean": None, "median_abs": None, "frac_abs_gt_30": None}
        absolute = np.abs(array)
        return {
            "mae": float(absolute.mean()),
            "mean": float(array.mean()),
            "median_abs": float(np.median(absolute)),
            "frac_abs_gt_30": float(np.mean(absolute > 30.0)),
        }

    yaw_a = np.asarray(yaw_d, dtype=np.float64)
    pitch_a = np.asarray(pitch_d, dtype=np.float64)
    roll_a = np.asarray(roll_d, dtype=np.float64)
    differ = [
        row["video_id"]
        for row in rows
        if row["pnp_label"] != row["fr_label"]
    ]
    summary = {
        "n_frames_both": int(yaw_a.size),
        "yaw": _angle_stats(yaw_d),
        "pitch": _angle_stats(pitch_d),
        "roll": _angle_stats(roll_d),
        "mae_yaw_deg": _mae(yaw_a),
        "mae_pitch_deg": _mae(pitch_a),
        "mae_roll_deg": _mae(roll_a),
        "median_abs_yaw_deg": None if yaw_a.size == 0 else float(np.median(np.abs(yaw_a))),
        "median_abs_pitch_deg": None if pitch_a.size == 0 else float(np.median(np.abs(pitch_a))),
        "median_abs_roll_deg": None if roll_a.size == 0 else float(np.median(np.abs(roll_a))),
        "attack_pnp": _rates(rows, "pnp_label"),
        "attack_facerecog2": _rates(rows, "fr_label"),
        "n_label_disagree": len(differ),
        "label_disagree": differ,
        "threshold": THRESHOLD,
        "seconds": time.time() - started,
    }
    payload = {"summary": summary, "videos": rows}
    out.mkdir(parents=True, exist_ok=True)
    path = out / "hpe_compare.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(
        f"\nКадры с обоими HPE: {summary['n_frames_both']}. "
        f"MAE yaw/pitch/roll = {summary['mae_yaw_deg']:.2f} / "
        f"{summary['mae_pitch_deg']:.2f} / {summary['mae_roll_deg']:.2f}°"
    )
    for name, block in (("solvePnP", summary["attack_pnp"]), ("facerecog2", summary["attack_facerecog2"])):
        print(
            f"{name}: APCER={block['APCER']:.3f} ({block['attack_as_live']}/{block['n_attack']}) "
            f"BPCER={block['BPCER']:.3f} ({block['live_as_attack']}/{block['n_live']})"
        )
    print(f"Вердикты разошлись на {len(differ)} клипах")
    print(f"Сравнение: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Сравнить HPE facerecog2 и solvePnP на папках живости")
    parser.add_argument("--data", type=Path, default=Path("android"))
    parser.add_argument("--out", type=Path, default=Path("python/pad_prototype/output"))
    args = parser.parse_args()
    compare(args.data, args.out)


if __name__ == "__main__":
    main()
