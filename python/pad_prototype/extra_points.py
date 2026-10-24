"""Дополнительные точки MediaPipe: параллакс носа и остаток губ.

Один проход Face Landmarker (тот же collect_faces, что в run.py), кэш
в output/landmark_cache. Дальше только формулы. Порог и решение run.py
не меняются.

Индексы сверены с FaceLandmarksConnections установленного mediapipe и с
канонической моделью canonical_face_model.obj (468 вершин, +Z вперёд, +Y вверх,
+X влево от человека). У модели нет имён вершин, роли такие:

- 1 — нижняя точка кончика (z=7.48). Её уже использует прототип.
- 4 — pronasale, максимум Z всей модели (z=7.59), сразу над 1 на цепочке носа.
- 48 и 278 — крылья: максимум |X| на контуре FACE_LANDMARKS_NOSE (z=5.81).
- 98 и 327 — ноздри: ниже и медиальнее крыльев (z=5.24), на том же контуре.
- 2 — subnasale: конец срединной цепочки 1-19-94-2, низ контура носа (z=6.06).
- 0 и 17 — середины внешнего контура губ (верх / низ).
- 13 и 14 — середины внутреннего контура губ.
- 159 и 145 — середины верхнего и нижнего века правого глаза человека.
- 386 и 374 — то же для левого глаза.

Не используются: 164 (фильтрум, не на контуре носа), 64/294 (соседние с крыльями
точки ободка ноздри, роль неоднозначна), 61/291 (уже в гомографии).
Губы в holdout не входят.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from pad_prototype.run import (
    THRESHOLD,
    FaceTracker,
    _errors,
    _json_default,
    _median,
    _parts,
    _pose_delta,
    collect_faces,
    discover_videos,
    pair_specificity,
)

CACHE_VERSION = 1
# Только отчёт о трёх текущих ошибках. В решающее правило эти id не входят.
REPORT_IDS = ("masks_with_nose/5_1", "real_1/8", "real_1/10")

# Равные веса, если weights is None. (2, 1, 1) — половина веса у кончика.
MEAN_PROBES: tuple[tuple[str, tuple[int, ...], tuple[float, ...] | None], ...] = (
    ("nose_tip", (1,), None),
    ("nose_wings", (48, 278), None),
    ("nose_tip_wings", (1, 48, 278), None),
    ("nose_tip_half_wings", (1, 48, 278), (2.0, 1.0, 1.0)),
    ("nose_tip_wings_sub", (1, 48, 278, 2), None),
    ("nose_pronasale", (4,), None),
    ("nose_pronasale_wings", (4, 48, 278), None),
    ("nose_tip_nostrils", (1, 98, 327), None),
)
RESIDUAL_PROBES: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("lip_outer", (0, 17)),
    ("lip_inner", (13, 14)),
    ("lip_four", (0, 13, 14, 17)),
    ("lip_upper", (0,)),
    ("lip_lower", (17,)),
    ("eye_upper", (159, 386)),
    ("eye_lower", (145, 374)),
    ("eye_four", (159, 145, 386, 374)),
)
DIAG_NAMES = (
    "nose_tip",
    "nose_tip_wings",
    "lip_inner",
    "lip_outer",
    "lip_four",
    "lip_lower",
    "eye_upper",
    "eye_four",
)
P90_NAMES = ("lip_inner", "lip_outer", "lip_four", "lip_lower", "eye_upper", "eye_four")

NOSE_FOR_RULES = (
    "nose_tip",
    "nose_tip_wings",
    "nose_tip_half_wings",
    "nose_tip_wings_sub",
    "nose_pronasale",
    "nose_pronasale_wings",
)
LIP_FOR_RULES = (
    "lip_inner",
    "lip_outer",
    "lip_four",
    "lip_lower",
    "lip_inner_p90",
    "eye_upper",
    "eye_upper_p90",
)


def _adjacency(connections) -> dict[int, set[int]]:
    graph: dict[int, set[int]] = {}
    for link in connections:
        graph.setdefault(link.start, set()).add(link.end)
        graph.setdefault(link.end, set()).add(link.start)
    return graph


def _simple_paths(graph: dict[int, set[int]], start: int, end: int) -> list[list[int]]:
    found: list[list[int]] = []

    def walk(node: int, path: list[int], seen: set[int]) -> None:
        if node == end:
            found.append(list(path))
            return
        for nxt in graph[node]:
            if nxt in seen:
                continue
            seen.add(nxt)
            path.append(nxt)
            walk(nxt, path, seen)
            path.pop()
            seen.remove(nxt)

    walk(start, [start], {start})
    return found


def _mids(paths: list[list[int]]) -> set[int]:
    shortest = min(len(path) for path in paths)
    return {path[len(path) // 2] for path in paths if len(path) == shortest and len(path) % 2 == 1}


def _chain_until_branch(graph: dict[int, set[int]], start: int) -> list[int]:
    path = [start]
    prev = None
    node = start
    while True:
        nxts = [item for item in graph[node] if item != prev]
        if len(nxts) != 1:
            break
        prev, node = node, nxts[0]
        path.append(node)
        if len(path) > 40:
            break
    return path


def verify_topology() -> None:
    """Середины контуров губ и век — из FaceLandmarksConnections, не из шпаргалки."""
    from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections as Mesh

    lips = _adjacency(Mesh.FACE_LANDMARKS_LIPS)
    right_eye = _adjacency(Mesh.FACE_LANDMARKS_RIGHT_EYE)
    left_eye = _adjacency(Mesh.FACE_LANDMARKS_LEFT_EYE)
    nose = _adjacency(Mesh.FACE_LANDMARKS_NOSE)
    checks = {
        "outer lips 61-291": (_mids(_simple_paths(lips, 61, 291)), {0, 17}),
        "inner lips 78-308": (_mids(_simple_paths(lips, 78, 308)), {13, 14}),
        "right eye 33-133": (_mids(_simple_paths(right_eye, 33, 133)), {145, 159}),
        "left eye 263-362": (_mids(_simple_paths(left_eye, 263, 362)), {374, 386}),
    }
    for name, (got, expect) in checks.items():
        if got != expect:
            raise SystemExit(f"Топология {name}: середины {sorted(got)}, ожидали {sorted(expect)}")
    bridge = _chain_until_branch(nose, 168)
    if bridge != [168, 6, 197, 195, 5, 4] or 1 not in nose[4]:
        raise SystemExit(f"Цепочка спинки носа {bridge}, соседи 4: {sorted(nose[4])}")
    under = _chain_until_branch(nose, 1)
    # 1 соединён и со спинкой (4), и с низом (19); уникальной цепочки нет.
    underside = [1]
    prev, node = 1, 19
    underside.append(node)
    while node != 2:
        nxts = [item for item in nose[node] if item != prev]
        if len(nxts) != 1:
            break
        prev, node = node, nxts[0]
        underside.append(node)
    if underside != [1, 19, 94, 2]:
        raise SystemExit(f"Цепочка под кончиком {underside}")
    nose_ids = set(nose)
    required = {1, 2, 4, 6, 48, 64, 98, 278, 294, 327}
    if not required <= nose_ids:
        raise SystemExit(f"В контуре носа нет {sorted(required - nose_ids)}")
    print("Топология MediaPipe: середины губ и век, спинка носа и цепочка 1-19-94-2 совпали", flush=True)


def _cache_file(cache_dir: Path, video_id: str) -> Path:
    return cache_dir / f"{video_id.replace('/', '__')}.npz"


def _pack(video_id: str, faces: list[dict]) -> dict:
    n = len(faces)
    points = np.zeros((n, 478, 2), np.float32)
    eye = np.zeros(n, np.float64)
    diag = np.zeros(n, np.float64)
    rotation = np.full((n, 3, 3), np.nan, np.float64)
    fr = np.full((n, 3), np.nan, np.float64)
    pnp = np.full((n, 3), np.nan, np.float64)
    for index, face in enumerate(faces):
        points[index] = face["points"]
        eye[index] = face["eye"]
        diag[index] = face["diag"]
        if face["rotation"] is not None:
            rotation[index] = face["rotation"]
        if face["fr"] is not None:
            fr[index] = np.asarray(face["fr"], dtype=np.float64)
        if face["pnp"] is not None:
            pnp[index] = np.asarray(face["pnp"], dtype=np.float64)
    return {
        "cache_version": np.int32(CACHE_VERSION),
        "video_id": np.array(video_id),
        "points": points,
        "eye": eye,
        "diag": diag,
        "rotation": rotation,
        "fr": fr,
        "pnp": pnp,
    }


def _unpack(blob) -> list[dict]:
    n = int(blob["points"].shape[0])
    faces = []
    for index in range(n):
        rotation = blob["rotation"][index]
        fr = blob["fr"][index]
        pnp = blob["pnp"][index]
        faces.append(
            {
                "points": blob["points"][index],
                "eye": float(blob["eye"][index]),
                "diag": float(blob["diag"][index]),
                "rotation": None if not np.isfinite(rotation).all() else rotation,
                "fr": None if not np.isfinite(fr).all() else (float(fr[0]), float(fr[1]), float(fr[2])),
                "pnp": None if not np.isfinite(pnp).all() else (float(pnp[0]), float(pnp[1]), float(pnp[2])),
            }
        )
    return faces


def _load_cached(path: Path, video_id: str) -> list[dict] | None:
    if not path.exists():
        return None
    blob = np.load(path)
    if int(blob["cache_version"]) != CACHE_VERSION:
        return None
    if str(blob["video_id"]) != video_id:
        return None
    if blob["points"].ndim != 3 or blob["points"].shape[1:] != (478, 2):
        return None
    return _unpack(blob)


def ensure_cache(videos: list[dict], cache_dir: Path, recompute: bool) -> list[list[dict]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    loaded: list[list[dict] | None] = []
    missing = []
    for video in videos:
        path = _cache_file(cache_dir, video["video_id"])
        faces = None if recompute else _load_cached(path, video["video_id"])
        loaded.append(faces)
        if faces is None:
            missing.append(video["video_id"])
    print(f"Клипов {len(videos)}, в кэше {len(videos) - len(missing)}, считать {len(missing)}", flush=True)
    if not missing:
        return loaded  # type: ignore[return-value]
    tracker = FaceTracker()
    pending = {video["video_id"] for video in videos if _load_cached(_cache_file(cache_dir, video["video_id"]), video["video_id"]) is None or recompute}
    done = 0
    for index, video in enumerate(videos):
        if video["video_id"] not in pending:
            continue
        done += 1
        started = time.time()
        faces = collect_faces(video, tracker)
        path = _cache_file(cache_dir, video["video_id"])
        np.savez_compressed(path, **_pack(video["video_id"], faces))
        loaded[index] = faces
        print(
            f"[{done}/{len(pending)}] {video['video_id']} лиц {len(faces)} за {time.time() - started:.1f}с",
            flush=True,
        )
    return loaded  # type: ignore[return-value]


def _frontal(face: dict) -> bool:
    if face["pnp"] is None or face["eye"] < 30:
        return False
    yaw, pitch, roll = face["pnp"]
    return abs(yaw) < 18.0 and abs(pitch) < 18.0 and abs(roll) < 18.0


def verify_geometry(per_video: list[list[dict]]) -> dict:
    """На анфасных кадрах крылья должны быть латеральнее ноздрей, губы и веки — по вертикали."""
    from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections as Mesh

    nose_ids = sorted({link.start for link in Mesh.FACE_LANDMARKS_NOSE} | {link.end for link in Mesh.FACE_LANDMARKS_NOSE})
    checks = {
        "wings_opposite": 0,
        "wings_beyond_nostrils": 0,
        "vertical_nose_lips": 0,
        "inner_lips": 0,
        "lids": 0,
        "extrema_48_278": 0,
    }
    extrema: dict[str, int] = {}
    n = 0
    example = None
    for faces in per_video:
        for face in faces:
            if not _frontal(face):
                continue
            pts = face["points"]
            n += 1
            x1, y1 = float(pts[1, 0]), float(pts[1, 1])
            if (float(pts[48, 0]) - x1) * (float(pts[278, 0]) - x1) < 0:
                checks["wings_opposite"] += 1
            if abs(float(pts[48, 0]) - x1) > abs(float(pts[98, 0]) - x1) and abs(float(pts[278, 0]) - x1) > abs(
                float(pts[327, 0]) - x1
            ):
                checks["wings_beyond_nostrils"] += 1
            ys = [float(pts[i, 1]) for i in (4, 1, 2, 0, 17)]
            if ys[0] < ys[1] < ys[2] < ys[3] < ys[4]:
                checks["vertical_nose_lips"] += 1
            if float(pts[0, 1]) < float(pts[13, 1]) < float(pts[14, 1]) < float(pts[17, 1]):
                checks["inner_lips"] += 1
            if float(pts[159, 1]) < float(pts[145, 1]) and float(pts[386, 1]) < float(pts[374, 1]):
                checks["lids"] += 1
            left = min(nose_ids, key=lambda i: float(pts[i, 0]))
            right = max(nose_ids, key=lambda i: float(pts[i, 0]))
            key = f"{left}/{right}"
            extrema[key] = extrema.get(key, 0) + 1
            if {left, right} == {48, 278}:
                checks["extrema_48_278"] += 1
            if example is None:
                example = {
                    "yaw": face["pnp"][0],
                    "pitch": face["pnp"][1],
                    "dx": {str(i): float(pts[i, 0] - pts[1, 0]) for i in (4, 2, 48, 278, 98, 327, 0, 13, 14, 17)},
                    "dy": {str(i): float(pts[i, 1] - pts[1, 1]) for i in (4, 2, 48, 278, 98, 327, 0, 13, 14, 17, 159, 145)},
                }
    rates = {name: (count / n if n else None) for name, count in checks.items()}
    print(f"Анфас-кадров для проверки ролей: {n}", flush=True)
    for name, rate in rates.items():
        print(f"  {name}: {rate}", flush=True)
    top = sorted(extrema.items(), key=lambda item: -item[1])[:6]
    print(f"  латеральные экстремумы контура носа: {top}", flush=True)
    return {"n_frontal": n, "rates": rates, "extrema_top": top, "example": example}


def _probe_xy(points: np.ndarray, indices: tuple[int, ...], weights: tuple[float, ...] | None) -> np.ndarray:
    pts = points[list(indices)].astype(np.float64)
    if weights is None:
        acc = pts.mean(axis=0)
    else:
        w = np.asarray(weights, dtype=np.float64)
        acc = (pts * w[:, None]).sum(axis=0) / w.sum()
    return acc.astype(np.float32)


def _point_error(homography: np.ndarray, src_xy: np.ndarray, dst_xy: np.ndarray) -> float:
    pred = cv2.perspectiveTransform(np.asarray(src_xy, np.float32).reshape(1, 1, 2), homography).reshape(2)
    return float(np.linalg.norm(pred - np.asarray(dst_xy, np.float32)))


def _group_error(homography: np.ndarray, src: np.ndarray, dst: np.ndarray, indices: tuple[int, ...]) -> float:
    idx = list(indices)
    pred = cv2.perspectiveTransform(src[idx].astype(np.float32).reshape(-1, 1, 2), homography).reshape(-1, 2)
    return float(np.linalg.norm(pred - dst[idx].astype(np.float32), axis=1).mean())


def _pair_errors(first: dict, second: dict) -> dict[str, float] | None:
    src_p, src_n, src_h = _parts(first["points"])
    dst_p, dst_n, dst_h = _parts(second["points"])
    homography, _ = cv2.findHomography(src_p.reshape(-1, 1, 2), dst_p.reshape(-1, 1, 2), 0)
    if homography is None:
        return None
    tip = _point_error(homography, src_n, dst_n)
    hold = float(
        np.linalg.norm(
            cv2.perspectiveTransform(src_h.reshape(-1, 1, 2), homography).reshape(-1, 2) - dst_h,
            axis=1,
        ).mean()
    )
    ref_tip, ref_hold = _errors(first, second)
    if not np.isfinite(ref_tip) or abs(tip - ref_tip) > 1e-3 or abs(hold - ref_hold) > 1e-3:
        raise RuntimeError(f"Расхождение с _errors: {tip} vs {ref_tip}, hold {hold} vs {ref_hold}")
    out = {"nose_tip_err": tip, "hold": hold}
    for name, indices, weights in MEAN_PROBES:
        src_xy = _probe_xy(first["points"], indices, weights)
        dst_xy = _probe_xy(second["points"], indices, weights)
        out[name] = _point_error(homography, src_xy, dst_xy)
    for name, indices in RESIDUAL_PROBES:
        out[name] = _group_error(homography, first["points"], second["points"], indices)
    return out


def _percentile(values: list[float], q: float) -> float:
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if array.size == 0:
        return float("nan")
    return float(np.percentile(array, q))


def score_faces(faces: list[dict]) -> dict:
    step = max(1, len(faces) // 8)
    chosen = faces[::step][:10]
    buckets: dict[str, list[float]] = {}
    diag_buckets: dict[str, list[float]] = {}
    n_pairs = 0
    for i in range(len(chosen)):
        for j in range(i + 1, len(chosen)):
            first, second = chosen[i], chosen[j]
            scale = max(first["eye"], second["eye"]) / (min(first["eye"], second["eye"]) + 1e-6)
            delta = _pose_delta(first, second, "pnp")
            if delta < 3.0 and scale < 1.05:
                continue
            errs = _pair_errors(first, second)
            if errs is None or not np.isfinite(errs["nose_tip_err"]):
                continue
            n_pairs += 1
            if not np.isfinite(errs["hold"]):
                continue
            diag = 0.5 * (first["diag"] + second["diag"]) + 1e-6
            den = errs["hold"] + 0.008 * diag
            names = [name for name, _, _ in MEAN_PROBES] + [name for name, _ in RESIDUAL_PROBES]
            for name in names:
                err = errs[name]
                if not np.isfinite(err):
                    continue
                buckets.setdefault(name, []).append(err / den)
                if name in DIAG_NAMES:
                    diag_buckets.setdefault(name, []).append(err / diag)
    ref_spec, ref_pairs = pair_specificity(faces, "pnp")
    scores = {name: _median(values) for name, values in buckets.items()}
    for name, values in diag_buckets.items():
        scores[f"{name}_diag"] = _median(values)
        if name in ("lip_inner", "lip_four", "eye_upper"):
            scores[f"{name}_diag_p90"] = _percentile(values, 90)
    for name in P90_NAMES:
        scores[f"{name}_p90"] = _percentile(buckets.get(name, []), 90)
    if not _close(scores.get("nose_tip", float("nan")), ref_spec) or int(ref_pairs) != n_pairs:
        raise RuntimeError(
            f"База не совпала с pair_specificity: {scores.get('nose_tip')} vs {ref_spec}, пары {n_pairs} vs {ref_pairs}"
        )
    return {"n_faces": len(faces), "n_pairs": n_pairs, "scores": scores}


def _close(left: float, right: float, tol: float = 1e-5) -> bool:
    if not np.isfinite(left) and not np.isfinite(right):
        return True
    return bool(np.isfinite(left) and np.isfinite(right) and abs(left - right) <= tol)


def _finite(value) -> float:
    if value is None:
        return float("nan")
    number = float(value)
    return number if np.isfinite(number) else float("nan")


def _label(score: float, n_faces: int, n_pairs: int, threshold: float) -> str:
    if n_faces < 5 or n_pairs < 3 or not np.isfinite(score):
        return "attack"
    return "live" if score >= threshold else "attack"


def _predict_single(row: dict, name: str, threshold: float) -> str:
    return _label(_finite(row["scores"].get(name)), row["n_faces"], row["n_pairs"], threshold)


def _predict_and(row: dict, nose: str, lip: str, t_nose: float, t_lip: float) -> str:
    if row["n_faces"] < 5 or row["n_pairs"] < 3:
        return "attack"
    n_score = _finite(row["scores"].get(nose))
    l_score = _finite(row["scores"].get(lip))
    if not np.isfinite(n_score) or not np.isfinite(l_score):
        return "attack"
    return "live" if n_score >= t_nose and l_score >= t_lip else "attack"


def _predict_or(row: dict, nose: str, lip: str, t_high: float, t_mid: float, t_lip: float) -> str:
    if row["n_faces"] < 5 or row["n_pairs"] < 3:
        return "attack"
    n_score = _finite(row["scores"].get(nose))
    if not np.isfinite(n_score):
        return "attack"
    if n_score >= t_high:
        return "live"
    l_score = _finite(row["scores"].get(lip))
    if np.isfinite(l_score) and n_score >= t_mid and l_score >= t_lip:
        return "live"
    return "attack"


def _rates(rows: list[dict], preds: list[str]) -> dict:
    n_live = sum(row["true_label"] == "live" for row in rows)
    n_attack = len(rows) - n_live
    attack_as_live = sum(row["true_label"] == "attack" and pred == "live" for row, pred in zip(rows, preds))
    live_as_attack = sum(row["true_label"] == "live" and pred != "live" for row, pred in zip(rows, preds))
    by_type: dict[str, dict[str, int]] = {}
    errors = []
    for row, pred in zip(rows, preds):
        bucket = by_type.setdefault(row["attack_type"], {"n": 0, "pred_live": 0})
        bucket["n"] += 1
        if pred == "live":
            bucket["pred_live"] += 1
        if pred != row["true_label"]:
            errors.append(row["video_id"])
    return {
        "n_live": n_live,
        "n_attack": n_attack,
        "attack_as_live": attack_as_live,
        "live_as_attack": live_as_attack,
        "APCER": attack_as_live / n_attack if n_attack else None,
        "BPCER": live_as_attack / n_live if n_live else None,
        "total_errors": attack_as_live + live_as_attack,
        "printed_flat_as_live": by_type.get("printed_flat", {}).get("pred_live", 0),
        "by_type_pred_live": by_type,
        "errors": errors,
    }


def _gap(rows: list[dict], name: str) -> dict:
    live = [_finite(row["scores"].get(name)) for row in rows if row["true_label"] == "live"]
    attack = [_finite(row["scores"].get(name)) for row in rows if row["true_label"] == "attack"]
    live = [value for value in live if np.isfinite(value)]
    attack = [value for value in attack if np.isfinite(value)]
    if not live or not attack:
        return {"min_live": None, "max_attack": None, "gap": None}
    min_live = min(live)
    max_attack = max(attack)
    return {
        "min_live": min_live,
        "max_attack": max_attack,
        "gap": min_live - max_attack,
        "min_live_id": next(row["video_id"] for row in rows if row["true_label"] == "live" and _close(_finite(row["scores"].get(name)), min_live, 1e-9)),
        "max_attack_id": next(
            row["video_id"] for row in rows if row["true_label"] == "attack" and _close(_finite(row["scores"].get(name)), max_attack, 1e-9)
        ),
    }


def _distribution(rows: list[dict], name: str) -> dict[str, dict]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        value = _finite(row["scores"].get(name))
        if np.isfinite(value):
            grouped.setdefault(row["attack_type"], []).append(value)
    out = {}
    for attack_type, values in sorted(grouped.items()):
        array = np.asarray(values, dtype=np.float64)
        out[attack_type] = {
            "n": int(array.size),
            "median": float(np.median(array)),
            "min": float(array.min()),
            "max": float(array.max()),
        }
    return out


def _rule_record(rows: list[dict], name: str, preds: list[str], baseline_wrong: set[str], score_name: str | None) -> dict:
    stats = _rates(rows, preds)
    wrong = set(stats["errors"])
    record = {
        "name": name,
        "predeclared_threshold": THRESHOLD if "1.8" in name else None,
        **{key: stats[key] for key in ("n_live", "n_attack", "attack_as_live", "live_as_attack", "APCER", "BPCER", "total_errors", "printed_flat_as_live")},
        "by_type_pred_live": stats["by_type_pred_live"],
        "errors": stats["errors"],
        "new_errors": sorted(wrong - baseline_wrong),
        "fixed": sorted(baseline_wrong - wrong),
        "known": {},
    }
    for video_id in REPORT_IDS:
        row = next(item for item in rows if item["video_id"] == video_id)
        pred = preds[rows.index(row)]
        record["known"][video_id] = {
            "true": row["true_label"],
            "pred": pred,
            "correct": pred == row["true_label"],
            "score": None if score_name is None else row["scores"].get(score_name),
        }
    if score_name is not None:
        record["gap"] = _gap(rows, score_name)
        record["median_by_type"] = _distribution(rows, score_name)
    return record


def build_rows(videos: list[dict], per_video: list[list[dict]], reference: dict[str, float]) -> list[dict]:
    rows = []
    worst = 0.0
    worst_id = None
    for video, faces in zip(videos, per_video):
        scored = score_faces(faces)
        ref = reference.get(video["video_id"])
        got = scored["scores"].get("nose_tip")
        diff = None if ref is None or got is None or not np.isfinite(got) else abs(float(got) - float(ref))
        if diff is not None and diff > worst:
            worst = diff
            worst_id = video["video_id"]
        base_pred = _label(_finite(got), scored["n_faces"], scored["n_pairs"], THRESHOLD)
        rows.append(
            {
                "video_id": video["video_id"],
                "attack_type": video["attack_type"],
                "true_label": video["label"],
                "n_faces": scored["n_faces"],
                "n_pairs": scored["n_pairs"],
                "scores": scored["scores"],
                "hpe_compare_pnp": ref,
                "abs_diff_vs_hpe_compare": diff,
                "baseline_pred": base_pred,
            }
        )
        flag = "" if diff is None or diff < 0.02 else "  РАСХОЖДЕНИЕ"
        print(
            f"{video['video_id']} baseline={None if got is None or not np.isfinite(got) else round(float(got), 4)} "
            f"ref={None if ref is None else round(float(ref), 4)} pairs={scored['n_pairs']}{flag}",
            flush=True,
        )
    print(f"Макс |pnp specificity - hpe_compare|: {worst:.6f} на {worst_id}", flush=True)
    return rows


def _fmt_threshold(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def evaluate(rows: list[dict]) -> list[dict]:
    baseline_preds = [_predict_single(row, "nose_tip", THRESHOLD) for row in rows]
    baseline_wrong = {row["video_id"] for row, pred in zip(rows, baseline_preds) if pred != row["true_label"]}
    records = [_rule_record(rows, "single:nose_tip>=1.80", baseline_preds, baseline_wrong, "nose_tip")]
    seen = {"single:nose_tip>=1.80"}

    def add(name: str, preds: list[str], score_name: str | None) -> None:
        if name in seen:
            return
        seen.add(name)
        records.append(_rule_record(rows, name, preds, baseline_wrong, score_name))

    single_names = [name for name, _, _ in MEAN_PROBES] + [name for name, _ in RESIDUAL_PROBES]
    single_names += [f"{name}_p90" for name in P90_NAMES]
    single_names += [f"{name}_diag" for name in DIAG_NAMES]
    single_names += ["lip_inner_diag_p90", "lip_four_diag_p90", "eye_upper_diag_p90"]
    ratio_thresholds = (1.2, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0, 2.2, 2.4)
    diag_thresholds = (0.004, 0.008, 0.012, 0.016, 0.02, 0.03, 0.05)
    for name in single_names:
        thresholds = diag_thresholds if name.endswith("_diag") or name.endswith("_diag_p90") else ratio_thresholds
        for threshold in thresholds:
            preds = [_predict_single(row, name, threshold) for row in rows]
            add(f"single:{name}>={_fmt_threshold(threshold)}", preds, name)

    nose_thresholds = (1.5, 1.6, 1.7, 1.8, 2.0, 2.2)
    mid_thresholds = (1.0, 1.2, 1.4, 1.5)
    lip_thresholds = (1.1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2)
    for nose in NOSE_FOR_RULES:
        for lip in LIP_FOR_RULES:
            for t_nose in nose_thresholds:
                for t_lip in lip_thresholds:
                    preds = [_predict_and(row, nose, lip, t_nose, t_lip) for row in rows]
                    add(
                        f"and:{nose}>={_fmt_threshold(t_nose)}&{lip}>={_fmt_threshold(t_lip)}",
                        preds,
                        None,
                    )
            for t_mid in mid_thresholds:
                for t_lip in lip_thresholds:
                    preds = [_predict_or(row, nose, lip, THRESHOLD, t_mid, t_lip) for row in rows]
                    add(
                        f"or:{nose}>=1.80|({nose}>={_fmt_threshold(t_mid)}&{lip}>={_fmt_threshold(t_lip)})",
                        preds,
                        None,
                    )
    return records


def _print_rule(record: dict) -> None:
    known = " ".join(
        f"{video_id.split('/')[-1]}:{info['pred'][0]}" for video_id, info in record["known"].items()
    )
    print(
        f"{record['name']}: APCER={record['APCER']:.3f} ({record['attack_as_live']}/{record['n_attack']}) "
        f"BPCER={record['BPCER']:.3f} ({record['live_as_attack']}/{record['n_live']}) "
        f"err={record['total_errors']} printFA={record['printed_flat_as_live']} "
        f"known[{known}] fixed={record['fixed']} new={record['new_errors']}",
        flush=True,
    )


def _print_distributions(rows: list[dict]) -> None:
    names = [name for name, _, _ in MEAN_PROBES] + ["lip_outer", "lip_inner", "lip_four", "lip_lower", "eye_upper", "eye_four"]
    names += ["lip_inner_diag", "lip_four_diag", "eye_upper_diag", "lip_inner_p90", "eye_upper_p90"]
    for name in names:
        print(f"\n{name}  {_gap(rows, name)}", flush=True)
        for attack_type, stats in _distribution(rows, name).items():
            print(
                f"  {attack_type:22} n={stats['n']:2} med={stats['median']:.3f} "
                f"min={stats['min']:.3f} max={stats['max']:.3f}",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Варианты параллакса по дополнительным точкам MediaPipe")
    parser.add_argument("--data", type=Path, default=Path("android"))
    parser.add_argument("--out", type=Path, default=Path("python/pad_prototype/output"))
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    verify_topology()
    videos = discover_videos(args.data)
    if not videos:
        raise SystemExit(f"Нет клипов в {args.data}")
    started = time.time()
    per_video = ensure_cache(videos, args.out / "landmark_cache", args.recompute)
    geometry = verify_geometry(per_video)
    reference = {}
    compare_path = args.out / "hpe_compare.json"
    if compare_path.exists():
        payload = json.loads(compare_path.read_text(encoding="utf-8"))
        reference = {
            row["video_id"]: row["pnp_specificity"]
            for row in payload["videos"]
            if row.get("pnp_specificity") is not None
        }
    rows = build_rows(videos, per_video, reference)
    diffs = [row["abs_diff_vs_hpe_compare"] for row in rows if row["abs_diff_vs_hpe_compare"] is not None]
    _print_distributions(rows)
    records = evaluate(rows)
    baseline_errors = records[0]["total_errors"]
    print("\nОсновные правила при пороге 1.80 и правила лучше базы", flush=True)
    interesting = []
    for record in records:
        primary = record["name"] in {
            "single:nose_tip>=1.80",
            "single:nose_wings>=1.80",
            "single:nose_tip_wings>=1.80",
            "single:nose_tip_half_wings>=1.80",
            "single:nose_tip_wings_sub>=1.80",
            "single:nose_pronasale>=1.80",
            "single:nose_pronasale_wings>=1.80",
            "single:nose_tip_nostrils>=1.80",
            "single:lip_inner>=1.80",
            "single:lip_outer>=1.80",
            "single:lip_four>=1.80",
            "single:lip_lower>=1.80",
            "single:eye_upper>=1.80",
            "single:eye_four>=1.80",
        }
        better = record["total_errors"] < baseline_errors and record["printed_flat_as_live"] == 0
        if primary or better:
            interesting.append(record)
    interesting.sort(key=lambda record: (record["total_errors"], record["attack_as_live"], record["live_as_attack"], record["name"]))
    # Полный список «лучше базы» может быть длинным: печатаем все с err<=2 и первичные.
    shown = 0
    for record in interesting:
        if record["total_errors"] > baseline_errors and not record["name"].startswith("single:"):
            continue
        if record["total_errors"] > 2 and record["name"] not in {
            "single:nose_tip>=1.80",
            "single:nose_wings>=1.80",
            "single:nose_tip_wings>=1.80",
            "single:nose_tip_half_wings>=1.80",
            "single:nose_tip_wings_sub>=1.80",
            "single:nose_pronasale>=1.80",
            "single:nose_pronasale_wings>=1.80",
            "single:nose_tip_nostrils>=1.80",
            "single:lip_inner>=1.80",
            "single:lip_outer>=1.80",
            "single:lip_four>=1.80",
            "single:lip_lower>=1.80",
            "single:eye_upper>=1.80",
            "single:eye_four>=1.80",
        }:
            continue
        _print_rule(record)
        shown += 1
        if shown >= 60:
            break
    print(f"Правил всего {len(records)}, напечатано {shown}", flush=True)
    finite_diffs = np.asarray(diffs, dtype=np.float64)
    summary = {
        "threshold": THRESHOLD,
        "n_videos": len(rows),
        "baseline_max_abs_diff_vs_hpe_compare": None if finite_diffs.size == 0 else float(finite_diffs.max()),
        "baseline_median_abs_diff_vs_hpe_compare": None if finite_diffs.size == 0 else float(np.median(finite_diffs)),
        "n_rules": len(records),
        "n_rules_fewer_errors_print_clean": sum(
            record["total_errors"] < baseline_errors and record["printed_flat_as_live"] == 0 for record in records
        ),
        "seconds": time.time() - started,
        "geometry": geometry,
        "indices": {
            "tip_current": 1,
            "pronasale": 4,
            "alae": [48, 278],
            "nostrils": [98, 327],
            "subnasale": 2,
            "lip_outer_centers": [0, 17],
            "lip_inner_centers": [13, 14],
            "eyelid_upper": [159, 386],
            "eyelid_lower": [145, 374],
            "holdout": [117, 346, 10, 152, 6],
            "homography": [33, 263, 133, 362, 61, 291],
            "skipped": {"philtrum": 164, "ala_nostril_ambiguous": [64, 294]},
        },
    }
    # В JSON кладём распределения и правила с err <= базы + 1, плюс все single при 1.80.
    kept = []
    for record in records:
        if record["name"].endswith(">=1.80") or record["total_errors"] <= baseline_errors + 1:
            slim = dict(record)
            slim.pop("by_type_pred_live", None)
            kept.append(slim)
    payload = {"summary": summary, "rules": kept, "videos": rows}
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "extra_points.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(f"Отчёт: {path}", flush=True)
    print(f"Время: {time.time() - started:.0f} с", flush=True)


if __name__ == "__main__":
    main()
