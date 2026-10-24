"""Два оценки поворота головы по одним и тем же 2D-точкам.

facerecog2 — линейное смещение кончика носа в системе линии глаз.
Формула и знаки как в HpeEvaluatorCommon::Process
(optimize_clear_nnt/facerecog2/src/hpe_engine/hpe_evaluator_common.cpp).
Коэффициенты baseline_vertical=0.5 и coeff_scale=80 — те, на которых
собран тест прямого лица (yaw=pitch=0) и поворот влево на 0.2 IPD (yaw=+16).
В git этих чисел нет: на main они читаются из бандла hpe_evaluator.json,
в снимке part_opt зашиты этими константами.

prototype — solvePnP шеститочечной модели, которая уже стоит воротами
движения в run.py. Из матрицы берутся рыскание, тангаж и крен осей головы.
Положительный yaw — поворот влево (нос к левому глазу человека),
положительный pitch — взгляд вверх, положительный roll — левый глаз
человека ниже правого. Так же, как у facerecog2.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

# Стороны человека, индексы MediaPipe Face Mesh.
NOSE = 1
CHIN = 152
MOUTH_RIGHT, MOUTH_LEFT = 61, 291
EYE_OUTER_RIGHT, EYE_INNER_RIGHT = 33, 133
EYE_OUTER_LEFT, EYE_INNER_LEFT = 263, 362

BASELINE_VERTICAL = 0.5
COEFF_SCALE = 80.0

# Нос, подбородок, глаз с меньшим x, глаз с большим x, уголки рта в том же порядке.
# Отрицательный X модели — левая сторона кадра.
MODEL_3D = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, -330.0, -65.0],
        [-225.0, 170.0, -135.0],
        [225.0, 170.0, -135.0],
        [-150.0, -150.0, -125.0],
        [150.0, -150.0, -125.0],
    ],
    dtype=np.float64,
)


def _average(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (np.asarray(a, dtype=np.float64) + np.asarray(b, dtype=np.float64)) / 2.0


def facerecog2_ypr_from_eyes(
    nose: np.ndarray,
    left_outer: np.ndarray,
    left_inner: np.ndarray,
    right_outer: np.ndarray,
    right_inner: np.ndarray,
) -> tuple[float, float, float] | None:
    """Yaw, pitch, roll в градусах. left/right — стороны человека."""
    left = _average(left_outer, left_inner)
    right = _average(right_outer, right_inner)
    eye_vec = left - right
    nose_vec = np.asarray(nose, dtype=np.float64) - (left + right) / 2.0
    ipd = float(np.linalg.norm(eye_vec))
    if ipd < 1.0 or not np.isfinite(ipd):
        return None
    roll_rad = math.atan2(float(eye_vec[1]), float(eye_vec[0]))
    cr, sr = math.cos(roll_rad), math.sin(roll_rad)
    yaw_proj = (float(nose_vec[0]) * cr + float(nose_vec[1]) * sr) / ipd
    pitch_proj = (-float(nose_vec[0]) * sr + float(nose_vec[1]) * cr) / ipd
    yaw = float(np.clip(yaw_proj * COEFF_SCALE, -180.0, 180.0))
    pitch = float(np.clip(-(pitch_proj - BASELINE_VERTICAL) * COEFF_SCALE, -90.0, 90.0))
    roll = math.fmod(math.degrees(roll_rad) + 180.0, 360.0) - 180.0
    return yaw, pitch, roll


def facerecog2_ypr(points: np.ndarray) -> tuple[float, float, float] | None:
    return facerecog2_ypr_from_eyes(
        points[NOSE],
        points[EYE_OUTER_LEFT],
        points[EYE_INNER_LEFT],
        points[EYE_OUTER_RIGHT],
        points[EYE_INNER_RIGHT],
    )


def _by_x(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (a, b) if float(a[0]) <= float(b[0]) else (b, a)


def ypr_from_rotation(rotation: np.ndarray) -> tuple[float, float, float]:
    """Оси модели: +X вправо кадра, +Y вверх, +Z к камере (нос впереди глаз)."""
    forward = rotation @ np.array([0.0, 0.0, 1.0])
    image_right = rotation @ np.array([1.0, 0.0, 0.0])
    yaw = math.degrees(math.atan2(float(forward[0]), -float(forward[2])))
    pitch = math.degrees(
        math.atan2(-float(forward[1]), math.hypot(float(forward[0]), float(forward[2])))
    )
    roll = math.degrees(math.atan2(float(image_right[1]), float(image_right[0])))
    return float(yaw), float(pitch), float(roll)


def prototype_pose(
    points: np.ndarray, shape: tuple[int, int]
) -> tuple[np.ndarray, tuple[float, float, float]] | None:
    """Матрица поворота и (yaw, pitch, roll). shape — (height, width) кадра."""
    height, width = shape
    eye_l, eye_r = _by_x(points[EYE_OUTER_RIGHT], points[EYE_OUTER_LEFT])
    mouth_l, mouth_r = _by_x(points[MOUTH_RIGHT], points[MOUTH_LEFT])
    image_points = np.vstack(
        [points[NOSE], points[CHIN], eye_l, eye_r, mouth_l, mouth_r]
    ).astype(np.float64)
    focal = float(max(width, height))
    camera = np.array(
        [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float64
    )
    ok, rvec, _tvec = cv2.solvePnP(
        MODEL_3D, image_points, camera, None, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    return rotation, ypr_from_rotation(rotation)


def prototype_ypr(
    points: np.ndarray, shape: tuple[int, int]
) -> tuple[float, float, float] | None:
    pose = prototype_pose(points, shape)
    if pose is None:
        return None
    return pose[1]


def angle_diff_deg(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0
