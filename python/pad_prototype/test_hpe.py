"""Формула facerecog2 на тех же синтетических точках, что test_facerecog2_hpe.cpp."""

from __future__ import annotations

import math
import unittest

import cv2
import numpy as np

from pad_prototype.hpe import (
    COEFF_SCALE,
    MODEL_3D,
    facerecog2_ypr_from_eyes,
    prototype_pose,
)


def _fr(right, left, nose):
    return facerecog2_ypr_from_eyes(nose, left, left, right, right)


class Facerecog2Cases(unittest.TestCase):
    def test_straight_is_zero(self):
        yaw, pitch, roll = _fr((100, 150), (200, 150), (150, 200))
        self.assertAlmostEqual(yaw, 0.0, places=5)
        self.assertAlmostEqual(pitch, 0.0, places=5)
        self.assertAlmostEqual(roll, 0.0, places=5)

    def test_left_turn_is_coeff_times_fifth_of_ipd(self):
        yaw, pitch, roll = _fr((120, 150), (220, 150), (190, 200))
        self.assertAlmostEqual(yaw, 0.2 * COEFF_SCALE, places=5)
        self.assertAlmostEqual(pitch, 0.0, places=5)
        self.assertAlmostEqual(roll, 0.0, places=5)
        self.assertAlmostEqual(yaw, 16.0, places=5)

    def test_right_turn_is_negative(self):
        yaw, pitch, roll = _fr((80, 150), (180, 150), (110, 200))
        self.assertAlmostEqual(yaw, -16.0, places=5)
        self.assertAlmostEqual(pitch, 0.0, places=5)

    def test_upward_matches_cpp_expectation(self):
        yaw, pitch, roll = facerecog2_ypr_from_eyes(
            (451, 672), (591, 676), (514, 667), (310, 685), (387, 672)
        )
        self.assertAlmostEqual(yaw, 0.0, delta=10)
        self.assertAlmostEqual(pitch, 41.0, delta=10)
        self.assertAlmostEqual(roll, 0.0, delta=10)

    def test_downward_matches_cpp_expectation(self):
        yaw, pitch, roll = facerecog2_ypr_from_eyes(
            (421, 1244), (593, 1050), (490, 1063), (249, 1050), (352, 1063)
        )
        self.assertAlmostEqual(yaw, 0.0, delta=10)
        self.assertAlmostEqual(pitch, -22.0, delta=10)
        self.assertAlmostEqual(roll, 0.0, delta=10)

    def test_rolled_matches_cpp_expectation(self):
        yaw, pitch, roll = facerecog2_ypr_from_eyes(
            (286, 644), (450, 634), (389, 609), (258, 552), (308, 573)
        )
        self.assertAlmostEqual(yaw, -20.0, delta=3)
        self.assertAlmostEqual(pitch, 0.0, delta=3)
        self.assertAlmostEqual(roll, 23.0, delta=3)


def _project(rotation, translation, points, size=640):
    camera = np.array(
        [[size, 0, size / 2], [0, size, size / 2], [0, 0, 1]], dtype=np.float64
    )
    changed = rotation @ points.T + translation.reshape(3, 1)
    projected = camera @ changed
    return (projected[:2] / projected[2]).T


def _mesh(rotation):
    """Шесть точек модели, разложенные в полный массив индексов MediaPipe."""
    rx = cv2.Rodrigues(np.array([math.pi, 0.0, 0.0]))[0]
    image = _project(rx @ rotation, np.array([0.0, 0.0, 900.0]), MODEL_3D)
    points = np.zeros((478, 2), dtype=np.float64)
    # Порядок MODEL_3D: нос, подбородок, image-left eye, image-right eye, рты.
    # У фронтального лица image-left — правый глаз человека (индекс 33).
    points[1] = image[0]
    points[152] = image[1]
    points[33] = image[2]
    points[263] = image[3]
    points[61] = image[4]
    points[291] = image[5]
    # Центр глаза совпадает с внешним углом: внутренних углов в модели нет.
    points[133] = image[2]
    points[362] = image[3]
    return points


def _rot_y(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rot_x(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


class PrototypeAxes(unittest.TestCase):
    def test_frontal_is_near_zero_and_yaw_sign_matches_facerecog2(self):
        from pad_prototype.hpe import facerecog2_ypr

        straight = _mesh(np.eye(3))
        yaw, pitch, roll = prototype_pose(straight, (640, 640))[1]
        self.assertAlmostEqual(yaw, 0.0, delta=1.5)
        self.assertAlmostEqual(pitch, 0.0, delta=1.5)
        self.assertAlmostEqual(roll, 0.0, delta=1.5)

        turned = _mesh(_rot_y(20))
        pnp = prototype_pose(turned, (640, 640))[1]
        fr = facerecog2_ypr(turned)
        self.assertGreater(pnp[0], 15)
        self.assertGreater(fr[0], 0)
        self.assertAlmostEqual(math.copysign(1, pnp[0]), math.copysign(1, fr[0]))

    def test_pitch_up_is_positive_like_facerecog2(self):
        from pad_prototype.hpe import facerecog2_ypr

        # Отрицательный поворот вокруг X модели поднимает нос в кадре.
        up = _mesh(_rot_x(-15))
        down = _mesh(_rot_x(15))
        self.assertGreater(prototype_pose(up, (640, 640))[1][1], 8)
        self.assertLess(prototype_pose(down, (640, 640))[1][1], -8)
        self.assertGreater(facerecog2_ypr(up)[1], facerecog2_ypr(down)[1])


if __name__ == "__main__":
    unittest.main()
