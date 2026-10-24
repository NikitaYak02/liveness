from __future__ import annotations

import numpy as np

from fa.landmarks import FaceLandmarks, landmarks_from_markup
from fa.qa import build_contact_sheet, landmark_nme


def test_landmarks_from_old_markup(face_img_and_landmarks):
    _, face = face_img_and_landmarks
    props = {name: [[float(x), float(y)]] for name, (x, y) in face.points.items()}
    props['face_rect'] = [[60.0, 80.0, 240.0, 320.0]]
    faces = landmarks_from_markup({'props': props})
    assert len(faces) == 1
    assert faces[0].get('kp_nose_tip') is not None
    assert faces[0].rect is not None


def test_landmark_nme_identical():
    pts = {'kp_pupil_left': (10.0, 10.0), 'kp_pupil_right': (30.0, 10.0), 'kp_nose_tip': (20.0, 25.0)}
    a = FaceLandmarks(points=pts)
    b = FaceLandmarks(points=dict(pts))
    assert landmark_nme(a, b) == 0.0


def test_landmark_nme_shifted():
    a = FaceLandmarks(
        points={'kp_pupil_left': (0.0, 0.0), 'kp_pupil_right': (100.0, 0.0), 'kp_nose_tip': (50.0, 50.0)}
    )
    b = FaceLandmarks(
        points={'kp_pupil_left': (0.0, 0.0), 'kp_pupil_right': (100.0, 0.0), 'kp_nose_tip': (50.0, 60.0)}
    )
    nme = landmark_nme(a, b)
    assert nme is not None
    assert 0.03 < nme < 0.05  # 10px / 100 interocular


def test_contact_sheet(face_img_and_landmarks):
    img, _ = face_img_and_landmarks
    sheet = build_contact_sheet([{'original': img, 'procedural': img}], cell_size=(64, 64))
    assert sheet.shape == (64, 128, 3)
