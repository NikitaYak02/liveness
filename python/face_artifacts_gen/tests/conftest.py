"""Shared fixtures: synthetic face image + landmarks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fa.landmarks import FaceLandmarks


@pytest.fixture
def face_img_and_landmarks():
    """Draw a crude face and return (img_bgr, FaceLandmarks)."""
    h, w = 480, 360
    img = np.full((h, w, 3), 200, dtype=np.uint8)
    # Face oval.
    cv2.ellipse(img, (180, 240), (120, 160), 0, 0, 360, (210, 190, 170), -1)
    # Eyes.
    cv2.ellipse(img, (140, 200), (22, 12), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (220, 200), (22, 12), 0, 0, 360, (255, 255, 255), -1)
    cv2.circle(img, (140, 200), 5, (40, 40, 40), -1)
    cv2.circle(img, (220, 200), 5, (40, 40, 40), -1)
    # Brows / nose / mouth / chin.
    cv2.line(img, (115, 175), (160, 170), (80, 60, 40), 3)
    cv2.line(img, (200, 170), (245, 175), (80, 60, 40), 3)
    cv2.line(img, (180, 200), (180, 250), (150, 120, 100), 2)
    cv2.ellipse(img, (180, 290), (35, 15), 0, 0, 360, (80, 60, 80), 2)
    cv2.circle(img, (180, 380), 3, (0, 0, 255), -1)

    pts = {
        'kp_eye_left_outer': (118.0, 200.0),
        'kp_eye_left_inner': (162.0, 200.0),
        'kp_eye_right_inner': (198.0, 200.0),
        'kp_eye_right_outer': (242.0, 200.0),
        'kp_pupil_left': (140.0, 200.0),
        'kp_pupil_right': (220.0, 200.0),
        'kp_eyebrow_left_outer': (115.0, 175.0),
        'kp_eyebrow_left_inner': (160.0, 170.0),
        'kp_eyebrow_left_center': (137.0, 172.0),
        'kp_eyebrow_right_inner': (200.0, 170.0),
        'kp_eyebrow_right_outer': (245.0, 175.0),
        'kp_eyebrow_right_center': (222.0, 172.0),
        'kp_nose_tip': (180.0, 250.0),
        'kp_nose_bridge': (180.0, 210.0),
        'kp_upper_lip_top': (180.0, 278.0),
        'kp_under_lip_bottom': (180.0, 302.0),
        'kp_mouth_left': (150.0, 290.0),
        'kp_mouth_right': (210.0, 290.0),
        'kp_chin': (180.0, 380.0),
        'kp_forehead': (180.0, 120.0),
        'kp_cheek_left': (90.0, 260.0),
        'kp_cheek_right': (270.0, 260.0),
    }
    face = FaceLandmarks(
        points=pts,
        rect=(60.0, 80.0, 240.0, 320.0),
        source='synthetic',
    )
    return img, face


@pytest.fixture
def tmp_dataset(tmp_path, face_img_and_landmarks):
    img, face = face_img_and_landmarks
    images = tmp_path / 'images'
    markup = tmp_path / 'markup'
    images.mkdir()
    markup.mkdir()
    img_path = images / 'face01.jpg'
    cv2.imwrite(str(img_path), img)
    props = {name: [[float(x), float(y)]] for name, (x, y) in face.points.items()}
    props['face_rect'] = [[60.0, 80.0, 240.0, 320.0]]
    mrk_path = markup / 'face01.jpg.json'
    mrk_path.write_text(json.dumps({'props': props}), encoding='utf-8')
    return tmp_path, img, face
