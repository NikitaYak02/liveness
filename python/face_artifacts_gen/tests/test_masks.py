from __future__ import annotations

import numpy as np

from fa.masks import (
    build_artifact_mask,
    build_bangs_mask,
    build_beard_mask,
    build_glasses_mask,
    build_occluder_mask,
)


def test_beard_mask_nonzero(face_img_and_landmarks):
    img, face = face_img_and_landmarks
    rng = np.random.default_rng(0)
    mask = build_beard_mask(img.shape[:2], face, rng)
    assert mask.shape == img.shape[:2]
    assert mask.dtype == np.uint8
    assert int((mask > 16).sum()) > 500


def test_bangs_mask_covers_forehead(face_img_and_landmarks):
    img, face = face_img_and_landmarks
    rng = np.random.default_rng(1)
    mask = build_bangs_mask(img.shape[:2], face, rng)
    fx, fy = face.points['kp_forehead']
    assert mask[int(fy) + 10, int(fx)] > 16


def test_glasses_mask_covers_pupils(face_img_and_landmarks):
    img, face = face_img_and_landmarks
    rng = np.random.default_rng(2)
    mask = build_glasses_mask(img.shape[:2], face, rng)
    for name in ('kp_pupil_left', 'kp_pupil_right'):
        x, y = face.points[name]
        assert mask[int(y), int(x)] > 16


def test_occluder_mask_inside_face(face_img_and_landmarks):
    img, face = face_img_and_landmarks
    rng = np.random.default_rng(3)
    mask = build_occluder_mask(img.shape[:2], face, rng)
    assert int((mask > 16).sum()) > 200


def test_build_artifact_mask_dispatch(face_img_and_landmarks):
    img, face = face_img_and_landmarks
    rng = np.random.default_rng(4)
    for art in ('beard', 'bangs', 'glasses_glare', 'occluder'):
        m = build_artifact_mask(art, img.shape[:2], face, rng)
        assert m.shape == img.shape[:2]
        assert m.max() > 0
