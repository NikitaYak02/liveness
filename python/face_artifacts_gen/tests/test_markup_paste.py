from __future__ import annotations

import numpy as np

from fa.io_utils import diff_mask, paste_back
from fa.markup import update_markup
from fa.masks import build_beard_mask, build_glasses_mask


def test_paste_back_preserves_outside(face_img_and_landmarks):
    img, face = face_img_and_landmarks
    # Hard binary mask (no soft edges) so outside pixels stay bit-exact.
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[180:260, 100:260] = 255
    edited = img.copy()
    edited[:] = (0, 0, 255)  # paint whole image red
    out = paste_back(img, edited, mask, feather=1)
    outside = mask == 0
    assert np.array_equal(out[outside], img[outside])
    inside = mask == 255
    assert inside.any()
    assert out[inside][:, 2].mean() > img[inside][:, 2].mean()


def test_diff_mask_detects_change(face_img_and_landmarks):
    img, _ = face_img_and_landmarks
    edited = img.copy()
    edited[100:150, 100:150] = 0
    m = diff_mask(img, edited, threshold=10)
    assert m[120, 120] == 255
    assert m[10, 10] == 0


def test_update_markup_keep_policy(face_img_and_landmarks):
    img, face = face_img_and_landmarks
    rng = np.random.default_rng(0)
    mask = build_glasses_mask(img.shape[:2], face, rng)
    props = {name: [[float(x), float(y)]] for name, (x, y) in face.points.items()}
    data = {'props': props}
    updated = update_markup(data, mask, 'glasses_glare', policy='keep')
    assert updated is not None
    occluded = updated['props'].get('occluded_keypoints') or []
    assert 'pupil_left' in occluded or 'eye_left_outer' in occluded
    # Coordinates kept.
    assert updated['props']['kp_pupil_left'][0][0] > 0


def test_update_markup_missing_policy(face_img_and_landmarks):
    img, face = face_img_and_landmarks
    rng = np.random.default_rng(1)
    mask = build_beard_mask(img.shape[:2], face, rng)
    props = {name: [[float(x), float(y)]] for name, (x, y) in face.points.items()}
    data = {'props': props}
    updated = update_markup(data, mask, 'beard', policy='missing')
    # Chin should be under beard mask → missing.
    chin = updated['props']['kp_chin'][0]
    assert chin[0] <= -500
