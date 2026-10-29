"""Internal face-keypoint scheme used by the NNT HRNet pipeline."""

from __future__ import annotations

import numpy as np

# Order matches face_kp_train_tools / HRNet_src.json TRAIN_KP_NAMES.
KP_NAMES = [
    "eye_left_outer",
    "eye_right_inner",
    "eye_left_inner",
    "eye_right_outer",
    "upper_lip_top",
    "under_lip_bottom",
    "mouth_right",
    "mouth_left",
    "nose_tip",
    "pupil_left",
    "pupil_right",
    "eyebrow_left_outer",
    "eyebrow_right_outer",
    "eyebrow_left_inner",
    "eyebrow_right_inner",
    "eyebrow_left_center",
    "eyebrow_right_center",
    "eyelid_left_upper_outer",
    "eyelid_right_upper_outer",
    "eyelid_left_upper_inner",
    "eyelid_right_upper_inner",
    "eyelid_left_lower_outer",
    "eyelid_right_lower_outer",
    "eyelid_left_lower_inner",
    "eyelid_right_lower_inner",
    "cheek_right",
    "cheek_left",
    "forehead",
    "chin",
    "nose_bridge",
]

NUM_LANDMARKS = len(KP_NAMES)  # 30
# Compact "base 11" used for detector augmentation (eyes / mouth / nose / pupils).
BASE_11_NAMES = KP_NAMES[:11]
NUM_BASE_11 = len(BASE_11_NAMES)
HEATMAP_CHANNELS = NUM_LANDMARKS + 1  # channel 0 unused, as in NNT struct_kp
CHANNEL_OFFSET = 1
IMAGE_SIZE = 128

# Skeleton edges from HRNet_src.json err_e (indices into KP_NAMES).
SKELETON_EDGES = [
    (0, 9), (9, 2),
    (3, 10), (10, 1),
    (7, 4), (4, 6), (6, 5), (5, 7),
    (11, 15), (15, 13),
    (12, 16), (16, 14),
    (0, 17), (17, 19), (19, 2),
    (0, 21), (21, 23), (23, 2),
    (3, 18), (18, 20), (20, 1),
    (3, 22), (22, 24), (24, 1),
    (25, 6), (26, 7),
    (27, 29), (29, 8),
    (28, 5),
]

# Horizontal flip permutation of KP_NAMES indices.
_FLIP_PAIRS = [
    (0, 3), (1, 2), (6, 7), (9, 10),
    (11, 12), (13, 14), (15, 16),
    (17, 18), (19, 20), (21, 22), (23, 24),
    (25, 26),
]
FLIP_INDICES = np.arange(NUM_LANDMARKS, dtype=np.int64)
for left, right in _FLIP_PAIRS:
    FLIP_INDICES[left], FLIP_INDICES[right] = right, left

# Interocular norm for NME: outer eye corners.
NORM_LEFT = KP_NAMES.index("eye_left_outer")
NORM_RIGHT = KP_NAMES.index("eye_right_outer")
