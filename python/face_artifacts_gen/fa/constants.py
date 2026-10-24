"""Shared constants for face artifact generation."""

from __future__ import annotations

from typing import Dict, List, Tuple

ARTIFACTS = ('beard', 'glasses_glare', 'bangs', 'occluder')
METHODS = ('flux_fill', 'sdxl_cn', 'instruct_edit', 'stylegan', 'procedural')

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')

# Default occluded-policy per artifact (overridable via CLI).
DEFAULT_OCCLUDED_POLICY: Dict[str, str] = {
    'beard': 'keep',
    'glasses_glare': 'keep',
    'bangs': 'missing',
    'occluder': 'missing',
}

# Keypoints typically covered by each artifact (used for markup updates).
ARTIFACT_KP_GROUPS: Dict[str, Tuple[str, ...]] = {
    'beard': (
        'kp_under_lip_bottom',
        'kp_mouth_left',
        'kp_mouth_right',
        'kp_upper_lip_top',
        'kp_chin',
        'kp_cheek_left',
        'kp_cheek_right',
    ),
    'glasses_glare': (
        'kp_eye_left_outer',
        'kp_eye_left_inner',
        'kp_eye_right_outer',
        'kp_eye_right_inner',
        'kp_pupil_left',
        'kp_pupil_right',
        'kp_eyelid_left_upper_outer',
        'kp_eyelid_left_upper_inner',
        'kp_eyelid_right_upper_outer',
        'kp_eyelid_right_upper_inner',
        'kp_eyelid_left_lower_outer',
        'kp_eyelid_left_lower_inner',
        'kp_eyelid_right_lower_outer',
        'kp_eyelid_right_lower_inner',
        'kp_nose_bridge',
    ),
    'bangs': (
        'kp_forehead',
        'kp_eyebrow_left_outer',
        'kp_eyebrow_left_inner',
        'kp_eyebrow_left_center',
        'kp_eyebrow_right_outer',
        'kp_eyebrow_right_inner',
        'kp_eyebrow_right_center',
    ),
    'occluder': (),  # decided dynamically from change_mask
}

# Per-artifact inpaint / instruct prompts.
ARTIFACT_PROMPTS: Dict[str, Dict[str, str]] = {
    'beard': {
        'prompt': (
            'natural full beard and mustache on a real person, realistic facial hair, '
            'same lighting, photorealistic, keep identity'
        ),
        'negative': (
            'cartoon, plastic, bald chin, clean shaven, deformed face, blurry, '
            'wrong identity, painted texture'
        ),
        'instruct': 'add a natural full beard and mustache, keep the same person and lighting',
    },
    'glasses_glare': {
        'prompt': (
            'person wearing eyeglasses with thick frames, strong specular glare and '
            'reflections on the lenses, photorealistic, same person'
        ),
        'negative': (
            'no glasses, clear lenses without glare, cartoon, deformed eyes, blurry, '
            'sunglasses completely opaque'
        ),
        'instruct': (
            'add thick-rimmed eyeglasses with strong glare and reflections on the lenses, '
            'keep the same person'
        ),
    },
    'bangs': {
        'prompt': (
            'natural hair bangs covering the forehead, realistic hair strands, '
            'same person, photorealistic'
        ),
        'negative': (
            'bald forehead, no bangs, cartoon, wig, deformed face, blurry'
        ),
        'instruct': 'add natural hair bangs covering the forehead, keep the same person',
    },
    'occluder': {
        'prompt': (
            'natural hand or everyday object partially occluding the face, '
            'realistic shadows, photorealistic, same person'
        ),
        'negative': (
            'fully covered face, cartoon, deformed face, blurry, plastic'
        ),
        'instruct': (
            'add a natural hand or everyday object partially occluding the face, '
            'keep the same person'
        ),
    },
}

# InsightFace 2d106 indices useful for region masks (approximate).
# Contour / jawline roughly indices 0..32 in 2d106.
INSIGHTFACE_JAW_IDX: List[int] = list(range(0, 33))
INSIGHTFACE_LEFT_EYEBROW = list(range(33, 42))
INSIGHTFACE_RIGHT_EYEBROW = list(range(42, 51))
INSIGHTFACE_NOSE = list(range(51, 66))
INSIGHTFACE_LEFT_EYE = list(range(66, 75))
INSIGHTFACE_RIGHT_EYE = list(range(75, 84))
INSIGHTFACE_MOUTH = list(range(84, 104))

CLIP_ARTIFACT_TEXTS: Dict[str, str] = {
    'beard': 'a person with a beard',
    'glasses_glare': 'a person wearing eyeglasses with lens glare',
    'bangs': 'a person with hair bangs on the forehead',
    'occluder': 'a face partially occluded by a hand or object',
}
