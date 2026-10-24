from __future__ import annotations

import numpy as np

from fa.masks import build_artifact_mask
from fa.methods.instruct_edit import InstructEditGenerator
from fa.methods.procedural import ProceduralGenerator
from fa.methods.stylegan import StyleGANGenerator, align_ffhq, _five_points


def _run(gen, img, face, artifact, seed=0):
    rng = np.random.default_rng(seed)
    mask = build_artifact_mask(artifact, img.shape[:2], face, rng)
    return gen.generate(img, face, artifact, mask, rng, seed=seed)


def test_procedural_all_artifacts(face_img_and_landmarks):
    img, face = face_img_and_landmarks
    gen = ProceduralGenerator(device='cpu')
    for art in ('beard', 'glasses_glare', 'bangs', 'occluder'):
        result = _run(gen, img, face, art, seed=10)
        assert result.image_bgr.shape == img.shape
        assert result.change_mask.shape == img.shape[:2]
        # Something should change.
        assert not np.array_equal(result.image_bgr, img)


def test_stylegan_fallback_aligned(face_img_and_landmarks):
    img, face = face_img_and_landmarks
    pts = _five_points(face)
    assert pts is not None
    aligned, M = align_ffhq(img, pts, out_size=256)
    assert aligned.shape == (256, 256, 3)
    assert M.shape == (2, 3)

    gen = StyleGANGenerator(device='cpu', out_size=256)
    result = _run(gen, img, face, 'beard', seed=3)
    assert result.image_bgr.shape == img.shape
    assert result.extra.get('aligned') is True


def test_instruct_procedural_fallback(face_img_and_landmarks):
    img, face = face_img_and_landmarks
    gen = InstructEditGenerator(device='cpu', backend='procedural_fallback')
    result = _run(gen, img, face, 'glasses_glare', seed=5)
    assert result.image_bgr.shape == img.shape
    assert result.extra.get('backend') == 'procedural_fallback'
