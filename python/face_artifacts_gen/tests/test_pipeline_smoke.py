from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from fa.pipeline import process_folder


def test_pipeline_procedural_smoke(tmp_dataset, tmp_path):
    input_root, img, face = tmp_dataset
    # Add a second image.
    images = input_root / 'images'
    markup = input_root / 'markup'
    cv2.imwrite(str(images / 'face02.jpg'), img)
    props = {name: [[float(x), float(y)]] for name, (x, y) in face.points.items()}
    props['face_rect'] = [[60.0, 80.0, 240.0, 320.0]]
    (markup / 'face02.jpg.json').write_text(json.dumps({'props': props}), encoding='utf-8')

    out = tmp_path / 'out'
    counts = process_folder(
        input_root=input_root,
        output_root=out,
        method='procedural',
        artifact='glasses_glare',
        device='cpu',
        limit=10,
        seed=0,
        skip_qa=True,
        save_masks=True,
    )
    assert counts['total'] == 2
    assert counts['ok'] == 2
    manifest = out / 'procedural' / 'glasses_glare' / 'manifest.jsonl'
    assert manifest.exists()
    lines = manifest.read_text(encoding='utf-8').strip().splitlines()
    assert len(lines) == 2
    # Output images exist.
    outs = list((out / 'procedural' / 'glasses_glare').rglob('*.jpg'))
    assert len(outs) >= 2


def test_pipeline_stylegan_and_instruct_fallback(tmp_dataset, tmp_path):
    input_root, _, _ = tmp_dataset
    out = tmp_path / 'out2'
    for method, artifact in (
        ('stylegan', 'beard'),
        ('instruct_edit', 'bangs'),
    ):
        kw = {}
        if method == 'instruct_edit':
            kw['backend'] = 'procedural_fallback'
        counts = process_folder(
            input_root=input_root,
            output_root=out,
            method=method,
            artifact=artifact,
            device='cpu',
            limit=1,
            seed=1,
            skip_qa=True,
            generator_kwargs=kw,
        )
        assert counts['ok'] >= 1
