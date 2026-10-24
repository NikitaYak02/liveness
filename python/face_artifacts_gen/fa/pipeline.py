"""End-to-end processing of an image folder."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .constants import ARTIFACTS, DEFAULT_OCCLUDED_POLICY, METHODS
from .io_utils import (
    ManifestRow,
    ManifestWriter,
    load_markup,
    relative_output_paths,
    resolve_pairs,
    write_image,
    write_json,
)
from .landmarks import InsightFaceLandmarker, resolve_face_landmarks
from .markup import update_markup
from .masks import build_artifact_mask
from .methods import create_generator
from .qa import ClipArtifactScorer, check_landmark_drift, write_contact_sheet_html


def process_folder(
    input_root: Path,
    output_root: Path,
    method: str,
    artifact: str,
    device: str = 'cuda',
    limit: int = 0,
    seed: Optional[int] = None,
    occluded_policy: Optional[str] = None,
    max_nme: float = 0.08,
    skip_qa: bool = False,
    use_clip: bool = False,
    save_masks: bool = True,
    require_markup: bool = False,
    generator_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, int]:
    if method not in METHODS:
        raise ValueError(f'unknown method {method!r}; choose from {METHODS}')
    if artifact not in ARTIFACTS:
        raise ValueError(f'unknown artifact {artifact!r}; choose from {ARTIFACTS}')

    input_root = Path(input_root)
    output_root = Path(output_root)
    out_dir = output_root / method / artifact
    out_dir.mkdir(parents=True, exist_ok=True)

    policy = occluded_policy or DEFAULT_OCCLUDED_POLICY.get(artifact, 'keep')
    pairs = resolve_pairs(input_root, require_markup=require_markup)
    if limit and limit > 0:
        pairs = pairs[:limit]
    if not pairs:
        logging.warning('No images found under %s', input_root)
        return {'total': 0}

    rng_master = np.random.default_rng(seed)
    # Landmarker is constructed lazily-safe (import happens on first detect).
    ctx = 0 if device.startswith('cuda') else -1
    landmarker = InsightFaceLandmarker(ctx_id=ctx)

    clip = ClipArtifactScorer(device=device) if use_clip else None
    gen = create_generator(method, device=device, **(generator_kwargs or {}))

    counts: Dict[str, int] = {
        'total': 0,
        'ok': 0,
        'rejected_nme': 0,
        'no_face': 0,
        'read_error': 0,
        'gen_error': 0,
    }
    contact_entries: List[Dict[str, str]] = []
    manifest_path = out_dir / 'manifest.jsonl'

    with ManifestWriter(manifest_path) as manifest:
        for idx, (img_path, markup_path) in enumerate(pairs):
            counts['total'] += 1
            t0 = time.time()
            img_seed = int(rng_master.integers(0, 2**31 - 1)) if seed is None else int(seed) + idx
            rng = np.random.default_rng(img_seed)

            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                logging.warning('read_error: %s', img_path)
                counts['read_error'] += 1
                manifest.write(
                    ManifestRow(
                        method=method,
                        artifact=artifact,
                        img_in=img_path,
                        img_out='',
                        markup_in=markup_path,
                        markup_out=None,
                        seed=img_seed,
                        prompt=None,
                        mask_path=None,
                        nme=None,
                        clip_score=None,
                        status='read_error',
                        elapsed_s=time.time() - t0,
                    )
                )
                continue

            markup_data = load_markup(markup_path)
            try:
                face = resolve_face_landmarks(img, markup_data, landmarker=landmarker)
            except Exception as exc:
                logging.warning('landmark failure %s: %s', img_path, exc)
                face = None

            if face is None:
                counts['no_face'] += 1
                manifest.write(
                    ManifestRow(
                        method=method,
                        artifact=artifact,
                        img_in=img_path,
                        img_out='',
                        markup_in=markup_path,
                        markup_out=None,
                        seed=img_seed,
                        prompt=None,
                        mask_path=None,
                        nme=None,
                        clip_score=None,
                        status='no_face',
                        elapsed_s=time.time() - t0,
                    )
                )
                continue

            mask = build_artifact_mask(artifact, img.shape[:2], face, rng)
            if int((mask > 16).sum()) < 20:
                logging.warning('empty mask for %s; skipping', img_path)
                counts['no_face'] += 1
                continue

            try:
                result = gen.generate(img, face, artifact, mask, rng, seed=img_seed)
            except Exception as exc:
                logging.exception('generation failed for %s: %s', img_path, exc)
                counts['gen_error'] += 1
                manifest.write(
                    ManifestRow(
                        method=method,
                        artifact=artifact,
                        img_in=img_path,
                        img_out='',
                        markup_in=markup_path,
                        markup_out=None,
                        seed=img_seed,
                        prompt=None,
                        mask_path=None,
                        nme=None,
                        clip_score=None,
                        status=f'gen_error:{exc}',
                        elapsed_s=time.time() - t0,
                    )
                )
                continue

            nme = None
            status = 'ok'
            if not skip_qa:
                accepted, nme = check_landmark_drift(
                    img, result.image_bgr, face, landmarker, max_nme=max_nme
                )
                if not accepted:
                    status = 'rejected_nme'
                    counts['rejected_nme'] += 1

            clip_score = None
            if clip is not None and status == 'ok':
                try:
                    clip_score = clip.score(result.image_bgr, artifact)
                except Exception as exc:
                    logging.debug('CLIP score failed: %s', exc)

            rel_img, rel_mrk = relative_output_paths(img_path, markup_path, input_root)
            # Keep images/ markup/ layout under method/artifact/
            out_img_path = out_dir / rel_img
            write_image(out_img_path, result.image_bgr)

            mask_path = None
            if save_masks:
                mask_rel = Path('masks') / Path(rel_img).with_suffix('.png')
                mask_path_full = out_dir / mask_rel
                write_image(mask_path_full, result.change_mask)
                mask_path = str(mask_path_full)

            out_mrk_path = None
            updated = update_markup(markup_data, result.change_mask, artifact, policy=policy)
            if updated is not None:
                if rel_mrk is not None:
                    out_mrk_path = out_dir / rel_mrk
                else:
                    # Synthesize markup path next to mirrored image.
                    parts = list(rel_img.parts)
                    if parts and parts[0] == 'images':
                        parts[0] = 'markup'
                        out_mrk_path = out_dir / Path(*parts).with_suffix(rel_img.suffix + '.json')
                    else:
                        out_mrk_path = out_dir / 'markup' / (rel_img.name + '.json')
                write_json(out_mrk_path, updated)

            if status == 'ok':
                counts['ok'] += 1

            # Contact sheet entry with paths relative to out_dir.
            try:
                orig_thumb = out_dir / 'contact' / 'originals' / Path(rel_img).name
                if not orig_thumb.exists():
                    write_image(orig_thumb, img)
                contact_entries.append(
                    {
                        'original': str(orig_thumb.relative_to(out_dir)),
                        'result': str(out_img_path.relative_to(out_dir)),
                        'method': method,
                        'artifact': artifact,
                        'nme': '' if nme is None else f'{nme:.4f}',
                        'status': status,
                    }
                )
            except Exception:
                pass

            manifest.write(
                ManifestRow(
                    method=method,
                    artifact=artifact,
                    img_in=img_path,
                    img_out=str(out_img_path),
                    markup_in=markup_path,
                    markup_out=str(out_mrk_path) if out_mrk_path else None,
                    seed=img_seed,
                    prompt=result.prompt,
                    mask_path=mask_path,
                    nme=nme,
                    clip_score=clip_score,
                    status=status,
                    elapsed_s=time.time() - t0,
                    extra=result.extra,
                )
            )
            logging.info(
                '[%d/%d] %s status=%s nme=%s %.2fs',
                idx + 1,
                len(pairs),
                Path(img_path).name,
                status,
                None if nme is None else f'{nme:.4f}',
                time.time() - t0,
            )

    if contact_entries:
        write_contact_sheet_html(out_dir / 'contact_sheet.html', contact_entries)

    try:
        gen.close()
    except Exception:
        pass

    logging.info('Done: %s', counts)
    return counts
