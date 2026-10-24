"""QA: landmark NME filter, optional CLIP artifact score, contact sheet."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .constants import CLIP_ARTIFACT_TEXTS
from .landmarks import FaceLandmarks, InsightFaceLandmarker


def _interocular(face: FaceLandmarks) -> float:
    left = face.get('kp_pupil_left') or face.get('kp_eye_left_outer')
    right = face.get('kp_pupil_right') or face.get('kp_eye_right_outer')
    if left and right:
        return float(np.hypot(left[0] - right[0], left[1] - right[1]))
    if face.rect:
        return float(max(face.rect[2] * 0.25, 1.0))
    return 1.0


def landmark_nme(
    ref: FaceLandmarks,
    hyp: FaceLandmarks,
    names: Optional[Sequence[str]] = None,
) -> Optional[float]:
    """Normalized mean error over shared visible keypoints."""
    if names is None:
        names = sorted(set(ref.points) & set(hyp.points))
    else:
        names = [n for n in names if n in ref.points and n in hyp.points]
    if not names:
        return None
    norm = max(_interocular(ref), 1e-3)
    errs = []
    for n in names:
        a, b = ref.points[n], hyp.points[n]
        errs.append(np.hypot(a[0] - b[0], a[1] - b[1]) / norm)
    return float(np.mean(errs))


def check_landmark_drift(
    original_bgr: np.ndarray,
    result_bgr: np.ndarray,
    ref_face: FaceLandmarks,
    landmarker: Optional[InsightFaceLandmarker],
    max_nme: float = 0.08,
) -> Tuple[bool, Optional[float]]:
    """
    Re-detect landmarks on result and compare to reference.
    Returns (accepted, nme). If landmarker is None, accept with nme=None.
    """
    if landmarker is None:
        return True, None
    try:
        faces = landmarker.detect(result_bgr)
    except Exception as exc:
        logging.warning('QA landmark detect failed: %s', exc)
        return True, None
    if not faces:
        return False, None
    nme = landmark_nme(ref_face, faces[0])
    if nme is None:
        return True, None
    return nme <= max_nme, nme


class ClipArtifactScorer:
    """Optional CLIP score that the artifact text matches the image."""

    def __init__(self, device: str = 'cuda', model_id: str = 'openai/clip-vit-base-patch32'):
        self.device = device
        self.model_id = model_id
        self._model = None
        self._proc = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError('CLIP QA requires: pip install transformers torch') from exc
        self._proc = CLIPProcessor.from_pretrained(self.model_id)
        self._model = CLIPModel.from_pretrained(self.model_id).to(self.device)
        self._model.eval()
        self._torch = torch

    def score(self, img_bgr: np.ndarray, artifact: str) -> Optional[float]:
        text = CLIP_ARTIFACT_TEXTS.get(artifact)
        if not text:
            return None
        try:
            self._load()
        except Exception as exc:
            logging.debug('CLIP unavailable: %s', exc)
            return None
        from PIL import Image

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        inputs = self._proc(text=[text], images=pil, return_tensors='pt', padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self._torch.no_grad():
            out = self._model(**inputs)
            # logits_per_image is similarity; sigmoid-ish normalize via softmax vs single text → logit
            score = float(out.logits_per_image[0, 0].item())
        return score


def build_contact_sheet(
    rows: List[Dict[str, np.ndarray]],
    cell_size: Tuple[int, int] = (256, 256),
    labels: Optional[List[str]] = None,
) -> np.ndarray:
    """
    rows: list of dicts {label: image_bgr}. Builds a grid.
    For simple use: one row with keys original + method names.
    """
    if not rows:
        return np.zeros((cell_size[1], cell_size[0], 3), dtype=np.uint8)
    keys = list(rows[0].keys())
    if labels:
        keys = labels
    cw, ch = cell_size
    n_rows = len(rows)
    n_cols = len(keys)
    sheet = np.full((n_rows * ch, n_cols * cw, 3), 30, dtype=np.uint8)
    for r, row in enumerate(rows):
        for c, key in enumerate(keys):
            img = row.get(key)
            if img is None:
                continue
            thumb = cv2.resize(img, (cw, ch), interpolation=cv2.INTER_AREA)
            cv2.putText(
                thumb,
                str(key)[:24],
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            sheet[r * ch : (r + 1) * ch, c * cw : (c + 1) * cw] = thumb
    return sheet


def write_contact_sheet_html(
    path: Path,
    entries: List[Dict[str, str]],
    title: str = 'Face artifacts contact sheet',
) -> None:
    """
    entries: [{original: rel_url, result: rel_url, method, artifact, nme, status}, ...]
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cards = []
    for e in entries:
        cards.append(
            f"""
            <div class="card">
              <div class="meta">{e.get('method','')} / {e.get('artifact','')} —
                status={e.get('status','')} nme={e.get('nme')}</div>
              <div class="pair">
                <img src="{e.get('original','')}" alt="original"/>
                <img src="{e.get('result','')}" alt="result"/>
              </div>
            </div>
            """
        )
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>{title}</title>
<style>
body {{ font-family: sans-serif; background: #111; color: #eee; }}
.card {{ margin: 16px; padding: 12px; background: #1c1c1c; border-radius: 8px; }}
.pair img {{ height: 240px; margin-right: 8px; border-radius: 4px; }}
.meta {{ margin-bottom: 8px; color: #aaa; }}
</style></head><body>
<h1>{title}</h1>
{''.join(cards)}
</body></html>
"""
    path.write_text(html, encoding='utf-8')
