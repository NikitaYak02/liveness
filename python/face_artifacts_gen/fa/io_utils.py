"""I/O helpers: pair loading, path mirroring, manifest writing."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .face_kp_common import (
    IMG_EXT,
    is_image_file,
    list_image_markup_pairs,
    resolve_markup_path,
)

from .constants import IMG_EXTS


@dataclass
class ManifestRow:
    method: str
    artifact: str
    img_in: str
    img_out: str
    markup_in: Optional[str]
    markup_out: Optional[str]
    seed: Optional[int]
    prompt: Optional[str]
    mask_path: Optional[str]
    nme: Optional[float]
    clip_score: Optional[float]
    status: str
    elapsed_s: float
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class ManifestWriter:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open('a', encoding='utf-8')

    def write(self, row: ManifestRow) -> None:
        self._fh.write(row.to_json() + '\n')
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> 'ManifestWriter':
        return self

    def __exit__(self, *args) -> None:
        self.close()


def list_images(root: Path) -> List[Path]:
    """Recursively list image files under root."""
    root = Path(root)
    out: List[Path] = []
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out.append(p)
    out.sort()
    return out


def resolve_pairs(
    input_root: Path,
    require_markup: bool = False,
) -> List[Tuple[str, Optional[str]]]:
    """
    Return (img_path, markup_path|None) pairs under input_root.

    Prefer face_kp_common.list_image_markup_pairs when markup exists.
    Images without markup are included unless require_markup=True.
    """
    input_root = Path(input_root)
    paired = list_image_markup_pairs(str(input_root))
    paired_set = {img for img, _ in paired}
    result: List[Tuple[str, Optional[str]]] = [(img, mrk) for img, mrk in paired]

    if not require_markup:
        search = input_root / 'images' if (input_root / 'images').is_dir() else input_root
        for img in list_images(search):
            s = str(img)
            if s in paired_set:
                continue
            mrk = resolve_markup_path(s)
            result.append((s, mrk if os.path.exists(mrk) else None))

    result.sort(key=lambda t: t[0])
    return result


def relative_output_paths(
    img_path: str,
    markup_path: Optional[str],
    input_root: Path,
) -> Tuple[Path, Optional[Path]]:
    img = Path(img_path)
    try:
        rel_img = img.relative_to(input_root)
    except ValueError:
        rel_img = Path(img.name)

    rel_mrk: Optional[Path] = None
    if markup_path:
        mrk = Path(markup_path)
        try:
            rel_mrk = mrk.relative_to(input_root)
        except ValueError:
            # Mirror images→markup structure if possible.
            parts = list(rel_img.parts)
            if parts and parts[0] == 'images':
                parts[0] = 'markup'
                rel_mrk = Path(*parts).with_suffix(rel_img.suffix + '.json')
            else:
                rel_mrk = Path(mrk.name)
    return rel_img, rel_mrk


def write_image(path: Path, img_bgr: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), img_bgr)
    if not ok:
        raise RuntimeError(f'failed to write image: {path}')


def write_json(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def load_markup(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except Exception as exc:
        logging.warning('failed to read markup %s: %s', path, exc)
        return None


def feather_mask(mask: np.ndarray, ksize: int = 15) -> np.ndarray:
    """Return float32 alpha in [0, 1] with soft edges."""
    ksize = max(3, int(ksize) | 1)
    m = mask.astype(np.float32)
    if m.max() > 1.0:
        m = m / 255.0
    return cv2.GaussianBlur(m, (ksize, ksize), 0)


def paste_back(
    original_bgr: np.ndarray,
    edited_bgr: np.ndarray,
    mask: np.ndarray,
    feather: int = 15,
) -> np.ndarray:
    """
    Blend edited into original using soft mask.
    Outside the (hard) mask original pixels are preserved when feather<=1;
    with feather>1 edges are softened and a thin border may blend.
    """
    if original_bgr.shape != edited_bgr.shape:
        edited_bgr = cv2.resize(
            edited_bgr,
            (original_bgr.shape[1], original_bgr.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    if feather is None or int(feather) <= 1:
        alpha = (mask.astype(np.float32) / (255.0 if mask.max() > 1 else 1.0))
    else:
        alpha = feather_mask(mask, feather)
    if alpha.ndim == 2:
        alpha = alpha[..., None]
    out = original_bgr.astype(np.float32) * (1.0 - alpha) + edited_bgr.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def diff_mask(
    original_bgr: np.ndarray,
    edited_bgr: np.ndarray,
    threshold: float = 18.0,
    morph_ksize: int = 5,
) -> np.ndarray:
    """Binary mask of regions that changed between original and edited."""
    if original_bgr.shape != edited_bgr.shape:
        edited_bgr = cv2.resize(
            edited_bgr,
            (original_bgr.shape[1], original_bgr.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    diff = cv2.absdiff(original_bgr, edited_bgr)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mask = (gray >= threshold).astype(np.uint8) * 255
    if morph_ksize > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return mask
