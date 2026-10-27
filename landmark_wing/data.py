"""Public datasets mapped onto the internal 30-point HRNet scheme."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .mapping import map_landmarks, mapped_to_arrays
from .scheme import FLIP_INDICES, IMAGE_SIZE, NUM_LANDMARKS

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DATASET_LANDMARKS = {
    "300w": NUM_LANDMARKS,
    "wflw": NUM_LANDMARKS,
    "lapa": NUM_LANDMARKS,
}


@dataclass(slots=True)
class LandmarkSample:
    image_path: Path
    raw_landmarks: np.ndarray  # native layout, used only for mapping
    bbox_xyxy: np.ndarray
    dataset: str
    split: str = ""


def expand_bbox(bbox_xyxy: np.ndarray, margin: float = 0.20) -> np.ndarray:
    x0, y0, x1, y1 = map(float, bbox_xyxy)
    width = max(x1 - x0, 1.0)
    height = max(y1 - y0, 1.0)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    side = max(width, height) * (1.0 + margin)
    half = 0.5 * side
    return np.array([cx - half, cy - half, cx + half, cy + half], dtype=np.float32)


def landmarks_to_bbox(landmarks: np.ndarray) -> np.ndarray:
    finite = landmarks[np.isfinite(landmarks).all(axis=1)]
    mins = finite.min(axis=0)
    maxs = finite.max(axis=0)
    return np.array([mins[0], mins[1], maxs[0], maxs[1]], dtype=np.float32)


def parse_pts(path: Path) -> np.ndarray:
    points: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith(("version:", "n_points:", "{", "}")):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                points.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue
    if not points:
        raise ValueError(f"no points in {path}")
    return np.asarray(points, dtype=np.float32)


def parse_lapa_landmarks(path: Path) -> np.ndarray:
    points = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            points.append([float(parts[0]), float(parts[1])])
    arr = np.asarray(points, dtype=np.float32)
    if arr.shape != (106, 2):
        raise ValueError(f"expected 106 points in {path}, got {arr.shape}")
    return arr


def _read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"failed to read image: {path}")
    return image


def crop_and_resize(
    image: np.ndarray,
    landmarks: np.ndarray,
    visibility: np.ndarray,
    bbox_xyxy: np.ndarray,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = map(float, expand_bbox(bbox_xyxy))
    height, width = image.shape[:2]
    x0_i = int(math.floor(x0))
    y0_i = int(math.floor(y0))
    x1_i = int(math.ceil(x1))
    y1_i = int(math.ceil(y1))

    pad_left = max(0, -x0_i)
    pad_top = max(0, -y0_i)
    pad_right = max(0, x1_i - width)
    pad_bottom = max(0, y1_i - height)
    if pad_left or pad_top or pad_right or pad_bottom:
        image = cv2.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=0,
        )
        x0_i += pad_left
        x1_i += pad_left
        y0_i += pad_top
        y1_i += pad_top
        landmarks = landmarks + np.array([pad_left, pad_top], dtype=np.float32)

    crop = image[y0_i:y1_i, x0_i:x1_i]
    if crop.size == 0:
        raise ValueError("empty crop")
    crop_h, crop_w = crop.shape[:2]
    resized = cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    scale_x = image_size / float(crop_w)
    scale_y = image_size / float(crop_h)
    points = landmarks.copy()
    points[:, 0] = (points[:, 0] - x0_i) * scale_x
    points[:, 1] = (points[:, 1] - y0_i) * scale_y
    return resized, points.astype(np.float32), visibility.astype(np.float32)


def _rotate(image: np.ndarray, landmarks: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    center = (width * 0.5, height * 0.5)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    ones = np.ones((landmarks.shape[0], 1), dtype=np.float32)
    points = np.concatenate([landmarks, ones], axis=1) @ matrix.T
    return rotated, points.astype(np.float32)


def _scale_translate(
    image: np.ndarray,
    landmarks: np.ndarray,
    scale: float,
    shift_x: float,
    shift_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    center = (width * 0.5, height * 0.5)
    matrix = np.array(
        [
            [scale, 0.0, (1.0 - scale) * center[0] + shift_x * width],
            [0.0, scale, (1.0 - scale) * center[1] + shift_y * height],
        ],
        dtype=np.float32,
    )
    warped = cv2.warpAffine(
        image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    ones = np.ones((landmarks.shape[0], 1), dtype=np.float32)
    points = np.concatenate([landmarks, ones], axis=1) @ matrix.T
    return warped, points.astype(np.float32)


def augment_crop(
    image: np.ndarray,
    landmarks: np.ndarray,
    visibility: np.ndarray,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angle = rng.uniform(-15.0, 15.0)
    image, landmarks = _rotate(image, landmarks, angle)
    scale = rng.uniform(0.85, 1.15)
    image, landmarks = _scale_translate(
        image, landmarks, scale, rng.uniform(-0.08, 0.08), rng.uniform(-0.08, 0.08)
    )
    if rng.random() < 0.5:
        image = np.ascontiguousarray(image[:, ::-1])
        width = image.shape[1]
        landmarks = landmarks.copy()
        landmarks[:, 0] = width - 1 - landmarks[:, 0]
        landmarks = landmarks[FLIP_INDICES]
        visibility = visibility[FLIP_INDICES]
    # Mild brightness / contrast on gray crop.
    image = image.astype(np.float32)
    image = image * (1.0 + rng.uniform(-0.2, 0.2)) + rng.uniform(-15.0, 15.0)
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image, landmarks, visibility


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    """Gray uint8 HxW -> float ``[1, H, W]`` in ``[-1, 1]``."""
    array = image.astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(array[None, ...].copy())


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Return RGB uint8 for visualization from a gray ``[1,H,W]`` tensor."""
    gray = ((tensor[0].detach().cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


class LandmarkDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[LandmarkSample],
        image_size: int = IMAGE_SIZE,
        augment: bool = False,
        seed: int = 0,
        min_visible: int = 20,
    ) -> None:
        if not samples:
            raise ValueError("dataset is empty")
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.min_visible = int(min_visible)
        self.num_landmarks = NUM_LANDMARKS
        self.samples = [s for s in samples if self._keep(s)]
        if not self.samples:
            raise ValueError("no samples with enough mapped keypoints")

    def _keep(self, sample: LandmarkSample) -> bool:
        mapped = map_landmarks(sample.raw_landmarks, sample.dataset)
        return len(mapped) >= self.min_visible

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        mapped = map_landmarks(sample.raw_landmarks, sample.dataset)
        landmarks, visibility = mapped_to_arrays(mapped)
        # Expand bbox to cover synthetic forehead / mapped points.
        bbox = sample.bbox_xyxy.copy()
        for x, y in landmarks[visibility > 0.5]:
            bbox[0] = min(bbox[0], x)
            bbox[1] = min(bbox[1], y)
            bbox[2] = max(bbox[2], x)
            bbox[3] = max(bbox[3], y)

        image = _read_gray(sample.image_path)
        image, landmarks, visibility = crop_and_resize(
            image, landmarks, visibility, bbox, self.image_size
        )
        if self.augment:
            rng = random.Random(self.seed + index * 9973)
            image, landmarks, visibility = augment_crop(image, landmarks, visibility, rng)

        # Drop points that left the crop after aug.
        inside = (
            (landmarks[:, 0] >= 0)
            & (landmarks[:, 0] < self.image_size)
            & (landmarks[:, 1] >= 0)
            & (landmarks[:, 1] < self.image_size)
        )
        visibility = visibility * inside.astype(np.float32)

        return {
            "image": image_to_tensor(image),
            "landmarks": torch.from_numpy(landmarks.astype(np.float32)),
            "visibility": torch.from_numpy(visibility.astype(np.float32)),
            "path": str(sample.image_path),
            "split": sample.split,
        }


def _find_image_for_stem(directory: Path, stem: str) -> Optional[Path]:
    for ext in IMG_EXTS:
        for candidate in (directory / f"{stem}{ext}", directory / f"{stem}{ext.upper()}"):
            if candidate.is_file():
                return candidate
    return None


def _collect_300w_subset(root: Path, relative_dirs: Iterable[str], split: str) -> list[LandmarkSample]:
    samples: list[LandmarkSample] = []
    for relative in relative_dirs:
        directory = root / relative
        if not directory.is_dir():
            continue
        for pts_path in sorted(directory.rglob("*.pts")):
            image_path = _find_image_for_stem(pts_path.parent, pts_path.stem)
            if image_path is None:
                continue
            landmarks = parse_pts(pts_path)
            if landmarks.shape[0] != 68:
                continue
            samples.append(
                LandmarkSample(
                    image_path=image_path,
                    raw_landmarks=landmarks,
                    bbox_xyxy=landmarks_to_bbox(landmarks),
                    dataset="300w",
                    split=split,
                )
            )
    return samples


def build_300w(root: Path, split: str) -> list[LandmarkSample]:
    root = Path(root)
    split = split.lower()
    layouts = {
        "train": ["afw", "AFW", "lfpw/trainset", "LFPW/trainset", "helen/trainset", "HELEN/trainset"],
        "common": ["lfpw/testset", "LFPW/testset", "helen/testset", "HELEN/testset"],
        "challenging": ["ibug", "IBUG"],
        "full": [
            "lfpw/testset", "LFPW/testset", "helen/testset", "HELEN/testset", "ibug", "IBUG",
        ],
        "val": [
            "lfpw/testset", "LFPW/testset", "helen/testset", "HELEN/testset", "ibug", "IBUG",
        ],
    }
    if split not in layouts:
        raise ValueError(f"unknown 300W split: {split}")
    samples = _collect_300w_subset(root, layouts[split], split)
    if not samples:
        raise FileNotFoundError(
            f"no 300W samples under {root} for split={split}. "
            "Expected AFW/LFPW/HELEN/IBUG folders with paired .jpg+.pts files."
        )
    return samples


def _parse_wflw_line(line: str, images_root: Path) -> Optional[LandmarkSample]:
    parts = line.strip().split()
    if len(parts) < 98 * 2 + 4 + 1:
        return None
    coords = np.asarray([float(v) for v in parts[: 98 * 2]], dtype=np.float32).reshape(98, 2)
    bbox = np.asarray([float(v) for v in parts[98 * 2 : 98 * 2 + 4]], dtype=np.float32)
    image_rel = parts[-1]
    image_path = images_root / image_rel
    if not image_path.is_file():
        alt = images_root / Path(image_rel).name
        if alt.is_file():
            image_path = alt
        else:
            return None
    return LandmarkSample(
        image_path=image_path,
        raw_landmarks=coords,
        bbox_xyxy=bbox,
        dataset="wflw",
        split="",
    )


def build_wflw(root: Path, split: str) -> list[LandmarkSample]:
    root = Path(root)
    split = split.lower()
    images_root = root / "WFLW_images"
    if not images_root.is_dir():
        images_root = root
    ann_dir_candidates = [
        root / "WFLW_annotations" / "list_98pt_rect_attr_train_test",
        root / "WFLW_annotations",
        root,
    ]
    filename = {
        "train": "list_98pt_rect_attr_train.txt",
        "val": "list_98pt_rect_attr_test.txt",
        "test": "list_98pt_rect_attr_test.txt",
    }.get(split)
    if filename is None:
        raise ValueError(f"unknown WFLW split: {split}")
    list_path = None
    for directory in ann_dir_candidates:
        candidate = directory / filename
        if candidate.is_file():
            list_path = candidate
            break
    if list_path is None:
        raise FileNotFoundError(f"WFLW annotation list {filename} not found under {root}")
    samples: list[LandmarkSample] = []
    with list_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            sample = _parse_wflw_line(line, images_root)
            if sample is not None:
                sample.split = split
                samples.append(sample)
    if not samples:
        raise FileNotFoundError(f"no WFLW samples parsed from {list_path}")
    return samples


def build_lapa(root: Path, split: str) -> list[LandmarkSample]:
    root = Path(root)
    split = split.lower()
    if split not in {"train", "val", "test"}:
        raise ValueError(f"unknown LaPa split: {split}")
    split_dir = root / split
    images_dir = split_dir / "images"
    landmarks_dir = split_dir / "landmarks"
    if not images_dir.is_dir() or not landmarks_dir.is_dir():
        raise FileNotFoundError(
            f"LaPa expects {split}/images and {split}/landmarks under {root}"
        )
    samples: list[LandmarkSample] = []
    for landmark_path in sorted(landmarks_dir.glob("*.txt")):
        image_path = _find_image_for_stem(images_dir, landmark_path.stem)
        if image_path is None:
            continue
        landmarks = parse_lapa_landmarks(landmark_path)
        samples.append(
            LandmarkSample(
                image_path=image_path,
                raw_landmarks=landmarks,
                bbox_xyxy=landmarks_to_bbox(landmarks),
                dataset="lapa",
                split=split,
            )
        )
    if not samples:
        raise FileNotFoundError(f"no LaPa samples under {split_dir}")
    return samples


def build_samples(dataset: str, root: Path, split: str) -> list[LandmarkSample]:
    name = dataset.lower()
    if name == "300w":
        return build_300w(root, split)
    if name == "wflw":
        return build_wflw(root, split)
    if name == "lapa":
        return build_lapa(root, split)
    raise ValueError(f"unknown dataset: {dataset}")


def make_dataset(
    dataset: str,
    root: Path,
    split: str,
    image_size: int = IMAGE_SIZE,
    augment: bool = False,
    seed: int = 0,
) -> LandmarkDataset:
    samples = build_samples(dataset, root, split)
    return LandmarkDataset(
        samples=samples,
        image_size=image_size,
        augment=augment,
        seed=seed,
    )


def default_train_val_splits(dataset: str) -> tuple[str, str]:
    name = dataset.lower()
    if name == "300w":
        return "train", "full"
    if name == "wflw":
        return "train", "test"
    if name == "lapa":
        return "train", "val"
    raise ValueError(f"unknown dataset: {dataset}")
