"""Public facial-landmark dataset loaders for Wing-loss training."""

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

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Horizontal flip index permutations for official layouts.
FLIP_INDICES_68 = np.array(
    [
        16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
        26, 25, 24, 23, 22, 21, 20, 19, 18, 17,
        27, 28, 29, 30, 35, 34, 33, 32, 31,
        45, 44, 43, 42, 47, 46, 39, 38, 37, 36, 41, 40,
        54, 53, 52, 51, 50, 49, 48, 59, 58, 57, 56, 55,
        64, 63, 62, 61, 60, 67, 66, 65,
    ],
    dtype=np.int64,
)

FLIP_INDICES_98 = np.array(
    [
        32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
        15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
        46, 45, 44, 43, 42, 50, 49, 48, 47, 37, 36, 35, 34, 33, 41, 40, 39, 38,
        51, 52, 53, 54, 59, 58, 57, 56, 55,
        72, 71, 70, 69, 68, 75, 74, 73, 64, 63, 62, 61, 60, 67, 66, 65,
        82, 81, 80, 79, 78, 77, 76, 87, 86, 85, 84, 83, 92, 91, 90, 89, 88,
        95, 94, 93, 97, 96,
    ],
    dtype=np.int64,
)

# Derived from MMPose configs/_base_/datasets/lapa.py swap map.
FLIP_INDICES_106 = np.array(
    [
        32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
        15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
        46, 45, 44, 43, 42, 50, 49, 48, 47, 37, 36, 35, 34, 33, 41, 40, 39, 38,
        51, 52, 53, 54,
        65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55,
        79, 78, 77, 76, 75, 82, 81, 80, 83,
        70, 69, 68, 67, 66, 73, 72, 71, 74,
        90, 89, 88, 87, 86, 85, 84,
        95, 94, 93, 92, 91,
        100, 99, 98, 97, 96,
        103, 102, 101,
        105, 104,
    ],
    dtype=np.int64,
)

NORM_INDICES = {
    68: (36, 45),
    98: (60, 72),
    106: (104, 105),
}

DATASET_LANDMARKS = {
    "300w": 68,
    "wflw": 98,
    "lapa": 106,
}


@dataclass(slots=True)
class LandmarkSample:
    image_path: Path
    landmarks: np.ndarray  # [N, 2] float32 in original image coords
    bbox_xyxy: np.ndarray  # [4] float32
    split: str = ""


def flip_indices_for(num_landmarks: int) -> np.ndarray:
    if num_landmarks == 68:
        return FLIP_INDICES_68
    if num_landmarks == 98:
        return FLIP_INDICES_98
    if num_landmarks == 106:
        return FLIP_INDICES_106
    raise ValueError(f"unsupported landmark count: {num_landmarks}")


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
    mins = landmarks.min(axis=0)
    maxs = landmarks.max(axis=0)
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


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"failed to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def crop_and_resize(
    image: np.ndarray,
    landmarks: np.ndarray,
    bbox_xyxy: np.ndarray,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray]:
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
            value=(0, 0, 0),
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
    return resized, points.astype(np.float32)


def _rotate_image_and_points(
    image: np.ndarray,
    landmarks: np.ndarray,
    angle_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    center = (width * 0.5, height * 0.5)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    ones = np.ones((landmarks.shape[0], 1), dtype=np.float32)
    homo = np.concatenate([landmarks, ones], axis=1)
    points = homo @ matrix.T
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
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    ones = np.ones((landmarks.shape[0], 1), dtype=np.float32)
    homo = np.concatenate([landmarks, ones], axis=1)
    points = homo @ matrix.T
    return warped, points.astype(np.float32)


def _color_jitter(image: np.ndarray, rng: random.Random) -> np.ndarray:
    out = image.astype(np.float32)
    brightness = 1.0 + rng.uniform(-0.2, 0.2)
    contrast = 1.0 + rng.uniform(-0.2, 0.2)
    saturation = 1.0 + rng.uniform(-0.2, 0.2)
    out = out * brightness
    mean = out.mean(axis=(0, 1), keepdims=True)
    out = (out - mean) * contrast + mean
    gray = out.mean(axis=2, keepdims=True)
    out = (out - gray) * saturation + gray
    return np.clip(out, 0, 255).astype(np.uint8)


def augment_crop(
    image: np.ndarray,
    landmarks: np.ndarray,
    flip_indices: np.ndarray,
    rng: random.Random,
    rotate_deg: float = 30.0,
    scale_range: tuple[float, float] = (0.85, 1.15),
    shift_frac: float = 0.08,
    flip_prob: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    angle = rng.uniform(-rotate_deg, rotate_deg)
    image, landmarks = _rotate_image_and_points(image, landmarks, angle)
    scale = rng.uniform(*scale_range)
    shift_x = rng.uniform(-shift_frac, shift_frac)
    shift_y = rng.uniform(-shift_frac, shift_frac)
    image, landmarks = _scale_translate(image, landmarks, scale, shift_x, shift_y)
    if rng.random() < flip_prob:
        image = np.ascontiguousarray(image[:, ::-1])
        width = image.shape[1]
        landmarks = landmarks.copy()
        landmarks[:, 0] = width - 1 - landmarks[:, 0]
        landmarks = landmarks[flip_indices]
    image = _color_jitter(image, rng)
    return image, landmarks


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    """RGB uint8 HxWxC -> float tensor CxHxW in [0, 1], ImageNet-normalized."""
    array = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    array = (array - mean) / std
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(3, 1, 1)
    array = (tensor * std + mean).clamp(0, 1).cpu().numpy()
    return (array.transpose(1, 2, 0) * 255.0).astype(np.uint8)


class LandmarkDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[LandmarkSample],
        image_size: int = 224,
        augment: bool = False,
        seed: int = 0,
    ) -> None:
        if not samples:
            raise ValueError("dataset is empty")
        self.samples = list(samples)
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.num_landmarks = int(self.samples[0].landmarks.shape[0])
        self.flip_indices = flip_indices_for(self.num_landmarks)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        image = _read_rgb(sample.image_path)
        landmarks = sample.landmarks.copy()
        image, landmarks = crop_and_resize(
            image, landmarks, sample.bbox_xyxy, self.image_size
        )
        if self.augment:
            rng = random.Random(self.seed + index * 9973)
            image, landmarks = augment_crop(image, landmarks, self.flip_indices, rng)
        return {
            "image": image_to_tensor(image),
            "landmarks": torch.from_numpy(landmarks.astype(np.float32)),
            "path": str(sample.image_path),
            "split": sample.split,
        }


def _find_image_for_stem(directory: Path, stem: str) -> Optional[Path]:
    for ext in IMG_EXTS:
        candidate = directory / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
        candidate = directory / f"{stem}{ext.upper()}"
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
                    landmarks=landmarks,
                    bbox_xyxy=landmarks_to_bbox(landmarks),
                    split=split,
                )
            )
    return samples


def build_300w(root: Path, split: str) -> list[LandmarkSample]:
    root = Path(root)
    split = split.lower()
    layouts = {
        "train": [
            "afw",
            "AFW",
            "lfpw/trainset",
            "LFPW/trainset",
            "helen/trainset",
            "HELEN/trainset",
        ],
        "common": [
            "lfpw/testset",
            "LFPW/testset",
            "helen/testset",
            "HELEN/testset",
        ],
        "challenging": ["ibug", "IBUG"],
        "full": [
            "lfpw/testset",
            "LFPW/testset",
            "helen/testset",
            "HELEN/testset",
            "ibug",
            "IBUG",
        ],
        "val": [
            "lfpw/testset",
            "LFPW/testset",
            "helen/testset",
            "HELEN/testset",
            "ibug",
            "IBUG",
        ],
    }
    if split not in layouts:
        raise ValueError(f"unknown 300W split: {split}")
    samples = _collect_300w_subset(root, layouts[split], split)
    if not samples:
        # Some dumps keep all images under one folder with nested names.
        samples = _collect_300w_subset(root, ["."], split)
        if split == "train":
            samples = [
                s
                for s in samples
                if any(token in str(s.image_path).lower() for token in ("afw", "train"))
            ]
    if not samples:
        raise FileNotFoundError(
            f"no 300W samples under {root} for split={split}. "
            "Expected AFW/LFPW/HELEN/IBUG folders with paired .jpg+.pts files."
        )
    return samples


def _parse_wflw_line(line: str, images_root: Path) -> Optional[LandmarkSample]:
    parts = line.strip().split()
    # 98 points * 2 + 4 bbox + 6 attributes + image path
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
        landmarks=coords,
        bbox_xyxy=bbox,
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
        raise FileNotFoundError(
            f"WFLW annotation list {filename} not found under {root}"
        )
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
                landmarks=landmarks,
                bbox_xyxy=landmarks_to_bbox(landmarks),
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
    image_size: int = 224,
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
