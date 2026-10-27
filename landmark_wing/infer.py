"""Run an HRNet face-keypoint checkpoint on a grayscale face crop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .data import crop_and_resize, expand_bbox, image_to_tensor
from .heatmaps import soft_argmax
from .model import HRNetFace
from .scheme import IMAGE_SIZE, KP_NAMES, NUM_LANDMARKS
from .vis import draw_landmarks


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        default=None,
        metavar=("X0", "Y0", "X1", "Y1"),
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def load_model(checkpoint_path: Path, device: str) -> tuple[HRNetFace, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    num_landmarks = int(payload.get("num_landmarks", NUM_LANDMARKS))
    if num_landmarks != NUM_LANDMARKS:
        raise RuntimeError(
            f"checkpoint has num_landmarks={num_landmarks}, expected {NUM_LANDMARKS}"
        )
    image_size = int(payload.get("image_size", IMAGE_SIZE))
    model = HRNetFace(num_landmarks=NUM_LANDMARKS, in_channels=1)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    meta = {
        "num_landmarks": NUM_LANDMARKS,
        "image_size": image_size,
        "dataset": payload.get("dataset"),
        "kp_names": KP_NAMES,
    }
    return model, meta


def crop_to_original(
    points_crop: np.ndarray,
    bbox_xyxy: np.ndarray,
    image_size: int,
) -> np.ndarray:
    x0, y0, x1, y1 = map(float, expand_bbox(bbox_xyxy))
    width = max(x1 - x0, 1.0)
    height = max(y1 - y0, 1.0)
    points = points_crop.astype(np.float32).copy()
    points[:, 0] = points[:, 0] / float(image_size) * width + x0
    points[:, 1] = points[:, 1] / float(image_size) * height + y0
    return points


@torch.inference_mode()
def predict_landmarks(
    model: HRNetFace,
    image_gray: np.ndarray,
    bbox_xyxy: np.ndarray,
    image_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dummy = np.zeros((NUM_LANDMARKS, 2), dtype=np.float32)
    vis = np.ones(NUM_LANDMARKS, dtype=np.float32)
    crop, _, _ = crop_and_resize(image_gray, dummy, vis, bbox_xyxy, image_size)
    tensor = image_to_tensor(crop).unsqueeze(0).to(device)
    heatmaps = model(tensor)
    pred_crop = soft_argmax(heatmaps)[0].cpu().numpy().astype(np.float32)
    pred_original = crop_to_original(pred_crop, bbox_xyxy, image_size)
    return pred_original, crop, pred_crop


def main() -> None:
    args = build_argparser().parse_args()
    model, meta = load_model(args.checkpoint, args.device)
    image_gray = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
    if image_gray is None:
        raise FileNotFoundError(args.image)
    if args.bbox is None:
        height, width = image_gray.shape[:2]
        bbox = np.array([0.0, 0.0, float(width - 1), float(height - 1)], dtype=np.float32)
    else:
        bbox = np.asarray(args.bbox, dtype=np.float32)

    points, crop, pred_crop = predict_landmarks(
        model,
        image_gray,
        bbox,
        image_size=int(meta["image_size"]),
        device=args.device,
    )
    named = {name: points[index].tolist() for index, name in enumerate(KP_NAMES)}
    result = {
        "image": str(args.image),
        "num_landmarks": meta["num_landmarks"],
        "dataset": meta.get("dataset"),
        "bbox_xyxy": bbox.tolist(),
        "landmarks": points.tolist(),
        "named_landmarks": named,
    }
    print(json.dumps(result))

    if args.out is not None:
        rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        overlay = draw_landmarks(rgb, pred_crop, target=None)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.out), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
