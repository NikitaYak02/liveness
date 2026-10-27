"""Run a Wing-loss landmark checkpoint on an image crop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .data import crop_and_resize, expand_bbox, image_to_tensor
from .model import MobileNetV2Landmarks
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


def load_model(checkpoint_path: Path, device: str) -> tuple[MobileNetV2Landmarks, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    num_landmarks = int(payload["num_landmarks"])
    image_size = int(payload.get("image_size", 224))
    model = MobileNetV2Landmarks(num_landmarks=num_landmarks, pretrained=False)
    state = payload["state_dict"]
    head_weight = state.get("head.weight")
    if head_weight is not None and head_weight.shape[0] != num_landmarks * 2:
        raise RuntimeError(
            f"checkpoint head size {head_weight.shape[0]} does not match "
            f"num_landmarks={num_landmarks}"
        )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    meta = {
        "num_landmarks": num_landmarks,
        "image_size": image_size,
        "dataset": payload.get("dataset"),
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
    model: MobileNetV2Landmarks,
    image_rgb: np.ndarray,
    bbox_xyxy: np.ndarray,
    image_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return original-image points, crop RGB, and crop-space points."""
    dummy = np.zeros((model.num_landmarks, 2), dtype=np.float32)
    crop, _ = crop_and_resize(image_rgb, dummy, bbox_xyxy, image_size)
    tensor = image_to_tensor(crop).unsqueeze(0).to(device)
    pred_crop = model(tensor)[0].cpu().numpy().astype(np.float32)
    pred_original = crop_to_original(pred_crop, bbox_xyxy, image_size)
    return pred_original, crop, pred_crop


def main() -> None:
    args = build_argparser().parse_args()
    model, meta = load_model(args.checkpoint, args.device)
    image_bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(args.image)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if args.bbox is None:
        height, width = image_rgb.shape[:2]
        bbox = np.array([0.0, 0.0, float(width - 1), float(height - 1)], dtype=np.float32)
    else:
        bbox = np.asarray(args.bbox, dtype=np.float32)

    points, crop, pred_crop = predict_landmarks(
        model,
        image_rgb,
        bbox,
        image_size=int(meta["image_size"]),
        device=args.device,
    )
    result = {
        "image": str(args.image),
        "num_landmarks": meta["num_landmarks"],
        "dataset": meta.get("dataset"),
        "bbox_xyxy": bbox.tolist(),
        "landmarks": points.tolist(),
    }
    print(json.dumps(result))

    if args.out is not None:
        overlay = draw_landmarks(crop, pred_crop, target=None)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.out), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
