from __future__ import annotations

import argparse

import torch

from .config import CooperativeScenarioConfig
from .flow import build_raft_model, compute_clipped_flow_magnitude, rgb_frame_to_tensor
from .landmarks import FaceAlignmentLandmarkDetector
from .model import DualResNet18Liveness
from .preprocessing import CooperativeVideoPreprocessor


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--landmark-device", default=None)
    parser.add_argument("--raft-variant", default="large", choices=("large", "small"))
    parser.add_argument("--raft-weights", default="DEFAULT")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    device = args.device
    config = CooperativeScenarioConfig()

    detector = FaceAlignmentLandmarkDetector(
        device=args.landmark_device or args.device,
        face_detector="sfd",
    )
    preprocessor = CooperativeVideoPreprocessor(detector=detector, config=config)
    raft_model = build_raft_model(
        device=device,
        variant=args.raft_variant,
        weights_name=args.raft_weights,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = DualResNet18Liveness(**checkpoint.get("model_kwargs", {}))
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device).eval()

    triplet = preprocessor.process_video(args.video)
    flow = compute_clipped_flow_magnitude(
        triplet.f1_rgb,
        triplet.f3_rgb,
        raft_model=raft_model,
        device=device,
        flow_size=config.crop_size,
        num_flow_updates=config.num_flow_updates,
        clip_fraction=config.clip_fraction,
    ).unsqueeze(0).to(device)
    rgb = rgb_frame_to_tensor(triplet.f2_rgb).unsqueeze(0).to(device)

    with torch.inference_mode():
        score = torch.sigmoid(model(flow, rgb)).item()

    prediction = "live" if score >= args.threshold else "spoof"
    print(f"score={score:.6f}")
    print(f"prediction={prediction}")
    print(f"threshold={args.threshold:.3f}")


if __name__ == "__main__":
    main()
