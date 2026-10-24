from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from tqdm import tqdm

from .config import CooperativeScenarioConfig
from .data import load_manifest
from .flow import build_raft_model, compute_clipped_flow_magnitude, rgb_frame_to_tensor
from .landmarks import FaceAlignmentLandmarkDetector
from .preprocessing import CooperativeVideoPreprocessor


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="JSONL/CSV with video_path,label[,id]")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--landmark-device", default=None)
    parser.add_argument("--raft-variant", default="large", choices=("large", "small"))
    parser.add_argument("--raft-weights", default="DEFAULT")
    parser.add_argument("--random-frame", action="store_true")
    parser.add_argument("--multi-resolution", action="store_true")
    parser.add_argument("--perspective-prob", type=float, default=0.0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    rng = random.Random(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = CooperativeScenarioConfig()
    detector = FaceAlignmentLandmarkDetector(
        device=args.landmark_device or args.device,
        face_detector="sfd",
    )
    preprocessor = CooperativeVideoPreprocessor(detector=detector, config=config)
    raft_model = build_raft_model(
        device=args.device,
        variant=args.raft_variant,
        weights_name=args.raft_weights,
    )

    manifest = load_manifest(args.manifest)
    failures = []
    for entry in tqdm(manifest, desc="Preparing"):
        output_path = out_dir / f"{entry.sample_id}.pt"
        if args.skip_existing and output_path.exists():
            continue

        try:
            triplet = preprocessor.process_video(
                entry.video_path,
                random_frame=args.random_frame,
                perspective_prob=args.perspective_prob,
                rng=rng,
            )
            flow_size = (
                rng.choice(config.flow_resolutions)
                if args.multi_resolution
                else config.crop_size
            )
            flow = compute_clipped_flow_magnitude(
                triplet.f1_rgb,
                triplet.f3_rgb,
                raft_model=raft_model,
                device=args.device,
                flow_size=flow_size,
                num_flow_updates=config.num_flow_updates,
                clip_fraction=config.clip_fraction,
            )
            rgb = rgb_frame_to_tensor(triplet.f2_rgb)
            payload = {
                "flow": flow,
                "rgb": rgb,
                "label": int(entry.label),
                "meta": {
                    "sample_id": entry.sample_id,
                    "video_path": entry.video_path,
                    "frame_indices": triplet.frame_indices,
                    "face_ratios": triplet.face_ratios,
                },
            }
            torch.save(payload, output_path)
        except Exception as exc:  # pragma: no cover - logging branch
            failures.append((entry.sample_id, str(exc)))

    if failures:
        failure_path = out_dir / "failures.txt"
        failure_path.write_text(
            "\n".join(f"{sample_id}\t{message}" for sample_id, message in failures),
            encoding="utf-8",
        )
        print(f"Completed with {len(failures)} failures. See {failure_path}")
    else:
        print("Completed without failures.")


if __name__ == "__main__":
    main()
