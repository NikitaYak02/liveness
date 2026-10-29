"""GPAvatar adapter: one avatar from all input photos, rendered from new viewpoints.

All photos are tracked (MICA shape, EMOCA expression, camera) and fused into one
tri-plane avatar. The expression stays the one of the most frontal photo; only
the head rotation changes. Rotation is applied around the FLAME origin (head
centre) relative to the frontal photo's tracked camera.

GPAvatar's camera: rays are origin = -R @ T, direction = R @ d_cam, so a world
point maps into the camera as R^T p + T. Rotating the head by Q in camera space
gives R' = R @ Q^T with T unchanged.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from common import (
    enter_repo, face_crop, job_args, load_job, log, pose_delta, read_bgr, save_frame, save_meta, sign, source_box,
)

SIZE = 512
YAW_SIGN = sign("gpavatar_yaw_sign", 1.0)
PITCH_SIGN = sign("gpavatar_pitch_sign", 1.0)


def rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    y, p = np.radians(YAW_SIGN * yaw_deg), np.radians(PITCH_SIGN * pitch_deg)
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    return ry @ rx


def to_tensor_rgb(img_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0


def main() -> int:
    args = job_args()
    job = load_job(args.job)
    enter_repo("GPAvatar")
    from core.data.tools import perspective_input
    from core.libs.lightning_track import TrackEngine
    from inference_tools import gpavatar_r2g

    device = "cuda"
    images = job["images"]
    crops = [to_tensor_rgb(face_crop(read_bgr(p), source_box(job, p), SIZE)) for p in images]

    track = TrackEngine(focal_length=12.0, device=device)
    matted, results = track.track_images(crops)
    tracked = [i for i in range(len(images)) if str(i) in results]
    if not tracked:
        raise SystemExit("GPAvatar tracking failed on every input photo")
    log(f"tracked {len(tracked)}/{len(images)} photos")

    calib = {"focal_length": 12.0, "principal_point": [0.0, 0.0]}
    feature_image = torch.stack([
        perspective_input(matted[i][None].to(device),
                          torch.as_tensor(results[str(i)]["transform_matrix"])[None].to(device), calib, fill=0.0)[0]
        for i in tracked
    ])
    inp_track = {
        "feature_image": feature_image.cpu(),
        "feature_shape": torch.stack([torch.as_tensor(results[str(i)]["mica_shape"]) for i in tracked]).mean(0),
        "frame_trans": torch.stack([torch.as_tensor(results[str(i)]["transform_matrix"]) for i in tracked]),
    }

    model = gpavatar_r2g().to(device).eval()
    with torch.no_grad():
        model.build_avatar(inp_track=inp_track)

    ref_path = job["frontal"]
    ref_idx = images.index(ref_path)
    if ref_idx not in tracked:
        ref_idx = tracked[0]
        ref_path = images[ref_idx]
    ref = results[str(ref_idx)]
    expression = torch.as_tensor(ref["emoca_expression"])[None].float().to(device)
    pose = torch.as_tensor(ref["emoca_pose"])[None].float().to(device)
    T_ref = torch.as_tensor(ref["transform_matrix"]).float()
    R_ref = T_ref[:, :3].numpy()

    for frame in job["frames"]:
        d_yaw, d_pitch = pose_delta(job, ref_path, frame)
        R = R_ref @ rotation(d_yaw, d_pitch).T
        trans = T_ref.clone()
        trans[:, :3] = torch.from_numpy(R).float()
        with torch.no_grad():
            out = model(expression=expression, pose=pose, transform_matrix=trans[None].to(device))[0]
        rgb = (out.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        save_frame(args.out, frame["index"], cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    save_meta(args.out, {"reference_photo": ref_path, "tracked_photos": [images[i] for i in tracked],
                         "yaw_sign": YAW_SIGN, "pitch_sign": PITCH_SIGN})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
