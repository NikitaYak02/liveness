"""FaceLift adapter: six generated views -> Gaussian head -> render along our trajectory.

FaceLift treats the input photo as the frontal camera (azimuth 270 deg, elevation 0
on a radius-2.7 orbit, z up). The head is turned by moving that camera:
azimuth = 270 + yaw delta, elevation = pitch delta, where the deltas are taken
relative to the input photo's 3DDFA-V2 pose. For a folder the most frontal
photo is used.
"""

from __future__ import annotations

import importlib
import os

import cv2
import numpy as np
import torch
import yaml

from common import enter_repo, job_args, load_job, log, pose_delta, save_frame, save_meta, sign

SIZE = 512
RADIUS = 2.7
HFOV = 50.0
YAW_SIGN = sign("facelift_yaw_sign", 1.0)
PITCH_SIGN = sign("facelift_pitch_sign", 1.0)


def orbit_camera(yaw_deg: float, pitch_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """(fxfycxcy, c2w) in FaceLift's OpenCV convention, same construction as get_turntable_cameras."""
    fx = SIZE / (2 * np.tan(np.deg2rad(HFOV) / 2.0))
    fxfycxcy = np.array([fx, fx, SIZE / 2.0, SIZE / 2.0], np.float32)
    azim = np.deg2rad(270.0 + YAW_SIGN * yaw_deg)
    elev = np.deg2rad(PITCH_SIGN * pitch_deg)
    cam_pos = np.array([RADIUS * np.cos(elev) * np.cos(azim), RADIUS * np.cos(elev) * np.sin(azim),
                        RADIUS * np.sin(elev)])
    forward = -cam_pos / np.linalg.norm(cam_pos)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = np.stack((right, -up / np.linalg.norm(up), forward), axis=1)
    c2w[:3, 3] = cam_pos
    return fxfycxcy, c2w


def main() -> int:
    args = job_args(lambda p: (p.add_argument("--seed", type=int, default=4),
                               p.add_argument("--steps", type=int, default=50)))
    job = load_job(args.job)
    repo = enter_repo("FaceLift")
    from einops import rearrange
    from easydict import EasyDict as edict
    from PIL import Image

    import inference as fl
    from gslrm.model.gaussians_renderer import render_opencv_cam
    from utils_folder.face_utils import preprocess_image

    device = torch.device("cuda:0")
    mv_ckpt, gs_ckpt, gs_cfg = fl.get_model_paths()
    pipe, generator, prompt_embeds = fl.initialize_mvdiffusion_pipeline(mv_ckpt, device)
    generator.manual_seed(args.seed)

    source = job["frontal"]
    image = preprocess_image(np.array(Image.open(source).convert("RGB")))
    log(f"generating 6 views for {source}")
    views = pipe(image, None, prompt_embeds=prompt_embeds, guidance_scale=3.0, num_images_per_prompt=1,
                 num_inference_steps=args.steps, generator=generator, eta=1.0).images
    views = views[1:] if len(views) == 7 else views
    sheet = np.concatenate([np.array(v) for v in views], axis=1)
    cv2.imwrite(os.path.join(args.out, "multiview.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

    # Free the diffusion model before loading GS-LRM so both fit into 24 GB.
    del pipe
    torch.cuda.empty_cache()

    model_cfg = edict(yaml.safe_load(open(gs_cfg)))
    module_name, class_name = model_cfg.model.class_name.rsplit(".", 1)
    model = importlib.import_module(module_name).__dict__[class_name](model_cfg)
    model.load_state_dict(torch.load(gs_ckpt, map_location="cpu")["model"])
    model = model.to(device)

    fxfycxcy, c2w = fl.setup_camera_parameters(device)
    lrm_input = torch.from_numpy(np.stack([np.array(v) for v in views])).float()[None].to(device) / 255
    lrm_input = rearrange(lrm_input, "b v h w c -> b v c h w")
    index = torch.stack([torch.zeros(lrm_input.size(1)).long(), torch.arange(lrm_input.size(1)).long()], dim=-1)
    batch = edict({"image": lrm_input, "c2w": c2w, "fxfycxcy": fxfycxcy, "index": index[None].to(device)})
    with torch.no_grad(), torch.autocast(enabled=True, device_type="cuda", dtype=torch.float16):
        result = model.forward(batch, create_visual=False, split_data=True)
    gaussians = result.gaussians[0].apply_all_filters(
        opacity_thres=0.04, scaling_thres=0.1, floater_thres=0.6,
        crop_bbx=[-0.91, 0.91, -0.91, 0.91, -1.0, 1.0], cam_origins=None, nearfar_percent=(0.0001, 1.0),
    )
    gaussians.save_ply(os.path.join(args.out, "gaussians.ply"))

    for frame in job["frames"]:
        d_yaw, d_pitch = pose_delta(job, source, frame)
        intr, pose = orbit_camera(d_yaw, d_pitch)
        with torch.no_grad():
            render = render_opencv_cam(gaussians, SIZE, SIZE, torch.from_numpy(pose).to(device),
                                       torch.from_numpy(intr).to(device))["render"]
        rgb = (render.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        save_frame(args.out, frame["index"], cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    save_meta(args.out, {"source": source, "repo": str(repo), "seed": args.seed,
                         "yaw_sign": YAW_SIGN, "pitch_sign": PITCH_SIGN})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
