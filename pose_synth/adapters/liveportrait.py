"""LivePortrait adapter: edit the head pose of the source photo directly.

No driving video: the source implicit keypoints are re-posed with a new
rotation (same as the "pose editing" sliders in the LivePortrait Gradio app),
then warped, decoded and pasted back into the full frame.

Targets are absolute 3DDFA-V2 angles. LivePortrait's own pose estimate differs
from 3DDFA-V2, so the adapter applies the *difference* between the target and
the source pose measured by 3DDFA-V2 on top of LivePortrait's source pose.
"""

from __future__ import annotations

import os

# Apple MPS lacks grid_sampler_3d; let it fall back to CPU (no effect on CUDA).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import enter_repo, job_args, load_job, log, read_bgr, save_frame, save_meta, sign

# LivePortrait angle = source angle + SIGN * (3DDFA-V2 delta).
YAW_SIGN = sign("liveportrait_yaw_sign", 1.0)
PITCH_SIGN = sign("liveportrait_pitch_sign", 1.0)


class LivePortraitEditor:
    def __init__(self):
        enter_repo("LivePortrait")
        from src.config.crop_config import CropConfig
        from src.config.inference_config import InferenceConfig
        from src.live_portrait_pipeline import LivePortraitPipeline

        force_cpu = not torch.cuda.is_available() and not torch.backends.mps.is_available()
        inference_cfg = InferenceConfig(flag_force_cpu=force_cpu, device_id=0)
        crop_cfg = CropConfig(flag_force_cpu=force_cpu, device_id=0)
        self.pipeline = LivePortraitPipeline(inference_cfg=inference_cfg, crop_cfg=crop_cfg)
        self.wrapper = self.pipeline.live_portrait_wrapper
        self.cropper = self.pipeline.cropper
        self.inference_cfg = inference_cfg

    @torch.no_grad()
    def prepare(self, path: str) -> dict:
        from src.utils.crop import prepare_paste_back
        from src.utils.io import resize_to_limit

        img_rgb = cv2.cvtColor(read_bgr(path), cv2.COLOR_BGR2RGB)
        img_rgb = resize_to_limit(img_rgb, 1280, 2)
        crop_info = self.cropper.crop_source_image(img_rgb, self.cropper.crop_cfg)
        if crop_info is None:
            raise RuntimeError(f"LivePortrait found no face in {path}")
        I_s = self.wrapper.prepare_source(crop_info["img_crop_256x256"])
        info = self.wrapper.get_kp_info(I_s)
        return {
            "img_rgb": img_rgb,
            "info": info,
            "f_s": self.wrapper.extract_feature_3d(I_s),
            "x_s": self.wrapper.transform_keypoint(info),
            "M_c2o": crop_info["M_c2o"],
            "mask": prepare_paste_back(self.inference_cfg.mask_crop, crop_info["M_c2o"],
                                       dsize=(img_rgb.shape[1], img_rgb.shape[0])),
            "pose": {k: float(info[k].item()) for k in ("pitch", "yaw", "roll")},
        }

    @torch.no_grad()
    def render(self, src: dict, pitch: float, yaw: float) -> np.ndarray:
        from src.utils.camera import get_rotation_matrix
        from src.utils.crop import paste_back

        info = src["info"]
        dev = self.wrapper.device
        as_t = lambda v: torch.tensor([[v]], dtype=torch.float32, device=dev)  # noqa: E731
        R_d = get_rotation_matrix(as_t(pitch), as_t(yaw), info["roll"].to(dev))
        x_d = info["kp"].to(dev) @ R_d + info["exp"].to(dev)
        x_d = x_d * info["scale"].to(dev)[..., None]
        x_d[:, :, 0:2] += info["t"].to(dev)[:, None, 0:2]
        x_d = self.wrapper.stitching(src["x_s"], x_d)
        out = self.wrapper.warp_decode(src["f_s"], src["x_s"], x_d)
        crop = self.wrapper.parse_output(out["out"])[0]
        full = paste_back(crop, src["M_c2o"], src["img_rgb"], src["mask"])
        return cv2.cvtColor(full, cv2.COLOR_RGB2BGR)


def main() -> int:
    args = job_args()
    job = load_job(args.job)
    editor = LivePortraitEditor()
    sources: dict[str, dict] = {}
    for frame in job["frames"]:
        path = frame["source"]
        if path not in sources:
            log(f"preparing {path}")
            sources[path] = editor.prepare(path)
        src = sources[path]
        ref = job["source_poses"].get(path)
        if ref is None:
            # No 3DDFA-V2 pose for this photo: treat LivePortrait's own estimate as the reference.
            ref = {"yaw": src["pose"]["yaw"] * YAW_SIGN, "pitch": src["pose"]["pitch"] * PITCH_SIGN}
        yaw = src["pose"]["yaw"] + YAW_SIGN * (frame["yaw"] - ref["yaw"])
        pitch = src["pose"]["pitch"] + PITCH_SIGN * (frame["pitch"] - ref["pitch"])
        save_frame(args.out, frame["index"], editor.render(src, pitch, yaw))
    save_meta(args.out, {"liveportrait_source_poses": {p: s["pose"] for p, s in sources.items()},
                         "yaw_sign": YAW_SIGN, "pitch_sign": PITCH_SIGN})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
