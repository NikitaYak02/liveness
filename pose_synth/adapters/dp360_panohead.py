"""PanoHead part of the DiffPortrait360 adapter (runs inside DiffPortrait360's 3DNoise folder).

Writes, for one cropped input photo and a list of cameras, one PNG per camera:
  <cond_dir>/0000000.png ...   pose conditions: a fixed PanoHead identity rendered at each camera
                               (same kind of "GAN_Generated" condition as the stock sphere32 set)
  <noise_dir>/0000000.png ...  3D-aware noise: the photo projected into PanoHead (latent
                               projection + pivotal tuning) rendered at the same cameras
The adapter later assembles these into the chunked layout inference.py expects.

Cameras are (horizontal, vertical) angles in radians for LookAtPoseSampler,
frontal = (pi/2, pi/2), radius 2.7, as in projector_withseg.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import PIL.Image
import torch

sys.path.insert(0, os.getcwd())

import dnnlib  # noqa: E402
import legacy  # noqa: E402
from camera_utils import LookAtPoseSampler  # noqa: E402
from projector_withseg import project, project_pti  # noqa: E402

INTRINSICS = [[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]]
CONDITION_SEED = 0


def camera(h: float, v: float, device) -> torch.Tensor:
    pose = LookAtPoseSampler.sample(h, v, torch.tensor([0, 0, 0.0], device=device), radius=2.7, device=device)
    return torch.cat([pose.reshape(-1, 16), torch.tensor(INTRINSICS, device=device).reshape(-1, 9)], 1)


def to_uint8(img: torch.Tensor) -> np.ndarray:
    return ((img.permute(1, 2, 0) + 1) * 127.5).clamp(0, 255).to(torch.uint8).cpu().numpy()


def load_g(network: str, device):
    with dnnlib.util.open_url(network) as fp:
        G = legacy.load_network_pkl(fp)["G_ema"].requires_grad_(False).to(device)
    G.rendering_kwargs["ray_start"] = 2.35
    return G


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--network", required=True)
    p.add_argument("--image", required=True, help="512x512 PanoHead-aligned crop")
    p.add_argument("--input-camera", required=True, help="json [h, v] of the photo")
    p.add_argument("--cameras", required=True, help="json list of [h, v]")
    p.add_argument("--cond-dir", required=True)
    p.add_argument("--noise-dir", required=True)
    p.add_argument("--num-steps", type=int, default=200)
    p.add_argument("--num-steps-pti", type=int, default=50)
    args = p.parse_args()

    device = torch.device("cuda")
    cameras = json.loads(args.cameras)
    h0, v0 = json.loads(args.input_camera)
    os.makedirs(args.cond_dir, exist_ok=True)

    G = load_g(args.network, device)
    with torch.no_grad():
        z = torch.from_numpy(np.random.RandomState(CONDITION_SEED).randn(1, G.z_dim)).to(device)
        ws_cond = G.mapping(z, camera(np.pi / 2, np.pi / 2, device), truncation_psi=0.7)
        for i, (h, v) in enumerate(cameras):
            img = G.synthesis(ws_cond, camera(h, v, device), noise_mode="const")["image"][0]
            PIL.Image.fromarray(to_uint8(img)).save(os.path.join(args.cond_dir, f"{i:07d}.png"))

    target = PIL.Image.open(args.image).convert("RGB").resize((G.img_resolution, G.img_resolution),
                                                              PIL.Image.LANCZOS)
    target = torch.tensor(np.array(target, dtype=np.uint8).transpose([2, 0, 1]), device=device)
    c0 = camera(h0, v0, device)
    w_steps, c_adj = project(G, target=target, c=c0.clone(), num_steps=args.num_steps, device=device, verbose=True)
    G_tuned = project_pti(G, target=target, w_pivot=w_steps[-1:], c=c_adj, num_steps=args.num_steps_pti,
                          device=device, verbose=True)[-1].to(device)
    ws = w_steps[-1].unsqueeze(0).to(device)

    os.makedirs(args.noise_dir, exist_ok=True)
    with torch.no_grad():
        for i, (h, v) in enumerate(cameras):
            img = G_tuned.synthesis(ws, camera(h, v, device), noise_mode="const")["image"][0]
            PIL.Image.fromarray(to_uint8(img)).save(os.path.join(args.noise_dir, f"{i:07d}.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
