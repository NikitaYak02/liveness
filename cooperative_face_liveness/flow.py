from __future__ import annotations

import torch
import torch.nn.functional as F


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def build_raft_model(
    device: str = "cuda",
    variant: str = "large",
    weights_name: str = "DEFAULT",
) -> torch.nn.Module:
    from torchvision.models.optical_flow import (
        Raft_Large_Weights,
        Raft_Small_Weights,
        raft_large,
        raft_small,
    )

    variant = variant.lower()
    if variant == "large":
        weights = None if weights_name.lower() == "none" else getattr(Raft_Large_Weights, weights_name)
        model = raft_large(weights=weights, progress=False)
    elif variant == "small":
        weights = None if weights_name.lower() == "none" else getattr(Raft_Small_Weights, weights_name)
        model = raft_small(weights=weights, progress=False)
    else:
        raise ValueError(f"Unsupported RAFT variant: {variant}")

    return model.to(device).eval()


def rgb_frame_to_tensor(frame_rgb) -> torch.Tensor:
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().div(255.0)
    mean = IMAGENET_MEAN.to(dtype=tensor.dtype)
    std = IMAGENET_STD.to(dtype=tensor.dtype)
    return (tensor - mean) / std


@torch.inference_mode()
def compute_clipped_flow_magnitude(
    frame1_rgb,
    frame3_rgb,
    raft_model: torch.nn.Module,
    device: str = "cuda",
    flow_size: int = 256,
    num_flow_updates: int = 3,
    clip_fraction: float = 0.20,
) -> torch.Tensor:
    image1 = torch.from_numpy(frame1_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    image2 = torch.from_numpy(frame3_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    if flow_size != frame1_rgb.shape[0]:
        image1 = F.interpolate(image1, size=(flow_size, flow_size), mode="bilinear", align_corners=False)
        image2 = F.interpolate(image2, size=(flow_size, flow_size), mode="bilinear", align_corners=False)

    image1 = image1.mul(2.0).sub(1.0).to(device)
    image2 = image2.mul(2.0).sub(1.0).to(device)

    try:
        list_of_flows = raft_model(image1, image2, num_flow_updates=num_flow_updates)
    except TypeError:
        list_of_flows = raft_model(image1, image2)
    flow = list_of_flows[-1]
    magnitude = torch.linalg.vector_norm(flow, ord=2, dim=1, keepdim=True)

    clip_value = clip_fraction * float(flow.shape[-1])
    magnitude = magnitude.clamp(max=clip_value).div(clip_value)

    if magnitude.shape[-1] != frame1_rgb.shape[0]:
        magnitude = F.interpolate(
            magnitude,
            size=(frame1_rgb.shape[0], frame1_rgb.shape[1]),
            mode="bilinear",
            align_corners=False,
        )
    return magnitude[0].cpu()
