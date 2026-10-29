#!/usr/bin/env bash
# FaceLift: 6-view diffusion + GS-LRM Gaussian head. Weights auto-download from HF on first run.
source "$(dirname "$0")/common.sh"

clone_repo FaceLift https://github.com/weijielyu/FaceLift.git 0a9420c1480db1e22daa6dee192b76a013295992
make_env facelift 3.11
command -v ffmpeg >/dev/null || echo "warning: ffmpeg not found (FaceLift's own video export needs it)" >&2
TORCH_INDEX="${PS_TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
pip_install packaging==24.2 typing-extensions==4.14.0
pip_install torch==2.4.0 torchvision==0.19.0 --index-url "$TORCH_INDEX"
pip_install transformers==4.44.2 "diffusers[torch]==0.30.3" huggingface-hub==0.35.3 xformers==0.0.27.post2 \
    accelerate==0.33.0 Pillow==10.4.0 opencv-python==4.10.0.84 scikit-image==0.21.0 lpips==0.1.4
pip_install facenet-pytorch --no-deps
pip_install rembg numpy==1.26.4 matplotlib==3.7.5 scikit-learn==1.3.2 einops==0.8.0 jaxtyping==0.2.19 \
    pytorch-msssim==1.0.0 easydict==1.13 pyyaml==6.0.2 termcolor==2.4.0 plyfile==1.0.3 tqdm \
    videoio==0.3.0 ffmpeg-python==0.2.0
pip_install --no-build-isolation git+https://github.com/graphdeco-inria/diff-gaussian-rasterization
echo "facelift ready -> $THIRD_PARTY/.venv_facelift/bin/python"
