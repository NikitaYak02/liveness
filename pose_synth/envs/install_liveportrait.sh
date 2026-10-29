#!/usr/bin/env bash
# LivePortrait (humans only): implicit-keypoint warping of the nearest source photo.
source "$(dirname "$0")/common.sh"

clone_repo LivePortrait https://github.com/KlingAIResearch/LivePortrait.git 9b294b3d0536135442ea73cb01e6cb3ca7029dd3
make_env liveportrait 3.10
# cu121 wheels; on older drivers try cu118 via PS_TORCH_INDEX.
TORCH_INDEX="${PS_TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
pip_install torch==2.3.0 torchvision==0.18.0 --index-url "$TORCH_INDEX"
cd "$THIRD_PARTY/LivePortrait"
pip_install -r requirements.txt huggingface_hub
# InsightFace only needs ONNX Runtime for the detector. onnxruntime-gpu often
# breaks on hosts whose system CUDA (e.g. 13.x) does not match the ORT build;
# CPU ORT is enough and keeps torch on the GPU.
uv pip uninstall -y onnxruntime-gpu onnxruntime >/dev/null 2>&1 || true
pip_install onnxruntime==1.18.1
python -c "from huggingface_hub import snapshot_download as d; \
d('KwaiVGI/LivePortrait', local_dir='pretrained_weights', allow_patterns=['liveportrait/*', 'insightface/*'])"
echo "liveportrait ready -> $THIRD_PARTY/.venv_liveportrait/bin/python"
