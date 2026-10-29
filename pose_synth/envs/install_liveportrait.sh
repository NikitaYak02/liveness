#!/usr/bin/env bash
# LivePortrait (humans only): implicit-keypoint warping of the nearest source photo.
source "$(dirname "$0")/common.sh"

clone_repo LivePortrait https://github.com/KlingAIResearch/LivePortrait.git 9b294b3d0536135442ea73cb01e6cb3ca7029dd3
make_env ps_liveportrait 3.10
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
cd "$THIRD_PARTY/LivePortrait"
pip install -r requirements.txt huggingface_hub
python -c "from huggingface_hub import snapshot_download as d; \
d('KwaiVGI/LivePortrait', local_dir='pretrained_weights', allow_patterns=['liveportrait/*', 'insightface/*'])"
echo "liveportrait ready (env ps_liveportrait)"
