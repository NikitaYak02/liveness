#!/usr/bin/env bash
# GPAvatar: FLAME tracking of 1..N photos + point-based neural head avatar.
source "$(dirname "$0")/common.sh"

clone_repo GPAvatar https://github.com/xg-chu/GPAvatar.git 3ca7d4419e978646005d4ace796f0d2655bfe71c
cd "$THIRD_PARTY/GPAvatar"
if ! conda env list | awk '{print $1}' | grep -qx ps_gpavatar; then
    conda env create -n ps_gpavatar -f environment.yml
fi
conda activate ps_gpavatar
pip install nerfacc==0.5.3 -f https://nerfacc-bucket.s3.us-west-2.amazonaws.com/whl/torch-2.0.0_cu118.html

# FLAME / EMOCA / MICA / matting assets.
[ -d demos/examples ] || bash ./build_resources.sh
download https://github.com/xg-chu/GPAvatar/releases/download/v1.0.0/one_model.ckpt checkpoints/one_model.ckpt
# inference_tools resolve the checkpoint relative to their own folder.
mkdir -p inference_tools/checkpoints
ln -sf "$PWD/checkpoints/one_model.ckpt" inference_tools/checkpoints/one_model.ckpt
echo "gpavatar ready (env ps_gpavatar)"
