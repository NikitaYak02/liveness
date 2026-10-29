#!/usr/bin/env bash
# GPAvatar: FLAME tracking of 1..N photos + point-based neural head avatar (no conda).
source "$(dirname "$0")/common.sh"

clone_repo GPAvatar https://github.com/xg-chu/GPAvatar.git 3ca7d4419e978646005d4ace796f0d2655bfe71c
make_env gpavatar 3.9
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
TORCH_INDEX="${PS_TORCH_INDEX:-https://download.pytorch.org/whl/cu118}"
pip_install pip setuptools wheel packaging
pip_install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url "$TORCH_INDEX"
pip_install fvcore iopath
# Facebook CDN often times out through uv's cache. Download the wheel ourselves
# (wget/curl retries), then install from the local file.
WHEEL_DIR="$THIRD_PARTY/wheels"
WHEEL="$WHEEL_DIR/pytorch3d-0.7.4-cp39-cp39-linux_x86_64.whl"
WHEEL_URL="https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt201/pytorch3d-0.7.4-cp39-cp39-linux_x86_64.whl"
mkdir -p "$WHEEL_DIR"
# Drop a truncated previous download so download() will refetch.
if [ -f "$WHEEL" ] && ! "$PS_PYTHON" -m zipfile -t "$WHEEL" >/dev/null 2>&1; then
    echo "removing corrupt $WHEEL" >&2
    rm -f "$WHEEL"
fi
p3d_ok=0
if download "$WHEEL_URL" "$WHEEL" && pip_install --no-cache-dir "$WHEEL"; then
    p3d_ok=1
fi
if [ "$p3d_ok" != 1 ]; then
    echo "pytorch3d wheel failed; building from source (slow)..." >&2
    pip_install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.4"
fi
pip_install mediapipe tqdm rich lmdb einops colored ninja av opencv-python scikit-image \
    onnxruntime-gpu onnx transformers pykalman pytorch-lightning==2.1.3
pip_install nerfacc==0.5.3 -f https://nerfacc-bucket.s3.us-west-2.amazonaws.com/whl/torch-2.0.0_cu118.html

cd "$THIRD_PARTY/GPAvatar"
[ -d demos/examples ] || PATH="$(dirname "$PS_PYTHON"):$PATH" bash ./build_resources.sh
download https://github.com/xg-chu/GPAvatar/releases/download/v1.0.0/one_model.ckpt checkpoints/one_model.ckpt
mkdir -p inference_tools/checkpoints
ln -sfn "$PWD/checkpoints/one_model.ckpt" inference_tools/checkpoints/one_model.ckpt
"$PS_PYTHON" -c "import torch, pytorch3d; print('gpavatar imports ok')"
echo "gpavatar ready -> $THIRD_PARTY/.venv_gpavatar/bin/python"
