#!/usr/bin/env bash
# FaceLift: 6-view diffusion + GS-LRM Gaussian head. Weights auto-download from HF on first run.
source "$(dirname "$0")/common.sh"

clone_repo FaceLift https://github.com/weijielyu/FaceLift.git 0a9420c1480db1e22daa6dee192b76a013295992
make_env facelift 3.11
command -v ffmpeg >/dev/null || echo "warning: ffmpeg not found (FaceLift's own video export needs it)" >&2
TORCH_INDEX="${PS_TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
pip_install packaging==24.2 typing-extensions==4.14.0 setuptools wheel ninja
pip_install torch==2.4.0 torchvision==0.19.0 --index-url "$TORCH_INDEX"
pip_install transformers==4.44.2 "diffusers[torch]==0.30.3" huggingface-hub==0.35.3 xformers==0.0.27.post2 \
    accelerate==0.33.0 Pillow==10.4.0 opencv-python==4.10.0.84 scikit-image==0.21.0 lpips==0.1.4
pip_install facenet-pytorch --no-deps
pip_install rembg numpy==1.26.4 matplotlib==3.7.5 scikit-learn==1.3.2 einops==0.8.0 jaxtyping==0.2.19 \
    pytorch-msssim==1.0.0 easydict==1.13 pyyaml==6.0.2 termcolor==2.4.0 plyfile==1.0.3 tqdm \
    videoio==0.3.0 ffmpeg-python==0.2.0 rich

# Arch/rolling: system CUDA may be newer than the torch wheel; skip the strict check.
"$PS_PYTHON" - <<'PY'
from pathlib import Path
import torch.utils.cpp_extension as ext
path = Path(ext.__file__)
text = path.read_text()
needle = "raise RuntimeError(CUDA_MISMATCH_MESSAGE.format(cuda_str_version, torch.version.cuda))"
if needle in text and "PS_ALLOW_CUDA_MISMATCH" not in text:
    path.write_text(text.replace(
        needle,
        "import os as _os\n"
        "        if _os.environ.get('PS_ALLOW_CUDA_MISMATCH') != '1':\n"
        "            " + needle + "\n"
        "        return",
    ))
    print(f"patched {path} for PS_ALLOW_CUDA_MISMATCH")
PY
export PS_ALLOW_CUDA_MISMATCH=1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9;9.0}"

# CUDA 12.4 nvcc rejects host GCC > 13. Arch's system g++ 15/16 is too new;
# micromamba g++ 14 (old cxx14 path) is also too new. Prefer <=13.
gcc_major() {
    local bin="$1" ver
    ver="$("$bin" -dumpfullversion 2>/dev/null || "$bin" -dumpversion 2>/dev/null || echo "")"
    echo "${ver%%.*}"
}

cxx_usable_for_cuda() {
    local bin="$1" major
    [ -x "$bin" ] || return 1
    case "$bin" in
        *clang++*) return 0 ;;
    esac
    major="$(gcc_major "$bin")"
    [ -n "$major" ] || return 1
    # Torch 2.4 + CUDA 12.4: GCC 9..13.
    [ "$major" -ge 9 ] && [ "$major" -le 13 ]
}

pick_host_cxx() {
    if [ -n "${PS_CUDAHOSTCXX:-}" ] && [ -x "${PS_CUDAHOSTCXX}" ]; then
        echo "$PS_CUDAHOSTCXX"
        return
    fi
    local cand path
    # Docker CUDA images usually ship a compatible plain g++; Arch needs g++-13.
    for cand in g++-13 g++-12 g++-11 g++ clang++; do
        path="$(command -v "$cand" 2>/dev/null || true)"
        if [ -n "$path" ] && cxx_usable_for_cuda "$path"; then
            echo "$path"
            return
        fi
    done
    local prefix="$THIRD_PARTY/cxx13"
    local mamba_cxx="$prefix/bin/x86_64-conda-linux-gnu-g++"
    if cxx_usable_for_cuda "$mamba_cxx"; then
        echo "$mamba_cxx"
        return
    fi
    echo "no GCC 9..13 / clang++; installing user-local g++ 13 via micromamba..." >&2
    local micromamba="$THIRD_PARTY/bin/micromamba"
    mkdir -p "$THIRD_PARTY/bin"
    if [ ! -x "$micromamba" ]; then
        curl -L "https://micro.mamba.pm/api/micromamba/linux-64/latest" | tar -xvj -C "$THIRD_PARTY/bin" --strip-components=1 bin/micromamba
    fi
    # Drop stale gcc14 toolchain if present — nvcc 12.4 refuses it.
    rm -rf "$THIRD_PARTY/cxx14"
    "$micromamba" create -y -p "$prefix" -c conda-forge "gxx_linux-64=13.*" "gcc_linux-64=13.*" cxx-compiler
    if cxx_usable_for_cuda "$mamba_cxx"; then
        echo "$mamba_cxx"
        return
    fi
    return 1
}

HOST_CXX="$(pick_host_cxx)" || {
    echo "FaceLift needs a host GCC <= 13 (CUDA 12.4) or clang++." >&2
    echo "Options:" >&2
    echo "  1) AUR/extra: install gcc13 and re-run" >&2
    echo "  2) sudo pacman -S clang && re-run" >&2
    echo "  3) set PS_CUDAHOSTCXX=/path/to/g++-13" >&2
    exit 1
}
export CUDAHOSTCXX="$HOST_CXX"
export CXX="$HOST_CXX"
case "$HOST_CXX" in
    *clang++*) export CC="${HOST_CXX/clang++/clang}" ;;
    *g++*)
        # conda compilers are named x86_64-conda-linux-gnu-g++; swap g++ -> gcc.
        export CC="${HOST_CXX/g++/gcc}"
        ;;
    *) export CC="$HOST_CXX" ;;
esac
# Ensure matching conda sysroot libs are findable when using micromamba cxx.
case "$HOST_CXX" in
    */cxx13/*)
        export PATH="$(dirname "$HOST_CXX"):$PATH"
        ;;
esac
echo "building rasterizer with host compiler $CUDAHOSTCXX (GCC $(gcc_major "$HOST_CXX"))"
# Clear a failed partial build from the previous gcc14 attempt.
rm -rf "$THIRD_PARTY/diff-gaussian-rasterization/build" \
       "$THIRD_PARTY/diff-gaussian-rasterization"/*.egg-info 2>/dev/null || true

# Upstream rasterizer misses <cstdint> under newer CUDA toolkits; patch a local checkout.
RAST="$THIRD_PARTY/diff-gaussian-rasterization"
if [ ! -d "$RAST/.git" ]; then
    git clone --recursive https://github.com/graphdeco-inria/diff-gaussian-rasterization "$RAST"
fi
git -C "$RAST" checkout -q 59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d || true
HDR="$RAST/cuda_rasterizer/rasterizer_impl.h"
if [ -f "$HDR" ] && ! grep -q '#include <cstdint>' "$HDR"; then
    sed -i '1a #include <cstdint>' "$HDR"
fi

pip_install --no-build-isolation "$RAST"
"$PS_PYTHON" -c "import torch, rich; print('facelift torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "facelift ready -> $THIRD_PARTY/.venv_facelift/bin/python"
