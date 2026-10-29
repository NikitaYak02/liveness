#!/usr/bin/env bash
# Run a command inside the CUDA 12.4 pose_synth container (GPU via docker run).
# Examples:
#   bash envs/docker_run.sh bash envs/install_all.sh
#   bash envs/docker_run.sh python run_pose_synth.py -i /data/frames -o /out/r1 --gpus 0,1
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
IMAGE="${PS_DOCKER_IMAGE:-pose_synth:cu124}"
VOLUME="${PS_DOCKER_VOLUME:-pose_synth_third_party}"

compose_build() {
    if docker compose version >/dev/null 2>&1; then
        docker compose build
    elif command -v docker-compose >/dev/null 2>&1; then
        docker-compose build
    else
        docker build -t "$IMAGE" .
    fi
}

if ! command -v docker >/dev/null 2>&1; then
    echo "docker not found; on Arch: sudo pacman -S docker docker-compose nvidia-container-toolkit" >&2
    exit 1
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    compose_build
fi
mkdir -p data out

# Prefer --gpus; fall back to nvidia runtime if the CLI has no --gpus flag.
GPU_ARGS=()
if docker run --help 2>&1 | grep -q -- '--gpus'; then
    GPU_ARGS=(--gpus all)
else
    GPU_ARGS=(--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all)
fi

# -i only when stdin is a TTY (so pipes still work).
TTY_ARGS=()
if [ -t 0 ]; then
    TTY_ARGS=(-it)
fi

exec docker run --rm "${TTY_ARGS[@]}" "${GPU_ARGS[@]}" \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e UV_HTTP_TIMEOUT=300 \
    -e PS_ALLOW_CUDA_MISMATCH=1 \
    -v "$ROOT:/workspace/pose_synth" \
    -v "$VOLUME:/workspace/pose_synth/third_party" \
    -v "$ROOT/data:/data" \
    -v "$ROOT/out:/out" \
    -w /workspace/pose_synth \
    "$IMAGE" \
    "$@"
