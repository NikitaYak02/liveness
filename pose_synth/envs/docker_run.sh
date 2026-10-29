#!/usr/bin/env bash
# Thin wrapper: run a command inside the CUDA 12.4 pose_synth container.
# Examples:
#   bash envs/docker_run.sh bash envs/install_all.sh
#   bash envs/docker_run.sh python run_pose_synth.py -i /data/frames -o /out/r1 --gpus 0,1
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

compose() {
    if docker compose version >/dev/null 2>&1; then
        docker compose "$@"
    elif command -v docker-compose >/dev/null 2>&1; then
        docker-compose "$@"
    else
        echo "Docker Compose not found." >&2
        echo "On Arch:" >&2
        echo "  sudo pacman -S docker docker-compose nvidia-container-toolkit" >&2
        echo "  sudo systemctl enable --now docker" >&2
        echo "  sudo usermod -aG docker \$USER   # then re-login" >&2
        echo "  sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker" >&2
        exit 1
    fi
}

if ! command -v docker >/dev/null 2>&1; then
    echo "docker not found; install: sudo pacman -S docker docker-compose nvidia-container-toolkit" >&2
    exit 1
fi

if ! docker image inspect pose_synth:cu124 >/dev/null 2>&1; then
    compose build
fi
mkdir -p data out
# --gpus is a `docker run` flag; compose v2 passes it via the compose file `gpus: all`.
exec compose run --rm pose_synth "$@"
