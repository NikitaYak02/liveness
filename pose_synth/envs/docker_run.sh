#!/usr/bin/env bash
# Thin wrapper: run a command inside the CUDA 12.4 pose_synth container.
# Examples:
#   bash envs/docker_run.sh bash envs/install_all.sh
#   bash envs/docker_run.sh python run_pose_synth.py -i /data/frames -o /out/r1 --gpus 0,1
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if ! docker compose version >/dev/null 2>&1; then
    echo "need docker compose v2" >&2
    exit 1
fi
if ! docker image inspect pose_synth:cu124 >/dev/null 2>&1; then
    docker compose build
fi
mkdir -p data out
exec docker compose run --rm --gpus all pose_synth "$@"
