#!/usr/bin/env bash
# Pick an interpreter for run_pose_synth.py inside the container:
# needs cv2 (+ torch when --kp-detector is used). Prefer image host venv, then 3ddfa.
set -euo pipefail
need_torch=0
for a in "$@"; do
    case "$a" in
        --kp-detector|--keypoints) need_torch=1 ;;
        face_alignment|insightface|hrnet) need_torch=1 ;;
    esac
done

candidates=(
    /opt/ps_host/bin/python
    /workspace/liveness/pose_synth/third_party/.venv_3ddfa/bin/python
)
chosen=""
for py in "${candidates[@]}"; do
    [ -x "$py" ] || continue
    "$py" -c "import cv2, numpy" >/dev/null 2>&1 || continue
    if [ "$need_torch" = 1 ]; then
        "$py" -c "import torch" >/dev/null 2>&1 || continue
    fi
    chosen="$py"
    break
done

if [ -z "$chosen" ]; then
    echo "no orchestrator python with cv2$([ "$need_torch" = 1 ] && echo '+torch'). Try:" >&2
    echo "  bash envs/docker_run.sh bash envs/install_3ddfa.sh" >&2
    echo "  or rebuild the image: docker build -t pose_synth:cu124 ." >&2
    exit 1
fi
echo "orchestrator: $chosen ($("$chosen" -V 2>&1))" >&2
exec "$chosen" "$@"
