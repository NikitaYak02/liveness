#!/usr/bin/env bash
# Installs every method into third_party/.venv_<method> (uv or python -m venv).
# A failing method does not stop the others.
cd "$(dirname "$0")"
if ! command -v uv >/dev/null 2>&1; then
    echo "tip: install uv (https://astral.sh/uv) for faster envs; falling back to python -m venv" >&2
fi
failed=()
for m in 3ddfa liveportrait gpavatar facelift dp360; do
    echo "=== $m"
    bash "install_$m.sh" || failed+=("$m")
done
if [ ${#failed[@]} -gt 0 ]; then
    echo "failed: ${failed[*]}" >&2
    exit 1
fi
echo "all methods ready; run_pose_synth.py will pick up third_party/.venv_<method>/bin/python"
