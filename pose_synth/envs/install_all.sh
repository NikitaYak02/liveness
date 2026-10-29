#!/usr/bin/env bash
# Installs every method into third_party/.venv_<method> (uv or python -m venv).
# A failing method does not stop the others.
#
# Docker note: envs/docker_run.sh mounts a named volume over third_party/. If that
# volume was seeded from host venvs, binaries are broken inside the container —
# make_env recreates them automatically. To wipe everything first:
#   PS_FORCE_RECREATE=1 bash envs/docker_run.sh bash envs/install_all.sh
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
