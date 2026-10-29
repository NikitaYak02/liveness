#!/usr/bin/env bash
# Minimal deps for the orchestrator itself (not the per-method adapters).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REQ="$ROOT/requirements.txt"
if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$(command -v python3 || command -v python)" -r "$REQ"
else
    python3 -m pip install -r "$REQ"
fi
echo "host ready: run with $(command -v python3 || command -v python) $ROOT/run_pose_synth.py ..."
