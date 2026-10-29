#!/usr/bin/env bash
# DiffPortrait360 (experimental on 24 GB): PanoHead projection + back-head + video diffusion.
source "$(dirname "$0")/common.sh"

clone_repo Diffportrait360 https://github.com/FreedomGu/Diffportrait360.git 91106e7098d8fbd84a1e397033136322a969c1eb
make_env dp360 3.9
cd "$THIRD_PARTY/Diffportrait360"
REQ=diffportrait360_release/code/requirements.txt
# dlib 19.24.6 fails to build on GCC 16 (std::ranges name clash). Our adapter does
# its own crop/pose, so skip dlib + face-alignment; install the rest.
FILTERED="$(mktemp)"
grep -viE '^(dlib|face-alignment)==' "$REQ" > "$FILTERED"
pip_install -r "$FILTERED" huggingface_hub
rm -f "$FILTERED"
# Adapter imports cv2 at top-level; pin explicitly in case requirements drift.
# setuptools provides pkg_resources (still imported by older transformers/CLIP).
# safetensors: Hub CLIP no longer ships pytorch_model.bin for all variants.
pip_install "opencv-python-headless>=4.7,<4.11" "setuptools<81" safetensors
"$PS_PYTHON" - <<'PY'
import cv2, torch, pkg_resources, safetensors
from pathlib import Path
from huggingface_hub import snapshot_download as d
root = Path("pretrained_weights")
d("gym890/diffportrait360", local_dir=str(root),
  allow_patterns=["easy-khair-180-gpc0.8-trans10-025000.pkl", "back_head-230000.th", "model_state-340000.th"])
d("openai/clip-vit-large-patch14", local_dir=str(root / "clip-vit-large-patch14"))
d("openai/clip-vit-large-patch14")  # HF cache for from_pretrained(id)
print("dp360 imports ok", cv2.__version__)
PY
rm -f "$THIRD_PARTY/dp360_status.json"
echo "dp360 ready -> $THIRD_PARTY/.venv_dp360/bin/python"
