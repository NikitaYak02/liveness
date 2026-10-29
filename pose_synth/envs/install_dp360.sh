#!/usr/bin/env bash
# DiffPortrait360 (experimental on 24 GB): PanoHead projection + back-head + video diffusion.
source "$(dirname "$0")/common.sh"

clone_repo Diffportrait360 https://github.com/FreedomGu/Diffportrait360.git 91106e7098d8fbd84a1e397033136322a969c1eb
make_env ps_dp360 3.9
cd "$THIRD_PARTY/Diffportrait360"
pip install -r diffportrait360_release/code/requirements.txt huggingface_hub
python -c "from huggingface_hub import snapshot_download as d; d('gym890/diffportrait360', local_dir='pretrained_weights', \
allow_patterns=['easy-khair-180-gpc0.8-trans10-025000.pkl', 'back_head-230000.th', 'model_state-340000.th'])"
# A fresh install forgets any earlier "does not fit" verdict.
rm -f "$THIRD_PARTY/dp360_status.json"
echo "dp360 ready (env ps_dp360)"
