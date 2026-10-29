"""Runs DiffPortrait360's inference.py (from its code/ folder), optionally in fp16.

With --half the model weights are cast to fp16 and the whole run happens under
CUDA autocast, so fp32 tensors built by the dataset are cast on the fly. All
other arguments go to inference.py unchanged.

Also ensures openai/clip-vit-large-patch14 is available: a relative empty
``openai/...`` folder (from a failed download) makes transformers treat the
Hub id as a local path and then fail looking for pytorch_model.bin.
"""

from __future__ import annotations

import os
import runpy
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.getcwd())

CLIP_ID = "openai/clip-vit-large-patch14"


def _clip_ready(path: Path) -> bool:
    if not path.is_dir():
        return False
    return bool(list(path.glob("model.safetensors"))
                or list(path.glob("pytorch_model.bin"))
                or list(path.glob("*.safetensors")))


def ensure_clip() -> Path:
    weights = Path(os.environ.get(
        "PS_DP360_WEIGHTS",
        Path(__file__).resolve().parents[1] / "third_party" / "Diffportrait360" / "pretrained_weights",
    ))
    clip_dir = weights / "clip-vit-large-patch14"

    # Transformers resolves Hub ids relative to cwd first; a broken local tree wins.
    shadow = Path("openai") / "clip-vit-large-patch14"
    if shadow.exists() and not _clip_ready(shadow):
        shutil.rmtree(shadow, ignore_errors=True)
        parent = shadow.parent
        if parent.is_dir() and not any(parent.iterdir()):
            try:
                parent.rmdir()
            except OSError:
                pass

    from huggingface_hub import snapshot_download

    if not _clip_ready(clip_dir):
        weights.mkdir(parents=True, exist_ok=True)
        snapshot_download(CLIP_ID, local_dir=str(clip_dir))
    # Also populate the default HF cache so bare from_pretrained(CLIP_ID) works.
    snapshot_download(CLIP_ID)

    if shadow.exists() and not _clip_ready(shadow):
        shutil.rmtree(shadow, ignore_errors=True)
    if not shadow.exists() and _clip_ready(clip_dir):
        shadow.parent.mkdir(parents=True, exist_ok=True)
        try:
            shadow.symlink_to(clip_dir.resolve())
        except OSError:
            shutil.copytree(clip_dir, shadow)

    if not (_clip_ready(clip_dir) or _clip_ready(shadow)):
        raise SystemExit(
            f"CLIP weights missing (tried {clip_dir}). "
            "Re-run envs/install_dp360.sh with network access."
        )
    return clip_dir if _clip_ready(clip_dir) else shadow.resolve()


half = "--half" in sys.argv
if half:
    sys.argv.remove("--half")
sys.argv[0] = "inference.py"

ensure_clip()

if half:
    import model_lib.ControlNet.cldm.model as cldm_model

    _create_model = cldm_model.create_model
    cldm_model.create_model = lambda *a, **kw: _create_model(*a, **kw).half()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        runpy.run_path("inference.py", run_name="__main__")
else:
    runpy.run_path("inference.py", run_name="__main__")
