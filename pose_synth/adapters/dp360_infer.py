"""Runs DiffPortrait360's inference.py (from its code/ folder), optionally in fp16.

With --half the model weights are cast to fp16 and the whole run happens under
CUDA autocast, so fp32 tensors built by the dataset are cast on the fly. All
other arguments go to inference.py unchanged.
"""

from __future__ import annotations

import os
import runpy
import sys

import torch

sys.path.insert(0, os.getcwd())

half = "--half" in sys.argv
if half:
    sys.argv.remove("--half")
sys.argv[0] = "inference.py"

if half:
    import model_lib.ControlNet.cldm.model as cldm_model

    _create_model = cldm_model.create_model
    cldm_model.create_model = lambda *a, **kw: _create_model(*a, **kw).half()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        runpy.run_path("inference.py", run_name="__main__")
else:
    runpy.run_path("inference.py", run_name="__main__")
