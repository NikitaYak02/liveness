"""Stand-in for a real adapter in orchestrator tests.

Invoked as `python fake_adapter.py <real adapter path> <adapter args>`; the
real adapter path tells which method is being faked.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

adapter = Path(sys.argv[1]).stem
p = argparse.ArgumentParser()
p.add_argument("--job")
p.add_argument("--out")
p.add_argument("--measure")
p.add_argument("--result")
args = p.parse_args(sys.argv[2:])

if args.measure:
    images = json.loads(Path(args.measure).read_text())
    result = {}
    for img in images:
        name = Path(img).stem
        # Generated frames are named by index; pretend the method reached 80% of the target.
        result[img] = {"yaw": float(len(name)), "pitch": 0.0, "roll": 0.0}
    Path(args.result).write_text(json.dumps(result))
    sys.exit(0)

if adapter == "gpavatar":
    print("CUDA out of memory (fake)")
    sys.exit(3)

job = json.loads(Path(args.job).read_text())
frames = Path(args.out) / "frames"
for frame in job["frames"]:
    img = np.full((64, 48, 3), 100 + int(frame["yaw"]), np.uint8)
    cv2.imwrite(str(frames / f"{frame['index']:04d}.png"), img)
