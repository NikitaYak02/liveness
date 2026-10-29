import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import run_pose_synth  # noqa: E402

FAKE = f"{sys.executable} {ROOT / 'tests' / 'fake_adapter.py'}"


class OrchestratorTest(unittest.TestCase):
    def test_end_to_end_with_fake_adapters(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            inp = tmp / "person"
            inp.mkdir()
            for name in ("a.png", "b.png"):
                cv2.imwrite(str(inp / name), np.zeros((32, 32, 3), np.uint8))
            out = tmp / "out"

            code = run_pose_synth.main([
                "-i", str(inp), "-o", str(out),
                "--methods", "3ddfa", "liveportrait", "gpavatar",
                "--mode", "yaw", "--yaw", "-20", "20", "--step", "10",
                "--gpus", "0,1",
                "--python", f"3ddfa={FAKE}",
                "--python", f"liveportrait={FAKE}",
                "--python", f"gpavatar={FAKE}",
            ])

            self.assertEqual(code, 1, "a failed method must make the run exit non-zero")
            self.assertTrue((out / "compare.mp4").stat().st_size > 0)
            self.assertTrue((out / "liveportrait" / "video.mp4").exists())
            self.assertIn("exit code 3", (out / "gpavatar" / "error.txt").read_text())

            sources = json.loads((out / "sources.json").read_text())
            self.assertEqual(len(sources), 2)

            job = json.loads((out / "liveportrait" / "job.json").read_text())
            self.assertEqual([f["yaw"] for f in job["frames"]], [0, -10, -20, -10, 0, 10, 20, 10, 0])

            records = json.loads((out / "3ddfa" / "poses.json").read_text())
            self.assertTrue(all(r["present"] and r["measured"] for r in records))

            summary = json.loads((out / "summary.json").read_text())
            self.assertIn("error", summary["gpavatar"])
            self.assertEqual(summary["liveportrait"]["measured"], len(records))


if __name__ == "__main__":
    unittest.main()
