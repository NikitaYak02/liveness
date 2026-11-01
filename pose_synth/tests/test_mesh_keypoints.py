import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ps.keypoints import payload_from_points68, resolve_kp_mode  # noqa: E402


class MeshKeypointsTest(unittest.TestCase):
    def test_resolve_auto(self):
        self.assertEqual(resolve_kp_mode("liveportrait", "auto"), "warp")
        self.assertEqual(resolve_kp_mode("facelift", "auto"), "mesh")
        self.assertEqual(resolve_kp_mode("3ddfa", "auto"), "mesh")
        self.assertEqual(resolve_kp_mode("liveportrait", "mesh"), "mesh")

    def test_payload_from_points68(self):
        points = [{"x": float(i), "y": float(i + 1), "visible": True} for i in range(68)]
        payload = payload_from_points68(points, "base11")
        self.assertIsNotNone(payload)
        self.assertEqual(len(payload["landmarks"]), 11)
        self.assertIn("bbox_xyxy", payload)
        # eye_left_outer comes from 300w index 45
        self.assertEqual(payload["landmarks"][0], [45.0, 46.0])


if __name__ == "__main__":
    unittest.main()
