import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from landmark_wing.scheme import BASE_11_NAMES, KP_NAMES, NUM_BASE_11  # noqa: E402
from ps.keypoints import scheme_names  # noqa: E402


class SchemeTest(unittest.TestCase):
    def test_base11(self):
        self.assertEqual(NUM_BASE_11, 11)
        self.assertEqual(BASE_11_NAMES[0], "eye_left_outer")
        self.assertEqual(BASE_11_NAMES[-1], "pupil_right")
        self.assertEqual(scheme_names("base11"), list(BASE_11_NAMES))
        self.assertEqual(len(scheme_names("all30")), len(KP_NAMES))


if __name__ == "__main__":
    unittest.main()
