import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from landmark_wing.scheme import BASE_11_NAMES, KP_NAMES, NUM_BASE_11  # noqa: E402
from ps.keypoints import DETECTORS, _fill_pupils_from_68, named_to_payload, scheme_names  # noqa: E402


class SchemeTest(unittest.TestCase):
    def test_base11(self):
        self.assertEqual(NUM_BASE_11, 11)
        self.assertEqual(BASE_11_NAMES[0], "eye_left_outer")
        self.assertEqual(BASE_11_NAMES[-1], "pupil_right")
        self.assertEqual(scheme_names("base11"), list(BASE_11_NAMES))
        self.assertEqual(len(scheme_names("all30")), len(KP_NAMES))
        self.assertIn("face_alignment", DETECTORS)

    def test_fan_pupil_fill(self):
        raw = np.zeros((68, 2), np.float32)
        raw[36:42] = np.array([[10, 10], [12, 8], [14, 8], [16, 10], [14, 12], [12, 12]], np.float32)
        raw[42:48] = raw[36:42] + np.array([20, 0], np.float32)
        named: dict = {}
        _fill_pupils_from_68(raw, named)
        self.assertAlmostEqual(named["pupil_right"][0], 13.0)
        payload = named_to_payload(
            {"eye_left_outer": (1.0, 2.0), "pupil_right": named["pupil_right"]},
            "base11",
            np.array([0, 0, 10, 10], np.float32),
        )
        self.assertEqual(payload["landmarks"][0], [1.0, 2.0])
        self.assertIsNone(payload["landmarks"][1])  # missing eye_right_inner


if __name__ == "__main__":
    unittest.main()
