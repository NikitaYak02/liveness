import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "adapters"))

from tddfa_geometry import (  # noqa: E402
    angle2matrix,
    facing_camera,
    fill_hidden_colors,
    matrix2angle,
    pose_degrees,
    visible_vertices,
    with_pose,
)


class EulerTest(unittest.TestCase):
    def test_round_trip(self):
        for yaw, pitch, roll in [(0, 0, 0), (60, -20, 5), (-75, 25, -10), (30, 10, 0)]:
            R = angle2matrix(*np.radians([yaw, pitch, roll]))
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
            got = np.degrees(matrix2angle(R))
            np.testing.assert_allclose(got, [yaw, pitch, roll], atol=1e-6)

    def test_with_pose_sets_requested_angles_and_keeps_scale(self):
        R = angle2matrix(*np.radians([10, 5, 3]))
        s, t = 0.0012, np.array([60.0, 55.0, 40.0])
        param = np.concatenate([np.concatenate([s * R, t[:, None]], 1).reshape(-1), np.zeros(50)]).astype(np.float32)
        out = with_pose(param, -70, 20, 3)
        pose = pose_degrees(out)
        self.assertAlmostEqual(pose["yaw"], -70, places=3)
        self.assertAlmostEqual(pose["pitch"], 20, places=3)
        self.assertAlmostEqual(pose["roll"], 3, places=3)
        self.assertAlmostEqual(np.linalg.norm(out[:3]), s, places=6)
        np.testing.assert_array_equal(out[12:], param[12:])

    def test_rotation_about_center_keeps_center_fixed(self):
        R = angle2matrix(*np.radians([0, 0, 0]))
        s, t = 2.0, np.array([1.0, 2.0, 3.0])
        param = np.concatenate([np.concatenate([s * R, t[:, None]], 1).reshape(-1), np.zeros(50)])
        c = np.array([5.0, -3.0, 7.0])
        out = with_pose(param, 45, 10, 0, center=c)
        P_old, P_new = param[:12].reshape(3, 4), out[:12].reshape(3, 4)
        np.testing.assert_allclose(P_old[:, :3] @ c + P_old[:, 3], P_new[:, :3] @ c + P_new[:, 3], atol=1e-4)


class VisibilityTest(unittest.TestCase):
    def test_front_point_hides_back_point(self):
        ver = np.array([[10.0, 10.0, 30.0], [10.0, 10.0, 40.0], [5.0, 50.0, 20.0]]).T
        vis = visible_vertices(ver, (64, 64), tolerance=1.0, cell=1)
        self.assertEqual(vis.tolist(), [False, True, True])

    def test_facing_camera_drops_grazing_and_handles_flipped_normals(self):
        normals = np.array([[0, 0, 1.0], [1.0, 0, 0.1], [0, 0, -1.0]])
        visible = np.array([True, True, False])
        self.assertEqual(facing_camera(normals, visible).tolist(), [True, False, False])
        self.assertEqual(facing_camera(-normals, visible).tolist(), [True, False, False])

    def test_hidden_colors_from_mirror(self):
        colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
        visible = np.array([True, False, False])
        mirror = np.array([1, 0, 1])
        out = fill_hidden_colors(colors, visible, mirror)
        np.testing.assert_array_equal(out[1], [1, 0, 0])
        np.testing.assert_array_equal(out[2], [1, 0, 0])


if __name__ == "__main__":
    unittest.main()
