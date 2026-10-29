import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ps import video  # noqa: E402
from ps.poses import build_trajectory  # noqa: E402
from ps.runner import GpuPool  # noqa: E402
from ps.source_select import most_frontal, nearest_source  # noqa: E402


class TrajectoryTest(unittest.TestCase):
    def test_yaw_sweep_starts_and_ends_frontal(self):
        poses = build_trajectory("yaw", (-60, 60), (-30, 30), 10)
        yaws = [p.yaw for p in poses]
        self.assertEqual(yaws[0], 0)
        self.assertEqual(yaws[-1], 0)
        self.assertEqual(min(yaws), -60)
        self.assertEqual(max(yaws), 60)
        self.assertTrue(all(p.pitch == 0 for p in poses))
        self.assertTrue(all(abs(a - b) <= 10 + 1e-9 for a, b in zip(yaws, yaws[1:])))

    def test_sweep_is_yaw_then_pitch_without_duplicate_frontal(self):
        yaw = build_trajectory("yaw", (-20, 20), (-10, 10), 10)
        pitch = build_trajectory("pitch", (-20, 20), (-10, 10), 10)
        sweep = build_trajectory("sweep", (-20, 20), (-10, 10), 10)
        self.assertEqual(len(sweep), len(yaw) + len(pitch) - 1)
        self.assertEqual(sweep[len(yaw) - 1], sweep[len(yaw) - 1].__class__(0.0, 0.0))

    def test_grid_is_serpentine(self):
        poses = build_trajectory("grid", (-10, 10), (-10, 10), 10)
        self.assertEqual(len(poses), 9)
        self.assertEqual([p.yaw for p in poses[:3]], [-10, 0, 10])
        self.assertEqual([p.yaw for p in poses[3:6]], [10, 0, -10])

    def test_uneven_step_still_reaches_limits(self):
        poses = build_trajectory("yaw", (-25, 25), (0, 0), 10)
        self.assertIn(-25.0, [p.yaw for p in poses])
        self.assertIn(25.0, [p.yaw for p in poses])

    def test_bad_mode(self):
        with self.assertRaises(ValueError):
            build_trajectory("spiral")


class SourceSelectTest(unittest.TestCase):
    images = ["front.jpg", "left.jpg", "right.jpg"]
    poses = {
        "front.jpg": {"yaw": 3.0, "pitch": 5.0},
        "left.jpg": {"yaw": -45.0, "pitch": 0.0},
        "right.jpg": {"yaw": 40.0, "pitch": -5.0},
    }

    def test_nearest_by_yaw(self):
        self.assertEqual(nearest_source({"yaw": -70, "pitch": 0}, self.images, self.poses), "left.jpg")
        self.assertEqual(nearest_source({"yaw": 60, "pitch": 20}, self.images, self.poses), "right.jpg")
        self.assertEqual(nearest_source({"yaw": 0, "pitch": -30}, self.images, self.poses), "front.jpg")

    def test_most_frontal(self):
        self.assertEqual(most_frontal(self.images, self.poses), "front.jpg")

    def test_unknown_poses_fall_back_to_first_image(self):
        unknown = {img: None for img in self.images}
        self.assertEqual(nearest_source({"yaw": 50, "pitch": 0}, self.images, unknown), "front.jpg")
        self.assertEqual(most_frontal(self.images, unknown), "front.jpg")

    def test_images_without_pose_are_ignored(self):
        poses = dict(self.poses, **{"right.jpg": None})
        self.assertEqual(nearest_source({"yaw": 60, "pitch": 0}, self.images, poses), "front.jpg")


class VideoTest(unittest.TestCase):
    def test_fit_tile_letterboxes(self):
        tile = video.fit_tile(np.full((100, 200, 3), 255, np.uint8), 64)
        self.assertEqual(tile.shape, (64, 64, 3))
        self.assertEqual(tile[0, 32].sum(), 0)
        self.assertEqual(tile[32, 32].sum(), 255 * 3)

    def test_missing_image_is_black(self):
        self.assertEqual(video.fit_tile(None, 32).sum(), 0)

    def test_grid_layout(self):
        self.assertEqual(video.grid_shape(5), (2, 3))
        self.assertEqual(video.grid_shape(4), (2, 2))
        tiles = [np.full((8, 8, 3), i, np.uint8) for i in range(5)]
        grid = video.compose_grid(tiles)
        self.assertEqual(grid.shape, (16, 24, 3))
        self.assertEqual(grid[12, 4, 0], 3)

    def test_label_tile_keeps_size(self):
        tile = np.zeros((256, 256, 3), np.uint8)
        out = video.label_tile(tile, "m", {"yaw": 10, "pitch": 0}, None)
        self.assertEqual(out.shape, tile.shape)
        self.assertGreater(out.sum(), 0)


class GpuPoolTest(unittest.TestCase):
    def test_request_larger_than_pool_gets_everything(self):
        pool = GpuPool([0])
        self.assertEqual(pool.acquire(2), [0])
        pool.release([0])
        self.assertEqual(pool.acquire(1), [0])

    def test_release_returns_cards(self):
        pool = GpuPool([0, 1])
        a = pool.acquire(1)
        b = pool.acquire(1)
        self.assertNotEqual(a, b)
        pool.release(a)
        pool.release(b)
        self.assertEqual(pool.acquire(2), [0, 1])


if __name__ == "__main__":
    unittest.main()
