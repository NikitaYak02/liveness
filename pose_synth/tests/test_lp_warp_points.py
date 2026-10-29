import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ps.lp_warp_points import (  # noqa: E402
    apply_tps,
    fit_tps,
    norm_to_pixel,
    pixel_to_norm,
    scale_points_resize_to_limit,
    transform_pts,
    warp_landmarks_fullframe,
)


class TPSTest(unittest.TestCase):
    def test_identity(self):
        rng = np.random.default_rng(0)
        ctrl = rng.uniform(-0.8, 0.8, size=(21, 2))
        params, src = fit_tps(ctrl, ctrl.copy())
        query = rng.uniform(-0.5, 0.5, size=(11, 2))
        out = apply_tps(params, src, query)
        np.testing.assert_allclose(out, query, atol=1e-5)

    def test_translation(self):
        rng = np.random.default_rng(1)
        ctrl = rng.uniform(-0.8, 0.8, size=(21, 2))
        delta = np.array([0.1, -0.05])
        params, src = fit_tps(ctrl, ctrl + delta)
        query = rng.uniform(-0.5, 0.5, size=(7, 2))
        out = apply_tps(params, src, query)
        np.testing.assert_allclose(out, query + delta, atol=1e-4)

    def test_norm_roundtrip(self):
        pts = np.array([[0.0, 0.0], [128.0, 64.0], [255.0, 255.0]])
        back = norm_to_pixel(pixel_to_norm(pts, 256), 256)
        np.testing.assert_allclose(back, pts, atol=1e-6)

    def test_resize_to_limit_scale(self):
        pts = np.array([[100.0, 200.0], [400.0, 800.0]])
        # already under limit → unchanged
        out = scale_points_resize_to_limit(pts, (900, 600), max_dim=1280)
        np.testing.assert_allclose(out, pts)
        # tall image scaled by height
        out = scale_points_resize_to_limit(pts, (2000, 1000), max_dim=1280)
        scale = 1280 / 2000
        np.testing.assert_allclose(out, pts * scale)

    def test_warp_fullframe_identity_motion(self):
        # Identity x_s==x_d and identity crop→full: landmarks stay put.
        m = np.eye(3)
        xs = np.zeros((21, 3))
        # spread control points in norm space
        ang = np.linspace(0, 2 * np.pi, 21, endpoint=False)
        xs[:, 0] = 0.5 * np.cos(ang)
        xs[:, 1] = 0.5 * np.sin(ang)
        xd = xs.copy()
        # landmarks in "full frame" that equal crop pixels when M=I
        lms = np.array([[100.0, 120.0], [200.0, 180.0], [150.0, 220.0]])
        # put them near centre in 512 crop → map to net 256 then to norm near 0
        out = warp_landmarks_fullframe(
            lms, x_s=xs, x_d=xd, m_c2o=m, crop_size=512.0,
        )
        np.testing.assert_allclose(out, lms, atol=0.5)

    def test_transform_pts(self):
        m = np.array([[2.0, 0.0, 10.0], [0.0, 2.0, 5.0], [0.0, 0.0, 1.0]])
        pts = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = transform_pts(pts, m)
        np.testing.assert_allclose(out, [[12.0, 9.0], [16.0, 13.0]])


if __name__ == "__main__":
    unittest.main()
