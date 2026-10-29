"""Warp 2D landmarks with LivePortrait's sparse implicit-keypoint motion.

LivePortrait does not expose a 2D image optical flow: the warping network builds a
3D deformation on the appearance feature volume, then SPADE synthesises a new crop.
The sparse motion that *drives* that field is the correspondence between the
transformed implicit keypoints ``x_s`` and ``x_d`` (xy in normalised crop space).

We approximate "points ride with the deformation" by a thin-plate spline that maps
``x_s[:, :2] → x_d[:, :2]`` and apply it to detector landmarks after projecting
them into the same crop / normalised coordinates.
"""

from __future__ import annotations

import numpy as np


def pairwise_sqdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances between rows of a (N,D) and b (M,D)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1)


def tps_basis(r2: np.ndarray) -> np.ndarray:
    """U(r) = r^2 log(r) for thin-plate spline (r^2 = squared distance)."""
    out = np.zeros_like(r2, dtype=np.float64)
    mask = r2 > 0
    r2m = r2[mask]
    out[mask] = r2m * np.log(np.sqrt(r2m))
    return out


def fit_tps(src: np.ndarray, dst: np.ndarray, reg: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Fit 2D TPS mapping src→dst. Returns (params (N+3,2), ctrl_src (N,2))."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError(f"src/dst must be (N,2), got {src.shape} / {dst.shape}")
    n = src.shape[0]
    if n < 3:
        raise ValueError("TPS needs at least 3 control points")
    k = tps_basis(pairwise_sqdist(src, src))
    p = np.hstack([np.ones((n, 1)), src])
    top = np.hstack([k + reg * np.eye(n), p])
    bot = np.hstack([p.T, np.zeros((3, 3))])
    l = np.vstack([top, bot])
    y = np.vstack([dst, np.zeros((3, 2))])
    params = np.linalg.solve(l, y)
    return params, src.copy()


def apply_tps(params: np.ndarray, ctrl: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Apply fitted TPS to query points (M,2)."""
    query = np.asarray(query, dtype=np.float64)
    ctrl = np.asarray(ctrl, dtype=np.float64)
    n = ctrl.shape[0]
    w, a = params[:n], params[n:]
    k = tps_basis(pairwise_sqdist(query, ctrl))
    return k @ w + np.hstack([np.ones((query.shape[0], 1)), query]) @ a


def pixel_to_norm(pts: np.ndarray, size: float) -> np.ndarray:
    """Crop pixel coords → LivePortrait normalised coords (align with make_coordinate_grid)."""
    size = float(size)
    return 2.0 * np.asarray(pts, dtype=np.float64) / max(size - 1.0, 1.0) - 1.0


def norm_to_pixel(pts: np.ndarray, size: float) -> np.ndarray:
    size = float(size)
    return (np.asarray(pts, dtype=np.float64) + 1.0) * 0.5 * max(size - 1.0, 1.0)


def transform_pts(pts: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Apply 2x3 or 3x3 affine to Nx2 points."""
    m = np.asarray(m, dtype=np.float64)
    pts = np.asarray(pts, dtype=np.float64)
    return pts @ m[:2, :2].T + m[:2, 2]


def invert_affine(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    if m.shape == (2, 3):
        m = np.vstack([m, [0.0, 0.0, 1.0]])
    return np.linalg.inv(m)


def scale_points_resize_to_limit(
    pts: np.ndarray,
    src_hw: tuple[int, int],
    max_dim: int = 1280,
) -> np.ndarray:
    """Map landmarks through LivePortrait's ``resize_to_limit`` (scale only; even-crop is bottom/right)."""
    h, w = int(src_hw[0]), int(src_hw[1])
    pts = np.asarray(pts, dtype=np.float64)
    if max_dim > 0 and max(h, w) > max_dim:
        if h > w:
            new_h = max_dim
            new_w = int(w * (max_dim / h))
        else:
            new_w = max_dim
            new_h = int(h * (max_dim / w))
        sx, sy = new_w / w, new_h / h
        return np.stack([pts[:, 0] * sx, pts[:, 1] * sy], axis=1)
    return pts.copy()


def warp_landmarks_fullframe(
    landmarks_xy: np.ndarray,
    *,
    x_s: np.ndarray,
    x_d: np.ndarray,
    m_c2o: np.ndarray,
    crop_size: float = 512.0,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Warp full-frame landmarks with sparse LP motion; returns (N,2) float, NaN if invalid.

    ``landmarks_xy`` must already be in the same full-frame coordinates as the
    LivePortrait canvas that ``m_c2o`` maps the crop into (i.e. after resize_to_limit).
    ``x_s`` / ``x_d`` are (K,3) implicit keypoints in normalised crop space.
    """
    pts = np.asarray(landmarks_xy, dtype=np.float64)
    out = np.full_like(pts, np.nan)
    if valid_mask is None:
        valid_mask = np.isfinite(pts).all(axis=1)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(pts).all(axis=1)
    if not valid_mask.any():
        return out

    m_c2o = np.asarray(m_c2o, dtype=np.float64)
    m_o2c = invert_affine(m_c2o)
    crop_pts = transform_pts(pts[valid_mask], m_o2c)

    xs = np.asarray(x_s, dtype=np.float64)[:, :2]
    xd = np.asarray(x_d, dtype=np.float64)[:, :2]
    params, ctrl = fit_tps(xs, xd)

    # Network keypoints live in the 256-input normalised space; crop affine is dsize
    # (usually 512). Convert crop pixels → 256 space before normalising.
    net_size = 256.0
    scale = net_size / float(crop_size)
    crop_net = crop_pts * scale
    query_norm = pixel_to_norm(crop_net, net_size)
    warped_norm = apply_tps(params, ctrl, query_norm)
    warped_net = norm_to_pixel(warped_norm, net_size)
    warped_crop = warped_net / scale
    warped_full = transform_pts(warped_crop, m_c2o)
    out[valid_mask] = warped_full
    return out
