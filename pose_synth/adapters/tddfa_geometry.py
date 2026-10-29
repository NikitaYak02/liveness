"""Pure-numpy geometry for the 3DDFA-V2 adapter (kept separate so it is testable
without the 3DDFA-V2 environment).

Euler convention follows 3DDFA-V2 ``utils/pose.py::matrix2angle``:
yaw = asin(R[2, 0]), pitch = atan2(R[2, 1], R[2, 2]), roll = atan2(R[1, 0], R[0, 0]).
"""

from __future__ import annotations

from math import asin, atan2, cos

import numpy as np


def matrix2angle(R: np.ndarray) -> tuple[float, float, float]:
    """Same as 3DDFA-V2 matrix2angle, returns (yaw, pitch, roll) in radians."""
    if R[2, 0] > 0.998:
        return np.pi / 2, atan2(-R[0, 1], -R[0, 2]), 0.0
    if R[2, 0] < -0.998:
        return -np.pi / 2, atan2(R[0, 1], R[0, 2]), 0.0
    x = asin(R[2, 0])
    y = atan2(R[2, 1] / cos(x), R[2, 2] / cos(x))
    z = atan2(R[1, 0] / cos(x), R[0, 0] / cos(x))
    return x, y, z


def angle2matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Inverse of matrix2angle (radians): R = Rz(roll) @ Ry(-yaw) @ Rx(pitch)."""
    cx, sx = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(-yaw), np.sin(-yaw)
    cz, sz = np.cos(roll), np.sin(roll)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def split_camera(param: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """param[:12] is a scaled rotation [sR | t]; returns (s, R, t)."""
    P = param[:12].reshape(3, 4)
    r1, r2 = P[0, :3], P[1, :3]
    s = (np.linalg.norm(r1) + np.linalg.norm(r2)) / 2.0
    r1, r2 = r1 / np.linalg.norm(r1), r2 / np.linalg.norm(r2)
    R = np.stack([r1, r2, np.cross(r1, r2)])
    return s, R, P[:, 3].copy()


def pose_degrees(param: np.ndarray) -> dict:
    _, R, _ = split_camera(param)
    yaw, pitch, roll = (float(np.degrees(a)) for a in matrix2angle(R))
    return {"yaw": yaw, "pitch": pitch, "roll": roll}


def with_pose(param: np.ndarray, yaw_deg: float, pitch_deg: float, roll_deg: float,
              center: np.ndarray | None = None) -> np.ndarray:
    """Copy of a 3DDFA-V2 parameter vector with the head rotation replaced.

    The model is rotated about ``center`` (model-space point, e.g. the mesh
    centroid) so the face stays in place instead of swinging around the BFM origin.
    """
    s, R_old, t = split_camera(param)
    R = angle2matrix(*np.radians([yaw_deg, pitch_deg, roll_deg]))
    if center is not None:
        c = np.asarray(center, np.float64).reshape(3)
        t = t + s * (R_old - R) @ c
    out = param.astype(np.float32).copy()
    out[:12] = np.concatenate([s * R, t[:, None]], axis=1).reshape(-1)
    return out


def visible_vertices(ver: np.ndarray, shape: tuple[int, int], tolerance: float | None = None,
                     cell: int = 2) -> np.ndarray:
    """Point z-buffer visibility for a dense mesh in image coordinates.

    ver is (3, N) with larger z closer to the camera (3DDFA-V2 convention).
    """
    h, w = shape
    x = np.clip(np.round(ver[0] / cell).astype(int), 0, (w - 1) // cell)
    y = np.clip(np.round(ver[1] / cell).astype(int), 0, (h - 1) // cell)
    z = ver[2]
    if tolerance is None:
        tolerance = 0.02 * float(np.ptp(ver[0]))
    zbuf = np.full(((h - 1) // cell + 1, (w - 1) // cell + 1), -np.inf, np.float32)
    np.maximum.at(zbuf, (y, x), z)
    inside = (ver[0] >= 0) & (ver[0] < w) & (ver[1] >= 0) & (ver[1] < h)
    return inside & (z >= zbuf[y, x] - tolerance)


def facing_camera(normals: np.ndarray, visible: np.ndarray, min_cos: float = 0.35) -> np.ndarray:
    """Vertices whose normal points at the camera by more than min_cos.

    Grazing vertices are technically visible but carry a smeared texture, so
    they are better recoloured from the mirror side. The normal orientation of
    the mesh is unknown, so it is flipped to agree with the visible majority.
    """
    n = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    nz = n[:, 2]
    if visible.any() and np.median(nz[visible]) < 0:
        nz = -nz
    return nz > min_cos


def sample_colors(img: np.ndarray, ver: np.ndarray) -> np.ndarray:
    """Bilinear BGR samples at vertex positions, float in [0, 1], shape (N, 3)."""
    h, w = img.shape[:2]
    x = np.clip(ver[0], 0, w - 1)
    y = np.clip(ver[1], 0, h - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    img = img.astype(np.float32)
    top = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
    bottom = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
    return ((top * (1 - fy) + bottom * fy) / 255.0).astype(np.float32)


def mirror_index(canonical: np.ndarray) -> np.ndarray:
    """For each vertex of the canonical (frontal) mesh, index of its left-right mirror.

    canonical is (3, N) in model space where x is the left-right axis.
    """
    from scipy.spatial import cKDTree

    pts = canonical.T.copy()
    center_x = np.median(pts[:, 0])
    mirrored = pts.copy()
    mirrored[:, 0] = 2 * center_x - mirrored[:, 0]
    _, idx = cKDTree(pts).query(mirrored)
    return idx


def fill_hidden_colors(colors: np.ndarray, visible: np.ndarray, mirror: np.ndarray) -> np.ndarray:
    """Hidden vertices take the colour of their mirror vertex when it is visible,
    otherwise the mean visible colour."""
    out = colors.copy()
    fallback = colors[visible].mean(axis=0) if visible.any() else np.full(3, 0.5, np.float32)
    hidden = ~visible
    from_mirror = hidden & visible[mirror]
    out[from_mirror] = colors[mirror[from_mirror]]
    out[hidden & ~from_mirror] = fallback
    return out
