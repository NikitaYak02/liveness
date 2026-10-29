"""3DDFA-V2 adapter.

Two modes:
  --job JOB --out DIR        rotate the fitted 3DMM mesh to each target pose
  --measure LIST --result R  head pose (yaw, pitch, roll) of each image in LIST

The rotated face is rendered with per-vertex colours sampled from the source
photo; vertices hidden in the source take their mirror vertex colour. It is
composited over the source frame with the original face region inpainted.
68 landmarks of each output frame, with visibility, go to DIR/landmarks.json.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from common import enter_repo, frames_dir, load_job, log, read_bgr, save_meta  # noqa: E402
from tddfa_geometry import (  # noqa: E402
    facing_camera,
    fill_hidden_colors,
    mirror_index,
    pose_degrees,
    sample_colors,
    visible_vertices,
    with_pose,
)

INPAINT_MAX_SIDE = 512


class Fitter:
    def __init__(self):
        repo = enter_repo("3DDFA_V2")
        import yaml
        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX

        cfg = yaml.safe_load(open(repo / "configs" / "mb1_120x120.yml"))
        cfg["checkpoint_fp"] = str(repo / cfg["checkpoint_fp"])
        cfg["bfm_fp"] = str(repo / cfg["bfm_fp"])
        self.detector = FaceBoxes_ONNX()
        self.tddfa = TDDFA_ONNX(**cfg)
        self.tri = self.tddfa.tri
        self.bfm_fp = cfg["bfm_fp"]
        self._bfm_model = None
        self._mirror = None

    def fit(self, img: np.ndarray, fallback_full_box: bool = False):
        """3DMM parameters of the largest face, or None."""
        boxes = self.detector(img)
        used_fallback = False
        if len(boxes) == 0:
            if not fallback_full_box:
                return None, None, False
            h, w = img.shape[:2]
            boxes = [[0.1 * w, 0.1 * h, 0.9 * w, 0.95 * h, 1.0]]
            used_fallback = True
        box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        params, rois = self.tddfa(img, [box])
        # A second pass cropped by the predicted landmarks is noticeably more stable.
        ver = self.tddfa.recon_vers(params, rois, dense_flag=False)[0]
        params, rois = self.tddfa(img, [ver], crop_policy="landmark")
        return params[0], rois[0], used_fallback

    def dense(self, param, roi):
        return self.tddfa.recon_vers([param], [roi], dense_flag=True)[0]

    def sparse(self, param, roi):
        return self.tddfa.recon_vers([param], [roi], dense_flag=False)[0]

    def model_space(self, param) -> np.ndarray:
        """Dense mesh in BFM model space (3, N) for this identity and expression."""
        from utils.tddfa_util import _parse_param

        _, _, alpha_shp, alpha_exp = _parse_param(param)
        bfm = self.bfm()
        v = bfm.u + bfm.w_shp @ alpha_shp + bfm.w_exp @ alpha_exp
        return v.reshape(3, -1, order="F")

    def bfm(self):
        if self._bfm_model is None:
            from bfm.bfm import BFMModel
            self._bfm_model = BFMModel(self.bfm_fp, shape_dim=40, exp_dim=10)
        return self._bfm_model

    def landmark_vertices(self) -> np.ndarray:
        """Dense-vertex index of each of the 68 landmarks (keypoints index flat xyz)."""
        return self.bfm().keypoints.reshape(-1, 3)[:, 0] // 3

    def mirror(self) -> np.ndarray:
        if self._mirror is None:
            self._mirror = mirror_index(self.bfm().u.reshape(3, -1, order="F"))
        return self._mirror


def measure(fitter: Fitter, images: list[str]) -> dict:
    result = {}
    for path in images:
        img = cv2.imread(path)
        if img is None:
            result[path] = None
            continue
        # Generated frames are tight face crops where FaceBoxes sometimes misses.
        param, roi, fallback = fitter.fit(img, fallback_full_box=True)
        if param is None:
            result[path] = None
            continue
        pose = pose_degrees(param)
        pose["fallback_box"] = fallback
        lms = fitter.sparse(param, roi)
        pose["box"] = [float(lms[0].min()), float(lms[1].min()), float(lms[0].max()), float(lms[1].max())]
        result[path] = pose
    return result


def inpaint_face(img: np.ndarray, ver: np.ndarray) -> np.ndarray:
    """Source frame with the face region filled from its surroundings."""
    h, w = img.shape[:2]
    hull = cv2.convexHull(ver[:2].T.astype(np.int32))
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    grow = max(3, int(0.06 * np.ptp(ver[0])))
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))

    scale = min(1.0, INPAINT_MAX_SIDE / max(h, w))
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    filled = cv2.inpaint(small, small_mask, 7, cv2.INPAINT_TELEA)
    filled = cv2.resize(filled, (w, h), interpolation=cv2.INTER_LINEAR)
    filled = cv2.GaussianBlur(filled, (0, 0), 2)
    soft = cv2.GaussianBlur(mask, (0, 0), 3).astype(np.float32)[..., None] / 255.0
    return (img * (1 - soft) + filled * soft).astype(np.uint8)


class SourceModel:
    """Everything derived once per source photo."""

    def __init__(self, fitter: Fitter, path: str):
        self.img = read_bgr(path)
        param, roi, _ = fitter.fit(self.img)
        if param is None:
            raise RuntimeError(f"no face found in {path}")
        self.param, self.roi = param, roi
        self.pose = pose_degrees(param)
        ver = fitter.dense(param, roi)
        visible = visible_vertices(ver, self.img.shape[:2])
        from Sim3DR import get_normal
        normals = get_normal(np.ascontiguousarray(ver.T), fitter.tri)
        visible &= facing_camera(normals, visible)
        colors = sample_colors(self.img, ver)
        self.colors = np.ascontiguousarray(fill_hidden_colors(colors, visible, fitter.mirror()))
        self.background = inpaint_face(self.img, ver)
        self.center = fitter.model_space(param).mean(axis=1)


def render_pose(fitter: Fitter, src: SourceModel, yaw: float, pitch: float):
    from Sim3DR import rasterize

    param = with_pose(src.param, yaw, pitch, src.pose["roll"], center=src.center)
    ver = fitter.dense(param, src.roi)
    frame = rasterize(np.ascontiguousarray(ver.T), fitter.tri, src.colors, bg=src.background.copy())

    lms = fitter.sparse(param, src.roi)
    vis_dense = visible_vertices(ver, frame.shape[:2])
    keypoints = fitter.landmark_vertices()
    landmarks = [
        {"x": float(lms[0, k]), "y": float(lms[1, k]), "visible": bool(vis_dense[keypoints[k]])}
        for k in range(lms.shape[1])
    ]
    return frame, landmarks, pose_degrees(param)


def run_job(fitter: Fitter, job: dict, out: str) -> None:
    sources: dict[str, SourceModel] = {}
    landmarks = {}
    frames = frames_dir(out)
    for frame in job["frames"]:
        path = frame["source"]
        if path not in sources:
            log(f"fitting {path}")
            sources[path] = SourceModel(fitter, path)
        image, lms, applied = render_pose(fitter, sources[path], frame["yaw"], frame["pitch"])
        cv2.imwrite(str(frames / f"{frame['index']:04d}.png"), image)
        landmarks[f"{frame['index']:04d}.png"] = {"source": path, "pose": applied, "points": lms}
    (Path(out) / "landmarks.json").write_text(json.dumps(landmarks, indent=1))
    save_meta(out, {"source_poses": {p: s.pose for p, s in sources.items()}})


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--job")
    p.add_argument("--out")
    p.add_argument("--measure")
    p.add_argument("--result")
    args = p.parse_args()

    fitter = Fitter()
    if args.measure:
        images = json.loads(Path(args.measure).read_text())
        Path(args.result).write_text(json.dumps(measure(fitter, images), indent=1))
        return 0
    if not (args.job and args.out):
        p.error("need --job and --out, or --measure and --result")
    run_job(fitter, load_job(args.job), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
