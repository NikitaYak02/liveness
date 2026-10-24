from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import CooperativeScenarioConfig
from .landmarks import FaceLandmarkDetector, FaceObservation, landmarks_to_bbox


@dataclass(slots=True)
class FrameDetection:
    frame_index: int
    frame_rgb: np.ndarray
    observation: FaceObservation
    face_height_ratio: float


@dataclass(slots=True)
class PreprocessedTriplet:
    f1_rgb: np.ndarray
    f2_rgb: np.ndarray
    f3_rgb: np.ndarray
    frame_indices: tuple[int, int, int]
    face_ratios: tuple[float, float, float]


def _ensure_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _rotate_image_and_points(
    image: np.ndarray,
    points: np.ndarray,
    angle_deg: float,
    center_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D(tuple(center_xy.tolist()), angle_deg, 1.0)
    rotated = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    rotated_points = cv2.transform(points[None, :, :], matrix)[0]
    return rotated, rotated_points


def _apply_random_perspective(
    image: np.ndarray,
    points: np.ndarray,
    jitter: float,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    src = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
        dtype=np.float32,
    )
    dx = jitter * w
    dy = jitter * h
    dst = src + np.array(
        [
            [rng.uniform(-dx, dx), rng.uniform(-dy, dy)],
            [rng.uniform(-dx, dx), rng.uniform(-dy, dy)],
            [rng.uniform(-dx, dx), rng.uniform(-dy, dy)],
            [rng.uniform(-dx, dx), rng.uniform(-dy, dy)],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    warped_points = cv2.perspectiveTransform(points[None, :, :], matrix)[0]
    return warped, warped_points


def _square_bbox_from_landmarks(landmarks: np.ndarray, margin: float) -> np.ndarray:
    x1, y1, x2, y2 = landmarks_to_bbox(landmarks)
    width = x2 - x1
    height = y2 - y1
    side = max(width, height) * (1.0 + 2.0 * margin)
    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)
    return np.array(
        [
            center_x - side * 0.5,
            center_y - side * 0.5,
            center_x + side * 0.5,
            center_y + side * 0.5,
        ],
        dtype=np.float32,
    )


def _crop_with_padding(image: np.ndarray, bbox_xyxy: np.ndarray, out_size: int) -> np.ndarray:
    x1, y1, x2, y2 = bbox_xyxy.astype(np.float32)
    x1_i = math.floor(x1)
    y1_i = math.floor(y1)
    x2_i = math.ceil(x2)
    y2_i = math.ceil(y2)

    pad_left = max(0, -x1_i)
    pad_top = max(0, -y1_i)
    pad_right = max(0, x2_i - image.shape[1])
    pad_bottom = max(0, y2_i - image.shape[0])

    if pad_left or pad_top or pad_right or pad_bottom:
        image = cv2.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_REFLECT101,
        )

    x1_i += pad_left
    x2_i += pad_left
    y1_i += pad_top
    y2_i += pad_top
    crop = image[y1_i:y2_i, x1_i:x2_i]
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)


def _preprocess_frame(
    frame_rgb: np.ndarray,
    landmarks: np.ndarray,
    config: CooperativeScenarioConfig,
    align_roll: bool,
) -> np.ndarray:
    work_frame = frame_rgb.copy()
    work_landmarks = landmarks.astype(np.float32).copy()

    if align_roll:
        left_eye = work_landmarks[list(config.eye_left_indices)].mean(axis=0)
        right_eye = work_landmarks[list(config.eye_right_indices)].mean(axis=0)
        eye_midpoint = 0.5 * (left_eye + right_eye)
        angle = math.degrees(math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
        work_frame, work_landmarks = _rotate_image_and_points(
            work_frame,
            work_landmarks,
            angle_deg=-angle,
            center_xy=eye_midpoint,
        )

    bbox_xyxy = _square_bbox_from_landmarks(work_landmarks, margin=config.crop_margin)
    return _crop_with_padding(work_frame, bbox_xyxy, out_size=config.crop_size)


class CooperativeVideoPreprocessor:
    def __init__(self, detector: FaceLandmarkDetector, config: CooperativeScenarioConfig | None = None) -> None:
        self.detector = detector
        self.config = config or CooperativeScenarioConfig()

    def scan_video(self, video_path: str | Path, max_frames: int | None = None) -> list[FrameDetection]:
        video_path = str(video_path)
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        detections: list[FrameDetection] = []
        frame_index = 0
        try:
            while True:
                ok, frame_bgr = capture.read()
                if not ok:
                    break
                if frame_index % self.config.scan_stride != 0:
                    frame_index += 1
                    continue
                frame_rgb = _ensure_rgb(frame_bgr)
                observation = self.detector.detect(frame_rgb)
                if observation is not None:
                    bbox_height = observation.bbox_xyxy[3] - observation.bbox_xyxy[1]
                    face_height_ratio = float(bbox_height) / float(frame_rgb.shape[0])
                    detections.append(
                        FrameDetection(
                            frame_index=frame_index,
                            frame_rgb=frame_rgb,
                            observation=observation,
                            face_height_ratio=face_height_ratio,
                        )
                    )
                frame_index += 1
                if max_frames is not None and frame_index >= max_frames:
                    break
        finally:
            capture.release()

        if len(detections) < 3:
            raise RuntimeError(
                f"Only {len(detections)} valid face detections were found in {video_path}. "
                "Need at least 3 to form the cooperative triplet."
            )
        return detections

    def select_triplet(
        self,
        detections: list[FrameDetection],
        random_frame: bool = False,
        rng: random.Random | None = None,
    ) -> tuple[FrameDetection, FrameDetection, FrameDetection]:
        rng = rng or random.Random()
        targets = self.config.target_face_heights

        if random_frame:
            ordered_by_ratio = sorted(detections, key=lambda item: item.face_height_ratio)
            pool = max(1, int(math.ceil(len(ordered_by_ratio) * self.config.frame_pool_quantile)))
            small_pool = ordered_by_ratio[:pool]
            large_pool = ordered_by_ratio[-pool:]

            for _ in range(32):
                f1 = rng.choice(small_pool)
                f3 = rng.choice(large_pool)
                if f1.frame_index >= f3.frame_index:
                    continue
                middle = [
                    item for item in detections if f1.frame_index < item.frame_index < f3.frame_index
                ]
                if not middle:
                    continue
                f2 = min(middle, key=lambda item: abs(item.face_height_ratio - targets[1]))
                return f1, f2, f3

        prefix_best = [0] * len(detections)
        prefix_best[0] = 0
        for idx in range(1, len(detections)):
            prev = prefix_best[idx - 1]
            current_err = abs(detections[idx].face_height_ratio - targets[0])
            prev_err = abs(detections[prev].face_height_ratio - targets[0])
            prefix_best[idx] = idx if current_err < prev_err else prev

        suffix_best = [0] * len(detections)
        suffix_best[-1] = len(detections) - 1
        for idx in range(len(detections) - 2, -1, -1):
            nxt = suffix_best[idx + 1]
            current_err = abs(detections[idx].face_height_ratio - targets[2])
            next_err = abs(detections[nxt].face_height_ratio - targets[2])
            suffix_best[idx] = idx if current_err < next_err else nxt

        best_triplet = None
        best_score = float("inf")
        for mid_idx in range(1, len(detections) - 1):
            left_idx = prefix_best[mid_idx - 1]
            right_idx = suffix_best[mid_idx + 1]
            score = (
                abs(detections[left_idx].face_height_ratio - targets[0])
                + abs(detections[mid_idx].face_height_ratio - targets[1])
                + abs(detections[right_idx].face_height_ratio - targets[2])
            )
            if score < best_score:
                best_score = score
                best_triplet = (detections[left_idx], detections[mid_idx], detections[right_idx])

        assert best_triplet is not None
        return best_triplet

    def process_video(
        self,
        video_path: str | Path,
        random_frame: bool = False,
        perspective_prob: float = 0.0,
        rng: random.Random | None = None,
    ) -> PreprocessedTriplet:
        rng = rng or random.Random()
        detections = self.scan_video(video_path)
        f1, f2, f3 = self.select_triplet(detections, random_frame=random_frame, rng=rng)

        frames = [f1.frame_rgb.copy(), f2.frame_rgb.copy(), f3.frame_rgb.copy()]
        landmarks = [
            f1.observation.landmarks.copy(),
            f2.observation.landmarks.copy(),
            f3.observation.landmarks.copy(),
        ]

        for idx in (0, 2):
            if perspective_prob > 0.0 and rng.random() < perspective_prob:
                frames[idx], landmarks[idx] = _apply_random_perspective(
                    frames[idx],
                    landmarks[idx],
                    jitter=self.config.perspective_jitter,
                    rng=rng,
                )

        processed = [
            _preprocess_frame(frames[0], landmarks[0], self.config, align_roll=True),
            _preprocess_frame(frames[1], landmarks[1], self.config, align_roll=False),
            _preprocess_frame(frames[2], landmarks[2], self.config, align_roll=True),
        ]

        return PreprocessedTriplet(
            f1_rgb=processed[0],
            f2_rgb=processed[1],
            f3_rgb=processed[2],
            frame_indices=(f1.frame_index, f2.frame_index, f3.frame_index),
            face_ratios=(f1.face_height_ratio, f2.face_height_ratio, f3.face_height_ratio),
        )
