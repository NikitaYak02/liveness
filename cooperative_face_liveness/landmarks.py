from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np


@dataclass(slots=True)
class FaceObservation:
    bbox_xyxy: np.ndarray
    landmarks: np.ndarray
    score: float = 1.0


class FaceLandmarkDetector(Protocol):
    def detect(self, frame_rgb: np.ndarray) -> Optional[FaceObservation]:
        ...


def landmarks_to_bbox(landmarks: np.ndarray) -> np.ndarray:
    min_xy = landmarks.min(axis=0)
    max_xy = landmarks.max(axis=0)
    return np.array([min_xy[0], min_xy[1], max_xy[0], max_xy[1]], dtype=np.float32)


class FaceAlignmentLandmarkDetector:
    """PyTorch-friendly facial landmark detector.

    The paper only states "neural key-point detector", so the concrete
    landmark network is an implementation choice. Here we use the
    `face-alignment` package because it exposes pretrained 2D FAN models
    and works well inside a PyTorch pipeline.
    """

    def __init__(
        self,
        device: str = "cuda",
        face_detector: str = "sfd",
        flip_input: bool = False,
    ) -> None:
        import os

        # face-alignment may torch.compile FAN; without Triton that hard-fails.
        os.environ["TORCHDYNAMO_DISABLE"] = "1"
        try:
            import face_alignment
        except ImportError as exc:
            raise ImportError(
                "Install `face-alignment` to use FaceAlignmentLandmarkDetector."
            ) from exc
        try:
            import torch._dynamo as _dynamo

            _dynamo.config.suppress_errors = True
        except Exception:
            pass

        if hasattr(face_alignment.LandmarksType, "TWO_D"):
            landmarks_type = face_alignment.LandmarksType.TWO_D
        else:
            landmarks_type = face_alignment.LandmarksType._2D

        self._fa = face_alignment.FaceAlignment(
            landmarks_type,
            face_detector=face_detector,
            device=device,
            flip_input=flip_input,
        )
        # Unwrap if compile already wrapped the net.
        net = getattr(self._fa, "face_alignment_net", None)
        orig = getattr(net, "_orig_mod", None)
        if orig is not None:
            self._fa.face_alignment_net = orig

    def detect(self, frame_rgb: np.ndarray) -> Optional[FaceObservation]:
        predictions = self._fa.get_landmarks_from_image(frame_rgb)
        if not predictions:
            return None

        landmarks = max(
            predictions,
            key=lambda points: float(np.prod(points.max(axis=0) - points.min(axis=0))),
        ).astype(np.float32)
        bbox_xyxy = landmarks_to_bbox(landmarks)
        return FaceObservation(bbox_xyxy=bbox_xyxy, landmarks=landmarks, score=1.0)
