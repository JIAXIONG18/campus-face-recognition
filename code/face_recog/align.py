"""人脸检测与五点对齐（YuNet + 相似变换）。"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "face_detection_yunet_2023mar.onnx"


class FaceAligner:
    def __init__(self, model_path: str | os.PathLike = DEFAULT_MODEL, score_threshold: float = 0.6):
        self.detector = cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320), score_threshold, 0.3, 5000
        )

    @staticmethod
    def _largest(faces: np.ndarray) -> np.ndarray:
        areas = faces[:, 2] * faces[:, 3]
        return faces[int(np.argmax(areas))]

    def detect(self, image_bgr: np.ndarray):
        h, w = image_bgr.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(image_bgr)
        if faces is None or len(faces) == 0:
            return None
        return self._largest(faces)

    def align(self, image_bgr: np.ndarray, size: int = 112):
        face = self.detect(image_bgr)
        if face is None:
            return None, None
        landmarks = face[4:14].reshape(5, 2).astype(np.float32)
        dst = ARCFACE_TEMPLATE * (size / 112.0)
        matrix, _ = cv2.estimateAffinePartial2D(landmarks, dst, method=cv2.LMEDS)
        if matrix is None:
            return None, None
        crop = cv2.warpAffine(image_bgr, matrix, (size, size), borderValue=0.0)
        return crop, face


def align_file(src: str | os.PathLike, dst: str | os.PathLike, aligner: FaceAligner, size: int = 112) -> bool:
    image = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if image is None:
        return False
    crop, _ = aligner.align(image, size)
    if crop is None:
        return False
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(dst), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]))
