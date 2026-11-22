"""推理服务：检测、对齐、特征提取与检索。"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from face_recog.align import FaceAligner
from face_recog.models import build_model
from liveness.train import LivenessNet
import config

APP_ROOT = ROOT / "code"
ENROLL_DIR = Path(os.environ.get("FACE_APP_ENROLL", ROOT / "data" / "enroll"))
DEFAULT_CHECKPOINT = Path(
    os.environ.get("FACE_APP_CHECKPOINT", ROOT / "experiments" / "runs" / "finetune_full" / "best.pt")
)
METRICS_PATH = Path(os.environ.get("FACE_APP_METRICS", ROOT / "experiments" / "metrics.json"))
LIVENESS_CHECKPOINT = Path(
    os.environ.get("FACE_APP_LIVENESS", ROOT / "experiments" / "runs" / "liveness" / "best.pt")
)


class FaceEngine:
    def __init__(self, checkpoint: Path | None = None, threshold: float | None = None):
        self.checkpoint = checkpoint or DEFAULT_CHECKPOINT
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(self.checkpoint, map_location="cpu")
        self.model = build_model(state.get("variant", "full"))
        self.model.load_state_dict(state["model"])
        self.model.to(self.device).eval()
        self.aligner = FaceAligner()
        base = threshold if threshold is not None else self._read_threshold()
        cfg = config.load()
        self.thresholds = {"考勤": base, "门禁": round(base + 0.10, 2)}
        self.thresholds.update(cfg.get("thresholds", {}))
        self.threshold = self.thresholds.get("考勤", base)
        self.liveness_threshold = float(cfg.get("liveness_threshold", 0.0))
        self.liveness = None
        liveness_ckpt = LIVENESS_CHECKPOINT
        if liveness_ckpt.exists():
            self.liveness = LivenessNet().to(self.device).eval()
            self.liveness.load_state_dict(torch.load(liveness_ckpt, map_location="cpu")["model"])
        self.gallery: list[dict] = []
        self.matrix: np.ndarray | None = None

    @staticmethod
    def _read_threshold() -> float:
        metrics = METRICS_PATH
        if metrics.exists():
            return float(json.loads(metrics.read_text(encoding="utf-8"))["threshold"])
        return 0.45

    def decode(self, blob: bytes):
        image = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("无法解析图像")
        return image

    def quality(self, image: np.ndarray, face) -> tuple[bool, str]:
        h, w = image.shape[:2]
        if face[2] < 48 or face[3] < 48:
            return False, "人脸过小，请靠近摄像头"
        crop = image[int(max(face[1], 0)) : int(min(face[1] + face[3], h)), int(max(face[0], 0)) : int(min(face[0] + face[2], w))]
        if crop.size and cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() < 25:
            return False, "图像模糊，请保持稳定"
        return True, ""

    @torch.no_grad()
    def embed(self, image: np.ndarray):
        crop, face = self.aligner.align(image)
        if crop is None:
            return None, None, None
        tensor = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = (tensor - 0.5) / 0.5
        tensor = torch.from_numpy(tensor.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        vector = F.normalize(self.model(tensor), dim=1)[0].cpu().numpy().astype(np.float32)
        return vector, face, crop

    @torch.no_grad()
    def liveness_score(self, crop: np.ndarray) -> float | None:
        if self.liveness is None:
            return None
        image = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        return float(self.liveness(tensor).item())

    def reload(self):
        from db import load_gallery

        rows = load_gallery()
        self.gallery = [dict(r) for r in rows]
        if rows:
            self.matrix = np.stack([np.frombuffer(r["feature"], dtype=np.float32) for r in rows])
            norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
            self.matrix = self.matrix / np.maximum(norms, 1e-12)
        else:
            self.matrix = None

    def identify(self, vector: np.ndarray, scene: str = "考勤"):
        start = time.perf_counter()
        if self.matrix is None or len(self.gallery) == 0:
            return None, 0.0, (time.perf_counter() - start) * 1000
        thr = self.thresholds.get(scene, self.threshold)
        scores = self.matrix @ vector
        best = int(np.argmax(scores))
        score = float(scores[best])
        match = self.gallery[best] if score >= thr else None
        return match, score, (time.perf_counter() - start) * 1000

    def save_enroll(self, user_id: str, blob: bytes, image: np.ndarray, face=None) -> str:
        folder = ENROLL_DIR / user_id
        folder.mkdir(parents=True, exist_ok=True)
        name = f"{int(time.time() * 1000)}.jpg"
        path = folder / name
        path.write_bytes(blob)
        return str(path.relative_to(ROOT))


engine: FaceEngine | None = None


def get_engine() -> FaceEngine:
    global engine
    if engine is None:
        engine = FaceEngine()
        engine.reload()
    return engine
