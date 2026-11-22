"""从 CelebA-Spoof parquet 抽取活体/攻击图片并生成训练清单。"""
from __future__ import annotations

import io
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "spoof"
OUT = ROOT / "data" / "spoof"


def extract(table, split, limit_per_class=None):
    rows = table.to_pylist()
    random.Random(2026 + hash(split) % 1000).shuffle(rows)
    counts = {"live": 0, "spoof": 0}
    samples = []
    for row in rows:
        cls = row["Class"]
        if cls not in counts:
            continue
        if limit_per_class and counts[cls] >= limit_per_class:
            continue
        image = cv2.imdecode(np.frombuffer(row["Filepath"]["bytes"], np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            continue
        h, w = image.shape[:2]
        scale = 320 / min(h, w)
        if scale < 1:
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        dst = OUT / split / cls / f"{cls}_{counts[cls]:05d}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        samples.append({"path": str(dst), "label": 1 if cls == "live" else 0})
        counts[cls] += 1
    return samples


def main():
    train = pq.read_table(RAW / "train-00000.parquet")
    valid = pq.read_table(RAW / "valid-00000.parquet")

    train_samples = extract(train, "train", limit_per_class=1750)
    valid_samples = extract(valid, "valid", limit_per_class=500)
    random.Random(0).shuffle(train_samples)
    cut = int(len(train_samples) * 0.85)
    manifest = {
        "train": train_samples[:cut],
        "valid": train_samples[cut:],
        "test": valid_samples,
    }
    (ROOT / "data" / "spoof_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in manifest.items():
        print(k, len(v), "live", sum(1 for s in v if s["label"] == 1), "spoof", sum(1 for s in v if s["label"] == 0))


if __name__ == "__main__":
    main()
