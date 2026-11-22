"""从 CelebA-Identity parquet 生成封闭集与冒充集清单（跨数据集测试）。"""
from __future__ import annotations

import collections
import io
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "celeba"
OUT = ROOT / "data" / "celeba"


def main():
    table = pq.read_table(RAW / "celeba_identity-00000.parquet")
    by_id = collections.defaultdict(list)
    for row in table.to_pylist():
        by_id[row["identity_id"]].append(row["img"]["bytes"])
    ids = [i for i, imgs in by_id.items() if len(imgs) >= 6]
    random.Random(2026).shuffle(ids)

    targets = ids[:50]
    impostors = ids[50:90]

    def save(identity, blob, index):
        image = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
        h, w = image.shape[:2]
        scale = 320 / min(h, w)
        if scale < 1:
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        dst = OUT / f"id_{identity}" / f"{index:03d}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return str(dst)

    gallery, probe, train, val, impostor = [], [], [], [], []
    for label, identity in enumerate(targets):
        blobs = by_id[identity][:6]
        random.Random(2026 + identity).shuffle(blobs)
        for index, blob in enumerate(blobs[:3]):
            path = save(identity, blob, index)
            train.append({"path": path, "label": label})
            gallery.append({"path": path, "label": label})
        for index, blob in enumerate(blobs[3:4]):
            val.append({"path": save(identity, blob, index + 3), "label": label})
        for index, blob in enumerate(blobs[4:6]):
            probe.append({"path": save(identity, blob, index + 4), "label": label})

    for label, identity in enumerate(impostors):
        for index, blob in enumerate(by_id[identity][:3]):
            impostor.append({"path": save(identity, blob, index), "label": label})

    manifest = {"train": train, "val": val, "gallery": gallery, "probe": probe, "impostor": impostor}
    (ROOT / "data" / "celeba_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print("train", len(train), "val", len(val), "gallery", len(gallery), "probe", len(probe), "impostor", len(impostor))


if __name__ == "__main__":
    main()
