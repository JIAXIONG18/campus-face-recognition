"""从 LFW parquet 生成对齐人脸与训练/评测划分。"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from face_recog.align import FaceAligner

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def identity_names(parquet_path: Path) -> list[str]:
    metadata = pq.read_metadata(parquet_path).metadata
    raw = metadata[b"huggingface"].decode("utf-8")
    return json.loads(raw)["info"]["features"]["label"]["names"]


def safe_name(path: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", path)


def worker(args):
    blob, dst, model = args
    aligner = FaceAligner(model)
    image = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return False
    crop, _ = aligner.align(image)
    if crop is None:
        return False
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(dst, crop, [cv2.IMWRITE_JPEG_QUALITY, 95]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", default=str(DATA / "lfw" / "lfw_data.parquet"))
    parser.add_argument("--out", default=str(DATA / "aligned"))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--targets", type=int, default=50)
    parser.add_argument("--impostors", type=int, default=50)
    args = parser.parse_args()

    parquet_path = Path(args.parquet)
    out_dir = Path(args.out)
    names = identity_names(parquet_path)
    table = pq.read_table(parquet_path).to_pylist()
    counts = Counter(row["label"] for row in table)
    ranked = [label for label, _ in counts.most_common()]

    target_ids = [i for i in ranked if counts[i] >= 10][: args.targets]
    used = set(target_ids)
    impostor_ids = [i for i in ranked if counts[i] >= 8 and i not in used][: args.impostors]
    used |= set(impostor_ids)
    pretrain_ids = [i for i in ranked if counts[i] >= 3 and i not in used]

    selected = set(target_ids) | set(impostor_ids) | set(pretrain_ids)
    tasks = []
    for row in table:
        label = row["label"]
        if label not in selected:
            continue
        dst = out_dir / str(label) / f"{safe_name(row['image']['path'])}.jpg"
        if dst.exists():
            continue
        tasks.append((row["image"]["bytes"], str(dst), str(ROOT / "code" / "models" / "face_detection_yunet_2023mar.onnx")))

    ok = 0
    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(worker, tasks, chunksize=16):
                ok += int(result)
    print(f"aligned {ok}/{len(tasks)} images -> {out_dir}")

    def files_of(label: int) -> list[str]:
        folder = out_dir / str(label)
        if not folder.exists():
            return []
        return sorted(str(p) for p in folder.glob("*.jpg"))

    def build(samples, mapping):
        return [[path, mapping[label]] for label, path in samples]

    pretrain_map = {label: i for i, label in enumerate(pretrain_ids)}
    pretrain = build([(label, p) for label in pretrain_ids for p in files_of(label)], pretrain_map)

    target_map = {label: i for i, label in enumerate(target_ids)}
    stage2 = {"train": [], "val": [], "test": [], "gallery": []}
    rng = random.Random(2026)
    for label in target_ids:
        files = files_of(label)
        rng.shuffle(files)
        files = files[:60]
        total = len(files)
        if total < 10:
            continue
        n_train = max(8, int(total * 0.7))
        n_val = max(1, int(total * 0.1))
        stage2["train"].extend((p, target_map[label]) for p in files[:n_train])
        stage2["val"].extend((p, target_map[label]) for p in files[n_train : n_train + n_val])
        stage2["test"].extend((p, target_map[label]) for p in files[n_train + n_val :])
        stage2["gallery"].extend((p, target_map[label]) for p in files[:5])

    impostor = []
    for idx, label in enumerate(impostor_ids):
        for p in files_of(label)[:5]:
            impostor.append((p, idx))

    splits = {
        "identity_names": {str(k): names[k] for k in selected},
        "pretrain_classes": len(pretrain_map),
        "target_classes": len(target_map),
        "pretrain": pretrain,
        "stage2": stage2,
        "impostor": impostor,
    }
    (DATA / "splits.json").write_text(json.dumps(splits, ensure_ascii=False), encoding="utf-8")
    print(
        f"pretrain identities={len(pretrain_map)} images={len(pretrain)} | "
        f"targets={len(target_map)} train={len(stage2['train'])} val={len(stage2['val'])} "
        f"test={len(stage2['test'])} gallery={len(stage2['gallery'])} | impostor images={len(impostor)}"
    )


if __name__ == "__main__":
    main()
