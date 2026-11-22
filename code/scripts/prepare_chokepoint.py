"""抽取 ChokePoint 门户 1 人脸并按身份生成训练/评测清单。"""
from __future__ import annotations

import json
import random
import re
import tarfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "chokepoint"
OUT = ROOT / "data" / "chokepoint"


def parse_groundtruth(xml_path: Path):
    tree = ET.parse(xml_path)
    frames = {}
    current = None
    for element in tree.getroot().iter():
        tag = element.tag
        if tag == "frame":
            current = element.get("number")
        elif tag == "person":
            if current is not None:
                frames[current] = element.get("id")
                current = None
    return frames


def find_sequence_dirs(root: Path):
    return [p for p in root.rglob("*") if p.is_dir() and re.search(r"_S\d+_C\d+(\.\d+)?$", p.name)]


def main():
    archive = RAW / "P1E.tar.xz"
    extracted = RAW / "P1E_extracted"
    if not extracted.exists():
        with tarfile.open(archive, "r:xz") as tar:
            tar.extractall(extracted)
        print("extracted", archive)

    by_subject = defaultdict(list)
    for seq_dir in find_sequence_dirs(extracted):
        for subject_dir in sorted(p for p in seq_dir.iterdir() if p.is_dir()):
            subject = subject_dir.name
            for image in sorted(subject_dir.glob("*")):
                if image.suffix.lower() in (".jpg", ".jpeg", ".png", ".pgm", ".ppm"):
                    by_subject[subject].append(image)

    subjects = sorted(by_subject)
    print("subjects", len(subjects), "frames", sum(len(v) for v in by_subject.values()))

    subjects = [s for s in subjects if len(by_subject[s]) >= 12]
    random.Random(2026).shuffle(subjects)
    targets = subjects[:20]
    impostors = subjects[20:25] if len(subjects) >= 25 else []

    def save(subject, path, index):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        h, w = image.shape[:2]
        scale = 160 / min(h, w) if min(h, w) > 160 else 1.0
        if scale < 1:
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        dst = OUT / subject / f"{index:04d}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return str(dst)

    train, val, gallery, probe, impostor = [], [], [], [], []
    for label, subject in enumerate(targets):
        paths = sorted(by_subject[subject])
        random.Random(2026 + label).shuffle(paths)
        saved = [(save(subject, p, i), p) for i, p in enumerate(paths)]
        saved = [s for s, _ in saved if s]
        n = len(saved)
        n_train = max(5, int(n * 0.7))
        n_val = max(1, int(n * 0.1))
        train += [{"path": p, "label": label} for p in saved[:n_train]]
        val += [{"path": p, "label": label} for p in saved[n_train : n_train + n_val]]
        probe += [{"path": p, "label": label} for p in saved[n_train + n_val :]]
        gallery += [{"path": p, "label": label} for p in saved[:5]]

    for label, subject in enumerate(impostors):
        for i, p in enumerate(sorted(by_subject[subject])[:5]):
            path = save(subject, p, i)
            if path:
                impostor.append({"path": path, "label": label})

    manifest = {"train": train, "val": val, "gallery": gallery, "probe": probe, "impostor": impostor}
    (ROOT / "data" / "chokepoint_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in manifest.items():
        print(k, len(v))


if __name__ == "__main__":
    main()
