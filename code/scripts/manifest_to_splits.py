"""把任意清单转为 splits.json，复用 train.py / evaluate.py。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def pairs(items):
    return [[s["path"], s["label"]] for s in items]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default=str(ROOT / "data" / "splits_cross.json"))
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    labels = [s["label"] for s in manifest["train"]]
    target_classes = max(labels) + 1
    splits = {
        "identity_names": {},
        "pretrain_classes": 0,
        "target_classes": target_classes,
        "pretrain": [],
        "stage2": {
            "train": pairs(manifest["train"]),
            "val": pairs(manifest.get("val", [])),
            "test": pairs(manifest["probe"]),
            "gallery": pairs(manifest["gallery"]),
        },
        "impostor": pairs(manifest.get("impostor", [])),
    }
    Path(args.out).write_text(json.dumps(splits, ensure_ascii=False), encoding="utf-8")
    print("wrote", args.out, "classes", target_classes)


if __name__ == "__main__":
    main()
