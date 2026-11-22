"""对任意 gallery/probe/impostor 清单评测（跨数据集测试用）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from face_recog.dataset import SimpleDataset
from face_recog.metrics import l2_normalize, pair_scores, roc_curve, eer
from face_recog.models import build_model

ROOT = Path(__file__).resolve().parents[2]


def load_model(path, device):
    state = torch.load(path, map_location="cpu")
    model = build_model(state.get("variant", "full"))
    model.load_state_dict(state["model"])
    return model.to(device).eval()


@torch.no_grad()
def embed_paths(model, paths, device, batch=64):
    loader = DataLoader(SimpleDataset([str(p) for p in paths]), batch_size=batch, num_workers=4)
    feats = []
    for images, _ in loader:
        images = images.to(device)
        v = F.normalize(model(images), dim=1)
        v = F.normalize(v + F.normalize(model(torch.flip(images, dims=[3])), dim=1), dim=1)
        feats.append(v.cpu())
    return torch.cat(feats).numpy()


def templates(features, labels):
    classes = np.unique(labels)
    return np.stack([l2_normalize(features[labels == c].mean(0, keepdims=True))[0] for c in classes]), classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--ckpt", default=str(ROOT / "experiments" / "runs" / "finetune_full" / "best.pt"))
    parser.add_argument("--threshold", type=float, default=0.70)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    model = load_model(Path(args.ckpt), device)

    gallery = embed_paths(model, [s["path"] for s in manifest["gallery"]], device)
    probe = embed_paths(model, [s["path"] for s in manifest["probe"]], device)
    gallery_labels = np.array([s["label"] for s in manifest["gallery"]])
    probe_labels = np.array([s["label"] for s in manifest["probe"]])
    impostor = embed_paths(model, [s["path"] for s in manifest.get("impostor", [])], device) if manifest.get("impostor") else None

    tmpl, classes = templates(gallery, gallery_labels)
    idx = {int(c): i for i, c in enumerate(classes)}
    similarity = probe @ tmpl.T
    predicted = classes[similarity.argmax(1)]
    rank1 = float((predicted == probe_labels).mean())
    genuine = similarity[np.arange(len(probe)), [idx[int(l)] for l in probe_labels]]

    pos, neg = pair_scores(probe, probe_labels, gallery, gallery_labels)
    fpr, tpr = roc_curve(pos, neg)
    result = {"rank1": rank1, "eer": eer(fpr, tpr), "n_probe": len(probe)}
    if impostor is not None:
        far = float(((impostor @ tmpl.T).max(1) >= args.threshold).mean())
        frr = float((genuine < args.threshold).mean())
        result.update({"far": far, "frr": frr, "threshold": args.threshold})
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in result.items()}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
