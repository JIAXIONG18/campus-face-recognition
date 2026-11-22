"""模型评测：Rank-1、ROC/EER、阈值与 FAR/FRR。"""
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
from face_recog.metrics import accuracy_at, eer, l2_normalize, pair_scores, roc_curve
from face_recog.models import build_model

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
EXP = ROOT / "experiments"


def load_model(path: Path, device):
    state = torch.load(path, map_location="cpu")
    model = build_model(state.get("variant", "full"))
    model.load_state_dict(state["model"])
    return model.to(device).eval()


@torch.no_grad()
def embed_paths(model, paths, device, batch_size: int = 64, tta: bool = True):
    loader = DataLoader(SimpleDataset([str(p) for p in paths]), batch_size=batch_size, num_workers=4)
    feats = []
    for images, _ in loader:
        images = images.to(device)
        vector = F.normalize(model(images), dim=1)
        if tta:
            vector = F.normalize(vector + F.normalize(model(torch.flip(images, dims=[3])), dim=1), dim=1)
        feats.append(vector.cpu())
    return torch.cat(feats).numpy()


def templates_of(features: np.ndarray, labels: np.ndarray):
    classes = np.unique(labels)
    templates = np.stack([l2_normalize(features[labels == c].mean(axis=0, keepdims=True))[0] for c in classes])
    return templates, classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=str(EXP / "runs" / "finetune_full" / "best.pt"))
    parser.add_argument("--out", default=str(EXP / "metrics.json"))
    parser.add_argument("--tag", default="")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--splits", default=str(DATA / "splits.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    model = load_model(Path(args.ckpt), device)

    def collect(key):
        items = splits["stage2"][key]
        return [p for p, _ in items], np.array([l for _, l in items])

    gallery_paths, gallery_labels = collect("gallery")
    probe_paths, probe_labels = collect("test")
    val_paths, val_labels = collect("val")
    impostor_paths = [p for p, _ in splits["impostor"]]

    gallery = embed_paths(model, gallery_paths, device)
    probe = embed_paths(model, probe_paths, device)
    val = embed_paths(model, val_paths, device)
    impostor = embed_paths(model, impostor_paths, device)

    templates, classes = templates_of(gallery, gallery_labels)
    label_index = {int(c): i for i, c in enumerate(classes)}

    similarity = probe @ templates.T
    predicted = classes[similarity.argmax(1)]
    rank1 = float((predicted == probe_labels).mean())
    genuine = similarity[np.arange(len(probe)), [label_index[int(l)] for l in probe_labels]]

    impostor_scores = (impostor @ templates.T).max(axis=1)
    calib, heldout = impostor_scores[::2], impostor_scores[1::2]
    if args.threshold is not None:
        threshold = float(args.threshold)
    else:
        threshold = float(np.quantile(calib, 0.99)) if len(calib) else 0.6
    far_impostor = float((impostor_scores >= threshold).mean()) if args.threshold is not None else float(
        (heldout >= threshold).mean()
    )
    frr = float((genuine < threshold).mean())

    positive, negative = pair_scores(probe, probe_labels, gallery, gallery_labels)
    ver_acc = accuracy_at(positive, negative, threshold)
    fpr, tpr = roc_curve(positive, negative)
    equal_error = eer(fpr, tpr)
    far = float((negative >= threshold).mean())
    tar_at_far = {}
    for level in (1e-1, 1e-2, 1e-3):
        idx = int(np.searchsorted(fpr, level))
        tar_at_far[str(level)] = float(tpr[min(idx, len(tpr) - 1)])

    per_identity = {}
    for label in np.unique(probe_labels):
        mask = probe_labels == label
        per_identity[int(label)] = float((predicted[mask] == label).mean())

    metrics = {
        "checkpoint": str(args.ckpt),
        "threshold": threshold,
        "rank1": rank1,
        "verification_accuracy": ver_acc,
        "far": far,
        "frr": frr,
        "far_impostor": far_impostor,
        "eer": equal_error,
        "tar_at_far": tar_at_far,
        "n_gallery": len(gallery_paths),
        "n_probe": len(probe_paths),
        "n_impostor": len(impostor_paths),
        "per_identity": per_identity,
    }
    out = Path(args.out)
    if args.tag:
        out = out.with_name(f"metrics_{args.tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    np.savez(
        EXP / "embeddings.npz",
        gallery=gallery,
        gallery_labels=gallery_labels,
        probe=probe,
        probe_labels=probe_labels,
        impostor=impostor,
        templates=templates,
        template_labels=classes,
    )
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_identity"}, indent=1))


if __name__ == "__main__":
    main()
