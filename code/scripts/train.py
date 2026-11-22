"""训练脚本：预训练 / 目标集微调 / 单阶段消融。"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from face_recog.dataset import FaceDataset
from face_recog.losses import ArcFaceLoss
from face_recog.models import build_model, count_params

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
EXP = ROOT / "experiments"


def set_seed(seed: int = 2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_classifier(model, loader, criterion, device):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        embeddings = model(images)
        loss, logits = criterion(embeddings, labels)
        loss_sum += loss.item() * labels.numel()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.numel()
    return loss_sum / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def embedding_accuracy(model, gallery_loader, probe_loader, device):
    model.eval()
    gallery, gallery_labels = [], []
    for images, labels in gallery_loader:
        gallery.append(F.normalize(model(images.to(device)), dim=1).cpu())
        gallery_labels.append(labels)
    gallery = torch.cat(gallery)
    gallery_labels = torch.cat(gallery_labels)
    correct = total = 0
    for images, labels in probe_loader:
        embeddings = F.normalize(model(images.to(device)), dim=1).cpu()
        predicted = gallery_labels[(embeddings @ gallery.T).argmax(1)]
        correct += (predicted == labels).sum().item()
        total += labels.numel()
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["pretrain", "finetune", "combined"], default="pretrain")
    parser.add_argument("--variant", default="full")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--scale", type=float, default=32.0)
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--init", default="")
    parser.add_argument("--tag", default="")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--splits", default=str(DATA / "splits.json"))
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = json.loads(Path(args.splits).read_text(encoding="utf-8"))

    if args.stage == "pretrain":
        samples = [tuple(x) for x in splits["pretrain"]]
        num_classes = splits["pretrain_classes"]
    elif args.stage == "finetune":
        samples = [tuple(x) for x in splits["stage2"]["train"]]
        num_classes = splits["target_classes"]
    else:
        samples = [tuple(x) for x in splits["pretrain"]]
        samples += [(p, int(l) + splits["pretrain_classes"]) for p, l in splits["stage2"]["train"]]
        num_classes = splits["pretrain_classes"] + splits["target_classes"]
    if args.limit:
        samples = samples[: args.limit]

    train_loader = DataLoader(
        FaceDataset(samples, train=True),
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        persistent_workers=args.workers > 0,
        pin_memory=True,
    )
    val_samples = [tuple(x) for x in splits["stage2"]["val"]]
    val_loader = DataLoader(FaceDataset(val_samples, train=False), batch_size=32, num_workers=2)
    gallery_samples = [tuple(x) for x in splits["stage2"]["gallery"]]
    gallery_loader = DataLoader(FaceDataset(gallery_samples, train=False), batch_size=32, num_workers=2)

    model = build_model(args.variant).to(device)
    if args.init:
        state = torch.load(args.init, map_location="cpu")
        missing, unexpected = model.load_state_dict(state["model"], strict=False)
        print(f"init from {args.init}, missing={len(missing)} unexpected={len(unexpected)}")
    criterion = ArcFaceLoss(
        model.head[-2].out_features, num_classes, scale=args.scale, margin=args.margin
    ).to(device)

    optimizer = torch.optim.SGD(
        list(model.parameters()) + list(criterion.parameters()),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.wd,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    tag = args.tag or f"{args.stage}_{args.variant}"
    out_dir = EXP / "runs" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_acc = 0.0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = correct = total = 0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                embeddings = model(images)
                loss, logits = criterion(embeddings, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * labels.numel()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.numel()

        scheduler.step()
        train_loss = loss_sum / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc = evaluate_classifier(model, val_loader, criterion, device)
        if args.stage == "pretrain":
            val_acc = embedding_accuracy(model, gallery_loader, val_loader, device)
            val_loss = 0.0
        record = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "elapsed": time.time() - start,
        }
        history.append(record)
        print(
            f"[{tag}] epoch {epoch:03d}/{args.epochs} loss={train_loss:.4f} acc={train_acc:.4f} "
            f"val_acc={val_acc:.4f} lr={record['lr']:.5f} {record['elapsed']:.0f}s",
            flush=True,
        )
        (out_dir / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")

        state = {
            "model": model.state_dict(),
            "variant": args.variant,
            "embedding_dim": model.head[-2].out_features,
            "num_classes": num_classes,
            "epoch": epoch,
            "val_acc": val_acc,
            "stage": args.stage,
        }
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(state, out_dir / "best.pt")
        torch.save(state, out_dir / "last.pt")

    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "tag": tag,
                "stage": args.stage,
                "variant": args.variant,
                "params": count_params(model),
                "num_classes": num_classes,
                "train_images": len(samples),
                "best_val_acc": best_acc,
                "epochs": args.epochs,
                "batch": args.batch,
                "lr": args.lr,
                "seconds": time.time() - start,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"done {tag}: best_val_acc={best_acc:.4f} params={count_params(model)/1e6:.2f}M")


if __name__ == "__main__":
    main()
