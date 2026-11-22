"""活体检测轻量网络与训练。"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]


class LivenessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            self._block(3, 16, 2),
            self._block(16, 32, 2),
            self._block(32, 64, 2),
            self._block(64, 128, 2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    @staticmethod
    def _block(in_ch, out_ch, stride):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


class LivenessDataset(Dataset):
    def __init__(self, samples, train=True):
        self.samples = samples
        self.train = train
        self.rng = random.Random(2026)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.train and self.rng.random() < 0.5:
            image = cv2.flip(image, 1)
        image = cv2.resize(image, (112, 112))
        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        return tensor, torch.tensor(float(label), dtype=torch.float32)


def train_loop(model, loader, optimizer, device):
    model.train()
    total = correct = 0
    loss_sum = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()
        pred = (logits > 0).float()
        loss_sum += loss.item() * labels.numel()
        correct += (pred == labels).sum().item()
        total += labels.numel()
    return loss_sum / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate_loop(model, loader, device, threshold=0.0):
    model.eval()
    labels, logits = [], []
    for images, y in loader:
        logits.append(model(images.to(device)).cpu())
        labels.append(y)
    labels = torch.cat(labels)
    logits = torch.cat(logits)
    pred = (logits > threshold).float()
    acc = float((pred == labels).float().mean())
    live = labels == 1
    far = float(((pred == 1) & (~live)).sum() / max((~live).sum(), 1))  # 攻击被误判为真实
    frr = float(((pred == 0) & live).sum() / max(live.sum(), 1))  # 真实被拒
    return acc, far, frr, labels, logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default=str(ROOT / "experiments" / "runs" / "liveness"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    samples = [(s["path"], s["label"]) for s in data["train"]]
    valid = [(s["path"], s["label"]) for s in data["valid"]]
    random.Random(0).shuffle(samples)

    train_loader = DataLoader(LivenessDataset(samples, True), batch_size=args.batch, shuffle=True, num_workers=4)
    valid_loader = DataLoader(LivenessDataset(valid, False), batch_size=args.batch, num_workers=2)

    model = LivenessNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    best = 0.0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_loop(model, train_loader, optimizer, device)
        scheduler.step()
        acc, far, frr, _, _ = evaluate_loop(model, valid_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_acc": acc, "far": far, "frr": frr})
        print(f"epoch {epoch:02d} loss={train_loss:.4f} acc={train_acc:.3f} val_acc={acc:.3f} FAR={far:.3f} FRR={frr:.3f} {time.time()-start:.0f}s", flush=True)
        if acc >= best:
            best = acc
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": acc}, out / "best.pt")
        (out / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    print("done best_val_acc", best)


if __name__ == "__main__":
    main()
