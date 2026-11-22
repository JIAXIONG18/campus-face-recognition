"""生成第二轮新增图表：活体 ROC、跨数据集对比、ChokePoint 样本。"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from face_recog.metrics import eer, roc_curve
from liveness.train import LivenessDataset, LivenessNet

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
FIG = EXP / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
BLUE = "#3b5bdb"
TEAL = "#0ca678"
ORANGE = "#e8590c"
GRAY = "#64748b"


def liveness():
    manifest = json.loads((ROOT / "data" / "spoof_manifest.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LivenessNet().to(device)
    model.load_state_dict(torch.load(EXP / "runs" / "liveness" / "best.pt", map_location=device)["model"])
    model.eval()
    loader = DataLoader(LivenessDataset([(s["path"], s["label"]) for s in manifest["test"]], False), batch_size=128)
    labels, logits = [], []
    with torch.no_grad():
        for images, y in loader:
            logits.append(model(images.to(device)).cpu())
            labels.append(y)
    labels = torch.cat(labels).numpy()
    logits = torch.cat(logits).numpy()
    pos, neg = logits[labels == 1], logits[labels == 0]
    fpr, tpr = roc_curve(pos, neg)
    e = eer(fpr, tpr)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].plot(fpr, tpr, color=BLUE, linewidth=1.6)
    axes[0].plot([0, 1], [0, 1], "--", color=GRAY, linewidth=0.8)
    axes[0].set_xlabel("误识率 FAR")
    axes[0].set_ylabel("通过率 TAR")
    axes[0].set_title(f"活体检测 ROC（EER={e*100:.2f}%）", fontsize=10)
    axes[0].grid(alpha=0.25, linestyle="--")
    bins = np.linspace(-4, 6, 60)
    axes[1].hist(neg, bins=bins, color="#ffc9c9", alpha=0.85, label="攻击样本")
    axes[1].hist(pos, bins=bins, color=TEAL, alpha=0.7, label="真实人脸")
    axes[1].set_xlabel("活体分数（logit）")
    axes[1].set_ylabel("样本数")
    axes[1].set_title("活体分数分布", fontsize=10)
    axes[1].legend(fontsize=8, frameon=False)
    fig.savefig(FIG / "fig_liveness.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("liveness saved, EER", round(e, 4))


def crossdomain():
    values = {"CelebA\n零样本": 0.1296, "ChokePoint\n零样本": 0.4810, "ChokePoint\n目标域微调": 0.9995}
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    colors = [ORANGE, ORANGE, TEAL]
    bars = ax.bar(list(values.keys()), [v * 100 for v in values.values()], color=colors, width=0.55)
    for bar, value in zip(bars, values.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 100 + 1, f"{value*100:.1f}%", ha="center", fontsize=9)
    ax.set_ylabel("Rank-1 准确率（%）")
    ax.set_title("跨数据集迁移测试", fontsize=10)
    ax.set_ylim(0, 108)
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    fig.savefig(FIG / "fig_crossdomain.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("crossdomain saved")


def chokepoint_samples():
    manifest = json.loads((ROOT / "data" / "chokepoint_manifest.json").read_text(encoding="utf-8"))
    gallery = manifest["gallery"][:10]
    import cv2

    fig, axes = plt.subplots(2, 5, figsize=(9, 3.9))
    for ax, item in zip(axes.ravel(), gallery):
        image = cv2.imread(item["path"], cv2.IMREAD_GRAYSCALE)
        ax.imshow(image, cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"ID {item['label']}", fontsize=8)
    fig.savefig(FIG / "fig_chokepoint_samples.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("chokepoint samples saved")


if __name__ == "__main__":
    liveness()
    crossdomain()
    chokepoint_samples()
