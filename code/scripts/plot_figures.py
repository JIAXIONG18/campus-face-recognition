"""生成论文所需图表。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from face_recog.metrics import pair_scores, roc_curve

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
FIG = EXP / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 200
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.edgecolor"] = "#94a3b8"
plt.rcParams["axes.linewidth"] = 0.8

BLUE = "#3b5bdb"
TEAL = "#0ca678"
ORANGE = "#e8590c"
GRAY = "#64748b"


def save(fig, name):
    path = FIG / name
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print("saved", path)


def box(ax, x, y, w, h, text, color="#eef2ff", edge=BLUE, size=10, weight="normal"):
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.1, edgecolor=edge, facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color="#1f2937", weight=weight)


def arrow(ax, x1, y1, x2, y2, color=GRAY, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=12,
            linewidth=1.1, color=color, connectionstyle=f"arc3,rad={rad}", linestyle=ls,
        )
    )


def training_curves():
    hist_p = json.loads((EXP / "runs" / "pretrain_full" / "history.json").read_text(encoding="utf-8"))
    hist_f = json.loads((EXP / "runs" / "finetune_full" / "history.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    axes[0].plot([r["epoch"] for r in hist_p], [r["train_loss"] for r in hist_p], color=BLUE)
    axes[0].set_title("(a) 预训练损失", fontsize=10)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].plot([r["epoch"] for r in hist_f], [r["train_loss"] for r in hist_f], color=TEAL, label="训练损失")
    axes[1].plot([r["epoch"] for r in hist_f], [r["val_loss"] for r in hist_f], color=ORANGE, label="验证损失")
    axes[1].set_title("(b) 微调阶段损失", fontsize=10)
    axes[1].set_xlabel("Epoch")
    axes[1].legend(fontsize=8, frameon=False)
    axes[2].plot([r["epoch"] for r in hist_f], [r["val_acc"] for r in hist_f], color=BLUE)
    axes[2].set_title("(c) 验证集识别准确率", fontsize=10)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_ylim(0, 1.05)
    for ax in axes:
        ax.grid(alpha=0.25, linestyle="--")
    save(fig, "fig_training.png")


def roc_and_scores():
    data = np.load(EXP / "embeddings.npz")
    metrics = json.loads((EXP / "metrics.json").read_text(encoding="utf-8"))
    positive, negative = pair_scores(data["probe"], data["probe_labels"], data["gallery"], data["gallery_labels"])
    fpr, tpr = roc_curve(positive, negative)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].plot(fpr, tpr, color=BLUE, linewidth=1.6)
    axes[0].plot([0, 1], [0, 1], "--", color=GRAY, linewidth=0.8)
    axes[0].set_xlabel("误识率 FAR")
    axes[0].set_ylabel("通过率 TAR")
    axes[0].set_title(f"ROC 曲线（EER={metrics['eer']*100:.2f}%）", fontsize=10)
    axes[0].grid(alpha=0.25, linestyle="--")
    bins = np.linspace(-0.2, 1.0, 60)
    axes[1].hist(negative, bins=bins, color="#c7d2fe", alpha=0.85, label="不同人")
    axes[1].hist(positive, bins=bins, color=TEAL, alpha=0.7, label="同一人")
    axes[1].axvline(metrics["threshold"], color=ORANGE, linestyle="--", linewidth=1.2, label="判定阈值")
    axes[1].set_xlabel("余弦相似度")
    axes[1].set_ylabel("样本数")
    axes[1].set_title("相似度分布", fontsize=10)
    axes[1].legend(fontsize=8, frameon=False)
    save(fig, "fig_roc.png")


def threshold_curve():
    data = np.load(EXP / "embeddings.npz")
    positive, negative = pair_scores(data["probe"], data["probe_labels"], data["gallery"], data["gallery_labels"])
    thresholds = np.linspace(-0.1, 0.9, 200)
    frr = [(positive < t).mean() for t in thresholds]
    far = [(negative >= t).mean() for t in thresholds]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.plot(thresholds, frr, color=BLUE, label="拒识率 FRR")
    ax.plot(thresholds, far, color=ORANGE, label="误识率 FAR")
    ax.set_xlabel("判定阈值")
    ax.set_ylabel("比率")
    ax.set_title("阈值与错误率的关系", fontsize=10)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(fontsize=8, frameon=False)
    save(fig, "fig_threshold.png")


def confusion():
    data = np.load(EXP / "embeddings.npz")
    templates, labels = data["templates"], data["template_labels"]
    probe, probe_labels = data["probe"], data["probe_labels"]
    best = labels[(probe @ templates.T).argmax(1)]
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    index = {v: i for i, v in enumerate(labels)}
    for true, pred in zip(probe_labels, best):
        matrix[index[true], index[pred]] += 1
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xlabel("预测身份编号")
    ax.set_ylabel("真实身份编号")
    ax.set_title("50 类识别混淆矩阵", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    save(fig, "fig_confusion.png")


def tsne_plot():
    data = np.load(EXP / "embeddings.npz")
    gallery = data["gallery"]
    labels = data["gallery_labels"]
    keep = np.isin(labels, np.unique(labels)[:12])
    points = TSNE(n_components=2, perplexity=18, random_state=7, init="pca").fit_transform(gallery[keep])
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    cmap = plt.get_cmap("tab20")
    for i, label in enumerate(np.unique(labels[keep])):
        mask = labels[keep] == label
        ax.scatter(points[mask, 0], points[mask, 1], s=26, color=cmap(i % 20), label=str(label), alpha=0.85)
    ax.set_title("12 个身份的特征分布（t-SNE）", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    save(fig, "fig_tsne.png")


def per_identity():
    metrics = json.loads((EXP / "metrics.json").read_text(encoding="utf-8"))
    values = np.array(list(metrics["per_identity"].values()))
    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(9, 2.9))
    ax.bar(range(len(values)), values[order], color=BLUE, width=0.82)
    ax.axhline(values.mean(), color=ORANGE, linestyle="--", linewidth=1, label=f"平均 {values.mean()*100:.1f}%")
    ax.set_xlabel("身份编号（按准确率排序）")
    ax.set_ylabel("Rank-1 准确率")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("各身份识别准确率", fontsize=10)
    save(fig, "fig_per_identity.png")


def model_architecture():
    fig, ax = plt.subplots(figsize=(12, 3.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    nodes = [
        (0.06, "输入\n112×112×3", "#f1f5f9"),
        (0.17, "Stem\nDSConv×2", "#e0e7ff"),
        (0.29, "Stage1\n64×56×56", "#dbeafe"),
        (0.41, "Stage2\n128×28×28", "#dbeafe"),
        (0.53, "Stage3\n256×14×14", "#dbeafe"),
        (0.65, "Stage4\n512×7×7", "#dbeafe"),
        (0.77, "SE 通道\n注意力", "#dcfce7"),
        (0.88, "GAP + FC\n512 → 128", "#fef3c7"),
        (0.97, "L2\n归一化", "#fee2e2"),
    ]
    for i, (x, text, color) in enumerate(nodes):
        box(ax, x, 0.5, 0.095, 0.42, text, color=color, size=8.5)
        if i < len(nodes) - 1:
            arrow(ax, x + 0.048, 0.5, nodes[i + 1][0] - 0.048, 0.5)
    ax.text(
        0.5, 0.06,
        "每个残差阶段由 2 个 BasicBlock 组成；Stage1 使用深度可分离卷积，Stage2–4 使用标准 3×3 卷积",
        ha="center", fontsize=8.5, color=GRAY,
    )
    save(fig, "fig_model.png")


def algorithm_flow():
    fig, ax = plt.subplots(figsize=(4.8, 9.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.axis("off")
    steps = [
        (0.93, "图像采集\n（摄像头 / 上传）"),
        (0.84, "人脸检测（YuNet）"),
        (0.75, "五点对齐\n相似变换到 112×112"),
        (0.66, "质量筛选\n尺寸 / 清晰度"),
        (0.57, "特征提取\n轻量化 ResNet-18"),
        (0.48, "L2 归一化\n128 维特征"),
        (0.39, "余弦相似度匹配\n遍历特征库"),
    ]
    for y, text in steps:
        box(ax, 0.5, y, 0.8, 0.07, text, size=9)
    for i in range(len(steps) - 1):
        arrow(ax, 0.5, steps[i][0] - 0.035, 0.5, steps[i + 1][0] + 0.035)
    box(ax, 0.5, 0.28, 0.8, 0.07, "最高相似度 ≥ 判定阈值？", color="#fef9c3", edge=ORANGE, size=9)
    arrow(ax, 0.5, 0.355, 0.5, 0.315)
    box(ax, 0.27, 0.12, 0.36, 0.08, "判为未注册人员\n不写入通过记录", color="#fff1f2", edge="#e03131", size=8.5)
    box(ax, 0.73, 0.12, 0.36, 0.08, "返回身份\n写入识别记录", color="#ecfdf5", edge=TEAL, size=8.5)
    arrow(ax, 0.38, 0.245, 0.30, 0.16, color="#e03131")
    arrow(ax, 0.62, 0.245, 0.70, 0.16, color=TEAL)
    ax.text(0.29, 0.215, "否", fontsize=9, color="#e03131", ha="right")
    ax.text(0.71, 0.215, "是", fontsize=9, color=TEAL, ha="left")
    ax.text(0.5, 0.995, "识别算法流程", ha="center", fontsize=11, weight="bold")
    save(fig, "fig_algorithm.png")


def system_architecture():
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    layers = [
        (0.79, "表现层", "Web 前端（HTML/CSS/JS）：识别页 · 录入页 · 记录页", "#e0e7ff"),
        (0.57, "服务层", "Flask：路由分发 · 业务逻辑 · 权限与参数校验", "#dbeafe"),
        (0.35, "算法层", "YuNet 检测对齐 · ResNet-18 特征提取 · 余弦匹配", "#dcfce7"),
        (0.13, "数据层", "SQLite：用户表 · 特征表 · 记录表；本地图片存储", "#fef3c7"),
    ]
    for y, title, desc, color in layers:
        box(ax, 0.5, y, 0.86, 0.15, "", color=color, edge="#94a3b8")
        ax.text(0.13, y + 0.035, title, fontsize=11, weight="bold", color="#1f2937")
        ax.text(0.13, y - 0.028, desc, fontsize=9, color="#334155")
    for y in (0.79, 0.57, 0.35):
        arrow(ax, 0.5, y - 0.075, 0.5, y - 0.105, style="<|-|>")
    ax.text(0.5, 0.985, "系统总体架构", ha="center", fontsize=12, weight="bold")
    save(fig, "fig_system.png")


def benchmark_chart():
    data = json.loads((EXP / "benchmark.json").read_text(encoding="utf-8"))
    names = {"baseline": "基线", "ds": "+深度可分离", "ds_se": "+SE 注意力", "full": "最终模型"}
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3))
    keys = list(names)
    params = [data[k]["params"] / 1e6 for k in keys]
    flops = [data[k]["flops"] / 1e9 for k in keys]
    latency = [data[k]["cpu_latency_ms"] for k in keys]
    for ax, values, title, color in zip(
        axes, [params, flops, latency],
        ["参数量（M）", "计算量（GFLOPs）", "CPU 单张推理（ms）"], [BLUE, TEAL, ORANGE],
    ):
        bars = ax.bar([names[k] for k in keys], values, color=color, width=0.6)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
        ax.grid(axis="y", alpha=0.2, linestyle="--")
    save(fig, "fig_benchmark.png")


if __name__ == "__main__":
    model_architecture()
    algorithm_flow()
    system_architecture()
    for fn in (training_curves, roc_and_scores, threshold_curve, confusion, tsne_plot, per_identity, benchmark_chart):
        try:
            fn()
        except Exception as exc:
            print("skip", fn.__name__, exc)
