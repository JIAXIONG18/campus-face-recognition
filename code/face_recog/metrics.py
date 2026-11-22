"""识别与验证指标。"""
from __future__ import annotations

import numpy as np


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def rank1_accuracy(probe: np.ndarray, probe_labels: np.ndarray, gallery: np.ndarray, gallery_labels: np.ndarray):
    similarity = l2_normalize(probe) @ l2_normalize(gallery).T
    best = similarity.argmax(axis=1)
    return float((gallery_labels[best] == probe_labels).mean()), similarity, gallery_labels[best]


def pair_scores(probe, probe_labels, gallery, gallery_labels):
    similarity = l2_normalize(probe) @ l2_normalize(gallery).T
    same = probe_labels[:, None] == gallery_labels[None, :]
    return similarity[same], similarity[~same]


def roc_curve(positive: np.ndarray, negative: np.ndarray, steps: int = 2000):
    scores = np.concatenate([positive, negative])
    labels = np.concatenate([np.ones_like(positive), np.zeros_like(negative)])
    order = np.argsort(-scores)
    scores, labels = scores[order], labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    tpr = tp / max(len(positive), 1)
    fpr = fp / max(len(negative), 1)
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    if len(tpr) > steps:
        idx = np.linspace(0, len(tpr) - 1, steps).astype(int)
        tpr, fpr = tpr[idx], fpr[idx]
    return fpr, tpr


def eer(fpr: np.ndarray, tpr: np.ndarray) -> float:
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fnp := fnr - fpr)))
    return float((fpr[idx] + fnr[idx]) / 2)


def threshold_at_far(positive: np.ndarray, negative: np.ndarray, target_far: float = 0.001):
    threshold = float(np.quantile(negative, 1 - target_far))
    far = float((negative >= threshold).mean())
    frr = float((positive < threshold).mean())
    return threshold, far, frr


def accuracy_at(positive: np.ndarray, negative: np.ndarray, threshold: float):
    return float(((positive >= threshold).sum() + (negative < threshold).sum()) / (len(positive) + len(negative)))
