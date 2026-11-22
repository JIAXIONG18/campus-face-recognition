"""评测指标单元测试（纯 numpy，不依赖模型）。"""
from __future__ import annotations

import numpy as np

from face_recog.metrics import eer, rank1_accuracy, roc_curve, threshold_at_far


def test_rank1_perfect_separation():
    gallery = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    probe = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    gallery_labels = np.array([0, 1])
    probe_labels = np.array([0, 1])

    accuracy, similarity, predicted = rank1_accuracy(probe, probe_labels, gallery, gallery_labels)

    assert accuracy == 1.0
    assert similarity.shape == (2, 2)
    assert predicted.tolist() == [0, 1]


def test_eer_on_separable_scores_is_zero():
    positive = np.linspace(0.8, 1.0, 100)
    negative = np.linspace(0.0, 0.4, 100)
    fpr, tpr = roc_curve(positive, negative)
    assert eer(fpr, tpr) < 1e-6


def test_threshold_at_far_controls_false_accepts():
    rng = np.random.default_rng(0)
    negative = rng.normal(0.3, 0.05, 5000)
    positive = rng.normal(0.8, 0.05, 5000)

    threshold, far, frr = threshold_at_far(positive, negative, target_far=0.01)

    assert 0.0 <= far <= 0.02
    assert frr < 0.001
    assert positive.mean() > threshold > negative.mean()
