"""模型结构与损失函数单元测试。"""
from __future__ import annotations

import torch

from face_recog.losses import ArcFaceLoss
from face_recog.models import build_model, count_flops, count_params


def test_all_variants_forward_shape():
    expected = {"baseline": 512, "ds": 512, "ds_se": 512, "full": 128}
    for variant, dim in expected.items():
        model = build_model(variant).eval()
        with torch.no_grad():
            output = model(torch.randn(2, 3, 112, 112))
        assert output.shape == (2, dim), variant
        assert torch.isfinite(output).all(), variant


def test_param_and_flop_budget():
    """最终模型控制在 11.5M 参数、1.3 GFLOPs 量级（论文实测值）。"""
    model = build_model("full")
    assert 11.0e6 < count_params(model) < 12.0e6
    assert 1.2e9 < count_flops(model) < 1.4e9


def test_arcface_loss_is_finite_and_differentiable():
    criterion = ArcFaceLoss(embedding_dim=128, num_classes=10, margin=0.2, scale=32.0)
    embeddings = torch.randn(4, 128, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3])
    loss, logits = criterion(embeddings, labels)
    loss.backward()

    assert logits.shape == (4, 10)
    assert torch.isfinite(loss)
    assert loss.item() > 0
    assert embeddings.grad is not None and torch.isfinite(embeddings.grad).all()
