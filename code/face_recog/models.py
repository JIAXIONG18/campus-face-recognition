"""轻量化 ResNet-18 人脸特征提取网络。"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.PReLU(out_ch)

    def forward(self, x):
        return self.act(self.bn(self.pointwise(self.depthwise(x))))


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = torch.sigmoid(self.fc2(F.relu(self.fc1(w), inplace=True)))
        return x * w


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, use_se: bool = False, ds: bool = False):
        super().__init__()
        conv = DepthwiseSeparableConv if ds else self._conv3x3
        self.conv1 = conv(in_ch, out_ch, stride)
        self.conv2 = conv(out_ch, out_ch, 1)
        if not ds:
            self.bn1 = nn.BatchNorm2d(out_ch)
            self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch) if use_se else None
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm2d(out_ch)
            )

    @staticmethod
    def _conv3x3(in_ch: int, out_ch: int, stride: int):
        return nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)

    def forward(self, x):
        if isinstance(self.conv1, DepthwiseSeparableConv):
            out = self.conv2(self.conv1(x))
        else:
            out = F.relu(self.bn1(self.conv1(x)), inplace=True)
            out = self.bn2(self.conv2(out))
        if self.se is not None:
            out = self.se(out)
        return F.relu(out + self.shortcut(x), inplace=True)


class LResNet18(nn.Module):
    """ResNet-18 变体：可选深度可分离卷积、SE 注意力与嵌入维度。"""

    def __init__(
        self,
        embedding_dim: int = 128,
        use_se: bool = True,
        ds_early: bool = True,
        drop_ratio: float = 0.4,
    ):
        super().__init__()
        if ds_early:
            self.stem = nn.Sequential(
                DepthwiseSeparableConv(3, 64, stride=2),
                DepthwiseSeparableConv(64, 64, stride=1),
            )
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(3, 64, 3, 2, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.PReLU(64),
                nn.Conv2d(64, 64, 3, 1, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.PReLU(64),
            )
        channels = [64, 128, 256, 512]
        blocks = []
        for stage, (in_ch, out_ch, stride) in enumerate(zip([64] + channels[:-1], channels, [1, 2, 2, 2])):
            for block in range(2):
                blocks.append(
                    BasicBlock(
                        in_ch if block == 0 else out_ch,
                        out_ch,
                        stride if block == 0 else 1,
                        use_se=use_se,
                        ds=ds_early and stage == 0,
                    )
                )
        self.body = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(drop_ratio),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(512),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.body(x)
        return self.head(x)


def build_model(variant: str = "full") -> LResNet18:
    presets = {
        "baseline": dict(embedding_dim=512, use_se=False, ds_early=False),
        "ds": dict(embedding_dim=512, use_se=False, ds_early=True),
        "ds_se": dict(embedding_dim=512, use_se=True, ds_early=True),
        "full": dict(embedding_dim=128, use_se=True, ds_early=True),
    }
    return LResNet18(**presets[variant])


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def count_flops(model: nn.Module, input_size: tuple[int, int, int] = (3, 112, 112)) -> int:
    total = 0
    handles = []

    def conv_hook(module, inp, out):
        nonlocal total
        kernel = module.kernel_size[0] * module.kernel_size[1]
        total += out.numel() * kernel * (module.in_channels // module.groups)

    def linear_hook(module, inp, out):
        nonlocal total
        total += module.in_features * out.numel()

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
    model.eval()
    model(torch.zeros(1, *input_size))
    for handle in handles:
        handle.remove()
    return total
