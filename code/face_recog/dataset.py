"""数据集与数据增广。"""
from __future__ import annotations

import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _augment(image: np.ndarray, rng: random.Random) -> np.ndarray:
    if rng.random() < 0.5:
        image = cv2.flip(image, 1)
    if rng.random() < 0.3:
        angle = rng.uniform(-10, 10)
        matrix = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, 1.0)
        image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REFLECT)
    if rng.random() < 0.5:
        alpha = rng.uniform(0.85, 1.15)
        beta = rng.uniform(-12, 12)
        image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        image = cv2.GaussianBlur(image, (3, 3), 0)
    return image


class FaceDataset(Dataset):
    def __init__(self, samples: list[tuple[str, int]], train: bool = True):
        self.samples = samples
        self.train = train
        self.rng = random.Random(2026)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.train:
            image = _augment(image, self.rng)
        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        if self.train and self.rng.random() < 0.25:
            top = self.rng.randint(0, 84)
            left = self.rng.randint(0, 84)
            tensor[:, top : top + 28, left : left + 28] = 0.0
        return tensor, label


class SimpleDataset(Dataset):
    def __init__(self, paths: list[str]):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int):
        image = cv2.imread(self.paths[index], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5
        return torch.from_numpy(image.transpose(2, 0, 1)).float(), 0
