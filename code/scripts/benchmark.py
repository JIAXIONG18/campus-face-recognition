"""模型复杂度与推理延迟测量。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from face_recog.models import build_model, count_flops, count_params

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"


def latency(model, device, runs: int = 100):
    model = model.to(device)
    model.eval()
    x = torch.randn(1, 3, 112, 112, device=device)
    with torch.no_grad():
        for _ in range(10):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(runs):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - start) / runs * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--out", default=str(EXP / "benchmark.json"))
    args = parser.parse_args()
    results = {}
    for variant in ["baseline", "ds", "ds_se", "full"]:
        model = build_model(variant)
        entry = {
            "params": count_params(model),
            "flops": count_flops(model),
            "cpu_latency_ms": latency(model, torch.device("cpu")),
        }
        if torch.cuda.is_available():
            entry["gpu_latency_ms"] = latency(model, torch.device("cuda"))
        results[variant] = entry
        print(variant, entry)

    rng = np.random.default_rng(0)
    gallery = rng.standard_normal((1000, 128)).astype(np.float32)
    gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)
    query = gallery[::100]
    start = time.perf_counter()
    for _ in range(100):
        _ = query @ gallery.T
    results["match_1000_ms"] = (time.perf_counter() - start) / 100 * 1000
    Path(args.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("match 1000 x 128:", results["match_1000_ms"], "ms")


if __name__ == "__main__":
    main()
