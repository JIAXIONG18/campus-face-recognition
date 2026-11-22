"""顺序执行四组消融实验并汇总指标。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
EXP = ROOT / "experiments"

VARIANTS = ["baseline", "ds", "ds_se", "full"]


def run(cmd):
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=CODE, check=True)


def main():
    summary = {}
    for variant in VARIANTS:
        tag = f"abl_{variant}"
        run(
            [
                sys.executable, "scripts/train.py", "--stage", "finetune", "--variant", variant,
                "--epochs", "30", "--batch", "32", "--lr", "0.005", "--margin", "0.2",
                "--scale", "32", "--workers", "0", "--tag", tag,
            ]
        )
        run(
            [
                sys.executable, "scripts/evaluate.py",
                "--ckpt", str(EXP / "runs" / tag / "best.pt"),
                "--out", str(EXP / f"metrics_{tag}.json"),
            ]
        )
        metrics = json.loads((EXP / f"metrics_{tag}.json").read_text(encoding="utf-8"))
        summary[variant] = {"rank1": metrics["rank1"], "eer": metrics["eer"]}
    (EXP / "ablation.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
