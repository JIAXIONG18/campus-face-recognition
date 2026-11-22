"""把论文公式渲染为透明/白底 PNG。"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "formulas"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"

FORMULAS = {
    "eq21": r"$y(i,j)=f\left(\sum_{m}\sum_{n}w(m,n)\,x(i+m,j+n)+b\right)$",
    "eq22": r"$y=f\left(F(x,\{W_i\})+x\right)$",
    "eq23": r"$L=-\frac{1}{N}\sum_{i}\log\frac{e^{s\cos(\theta_{y_i}+m)}}"
            r"{e^{s\cos(\theta_{y_i}+m)}+\sum_{j\neq y_i}e^{s\cos\theta_j}}$",
    "eq51": r"$\mathrm{Acc}_{\mathrm{Rank1}}=\frac{N_{\mathrm{correct}}}{N_{\mathrm{probe}}}$",
    "eq52": r"$\mathrm{FAR}=\frac{FP}{FP+TN},\qquad \mathrm{FRR}=\frac{FN}{FN+TP}$",
}

SIZES = {"eq21": (7.2, 1.0), "eq22": (4.4, 0.9), "eq23": (7.4, 1.7), "eq51": (4.2, 1.0), "eq52": (7.0, 0.9)}

for name, tex in FORMULAS.items():
    fig = plt.figure(figsize=SIZES[name])
    fig.text(0.5, 0.5, tex, ha="center", va="center", fontsize=16)
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=400, transparent=False, facecolor="white", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("saved", path)
