#!/usr/bin/env python3
"""Replot the profile likelihood using a symmetrized second difference."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    type=Path,
    default=PAPER / "reproduction" / "profile" / "fig_profile.pt",
)
parser.add_argument(
    "--output",
    type=Path,
    default=PAPER / "figures" / "fig_profile.pdf",
)
args = parser.parse_args()

data = torch.load(args.input, map_location="cpu", weights_only=False)
ts = np.array(data["ts"], dtype=float)
zero = int(np.argmin(np.abs(ts)))
t_positive = ts[zero:]


def symmetrize(curve):
    values = np.array(curve, dtype=float)
    return np.array(
        [
            (values[zero + offset] + values[zero - offset]) / 2.0
            for offset in range(len(t_positive))
        ]
    )


colors = {
    "turnover (D+L) [soft]": "#1f77b4",
    "net (D-L) [stiff]": "#d62728",
    "transfer (T)": "#2ca02c",
}
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
for name, curve in data["curves"].items():
    axes[0].plot(
        t_positive,
        symmetrize(curve),
        "-o",
        ms=3,
        lw=1.8,
        color=colors.get(name, "k"),
        label=name,
    )
axes[0].set_title("Symmetrized curvature (global)")
axes[0].set_xlabel(r"step $|t|$ along unit direction")
axes[0].set_ylabel(
    r"$\frac{1}{2}\,[L(\hat\theta{+}t v){+}L(\hat\theta{-}t v){-}2L(\hat\theta)]$"
)
axes[0].legend(frameon=False, fontsize=8)
axes[0].grid(alpha=0.25)
axes[0].set_ylim(bottom=0)

for name, curve in data["branch_curves"].items():
    linestyle = "-" if "turn" in name else "--"
    color = "#1f77b4" if "turn" in name else "#d62728"
    axes[1].plot(
        t_positive,
        symmetrize(curve),
        linestyle,
        lw=1.3,
        color=color,
        alpha=0.7,
    )
axes[1].plot([], [], "-", color="#1f77b4", label="turnover (per branch)")
axes[1].plot([], [], "--", color="#d62728", label="net (per branch)")
axes[1].set_title("Symmetrized curvature (per branch)")
axes[1].set_xlabel(r"step $|t|$")
axes[1].legend(frameon=False, fontsize=8)
axes[1].grid(alpha=0.25)
axes[1].set_ylim(bottom=0)

args.output.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(args.output)

curvatures = {key: symmetrize(value) for key, value in data["curves"].items()}
turnover = next(key for key in curvatures if "turn" in key)
net = next(key for key in curvatures if "net" in key)
transfer = next(key for key in curvatures if "transfer" in key)
print(
    "symmetrized curvature at |t|=1.5: "
    f"turnover={curvatures[turnover][-1]:.1f} "
    f"net={curvatures[net][-1]:.1f} "
    f"transfer={curvatures[transfer][-1]:.1f}"
)
print(
    f"net/turnover = "
    f"{curvatures[net][-1] / max(curvatures[turnover][-1], 1e-9):.1f}x; "
    f"all >=0: {all((curvatures[key] >= -1e-6).all() for key in curvatures)}"
)
print(f"saved {args.output}")
