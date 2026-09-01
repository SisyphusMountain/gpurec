#!/usr/bin/env python3
"""Plot gpurec/AleRax wall-clock from the versioned headline CSV."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=HERE.parent / "results" / "headline.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO
        / "papers"
        / "gpu-reconciliation"
        / "figures"
        / "fig_speed.pdf",
    )
    return parser.parse_args()


args = parse_args()
with args.results.open(newline="") as handle:
    rows = list(csv.DictReader(handle))

gpu = [row for row in rows if row["tool"] == "gpurec"]
cpu = next(row for row in rows if row["tool"] == "AleRax")
points = [
    (int(row["families"]), float(row["fit_seconds"]))
    for row in gpu
]
labels = [row["subset"] for row in gpu]
alerax = (int(cpu["families"]), float(cpu["fit_seconds"]))

ns = [n for n, _ in points]
ts = [t for _, t in points]
fig, ax = plt.subplots(figsize=(4.2, 3.2))
ax.loglog(
    ns,
    ts,
    "o-",
    color="#1f5fae",
    lw=1.8,
    ms=7,
    label="gpurec (RTX 4090)",
    zorder=3,
)
ax.loglog(
    [alerax[0]],
    [alerax[1]],
    "s",
    color="#c0392b",
    ms=9,
    label="AleRax (24 CPU cores)",
    zorder=4,
)
ax.annotate(
    "AleRax ($\\sim$23$\\times$ slower)",
    alerax,
    textcoords="offset points",
    xytext=(-6, 6),
    fontsize=7,
    ha="right",
    va="bottom",
    color="#c0392b",
)
ax.annotate(
    "",
    xy=points[-1],
    xytext=alerax,
    arrowprops=dict(arrowstyle="<->", color="#7f8c8d", lw=1.0),
    zorder=1,
)
for (n, runtime), label in zip(points, labels):
    ax.annotate(
        f"{label}\n{n} fam",
        (n, runtime),
        textcoords="offset points",
        xytext=(8, -4),
        fontsize=7,
        va="top",
    )
ax.set_xlabel("number of gene families ($\\geq 4$ species)")
ax.set_ylabel("wall-clock (s)")
ax.set_title("Genewise DTL-rate fitting: gpurec vs AleRax", fontsize=9)
ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
ax.legend(fontsize=7, loc="upper left")
fig.tight_layout()
args.output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.output)
print(
    f"wrote {args.output}  "
    f"(throughput {ns[0] / ts[0]:.1f} -> {ns[-1] / ts[-1]:.1f} fam/s)"
)
