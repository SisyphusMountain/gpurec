"""Figure for paper S4.5: held-out NLL vs tree-smoothing prior strength kappa, Hogenom-1055 species-wise,
K=5-fold CV. Pure CPU: reads the two existing run_cv state.pt grids (coarse + refined), merges them
(refined overrides on shared kappa), marks kappa*=argmin, and the kappa=0 unpenalized-MLE baseline.
No GPU, no gpurec import. Run: PYTHONNOUSERSITE=1 /home/enzo/miniforge3/bin/python plot_fig_cv.py
"""
import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = "/home/enzo/Documents/git/gpurec/kernel-bench/paper/figures/fig_cv.pdf"

coarse = torch.load(f"{HERE}/runs/cv_1055/state.pt", weights_only=False, map_location="cpu")["cv"]
refined = torch.load(f"{HERE}/runs/cv_1055_refined/state.pt", weights_only=False, map_location="cpu")["cv"]
cv = {**coarse, **refined}            # refined overrides coarse on shared kappa {1, 0.1, 0}

mle = cv[0.0]                          # unpenalized MLE (kappa = 0)
ks = sorted(k for k in cv if k > 0)    # positive kappa, ascending
ystar_k = min(ks, key=lambda k: cv[k]) # = 0.03
ystar = cv[ystar_k]
gain = mle - ystar                     # held-out NLL improvement of the prior over the MLE

print(f"kappa grid: {ks}")
print(f"kappa* = {ystar_k} (held-out {ystar:.2f}); MLE(kappa=0) = {mle:.2f}; gain = {gain:.2f} NLL")

fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.set_xscale("log")
ax.plot(ks, [cv[k] for k in ks], "o-", color="#1f77b4", lw=1.8, ms=5,
        label="tree-smoothing prior ($K{=}5$ CV)")
ax.scatter([ystar_k], [ystar], s=200, marker="*", color="#d62728", zorder=6,
           label=rf"$\kappa^\star = {ystar_k:g}$")
ax.axhline(mle, ls="--", color="#888888", lw=1.2)
ax.text(ks[-1], mle, "  unpenalized MLE", va="center", ha="left", fontsize=9, color="#555555")
ax.annotate(rf"$-{gain:.0f}$ NLL", xy=(ystar_k, ystar), xytext=(ystar_k, ystar + 0.45 * gain),
            ha="center", fontsize=9, color="#d62728",
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1))

ax.set_xlabel(r"tree-smoothing prior strength $\kappa$")
ax.set_ylabel("held-out NLL (mean over 5 folds)")
ax.grid(True, which="both", alpha=0.25)
ax.legend(frameon=False, loc="upper center")
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT)
print(f"saved -> {OUT}")
