"""Profile-likelihood slices along the stiff/soft duplication-loss axes (GPT 5.5 Pro #2 leverage).

Turns the eigenvalue claim of Fig 2 into interpretable likelihood geometry: at the converged archaeal
rate fit theta_hat, evaluate the DATA NLL along
   turnover  (delta_D = delta_L  = +1, delta_T=0)   [soft]
   net       (delta_D = -delta_L = +1, delta_T=0)   [stiff]
   transfer  (delta_T = +1)                          [decoupled, shallow]
both GLOBALLY (all branches together) and for a few representative branches, over a 1-D grid of step t
(unit-normalized directions, so curvatures are comparable). Plot -> fig_profile.pdf.

Env: FAMILIES(0=all) THETA(<path>|CERTIFIED) NPTS(21) TMAX(1.5) OUT(figures/fig_profile.pdf).
"""
import os, sys, math
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import run_cv
from run_cv import DATASETS, build_model, _CV_SO
from gpurec import SolverOptions
from gpurec.solver.value_and_grad import make_value_and_grad
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
DEV = "cuda"
CERT = os.environ.get(
    "THETA",
    str(HERE / "_artifacts" / "certified_v2" / "archaea_lam0.03_certified.pt"),
)


def main():
    n_fam = int(os.environ.get("FAMILIES", "0"))
    npts = int(os.environ.get("NPTS", "21")); tmax = float(os.environ.get("TMAX", "1.5"))
    out = os.environ.get(
        "OUT",
        str(REPO / "papers" / "gpu-reconciliation" / "figures" / "fig_profile.pdf"),
    )
    ds = DATASETS["archaea"]; run_cv._SP_TREE = ds["species_tree"]
    so = SolverOptions(**_CV_SO); so.validate()
    paths = ds["families"](None if n_fam <= 0 else n_fam)
    model = build_model(paths, so); bs = model.batch_statics
    S = int(model.species_helpers["S"]); rw = model.receiver_weights.detach().clone()
    saved = torch.load(CERT, weights_only=False, map_location=DEV)
    theta_saved = saved.get("theta", saved.get("theta_newton"))
    if theta_saved is None:
        raise KeyError(f"{CERT} contains neither 'theta' nor 'theta_newton'")
    th = theta_saved.to(DEV).float().reshape(S, 3)
    f = make_value_and_grad(bs, rw, theta_shape=(S, 3))                 # data NLL (no penalty), receiver uniform
    def loss(theta): return float(f(theta.reshape(-1))[0])
    L0 = loss(th)
    print(f"[profile] S={S} branches, {len(paths)} families, L0(data NLL)={L0:.2f}", flush=True)

    ts = torch.linspace(-tmax, tmax, npts)
    def slice_along(vec):                                              # vec: [S,3], unit-normalized
        v = vec / vec.norm()
        return [loss(th + float(t) * v) - L0 for t in ts]

    # global directions (all branches together)
    g_turn = torch.zeros(S, 3, device=DEV); g_turn[:, 0] = 1; g_turn[:, 1] = 1
    g_net = torch.zeros(S, 3, device=DEV); g_net[:, 0] = 1; g_net[:, 1] = -1
    g_tau = torch.zeros(S, 3, device=DEV); g_tau[:, 2] = 1
    curves = {}
    for name, v in [("turnover (D+L) [soft]", g_turn), ("net (D-L) [stiff]", g_net), ("transfer (T)", g_tau)]:
        curves[name] = slice_along(v)
        c = curves[name]
        print(f"  global {name:24s}: dNLL at +-{tmax} = {c[0]:.1f}/{c[-1]:.1f}", flush=True)

    # a few representative leaf branches (the strongest D-L coupled ones are interesting; use first 3 leaves)
    branch_curves = {}
    for b in [0, 1, 2]:
        for tag, dD, dL in [("turn", 1, 1), ("net", 1, -1)]:
            v = torch.zeros(S, 3, device=DEV); v[b, 0] = dD; v[b, 1] = dL
            branch_curves[f"branch{b}-{tag}"] = slice_along(v)

    torch.save(dict(ts=ts, L0=L0, curves=curves, branch_curves=branch_curves, S=S, n=len(paths)),
               out.replace(".pdf", ".pt"))

    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8))
    colors = {"turnover (D+L) [soft]": "#1f77b4", "net (D-L) [stiff]": "#d62728", "transfer (T)": "#2ca02c"}
    for name, c in curves.items():
        ax[0].plot(ts.numpy(), c, "-o", ms=3, lw=1.8, color=colors[name], label=name)
    ax[0].set_title("Global profile likelihood"); ax[0].set_xlabel("step $t$ along unit direction")
    ax[0].set_ylabel(r"$\Delta$ data NLL"); ax[0].legend(frameon=False, fontsize=8); ax[0].grid(alpha=.25)
    for k, c in branch_curves.items():
        ls = "-" if "turn" in k else "--"; col = "#1f77b4" if "turn" in k else "#d62728"
        ax[1].plot(ts.numpy(), c, ls, lw=1.3, color=col, alpha=.7)
    ax[1].plot([], [], "-", color="#1f77b4", label="turnover (per branch)")
    ax[1].plot([], [], "--", color="#d62728", label="net (per branch)")
    ax[1].set_title("Per-branch profile likelihood"); ax[1].set_xlabel("step $t$")
    ax[1].legend(frameon=False, fontsize=8); ax[1].grid(alpha=.25)
    fig.tight_layout(); os.makedirs(os.path.dirname(out), exist_ok=True); fig.savefig(out)
    print(f"[saved] {out}", flush=True)


if __name__ == "__main__":
    main()
