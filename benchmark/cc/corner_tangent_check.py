"""Forward-mode check at a rate corner: root-row tangents vs finite differences of the root rows (fp64).

Independent of the backward pass: if the tangent matches FD, the forward and tangent kernels are
right and any gradient error lives in the adjoint/VJP kernels.
Usage: corner_tangent_check.py --species S --families LIST --corner=D,L,T --step 1e-3
"""
from __future__ import annotations

import argparse
import sys

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True); ap.add_argument("--families", required=True)
    ap.add_argument("--corner", required=True); ap.add_argument("--step", required=True, type=float)
    args = ap.parse_args()
    from gpurec.api.model import GeneReconModel
    from gpurec.api.solver_options import SolverOptions
    from gpurec.fit.genewise_fit import _BASE_SOLVER
    from gpurec.solver.value_and_grad import forward_solve
    from gpurec.solver.hvp.forward_tangent import jvp_root_scores
    paths = [ln.strip() for ln in open(args.families) if ln.strip() and not ln.startswith("#")]
    so = SolverOptions(**{**_BASE_SOLVER, "pi_iters": 16, "neumann_terms": 16,
                          "forward_self_loop": "exact", "adjoint_self_loop": "exact"})
    m = GeneReconModel(args.species, paths, mode="genewise", device="cuda", dtype=torch.float64,
                       solver_options=so, clade_budget=None)
    m.receiver_weights.requires_grad_(False)
    static = m.batch_statics[0]
    rw = m.receiver_weights.detach()
    d, l, t = (float(x) for x in args.corner.split(","))
    theta = torch.tensor([d, l, t], dtype=torch.float64, device="cuda").repeat(len(paths), 1).contiguous()
    th0 = m._theta_for_static(static, theta)
    loss0, sv = forward_solve([static], th0, rw)
    root0 = sv["root_rows"].double()
    q = torch.softmax(root0 * torch.log(torch.tensor(2.0, dtype=torch.float64, device="cuda")), dim=-1)
    print(f"[tan] corner D={d} L={l} T={t} NLL={float(loss0.sum()):.6f} root rows {tuple(root0.shape)}", flush=True)
    for k, name in enumerate(("D", "L", "T")):
        v = torch.zeros_like(th0); v[..., k] = 1.0
        tang = jvp_root_scores(static, th0, v, sv, primal_gene_split=None,
                               leaf_fm_log=getattr(static, "leaf_fm_log", None)).double()
        e = torch.zeros_like(th0); e[..., k] = args.step
        _, svp = forward_solve([static], th0 + e, rw); rp = svp["root_rows"].double().clone()
        _, svm = forward_solve([static], th0 - e, rw); rm = svm["root_rows"].double().clone()
        fd = (rp - rm) / (2 * args.step)
        both = torch.isfinite(fd) & torch.isfinite(tang) & torch.isfinite(root0)
        diff = torch.where(both, (tang - fd).abs(), torch.zeros_like(fd))
        rmax = torch.where(torch.isfinite(root0), root0, torch.full_like(root0, -float("inf"))).amax(dim=-1, keepdim=True)
        depth = rmax - root0
        i = int(diff.argmax()); r, c = divmod(i, root0.shape[-1])
        head = both & (depth < 30)
        print(f"[tan]  d/d{name}: max |tangent - FD| over root lanes {float(diff.max()):.3e} at lane {c} (depth {float(depth[r, c]):.1f}; "
              f"tangent {float(tang[r, c]):+.6e}, FD {float(fd[r, c]):+.6e}); within 30 orders of the max: {float(diff[head].max()):.3e}; "
              f"root-part NLL derivative: tangent {float(-(q * tang).sum()):+.6e} vs FD {float(-(q * fd).sum()):+.6e}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
