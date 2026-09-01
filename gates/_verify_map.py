"""Phase-2 verification gates for the MAP/CV harness (small live-hogenom model).

1. ADDITIVITY: at a fixed theta, NLL(train_sub)+NLL(test_sub) == NLL(full) and the gradients add
   (validates the data-subsetting masking + shared-E premise the CV relies on).
2. PRIOR GRAD: grad(prior on) - grad(prior off) == lam*(theta-theta_ref) exactly.
3. FINITE DIFFERENCE: directional FD of the MAP loss matches the analytic MAP gradient.

    python -m gates._verify_map
"""

from __future__ import annotations

import glob
import math

import torch

from gpurec import GeneReconModel, SolverOptions
from gpurec.api._execution import stream_batches
from gpurec.fit.map_cv import _DEFAULT_SO
from gpurec.solver.value_and_grad import make_value_and_grad
from gates._paths import HOGENOM_ROOT

_ROOT = str(HOGENOM_ROOT)
_SP = (f"{_ROOT}/runs/MFP/true_start_ufboot1000/"
       "run_--gene-tree-samples_100_--per-family-rates_1/alegenerax/species_trees/"
       "starting_species_tree.newick")


def _build(paths, so, device):
    return GeneReconModel(_SP, [str(p) for p in paths], mode="specieswise", device=device,
                          solver_options=so)


def run(n=30, n_train=22, device="cuda", pi_iters=64, neumann_terms=64):
    # The FD gate needs a CONVERGED solver: at the fixture's pi_iters=16/neumann=16 the forward
    # is biased and the backward truncation differs, so FD-of-loss vs gradient disagree ~5% (a
    # known truncation artifact, not a bug). pi>=64/neumann>=64 converges both -> FD matches the
    # fp32 floor. (Additivity + prior gates are convergence-independent.)
    so = SolverOptions(**{**_DEFAULT_SO, "pi_iters": pi_iters, "neumann_terms": neumann_terms})
    so.validate()
    trees = sorted(glob.glob(f"{_ROOT}/families/*/gene_trees/ufboot1000.MFP.geneTree.newick"))[:n]
    full = _build(trees, so, device)
    train = _build(trees[:n_train], so, device)
    test = _build(trees[n_train:], so, device)
    S = int(full.species_helpers["S"])
    torch.manual_seed(0)
    theta = (torch.full((S, 3), math.log2(0.1), device=device)
             + 0.2 * torch.randn(S, 3, device=device))
    rw = torch.zeros(S, device=device)

    def ng(m):
        L, g, _, _ = stream_batches(m.batch_statics, theta, rw, torch.zeros_like(rw), genewise=False, need_grad=True)
        return float(L), g.reshape(-1)

    Lf, gf = ng(full)
    LA, gA = ng(train)
    LB, gB = ng(test)
    add_loss = abs(Lf - (LA + LB)) / max(1.0, abs(Lf))
    add_grad = float((gf - (gA + gB)).norm() / gf.norm())
    ok1 = add_loss < 1e-4 and add_grad < 1e-3
    print(f"[1 additivity] NLL rel={add_loss:.2e}  grad rel_L2={add_grad:.2e}  -> {'PASS' if ok1 else 'FAIL'}")

    # 2. prior grad term: grad(on) - grad(off) == lam*(theta-theta_ref)
    lam = 12.0
    ref = torch.full((S, 3), math.log2(0.1), device=device)
    f_off = make_value_and_grad(train.batch_statics, rw, theta_shape=(S, 3))
    f_on = make_value_and_grad(train.batch_statics, rw, theta_shape=(S, 3), prior=(lam, ref))
    _, g_off, _, _ = f_off(theta.reshape(-1))
    L_on, g_on, _, _ = f_on(theta.reshape(-1))
    analytic = lam * (theta.reshape(-1) - ref.reshape(-1))
    # the data-grad part has atomic noise between the two calls; compare to the analytic prior term
    prior_err = float((g_on - g_off - analytic).norm() / analytic.norm().clamp_min(1e-30))
    L_off_expect = LA + 0.5 * lam * float(((theta - ref) ** 2).sum())
    loss_err = abs(L_on - L_off_expect) / abs(L_on)
    ok2 = prior_err < 5e-3 and loss_err < 1e-3
    print(f"[2 prior grad] (g_on-g_off) vs lam*(th-ref) rel={prior_err:.2e}  loss rel={loss_err:.2e}"
          f"  -> {'PASS' if ok2 else 'FAIL'}")

    # 3. finite difference of MAP loss vs analytic MAP grad, FP64 + eps=1e-6 (kbench verify.py
    #    recipe: the fp64 forward is bit-exact deterministic, so central FD is a trustworthy
    #    oracle; fp32 is too noisy for a scalar-loss FD). Analytic grad also computed in fp64.
    theta64 = theta.double()
    ref64 = ref.double()
    f_on64 = make_value_and_grad(train.batch_statics, rw.double(), theta_shape=(S, 3),
                                 prior=(lam, ref64))
    _, g_map, _, _ = f_on64(theta64.reshape(-1))
    g_map = g_map.double()
    th0 = theta64.reshape(-1)
    rels = []
    for s in range(3):
        gen = torch.Generator(device=device).manual_seed(100 + s)
        v = torch.randn(3 * S, generator=gen, device=device, dtype=torch.float64)
        v /= v.norm()
        eps = 1e-6
        Lp, _, _, _ = f_on64(th0 + eps * v, want_grad=False)
        Lm, _, _, _ = f_on64(th0 - eps * v, want_grad=False)
        fd = (Lp - Lm) / (2 * eps)
        ana = float(torch.dot(g_map, v))
        rel = abs(fd - ana) / max(1.0, abs(ana))
        rels.append(rel)
        print(f"  FD dir {s}: fd={fd:.6f} analytic={ana:.6f} rel={rel:.2e}")
    ok3 = max(rels) < 1e-3  # fp64 converged -> kbench-grade FD agreement
    print(f"[3 FD] max rel={max(rels):.2e}  -> {'PASS' if ok3 else 'FAIL'}")

    ok = ok1 and ok2 and ok3
    print(f"\n[verify_map] {'ALL PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
