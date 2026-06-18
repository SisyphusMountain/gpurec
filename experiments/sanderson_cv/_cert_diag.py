"""Diagnose the PD certificate: on the single-batch 48-family smoke refit thetas, compare the
FD-of-gradient HVP against the VERIFIED analytic exact-HVP (ground truth, single-batch only), and
sweep Lanczos m / FD eps. Decides whether the smoke-cert failure was just m too small (cheap fix) or
genuine FD noise (=> build multi-batch exact-HVP)."""
from __future__ import annotations
import sys, time
from pathlib import Path
import torch

from gpurec import GeneReconModel, SolverOptions
from gpurec.optim.value_and_grad import make_value_and_grad, forward_solve
from gpurec.optim.hvp_exact import make_exact_hvp
from gpurec.optim.cg import lanczos_min_eigpair

HERE = Path(__file__).resolve().parent
ROOT = Path("/home/enzo/Documents/git/gpurec/gpurec/tests/data/alerax_hogenom_core/hogenom")
SP = ROOT/"runs/MFP/true_start_ufboot1000/run_--gene-tree-samples_100_--per-family-rates_1/alegenerax/species_trees/starting_species_tree.newick"
SO = dict(e_max_iter=2000,e_tol=1e-8,pi_iters=64,neumann_terms=64,
   self_loop_solver="neumann",bicgstab_max_iter=500,bicgstab_tol=1e-7,bicgstab_breakdown_tol=1e-30,
   adjoint_pruning_threshold=1e-6,use_adjoint_pruning=True,pibar_side_threshold=0.0)


def lap_apply(u, child, par, lam):
    """lam * L u  for the tree graph-Laplacian (same structure as the penalty gradient)."""
    u = u.reshape(-1, 3)
    out = torch.zeros_like(u)
    diff = u.index_select(0, child) - u.index_select(0, par)
    out.index_add_(0, child, diff); out.index_add_(0, par, -diff)
    return (lam * out).reshape(-1)


def main(lam_idx=0):
    so = SolverOptions(**SO); so.validate()
    fams = [l.strip() for l in open(HERE/"families_1055.txt")][:48]
    paths = [str(ROOT/"families"/f/"gene_trees/ufboot1000.MFP.geneTree.newick") for f in fams]
    m = GeneReconModel(str(SP), paths, mode="specieswise", device="cuda", solver_options=so)
    S = int(m.species_helpers["S"]); rw = m.receiver_weights.detach().clone()
    sp = m.species_helpers["sp_parent"].long()
    child = (sp >= 0).nonzero(as_tuple=True)[0]; par = sp[child]
    p = S*3

    ck = torch.load(HERE/"runs/smoke/ckpt"/f"refit_lam{lam_idx}.pt", weights_only=False)
    theta = ck["theta"].cuda().float(); lam = float(ck["lam"])
    print(f"\n===== refit theta at lam={lam} (48 fam, single batch), p={p} =====")

    # exact analytic HVP (ground truth) -- fp32
    _loss, sv = forward_solve(m.batch_statics, theta, rw)
    hvp_ex = make_exact_hvp(m.batch_statics, theta, rw, sv)
    Av_ex = lambda u: hvp_ex(u.float()).double() + lap_apply(u.double(), child, par, lam)

    # FD HVP via central diff of the penalized gradient
    f = make_value_and_grad(m.batch_statics, rw, theta_shape=(S,3), tree_penalty=(lam, sp))
    tvec = theta.reshape(-1)
    def make_fd(eps, K=1):
        fK = make_value_and_grad(m.batch_statics, rw, theta_shape=(S,3), tree_penalty=(lam, sp), grad_avg_K=K)
        def Av(u):
            u = u.to(tvec.dtype)
            _,gp,_,_ = fK((tvec+eps*u).contiguous()); _,gm,_,_ = fK((tvec-eps*u).contiguous())
            return ((gp.double()-gm.double())/(2*eps))
        return Av

    # 1) direct HVP accuracy on a random unit vector: exact vs FD
    torch.manual_seed(0); u = torch.randn(p, device="cuda", dtype=torch.float64); u/=u.norm()
    Hu_ex = Av_ex(u)
    print("  direct Hu rel-err (FD vs exact) on random u:")
    for eps in (1e-3, 3e-3, 1e-2, 3e-2):
        Hu_fd = make_fd(eps)(u)
        print(f"    eps={eps:<6g} rel={float((Hu_fd-Hu_ex).norm()/Hu_ex.norm()):.3e}")
    for K in (4,):
        Hu_fd = make_fd(1e-2, K=K)(u)
        print(f"    eps=1e-2 K={K} rel={float((Hu_fd-Hu_ex).norm()/Hu_ex.norm()):.3e}")

    # 2) lanczos min-eig: exact ground truth, then FD at several (m, eps)
    le, ve = lanczos_min_eigpair(Av_ex, p, m=120, seed=0)
    re = float((Av_ex(ve)-le*ve).norm())
    print(f"  EXACT lanczos m=120:  lam_min={le:+.4e}  resid={re:.2e}")
    for eps in (1e-2, 3e-2):
        for mm in (60, 120):
            t0=time.perf_counter()
            lf, vf = lanczos_min_eigpair(make_fd(eps), p, m=mm, seed=0)
            rf = float((make_fd(eps)(vf)-lf*vf).norm())
            print(f"  FD eps={eps:<6g} m={mm:<4d} lam_min={lf:+.4e}  resid={rf:.2e}  ({time.perf_counter()-t0:.0f}s)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
