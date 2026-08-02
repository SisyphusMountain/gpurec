"""PD certificate for the Sanderson-CV refits, via the VERIFIED analytic exact-HVP summed over
batches (the FD-of-gradient HVP cannot resolve the near-zero bottom eigenvalue -- see _cert_diag).

H_total u = sum_b (exact_hvp_b)(u) + lam * L u   (L = species-tree graph Laplacian).
Smallest eigenvalue via Lanczos; lam_min > 0 with a small Ritz residual => certified true local min.

Run AFTER the CV finishes (needs the whole GPU): reads ckpt/refit_lam*.pt + state.pt, writes the
certified lam_min/residual back into state.pt and prints the table.

  python experiments/sanderson_cv/certify.py --outdir experiments/sanderson_cv/runs/cv_1055
  python experiments/sanderson_cv/certify.py --selftest 256   # exact-multibatch HVP == FD HVP check
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import torch

from gpurec import GeneReconModel, SolverOptions
from gpurec.solver.value_and_grad import make_value_and_grad, forward_solve
from gpurec.solver.hvp.exact import make_exact_hvp
from gpurec.solver.krylov import lanczos_min_eigpair

HERE = Path(__file__).resolve().parent
ROOT = Path("/home/enzo/Documents/git/gpurec/gpurec/tests/data/alerax_hogenom_core/hogenom")
SP = ROOT/"runs/MFP/true_start_ufboot1000/run_--gene-tree-samples_100_--per-family-rates_1/alegenerax/species_trees/starting_species_tree.newick"
SO = dict(e_max_iter=2000,e_tol=1e-8,pi_iters=64,neumann_terms=64,
   bicgstab_max_iter=500,bicgstab_tol=1e-7,bicgstab_breakdown_tol=1e-30,
   adjoint_pruning_threshold=1e-6,use_adjoint_pruning=True,pibar_side_threshold=0.0)


def family_paths(n=None):
    fams = [l.strip() for l in open(HERE/"families_1055.txt") if l.strip()]
    return [str(ROOT/"families"/f/"gene_trees/ufboot1000.MFP.geneTree.newick") for f in (fams[:n] if n else fams)]


def _lap_edges(sp_parent):
    sp = sp_parent.long(); child = (sp >= 0).nonzero(as_tuple=True)[0]
    return child.contiguous(), sp[child].contiguous()


def lap_apply(u, child, par, lam):
    u = u.reshape(-1, 3); out = torch.zeros_like(u)
    diff = u.index_select(0, child) - u.index_select(0, par)
    out.index_add_(0, child, diff); out.index_add_(0, par, -diff)
    return (lam * out).reshape(-1)


def make_exact_multibatch_hvp(model, theta, rw, child, par, lam):
    """sum_b exact_hvp_b(u) + lam*L u. Builds one adjoint cache per batch (kept resident)."""
    hvps = []
    for static in model.batch_statics:
        _loss, sv = forward_solve([static], theta, rw)
        hvps.append(make_exact_hvp([static], theta, rw, sv))
    theta32 = theta.float()

    def Av(u):
        acc = torch.zeros(theta.numel(), dtype=torch.float64, device=theta.device)
        u32 = u.float()
        for h in hvps:
            acc += h(u32).double()
        acc += lap_apply(u.double(), child, par, lam)
        return acc
    return Av


def selftest(n):
    """Multi-batch exact HVP must match the FD HVP (trustworthy ~0.5% at eps=1e-2) on random u."""
    so = SolverOptions(**SO); so.validate()
    m = GeneReconModel(str(SP), family_paths(n), mode="specieswise", device="cuda", solver_options=so)
    S = int(m.species_helpers["S"]); rw = m.receiver_weights.detach().clone()
    sp = m.species_helpers["sp_parent"]; child, par = _lap_edges(sp)
    p = S*3; lam = 10.0
    print(f"[selftest] n={n} families -> {len(m.batch_statics)} batches, p={p}, lam={lam}")
    torch.manual_seed(0); theta = (0.3*torch.randn(S,3,device="cuda")).float()
    Av_ex = make_exact_multibatch_hvp(m, theta, rw, child, par, lam)
    f = make_value_and_grad(m.batch_statics, rw, theta_shape=(S,3), tree_penalty=(lam, sp))
    tvec = theta.reshape(-1); eps = 1e-2
    def Av_fd(u):
        u=u.float(); _,gp,_,_=f((tvec+eps*u).contiguous()); _,gm,_,_=f((tvec-eps*u).contiguous())
        return ((gp.double()-gm.double())/(2*eps))
    u = torch.randn(p, device="cuda", dtype=torch.float64); u/=u.norm()
    He, Hf = Av_ex(u), Av_fd(u)
    rel = float((He-Hf).norm()/He.norm())
    print(f"[selftest] exact-multibatch vs FD(eps=1e-2) Hu rel-err = {rel:.3e}  "
          f"-> {'PASS (summation correct)' if rel < 0.02 else 'FAIL'}")
    return rel < 0.02


def certify(outdir, m_lanczos=120, families=1055):
    outdir = Path(outdir)
    state = torch.load(outdir/"state.pt", weights_only=False)
    so = SolverOptions(**SO); so.validate()
    model = GeneReconModel(str(SP), family_paths(families), mode="specieswise", device="cuda", solver_options=so)
    S = int(model.species_helpers["S"]); rw = model.receiver_weights.detach().clone()
    sp = model.species_helpers["sp_parent"]; child, par = _lap_edges(sp); p = S*3
    ckpts = sorted(outdir.glob("ckpt/refit_lam*.pt"), key=lambda x: int(x.stem.split("lam")[1]))
    print(f"[certify] {len(ckpts)} refit thetas, p={p}, {len(model.batch_statics)} batches, m={m_lanczos}")
    print(f"{'lam':>10} {'lam_min':>13} {'resid':>10} {'PD':>6}  {'t(s)':>6}")
    for ck in ckpts:
        d = torch.load(ck, weights_only=False); theta = d["theta"].cuda().float(); lam = float(d["lam"])
        t0 = time.perf_counter()
        Av = make_exact_multibatch_hvp(model, theta, rw, child, par, lam)
        lam_min, v = lanczos_min_eigpair(Av, p, m=m_lanczos, seed=0)
        resid = float((Av(v) - lam_min*v).norm())
        pd = bool(lam_min > 0 and resid < 0.1*max(1.0, abs(lam_min)))
        dt = time.perf_counter()-t0
        print(f"{lam:>10g} {lam_min:>+13.5e} {resid:>10.2e} {str(pd):>6}  {dt:>6.0f}")
        if str(lam) in state.get("refit", {}):
            state["refit"][str(lam)].update(lam_min=lam_min, ritz_resid=resid, certified_pd=pd,
                                             cert_method="exact_multibatch", cert_m=m_lanczos)
        torch.cuda.empty_cache()
    torch.save(state, outdir/"state.pt")
    print(f"[certify] wrote certified lam_min/resid back into {outdir/'state.pt'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(HERE/"runs"/"cv_1055"))
    ap.add_argument("--families", type=int, default=1055)
    ap.add_argument("--m", type=int, default=120)
    ap.add_argument("--selftest", type=int, default=0)
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if selftest(a.selftest) else 1)
    certify(a.outdir, m_lanczos=a.m, families=a.families)
