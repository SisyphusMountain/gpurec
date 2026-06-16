"""Saddle-escape + Newton-polish for a Sanderson-CV checkpoint, via the analytic exact-HVP.

If a CV refit is only a saddle (lam_min<0 -- which Adam->L-BFGS cannot escape, since L-BFGS keeps a
PD Hessian model), this descends the most-negative-curvature eigenvector, L-BFGS re-converges, then
takes ONE Newton step (delta = -H^-1 g, H now PD) to drive |g|->0, and re-certifies.

EFFICIENCY: the exact-HVP cache is built ONCE per point and the operator only APPLIES it. (The early
ad-hoc version rebuilt forward_solve+make_exact_hvp on every Lanczos iteration -- ~3x slower; fixed
here.) For small p the exact Hessian is formed (definitive eigh); for large p, converged Lanczos.

Capture-driven so it runs locally (archaea) or on the cluster (hogenom). Env: CAP, THETA, LAM,
[FULL_HESSIAN=1|0 default auto: full if p<=1200].
"""
from __future__ import annotations
import os, time
import numpy as np
import torch
from scipy.optimize import minimize
from gpurec.optim.value_and_grad import forward_solve, make_value_and_grad
from gpurec.optim.hvp_exact import make_exact_hvp
from gpurec.optim.cg import lanczos_min_eigpair
from gpurec.api._execution import stream_batches

DEV = "cuda"


def make_lap(child, par, lam):
    def lap(v):
        u = v.reshape(-1, 3); out = torch.zeros_like(u)
        d = u.index_select(0, child) - u.index_select(0, par)
        out.index_add_(0, child, d); out.index_add_(0, par, -d)
        return (lam * out).reshape(-1)
    return lap


def build_hvp_once(batch_statics, theta2d, rw, lap, p):
    """Build per-batch exact-HVP caches ONCE at theta2d; return Av(v) that only APPLIES them."""
    hvps = []
    for st in batch_statics:
        _l, sv = forward_solve([st], theta2d, rw)
        hvps.append(make_exact_hvp([st], theta2d, rw, sv))
    def Av(v):
        acc = torch.zeros(p, dtype=torch.float64, device=DEV)
        for h in hvps:
            acc += h(v).double()
        return acc + lap(v)
    return Av


def bottom(Av, p, full):
    """Return (lam_min, v_min, info). full=True forms the exact Hessian (eigh, info=n_neg);
    else converged Lanczos (info=ritz_residual)."""
    if full:
        H = torch.zeros(p, p, dtype=torch.float64, device=DEV)
        I = torch.eye(p, dtype=torch.float64, device=DEV)
        for i in range(p):
            H[:, i] = Av(I[:, i])
        asym = float((H - H.T).abs().max()); H = 0.5 * (H + H.T)
        mu, V = torch.linalg.eigh(H)
        return float(mu[0]), V[:, 0], dict(n_neg=int((mu < 0).sum()), asym=asym, H=H, mu=mu)
    lam, v = lanczos_min_eigpair(Av, p, m=min(300, p), seed=0)
    return lam, v, dict(resid=float((Av(v) - lam * v).norm()))


def newton_polish(th0, bs, rw, lap, p, S, grad, full, max_iter=10, tol=1e-3, verbose=True,
                  final_cert=True):
    """Iterated DAMPED Newton with a backtracking LINE SEARCH on ||g||.

    A single undamped step delta=-H^-1 g overshoots at near-flat points (lam_min~0): H^-1 amplifies
    the soft direction by ~1/lam_min, the full step leaves the quadratic-trust region and ||g|| can
    *increase*. So each iteration rebuilds the (PD) Hessian, computes the Newton direction, and accepts
    the largest alpha in {1,1/2,1/4,...} that DECREASES ||g|| (d=-H^-1 g is a descent direction for
    1/2||g||^2). Iterates until ||g||<tol or no alpha helps. Returns (theta, gnorm, lam_min, info,
    n_iter, total_step)."""
    from gpurec.optim.cg import cg_solve
    th = th0.clone().reshape(-1); total = 0.0; n_done = 0
    g = grad(th); gn = float(g.norm())
    for n in range(1, max_iter + 1):
        if gn < tol:
            break
        # CG-solve the Newton direction via the HVP operator (cheap -- no full 357x357 Hessian
        # in the loop; the definitive eigh-cert is formed once after the loop). H is PD here.
        Av = build_hvp_once(bs, th.reshape(S, 3), rw, lap, p)
        d, _it, _conv = cg_solve(Av, -g, tol=1e-8, max_iter=400)
        alpha, ok, g_try, gnt = 1.0, False, g, gn
        for _ in range(30):
            g_try = grad(th + alpha * d); gnt = float(g_try.norm())
            if gnt < gn:
                ok = True; break
            alpha *= 0.5
        if not ok:
            if verbose:
                print(f"    [newton] iter {n}: no ||g||-decreasing step; stop at |g|={gn:.3e}", flush=True)
            break
        th = th + alpha * d; total += float((alpha * d).norm()); g, gn = g_try, gnt; n_done = n
        if verbose:
            print(f"    [newton] iter {n}: alpha={alpha:g}  |g|->{gn:.3e}", flush=True)
    if final_cert:
        Av = build_hvp_once(bs, th.reshape(S, 3), rw, lap, p)
        lm_final, _v, info_final = bottom(Av, p, full)
    else:
        lm_final, info_final = float("nan"), {}   # CV mode: the |g|<tol fit is what we need, skip the
        #                                            redundant full-Hessian PD certificate (the dominant cost)
    return th, gn, lm_final, info_final, n_done, total


def run(bs, rw, sp, S, theta_path, lam, full=None, out_path=None, meta=None,
        final_cert=True, polish_tol=1e-3, max_polish=10):
    p = 3 * S
    rw = rw.to(DEV).double(); sp = sp.to(DEV).long()
    child = (sp >= 0).nonzero(as_tuple=True)[0].contiguous(); par = sp[child].contiguous()
    lap = make_lap(child, par, lam)
    if full is None:
        full = p <= 1200
    theta = torch.load(theta_path, map_location=DEV, weights_only=False)["theta"].to(DEV).double()
    fg = make_value_and_grad(bs, rw, theta_shape=(S, 3), tree_penalty=(lam, sp))
    grad = lambda thf: fg(thf)[1].double()
    def loss(thf):
        l, _, _ = stream_batches(bs, thf.reshape(S, 3), rw, genewise=False, need_grad=False)
        d = thf.reshape(S, 3); diff = d.index_select(0, child) - d.index_select(0, par)
        return float(l) + 0.5 * lam * float((diff * diff).sum())

    print(f"[saddle_escape] p={p} batches={len(bs)} lam={lam} full_hessian={full}", flush=True)
    tsf = theta.reshape(-1)
    t0 = time.time()
    Av0 = build_hvp_once(bs, theta, rw, lap, p)               # built ONCE
    lam_min, v0, info = bottom(Av0, p, full)
    F0 = loss(tsf); g0n = float(grad(tsf).norm())
    print(f"  checkpoint: F={F0:.4f} |g|={g0n:.4e} "
          f"lam_min={lam_min:+.5e} {info if 'H' not in info else {k:info[k] for k in info if k not in ('H','mu')}} ({time.time()-t0:.0f}s)", flush=True)
    def lbfgs(x0):
        def fun(x):
            x = torch.tensor(x, device=DEV, dtype=torch.float64); l, g, _, _ = fg(x)
            return float(l), g.double().cpu().numpy()
        r = minimize(fun, x0.cpu().numpy(), jac=True, method="L-BFGS-B", bounds=None,
                     options=dict(maxiter=300, maxcor=50, ftol=1e-16, gtol=1e-12))
        return torch.tensor(r.x, device=DEV, dtype=torch.float64)

    is_saddle = lam_min < 0
    if is_saddle:                        # descend the most-negative-curvature eigenvector, re-converge
        a_best = min(((loss(tsf + a * v0), a) for a in (-4, -2, -1, 1, 2, 4)), key=lambda z: z[0])[1]
        th_cur = lbfgs(tsf + a_best * v0)
        Av_cur = build_hvp_once(bs, th_cur.reshape(S, 3), rw, lap, p)
        lm_cur, _, info_cur = bottom(Av_cur, p, full)
        print(f"  escaped (a={a_best:+g}): F={loss(th_cur):.4f} |g|={float(grad(th_cur).norm()):.4e} lam_min={lm_cur:+.5e}", flush=True)
    else:                                # already PD -- just Newton-polish to |g|->0
        th_cur, Av_cur, lm_cur, info_cur = tsf.clone(), Av0, lam_min, info
        print("  already PD; Newton-polishing to drive |g|->0", flush=True)
    g_cur = grad(th_cur)
    # line-searched, iterated Newton (robust at near-flat points; see newton_polish docstring)
    th_star, gsn, lm_star, info_star, n_newton, step_total = newton_polish(
        th_cur, bs, rw, lap, p, S, grad, full, max_iter=max_polish, tol=polish_tol,
        final_cert=final_cert)
    print(f"  Newton polish: {n_newton} line-searched iters, total step {step_total:.4f}, "
          f"|g| {float(g_cur.norm()):.3e}->{gsn:.3e}", flush=True)
    F_star = loss(th_star)
    # final_cert=False (CV mode): the |g|<polish_tol fit is the deliverable; lm_star is NaN, so report
    # convergence by gradient only (the saddle was already detected+escaped at the start point).
    certified = (gsn < 1e-2 and lm_star > 0) if final_cert else None
    tag = ("ZERO-GRAD + PD (true local min)" if certified
           else f"converged |g|={gsn:.1e} (PD cert skipped)" if certified is None
           else "not fully there")
    print(f"\n=== {'ESCAPE+' if is_saddle else ''}NEWTON (lam={lam}): F {F0:.4f} -> {F_star:.4f}   "
          f"lam_min {lam_min:+.5e} -> {lm_star:+.6e}   |g|->{gsn:.3e}   -> {tag} ===", flush=True)
    res = dict(
        lam=lam, full_hessian=full, is_saddle=is_saddle, certified=certified,
        theta_saddle=theta.reshape(S, 3).cpu(), theta_escaped=th_cur.reshape(S, 3).cpu(),
        theta_newton=th_star.reshape(S, 3).cpu(),
        lam_min_saddle=lam_min, lam_min_escaped=lm_cur, lam_min_newton=lm_star,
        gnorm_saddle=g0n, gnorm_escaped=float(g_cur.norm()), gnorm_newton=gsn,
        loss_saddle=F0, loss_escaped=loss(th_cur), loss_newton=F_star,
        newton_step_total=step_total, newton_iters=n_newton, n_neg_newton=info_star.get("n_neg"),
        ritz_resid_newton=info_star.get("resid"), meta=meta or {})
    if out_path:
        torch.save(res, out_path); print(f"  saved checkpoint -> {out_path}", flush=True)
    return res


def _model_from_env():
    """Resolve (batch_statics, rw, sp, S, meta) from CAP (a capture) or from the in-repo DATASET."""
    if os.environ.get("CAP"):
        cap = torch.load(os.environ["CAP"], map_location=DEV, weights_only=False)
        return (cap["batch_statics"], cap["rw"], cap["sp_parent"], int(cap["S"]),
                dict(source="capture", cap=os.environ["CAP"]))
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from run_cv import DATASETS, _CV_SO
    from gpurec import GeneReconModel, SolverOptions
    ds = os.environ.get("DATASET", "archaea"); n = int(os.environ.get("FAMILIES", "256"))
    so = SolverOptions(**_CV_SO); so.validate()
    m = GeneReconModel(str(DATASETS[ds]["species_tree"]), [str(x) for x in DATASETS[ds]["families"](n)],
                       mode="specieswise", device=DEV, solver_options=so)
    return (m.batch_statics, m.receiver_weights.detach(), m.species_helpers["sp_parent"],
            int(m.species_helpers["S"]), dict(source="dataset", dataset=ds, families=n))


if __name__ == "__main__":
    fenv = os.environ.get("FULL_HESSIAN")
    bs, rw, sp, S, meta = _model_from_env()
    run(bs, rw, sp, S, os.environ["THETA"], float(os.environ.get("LAM", "0.03")), out_path=os.environ.get("OUT"),
        meta=meta,
        full=(None if fenv is None else bool(int(fenv))))
