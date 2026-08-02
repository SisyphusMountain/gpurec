"""Bound-constrained specieswise minimum at lam=0.03 on archaea, to a CERTIFIED KKT point.

Targets the BOX-CONSTRAINED objective  argmin_theta F(theta)=NLL+0.5*lam*||L theta||^2  s.t.
theta in [log2(min_rate), log2(max_rate)]  (rate box, default [1e-6, 2.0]). The box pins the runaway
saturated rates (the unconstrained lam=0.03 optimum overfits to rate~4000 in ~38% of coords, the D-L
turnover direction = the near-singular Hessian direction). Pinning them removes the near-singular
directions -> a WELL-CONDITIONED, certifiable minimum (zero PROJECTED gradient).

Key difference from optimize_specieswise_matrixfree.py: that script uses family_chunk_size=1 (one
batch/family) to build a per-family empirical-Fisher preconditioner -> 5446 solves/step = hours on full
archaea. Here we CHUNK families (fast loss/grad/HVP) and rely on the BOX for conditioning -> plain
trust-region Steihaug-CG (no preconditioner). If CG saturates, set PRECOND=penalty (analytic 0-HVP
penalty diagonal lam*deg*I).

Outputs theta + |Pg| + the BOUND-CONSTRAINED KKT certificate: lam_min on the FREE (inactive) subspace
(reduced Hessian PD?) + the active-set sign check.

Env: DATASET(archaea) FAMILIES(0=all) LAM(0.03) MIN_RATE(1e-6) MAX_RATE(2.0) ADAM(40) ADAM_LR(1.0)
     NEWTON(40) GTOL(1e-3) CG_MAX(200) CHUNK(300) PRECOND(none|penalty) INIT(zeros|<path>) CERT(1)
     CERT_M(160) OUT. SADDLE_DTYPE=float32|float64.
"""
from __future__ import annotations
import os, sys, glob, time, math
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
DEV = "cuda"
os.environ.setdefault("SADDLE_DTYPE", "float32")
os.environ.setdefault("GPUREC_MEMORY_POLICY_RESERVE_GIB", "0")
DTYPE = torch.float32 if os.environ["SADDLE_DTYPE"] == "float32" else torch.float64

from run_cv import DATASETS, _CV_SO
from gpurec import GeneReconModel, SolverOptions
from gpurec.fit.optimize import Schedule
from gpurec.solver.value_and_grad import make_value_and_grad
from gpurec.solver.krylov import steihaug_cg
from gpurec.optimization import clamp_log_rate_, project_rate_gradient_, log2_rate_bounds
from saddle_escape import make_lap, build_hvp_once


def atomic_save(obj, path):
    """Crash-safe save: write to a temp file then os.replace (atomic on POSIX). For A100 wall-kill
    safety -- a half-written checkpoint never clobbers the last good one."""
    if not path:
        return
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def proj_gnorm(theta_flat, g_flat, S, *, min_rate, max_rate):
    g = g_flat.clone()
    project_rate_gradient_(theta_flat.reshape(S, 3), g.reshape(S, 3), min_rate=min_rate, max_rate=max_rate)
    return float(g.norm()), g


def adam_warmup(f, theta, *, steps, lr, S, min_rate, max_rate, log_every=10):
    leaf = theta.clone().requires_grad_(True)
    opt = torch.optim.Adam([leaf], lr=lr); sched = Schedule("adaptive", lr, t_max=steps)
    t0 = time.perf_counter()
    for it in range(int(steps)):
        loss, g, _, _ = f(leaf.detach().reshape(-1))
        opt.param_groups[0]["lr"] = sched.update(loss, g); leaf.grad = g.reshape(S, 3); opt.step()
        with torch.no_grad():
            clamp_log_rate_(leaf, min_rate=min_rate, max_rate=max_rate)
        if it % log_every == 0 or it == steps - 1:
            pg, _ = proj_gnorm(leaf.detach().reshape(-1), g, S, min_rate=min_rate, max_rate=max_rate)
            print(f"  [adam {it:3d}] loss={loss:.3f} |Pg|={pg:.3e} lr={opt.param_groups[0]['lr']:.3g} "
                  f"({time.perf_counter()-t0:.1f}s)", flush=True)
    return leaf.detach()


def newton_tr(f, theta, bs, rw, lap, S, lam, *, min_rate, max_rate, max_iter, gtol, cg_max,
              Minv=None, delta0=1.0, delta_max=20.0):
    """Box-projected trust-region Steihaug-CG Newton to |Pg|<gtol (no preconditioner unless Minv given)."""
    p = 3 * S; theta = theta.detach().reshape(-1).clone(); delta = float(delta0)
    t0 = time.perf_counter(); loss, g, _, _ = f(theta)
    for it in range(1, int(max_iter) + 1):
        pg_norm, _ = proj_gnorm(theta, g, S, min_rate=min_rate, max_rate=max_rate)
        print(f"  [newton {it:3d}] loss={loss:.4f} |Pg|={pg_norm:.3e} delta={delta:.2g} "
              f"({time.perf_counter()-t0:.1f}s)", flush=True)
        if pg_norm < gtol:
            print(f"  [newton] CONVERGED |Pg|={pg_norm:.3e} < {gtol:g}", flush=True); break
        Av = build_hvp_once(bs, theta.reshape(S, 3), rw, lap, p)
        cg_tol = max(min(0.5, pg_norm ** 0.5) * pg_norm, 1e-8)
        step, Ap, cg_it, status = steihaug_cg(Av, -g.to(DTYPE), delta, tol=cg_tol, max_iter=cg_max, Minv=Minv)
        pred = -float(g.to(DTYPE) @ step) - 0.5 * float(step @ Ap)
        theta_try = theta + step
        with torch.no_grad():
            clamp_log_rate_(theta_try.reshape(S, 3), min_rate=min_rate, max_rate=max_rate)
        loss_try, g_try, _, _ = f(theta_try)
        actual = loss - loss_try
        rho = actual / pred if pred > 0 else (1.0 if actual > 0 else -1.0)
        sn = float(step.norm())
        # Indefinite-Hessian-aware radius rule: a negative-curvature step's quadratic model overpredicts
        # the decrease (model -> -inf along neg curv), so rho is always low even when the step genuinely
        # lowers the loss -- shrinking on rho<0.25 collapses the trust region at saddles. Instead: shrink
        # ONLY on a failed (non-decreasing) step; GROW on an accepted boundary/neg-curv step so the TR
        # walks down the negative-curvature direction (saddle escape via repeated boundary steps).
        accepted = (actual > 0)
        if not accepted:
            delta = 0.25 * delta
        elif status in ("boundary", "neg_curv"):
            delta = min(2.0 * delta, delta_max)
        # accepted interior step -> keep delta
        if accepted:
            theta, loss, g = theta_try, loss_try, g_try; tag = "accept"
        else:
            tag = "reject"
        print(f"      cg_it={cg_it:3d} status={status:9s} |step|={sn:.2e} rho={rho:+.3f} -> {tag}", flush=True)
    pg_norm, pg = proj_gnorm(theta, g, S, min_rate=min_rate, max_rate=max_rate)
    return theta.reshape(S, 3), float(loss), pg_norm, g


def free_subspace_lanczos(Av, free_mask, p, *, m=160, seed=0):
    """Smallest eigenpair of H restricted to the FREE subspace (active coords masked out). Lanczos
    started IN the free subspace (q0 = mask*randn) with Av_free(v)=mask*Av(mask*v), so the Krylov space
    stays free and the active subspace's zero-eigenvalues are NOT picked up. Returns
    (lam_min_free, v_min, resid); v_min is the bottom Ritz vector (fp64, supported on the free set)."""
    from scipy.linalg import eigh_tridiagonal
    g = torch.Generator(device=DEV).manual_seed(seed)
    msk = free_mask.to(torch.float64)
    q = (torch.randn(p, generator=g, device=DEV, dtype=torch.float64) * msk)
    q = q / q.norm()
    Avf = lambda v: (Av((v * msk).to(DTYPE)).to(torch.float64) * msk)
    Q, alphas, betas = [], [], []; beta, q_prev = 0.0, torch.zeros_like(q)
    for _ in range(int(m)):
        w = Avf(q) - beta * q_prev; a = float(torch.dot(w, q)); w = w - a * q
        for qq in Q:
            w = w - torch.dot(w, qq) * qq
        Q.append(q.clone()); alphas.append(a); b = float(w.norm())
        if b < 1e-12:
            break
        q_prev, q, beta = q, w / b, b; betas.append(b)
    n = len(alphas)
    wv, Sv = eigh_tridiagonal(np.array(alphas), np.array(betas[:n-1]), eigvals_only=False)
    s = torch.tensor(Sv[:, 0], device=DEV, dtype=torch.float64); v = torch.zeros(p, device=DEV, dtype=torch.float64)
    for i, qi in enumerate(Q):
        v += s[i] * qi
    v = v / v.norm()
    resid = float((Avf(v) - wv[0] * v).norm())
    return float(wv[0]), v, resid


def binding_mask(theta_flat, g_flat, lo, hi, *, atol=1e-4):
    """KKT-binding coords: at a box edge AND the gradient pushes further into the wall (cannot move).
    Free = complement. |g * free| equals the projected-gradient norm |Pg|."""
    at_hi = theta_flat >= hi - atol
    at_lo = theta_flat <= lo + atol
    return (at_hi & (g_flat < 0)) | (at_lo & (g_flat > 0))


def escape_along(f, theta, v, S, *, min_rate, max_rate, n=10, t_lo=2e-2, t_hi=5.0):
    """Bidirectional box-projected line search along a (unit) negative-curvature direction v. Rides
    BOTH +/-v over a geometric grid of step sizes, clamps each trial into the box, returns the lowest-loss
    trial (loss, theta, t, sign). The linear term is ~0 (we only call this when |Pg|<gtol), so a negative
    curvature gives a quadratic decrease either way until the box/high-order terms stop it."""
    vd = v.to(DTYPE)
    loss0 = float(f(theta)[0]); best = (loss0, theta.clone(), 0.0, 0.0)
    ts = [t_lo * (t_hi / t_lo) ** (k / (n - 1)) for k in range(n)]
    for s in (1.0, -1.0):
        for t in ts:
            th = (theta + s * t * vd).reshape(S, 3).clone()
            with torch.no_grad():
                clamp_log_rate_(th, min_rate=min_rate, max_rate=max_rate)
            th = th.reshape(-1)
            l = float(f(th)[0])
            if l < best[0]:
                best = (l, th, t, s)
    return best


def projected_newton_escape(f, theta, bs, rw, lap, S, lam, *, lo, hi, min_rate, max_rate,
                            max_iter, gtol, cg_max, Minv=None, delta0=1.0, delta_max=20.0,
                            esc_tol=1e-3, esc_m=160, max_escape=20, ckpt=None):
    """Bound-constrained minimizer that does NOT stall at saddles. Each iteration:

      1. binding/free split (KKT) -> reduced gradient g_free; |Pg|=|g_free|.
      2. if |Pg|>=gtol: a FREE-SUBSPACE trust-region Steihaug-CG step. The CG operator and RHS are both
         masked to the free set (Avf(v)=free*Av(free*v), b=-g_free), so the step lives in the free
         subspace and never pushes binding coords into the wall (the old full-space CG's stall). Steihaug
         rides reduced negative curvature to the TR boundary -> escapes mild saddles for free.
      3. if |Pg|<gtol: compute the bottom eigenpair of the REDUCED Hessian (free-subspace Lanczos).
            lam_min_free > -esc_tol  -> CONVERGED (KKT point, reduced-H PSD).
            lam_min_free <= -esc_tol -> ESCAPE along v_min (bidirectional box line search), reset TR, loop.
         This is the safety net for the saddle the TR-CG can converge to when the reduced gradient is
         already tiny (Newton is saddle-attracted; the eigen-escape is what guarantees we leave).

    A TR step whose predicted/actual decrease falls below the loss-eval resolution (fp32: ~eps*|loss|
    ~3e-3 at loss~2.7e4) becomes pure noise -> rho explodes, every step rejects, delta collapses (the
    fp32 floor seen at |Pg|~2e-3). A consecutive-reject streak (stall_max) routes us into the SAME
    curvature-check/escape path as |Pg|<gtol, so the loop terminates cleanly (PSD -> resolution-limited
    stop) or escapes a saddle instead of spinning. fp64 pushes this floor to ~1e-9 (the real endgame).

    Loss decreases every accepted/escape step and is bounded below -> terminates. Returns
    (theta2d, loss, |Pg|, g, info)."""
    p = 3 * S
    theta = theta.detach().reshape(-1).clone(); delta = float(delta0)
    t0 = time.perf_counter(); loss, g, _, _ = f(theta)
    n_escape = 0; last_lmf = None; stall = 0; stall_max = 6; status_out = "max_iter"

    def curv_escape(free, reason):
        """Reduced-Hessian curvature check + negative-curvature escape. Mutates outer theta/loss/g/
        delta/n_escape/last_lmf. Returns one of 'converged' | 'escaped' | 'maxesc' | 'stuck'."""
        nonlocal theta, loss, g, delta, n_escape, last_lmf
        Av = build_hvp_once(bs, theta.reshape(S, 3), rw, lap, p)
        lam_min, v_min, resid = free_subspace_lanczos(Av, free, p, m=esc_m)
        last_lmf = (lam_min, resid)
        print(f"      [curv:{reason}] lam_min_free={lam_min:+.5e} (resid={resid:.2e}, m={esc_m})", flush=True)
        if lam_min > -esc_tol:
            return "converged"
        if n_escape >= max_escape:
            print(f"  [stop] hit max_escape={max_escape}; lam_min_free still {lam_min:+.3e}", flush=True)
            return "maxesc"
        l_new, th_new, t_b, s_b = escape_along(f, theta, v_min, S, min_rate=min_rate, max_rate=max_rate)
        if l_new < loss - 1e-7:
            drop = loss - l_new
            theta = th_new; loss, g, _, _ = f(theta); n_escape += 1; delta = max(delta, 1.0)
            atomic_save(dict(theta=theta.reshape(S, 3).cpu(), loss=loss, n_escape=n_escape, kind="escape"), ckpt)
            print(f"      [escape #{n_escape}] sign={s_b:+.0f} t={t_b:.3g} drop={drop:.3g} "
                  f"-> loss={loss:.4f} (from saddle mu={lam_min:+.3e})", flush=True)
            return "escaped"
        print(f"  [stop] saddle mu={lam_min:+.3e} but NO box-line-search decrease along v_min", flush=True)
        return "stuck"

    for it in range(1, int(max_iter) + 1):
        bind = binding_mask(theta, g, lo, hi)
        free = (~bind).to(DTYPE)
        pg = float((g * free).norm())
        print(f"  [it {it:3d}] loss={loss:.4f} |Pg|={pg:.3e} bind={int(bind.sum())}/{p} delta={delta:.2g} "
              f"esc={n_escape} ({time.perf_counter()-t0:.1f}s)", flush=True)
        if pg < gtol or stall >= stall_max:
            reason = "grad" if pg < gtol else "stall"
            res = curv_escape(free, reason)
            stall = 0
            if res == "escaped":
                continue
            if res == "converged":
                status_out = "converged" if pg < gtol else "floor"
                print(f"  [done] {'|Pg|<gtol' if pg < gtol else 'fp-resolution floor'} and reduced-H PSD "
                      f"(|Pg|={pg:.3e}, lam_min_free={last_lmf[0]:+.3e})", flush=True)
            else:
                status_out = res
            break
        # ---- free-subspace trust-region Steihaug-CG step ----
        g_free = g * free
        Av = build_hvp_once(bs, theta.reshape(S, 3), rw, lap, p)
        Avf = (lambda v: free * Av(free * v))
        Minv_free = None if Minv is None else (lambda r: free * Minv(free * r))
        cg_tol = max(min(0.5, pg ** 0.5) * pg, 1e-8)
        step, Ap, cg_it, status = steihaug_cg(Avf, -g_free.to(DTYPE), delta, tol=cg_tol, max_iter=cg_max, Minv=Minv_free)
        pred = -float(g_free.to(DTYPE) @ step) - 0.5 * float(step @ Ap)
        theta_try = theta + step
        with torch.no_grad():
            clamp_log_rate_(theta_try.reshape(S, 3), min_rate=min_rate, max_rate=max_rate)
        loss_try, g_try, _, _ = f(theta_try)
        actual = loss - loss_try
        rho = actual / pred if pred > 0 else (1.0 if actual > 0 else -1.0)
        sn = float(step.norm())
        accepted = (actual > 0)
        if not accepted:
            delta = 0.25 * delta; stall += 1
        else:
            if status in ("boundary", "neg_curv"):
                delta = min(2.0 * delta, delta_max)
            theta, loss, g = theta_try, loss_try, g_try; stall = 0
            atomic_save(dict(theta=theta.reshape(S, 3).cpu(), loss=loss, n_escape=n_escape, kind="newton", it=it), ckpt)
        print(f"      cg_it={cg_it:3d} status={status:9s} |step|={sn:.2e} rho={rho:+.3f} "
              f"-> {'accept' if accepted else 'reject'}", flush=True)
    pg_norm, _ = proj_gnorm(theta, g, S, min_rate=min_rate, max_rate=max_rate)
    return theta.reshape(S, 3), float(loss), pg_norm, g, dict(n_escape=n_escape, last_lam_min_free=last_lmf,
                                                              status=status_out)


def main():
    ds = os.environ.get("DATASET", "archaea")
    _fam = int(os.environ.get("FAMILIES", "0")); nfam = None if _fam == 0 else _fam
    lam = float(os.environ.get("LAM", "0.03"))
    min_rate = float(os.environ.get("MIN_RATE", "1e-6")); max_rate = float(os.environ.get("MAX_RATE", "2.0"))
    adam_steps = int(os.environ.get("ADAM", "40")); adam_lr = float(os.environ.get("ADAM_LR", "1.0"))
    newton = int(os.environ.get("NEWTON", "40")); gtol = float(os.environ.get("GTOL", "1e-3"))
    cg_max = int(os.environ.get("CG_MAX", "200")); chunk = int(os.environ.get("CHUNK", "300"))
    precond = os.environ.get("PRECOND", "none"); do_cert = os.environ.get("CERT", "1") == "1"
    cert_m = int(os.environ.get("CERT_M", "160")); init = os.environ.get("INIT", "zeros")
    esc_tol = float(os.environ.get("ESC_TOL", "1e-3")); esc_m = int(os.environ.get("ESC_M", str(cert_m)))
    max_escape = int(os.environ.get("MAX_ESCAPE", "20"))
    out = os.environ.get("OUT", os.path.join(HERE, "runs", f"bounded_{ds}_n{_fam}_lam{lam:g}.pt"))
    os.makedirs(os.path.dirname(out), exist_ok=True)

    so = SolverOptions(**{**_CV_SO, "pi_iters": 64, "neumann_terms": 64}); so.validate()
    paths = DATASETS[ds]["families"](nfam)
    t0 = time.perf_counter()
    model = GeneReconModel(str(DATASETS[ds]["species_tree"]), [str(x) for x in paths], mode="specieswise",
                           device=DEV, solver_options=so, clade_budget=80000, family_chunk_size=chunk)
    S = int(model.species_helpers["S"]); p = 3 * S
    rw = model.receiver_weights.detach(); sp = model.species_helpers["sp_parent"].detach().reshape(-1).long()
    bs = model.batch_statics
    child = (sp >= 0).nonzero(as_tuple=True)[0].to(DEV); parent = sp[child].to(DEV)
    lap = make_lap(child, parent, lam)
    lo, hi = log2_rate_bounds(min_rate, max_rate)
    print(f"=== bounded specieswise : {ds} n={len(paths)} lam={lam} dtype={DTYPE} box=[{lo:.2f},{hi:.2f}] "
          f"chunk={chunk} nbatch={len(bs)} precond={precond} build={time.perf_counter()-t0:.1f}s ===", flush=True)

    f = make_value_and_grad(bs, rw, theta_shape=(S, 3), tree_penalty=(lam, sp))
    ckpt = out + ".ckpt"
    resume = os.environ.get("RESUME", "0") == "1" and os.path.exists(ckpt)
    if resume:
        rc = torch.load(ckpt, map_location=DEV, weights_only=False)
        theta = rc["theta"].to(DEV).to(DTYPE).reshape(S, 3)
        with torch.no_grad():
            clamp_log_rate_(theta, min_rate=min_rate, max_rate=max_rate)
        print(f"  RESUME from {os.path.basename(ckpt)} (loss={rc.get('loss')}, n_escape={rc.get('n_escape')})", flush=True)
        adam_steps = 0  # already warm; skip Adam on resume
    elif init == "zeros":
        theta = torch.zeros((S, 3), device=DEV, dtype=DTYPE)
    else:
        theta = torch.load(init, map_location=DEV, weights_only=False)["theta"].to(DEV).to(DTYPE).reshape(S, 3)
        with torch.no_grad():
            clamp_log_rate_(theta, min_rate=min_rate, max_rate=max_rate)
        print(f"  init from {os.path.basename(init)} (clamped into box)", flush=True)

    Minv = None
    if precond == "penalty":
        from optimize_specieswise_matrixfree import laplacian_degree
        deg = laplacian_degree(sp).to(DEV)
        diag = (lam * deg).clamp(min=1e-3).reshape(S, 1).expand(S, 3).reshape(-1).to(DTYPE)
        Minv = lambda v: v / diag

    if adam_steps > 0:
        print("--- Adam warmup (box-projected) ---", flush=True)
        theta = adam_warmup(f, theta, steps=adam_steps, lr=adam_lr, S=S, min_rate=min_rate, max_rate=max_rate)
    print("--- box projected-Newton-CG + negative-curvature escape ---", flush=True)
    theta, loss, pg, g, info = projected_newton_escape(
        f, theta, bs, rw, lap, S, lam, lo=lo, hi=hi, min_rate=min_rate, max_rate=max_rate,
        max_iter=newton, gtol=gtol, cg_max=cg_max, Minv=Minv, esc_tol=esc_tol, esc_m=esc_m,
        max_escape=max_escape, ckpt=ckpt)
    print(f"  [solver] escapes used={info['n_escape']} last_lam_min_free={info['last_lam_min_free']}", flush=True)
    frac_extreme = float((theta.abs() > 5).float().mean())
    print(f"\n=== SOLVE DONE  loss={loss:.4f}  |Pg|={pg:.3e}  frac|t|>5={frac_extreme:.3f} ===", flush=True)

    rec = dict(theta=theta.cpu(), loss=loss, proj_gnorm=pg, lam=lam, dataset=ds, n_families=len(paths),
               box=(min_rate, max_rate))
    # ---- bound-constrained KKT certificate ----
    if do_cert:
        tv = theta.reshape(-1)
        atol = 1e-4
        at_hi = (tv >= hi - atol); at_lo = (tv <= lo + atol)
        active = at_hi | at_lo; free_mask = (~active).to(DTYPE)
        n_act = int(active.sum())
        # active-set sign check: at upper box g<=0 (wants to push higher, can't); at lower box g>=0
        gd = g.reshape(-1)
        bad_hi = int(((at_hi) & (gd > 1e-3)).sum()); bad_lo = int(((at_lo) & (gd < -1e-3)).sum())
        print(f"\n[cert] active(box)={n_act}/{p} ({n_act/p:.3f})  hi={int(at_hi.sum())} lo={int(at_lo.sum())} "
              f"| KKT sign violations: hi={bad_hi} lo={bad_lo}", flush=True)
        # reduced Hessian PD on the FREE subspace
        Av = build_hvp_once(bs, theta.reshape(S, 3), rw, lap, p)
        lam_min_free, _v_min, resid = free_subspace_lanczos(Av, free_mask, p, m=cert_m)
        pd = lam_min_free > 0 and resid < 0.1 * max(abs(lam_min_free), 1e-6)
        print(f"[cert] lam_min_free={lam_min_free:+.5e} (Ritz resid={resid:.3e}, m={cert_m})  "
              f"reduced-Hessian PD={pd}", flush=True)
        verdict = (pd and bad_hi == 0 and bad_lo == 0 and pg < max(gtol * 5, 1e-2))
        print(f"[cert] VERDICT: {'CERTIFIED bound-constrained local min (|Pg| small, free-H PD, KKT signs ok)' if verdict else 'NOT fully certified -- see above'}", flush=True)
        rec.update(active=int(n_act), frac_active=n_act/p, lam_min_free=lam_min_free, ritz_resid=resid,
                   kkt_viol=(bad_hi, bad_lo), certified=bool(verdict))
    torch.save(rec, out)
    print(f"  saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
