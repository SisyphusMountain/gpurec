"""Bound-constrained JOINT (theta, alpha) minimum on archaea -> KKT certificate -> receiver-weight
Fisher / uncertainty.

Extends the bound-constrained theta-only solver (converge_bounded_archaea.py) to the JOINT variable
z = [theta.reshape(-1); alpha], where alpha in R^S are the per-species receiver (transfer-recipient)
logits. Both blocks are boxed:

  * theta (log2 D/T/L rates): theta in [log2(min_rate), log2(max_rate)]   (default rate [0.05, 2.0])
  * alpha (receiver logits):  alpha_i >= ALPHA_FLOOR                       (default -log2(100*S))

The alpha box is the receiver-weight analogue of the theta rate-floor: it pins the runaway w_i->0
(a species' recipient weight collapsing). The floor -log2(100*S) is so low it acts as a pure SAFETY
RAIL (a species kept above 1/(100*S) of the recipient mass) and in practice almost never binds, so
the only ACTIVE box constraints are the theta coords -- the joint free subspace is
(theta non-active) (+) (alpha gauge-fixed, all free).

GAUGE: alpha enters the NLL only via a full log_softmax (w = softmax(alpha)), so the loss is exactly
invariant under alpha -> alpha + c*1; the joint Hessian is singular along [0; 1_S]. The solver/cert
work in the gauge-fixed subspace via P_free = blockdiag(free_theta_mask, I_S - 11^T/S) (active-set
mask on theta AND mean-subtract on alpha), reusing the verified S9 consumers
(gpurec.solver.curvature.receiver) with their proj= hook. The joint analytic exact HVP (make_exact_hvp) is summed
across batches (multi-batch full archaea).

Pipeline: warm theta (certified theta-only min) -> short joint Adam warmup (alpha off uniform) ->
box-projected gauge Newton-CG to |P g| < gtol -> KKT cert (free-subspace reduced-Hessian PD via
deflated gauge Lanczos) -> receiver_information (alpha s.e. + recipient-prob s.e.). Atomic checkpoint
+ resume for A100 wall-kills.

Env: DATASET(archaea) FAMILIES(0=all) LAM(0.03) MIN_RATE(0.05) MAX_RATE(2.0) ALPHA_FLOOR(auto)
     INIT_THETA(<path>|zeros) ADAM(60) ADAM_LR(0.3) NEWTON(40) GTOL(1e-3) CG_MAX(60) CERT(1)
     CERT_M(200) FISHER(1) FISHER_SPECIES(0=all) SEED(0) OUT RESUME(1) SADDLE_DTYPE(float32|float64).
"""
from __future__ import annotations

import glob
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DEV = "cuda"
os.environ.setdefault("SADDLE_DTYPE", "float64")
os.environ.setdefault("GPUREC_MEMORY_POLICY_RESERVE_GIB", "0")
DTYPE = torch.float32 if os.environ["SADDLE_DTYPE"] == "float32" else torch.float64

from gpurec import GeneReconModel, SolverOptions
from gpurec.fit.optimize import first_order
from gpurec.solver.value_and_grad import forward_solve, make_value_and_grad, free_cuda_cache_if_tight
from gpurec.solver.hvp.exact import make_exact_hvp
from gpurec.solver.krylov import cg_witness, lanczos_extremes
from gpurec.optimization import clamp_log_rate_, project_rate_gradient_, log2_rate_bounds
from gpurec.solver.curvature.receiver import (
    certify_joint_min, proj_alpha, receiver_information,
)

# ------------------------------------------------------------------------------- archaea fixture
DATA = "/home/enzo/Documents/git/gpurec/gpurec/tests/data"
ARCHAEA_ROOT = os.environ.get("GPUREC_ARCHAEA_ROOT", f"{DATA}/alerax_archaea_davin2017")
_SP_TREE = f"{ARCHAEA_ROOT}/species_reference/reference_species_tree.newick"
# converged solver (matches the CV / bounded-theta runs)
_CV_SO = dict(e_max_iter=2000, e_tol=1e-8, pi_iters=64, neumann_terms=64,
              bicgstab_max_iter=500, bicgstab_tol=1e-7,
              bicgstab_breakdown_tol=1e-30, adjoint_pruning_threshold=1e-6,
              use_adjoint_pruning=True, pibar_side_threshold=0.0)


def _archaea_families(n):
    fs = sorted(glob.glob(f"{ARCHAEA_ROOT}/ale_gene_tree_distributions/main_families_ge4seq/*.ale"))
    return fs if (n is None or n <= 0) else fs[:n]


def atomic_save(obj, path):
    if not path:
        return
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ----------------------------------------------------------------- multi-batch joint HVP + penalty
def make_tree_lap(child, parent, lam):
    """GBM tree-Laplacian penalty HVP on the theta block: lam * L (acts per species, all 3 cats)."""
    def lap(vt):
        u = vt.reshape(-1, 3)
        out = torch.zeros_like(u)
        d = u.index_select(0, child) - u.index_select(0, parent)
        out.index_add_(0, child, d)
        out.index_add_(0, parent, -d)
        return (lam * out).reshape(-1)
    return lap


def build_joint_hvp_multibatch(batch_statics, theta2d, alpha, lap, theta_numel, S):
    """Sum the analytic joint (theta, alpha) exact HVP over batches (full archaea is multi-batch) and
    add the theta-block penalty. Builds per-batch caches ONCE; returns Hu(u) over z=[theta; alpha]."""
    hvps = []
    for st in batch_statics:
        _l, sv = forward_solve([st], theta2d, alpha)
        hvps.append(make_exact_hvp([st], theta2d, alpha, sv))
        free_cuda_cache_if_tight()
    p = theta_numel + S

    def Hu(u):
        u = u.to(theta2d.dtype)
        acc = torch.zeros(p, dtype=torch.float64, device=DEV)
        for h in hvps:
            acc += h(u).to(torch.float64)
        acc[:theta_numel] += lap(u[:theta_numel].to(torch.float64))
        return acc
    return Hu


# ----------------------------------------- receiver-weight FLOOR via softmax reparametrization
def make_reparam(S, *, floor_frac=1e-2):
    """Per-species recipient-probability floor. The model is fed alpha = log(w) with

        w = c*1 + (1 - S*c)*softmax(beta),   c = floor_frac/S = 1/(100 S),

    so EVERY w_i >= c = 1/(100 S) automatically (dominant sinks w_i->1 still allowed), and the
    optimizer works in the UNCONSTRAINED, gauge-fixed beta -- no box, no active set on the receiver
    block (this replaces the broken lower-only logit clamp, which a softmax shift defeats). The joint
    beta-Hessian is the Gauss-Newton wrap J^T H_aa J (J = dalpha/dbeta); it is EXACT at the minimum,
    where the receiver gradient g_a = 0 (gauge gives 1^T g_a = 0, stationarity gives g_a in
    null(J^T) = span(w), and span(w) ∩ {1^T x = 0} = {0})."""
    c = float(floor_frac) / S
    om = 1.0 - S * c

    def w_of_beta(beta):
        return c + om * torch.softmax(beta.double(), dim=-1)

    def alpha_of_beta(beta):
        return torch.log(w_of_beta(beta))

    def Jv(beta, vb):                                   # (dalpha/dbeta) vb
        s = torch.softmax(beta.double(), dim=-1)
        w = c + om * s
        return om * s * (vb.double() - (s * vb.double()).sum()) / w

    def JT(beta, u):                                    # (dalpha/dbeta)^T u
        s = torch.softmax(beta.double(), dim=-1)
        w = c + om * s
        uw = u.double() / w
        return om * s * (uw - (s * uw).sum())

    def dw_jac(beta):                                   # dw/dbeta = om (diag(s) - s s^T)  [S x S]
        s = torch.softmax(beta.double(), dim=-1)
        return om * (torch.diag(s) - torch.outer(s, s))

    return dict(c=c, om=om, w_of_beta=w_of_beta, alpha_of_beta=alpha_of_beta, Jv=Jv, JT=JT, dw_jac=dw_jac)


def make_Hbeta(batch_statics, theta2d, beta, rp, lap, theta_numel, S):
    """Gauss-Newton joint Hessian in z=[theta; beta]: wrap the analytic alpha-space joint HVP with the
    beta->alpha Jacobian. theta block is exact; the receiver block is J^T H_aa J (exact at the min)."""
    al = rp["alpha_of_beta"](beta)
    Ha = build_joint_hvp_multibatch(batch_statics, theta2d, al, lap, theta_numel, S)

    def Hb(v):
        vt, vb = v[:theta_numel], v[theta_numel:]
        Hr = Ha(torch.cat([vt, rp["Jv"](beta, vb)]))
        return torch.cat([Hr[:theta_numel], rp["JT"](beta, Hr[theta_numel:])])
    return Hb


# -------------------------------------------------------------------------------- box + gauge proj
def make_P_free(free_theta_mask, theta_numel):
    """P_free(v) = [free_theta_mask (.) v_theta ; v_alpha - mean(v_alpha)]  -- active-set mask on
    theta AND gauge-fix on alpha. Orthogonal projector onto the free joint subspace."""
    def P(v):
        vt = v[:theta_numel] * free_theta_mask
        va = proj_alpha(v[theta_numel:])
        return torch.cat([vt, va])
    return P


def binding_theta_mask(theta_flat, g_theta_flat, lo, hi, atol=1e-6):
    """Active (binding) theta coords: at a box edge with the gradient pushing further out."""
    at_lo = (theta_flat <= lo + atol) & (g_theta_flat > 0)
    at_hi = (theta_flat >= hi - atol) & (g_theta_flat < 0)
    return at_lo | at_hi


def proj_grad_joint(g_z, theta_flat, theta_numel, S, lo, hi):
    """Projected gradient: zero theta box-active outward components + gauge-fix the alpha block."""
    g = g_z.double().clone()
    project_rate_gradient_(theta_flat.reshape(S, 3),
                           g[:theta_numel].reshape(S, 3),
                           min_rate=2.0 ** lo, max_rate=2.0 ** hi)
    g[theta_numel:] = proj_alpha(g[theta_numel:])
    return g


# ---------------------------------------------------------------------------- bounded gauge Newton
def bounded_gauge_newton(batch_statics, vg, rp, z0, *, theta_numel, S, lo, hi, lap,
                         child, parent, lam, sigma=0.01, lanczos_m=10, nu=1.5, omega=1.5,
                         max_bumps=3, max_cg=60, c1=1e-4, ls_max=25, gtol=1e-3, max_newton=40,
                         seed=0, ckpt_path=None, verbose=True):
    """Box-projected gauge Newton on z=[theta; beta]: theta is boxed [lo,hi]; beta is the
    UNCONSTRAINED gauge-fixed receiver reparam variable (the w-floor is built into rp). The receiver
    block has NO box/active set -- only the gauge (mean-zero beta) is projected. Returns (theta, beta,
    |Pg|, F, history)."""
    z = z0.double().clone()
    z[theta_numel:] = proj_alpha(z[theta_numel:])          # land beta on the gauge slice

    def split(zv):
        return zv[:theta_numel].reshape(S, 3).contiguous(), zv[theta_numel:].contiguous()

    def clamp_z(zv):
        th, be = split(zv)
        clamp_log_rate_(th, min_rate=2.0 ** lo, max_rate=2.0 ** hi)
        return torch.cat([th.reshape(-1), proj_alpha(be)])  # theta box + beta gauge-fix (no beta box)

    def vg_beta(zv, want_grad=True):
        th, be = split(zv)
        out = vg(torch.cat([th.reshape(-1), rp["alpha_of_beta"](be)]), want_grad=want_grad)
        if not want_grad:
            return out[0], None, None, None
        g_z = out[1].double()
        return out[0], torch.cat([g_z[:theta_numel], rp["JT"](be, g_z[theta_numel:])]), None, None

    def build_Hbeta(zv):
        th, be = split(zv)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return make_Hbeta(batch_statics, th.reshape(S, 3), be, rp, lap, theta_numel, S)

    z = clamp_z(z)
    F, g_z, _, _ = vg_beta(z)
    th_flat = z[:theta_numel]
    gP = proj_grad_joint(g_z, th_flat, theta_numel, S, lo, hi)

    # initial lam_damp from a rough top-eigenvalue estimate of the gauge+box operator
    free_mask = (~binding_theta_mask(th_flat, g_z[:theta_numel].double(), lo, hi)).double()
    P = make_P_free(free_mask, theta_numel)
    Hu = build_Hbeta(z)
    Av0 = lambda v: P(Hu(P(v)))
    gen = torch.Generator(device=DEV).manual_seed(seed)
    start = P(torch.randn(theta_numel + S, generator=gen, device=DEV, dtype=torch.float64))
    _, lam_max = lanczos_extremes(Av0, theta_numel + S, m=lanczos_m, device=DEV, start=start)
    lam_max = max(lam_max, 1e-9)
    lam_damp = sigma * lam_max
    # NB: floor RELATIVE TO lam_max would over-damp -- lam_max is set by the stiff theta directions
    # (~5000 on full archaea) while the soft/free curvature is O(lam_min_free)~0.3, so a 1e-4*lam_max
    # floor (~0.5) pins lam_damp ABOVE the real curvature -> linear, not Newton, convergence. Use a
    # tiny absolute-ish floor so lam_damp adapts down to a near-Newton step; the cg_witness re-damps
    # on any negative curvature, so collapsing toward 0 on the PD free subspace is safe.
    lam_floor = max(1e-9 * lam_max, 1e-9)
    lam_ceil = 10.0 * lam_max
    if verbose:
        print(f"[bj] S={S} p={theta_numel + S}  lam_max~{lam_max:.3f}  lam_damp0={lam_damp:.4f}", flush=True)

    history = []
    hvp_stale = False
    # F-plateau stop: an under-identified joint (theta,beta) endgame can floor |Pg| well above gtol
    # (CG saturates max_iter, F freezes) and would otherwise grind to max_newton, wasting the wall and
    # never reaching the cert/Fisher stage. Stop once F improves by < ftol (relative) for `patience`
    # consecutive accepted steps. gtol still breaks first for well-conditioned problems that reach it.
    ftol = float(os.environ.get("FTOL", "1e-7"))
    plateau_patience = int(os.environ.get("PLATEAU_PATIENCE", "3"))
    stall = 0
    for k in range(int(max_newton)):
        th_flat = z[:theta_numel]
        gnorm = float(gP.norm())
        n_active = int(binding_theta_mask(th_flat, g_z[:theta_numel].double(), lo, hi).sum())
        history.append({"newton": k, "F": F, "gnorm": gnorm, "lam_damp": lam_damp, "active": n_active})
        if verbose:
            print(f"[bj {k:2d}] F={F:.6f}  |Pg|={gnorm:.4e}  active={n_active}/{theta_numel}  "
                  f"lam={lam_damp:.3e}", end="", flush=True)
        if gnorm < gtol:
            if verbose:
                print("  converged", flush=True)
            break
        if hvp_stale:
            Hu = None
            free_cuda_cache_if_tight(min_free_gib=8.0)
            Hu = build_Hbeta(z)
            hvp_stale = False
        free_mask = (~binding_theta_mask(th_flat, g_z[:theta_numel].double(), lo, hi)).double()
        P = make_P_free(free_mask, theta_numel)
        Hz = lambda v: P(Hu(P(v)))

        eta = min(0.1, gnorm ** 0.5)
        p_step, cg_iters, status, cert = None, 0, "", None
        for _bump in range(int(max_bumps) + 1):
            Av = lambda v, ld=lam_damp: Hz(v) + ld * P(v)
            p_step, cg_iters, status, cert = cg_witness(Av, -gP, tol=eta * gnorm, max_iter=max_cg)
            if status != "neg_curv":
                break
            lam_damp = min(lam_ceil, nu * (lam_damp - cert))
        if status == "neg_curv":
            p_step = -gP / lam_damp
            status = "fallback_gd"
        p_step = P(p_step)
        gp = float(torch.dot(gP, p_step))
        if gp >= 0.0:
            p_step = -gP / lam_damp
            gp = -gnorm * gnorm / lam_damp
            status += "+gd"

        alpha_ls, accepted = 1.0, False
        for _ in range(int(ls_max)):
            trial = clamp_z(z + alpha_ls * p_step)
            Ft, _, _, _ = vg_beta(trial, want_grad=False)
            if Ft <= F + c1 * alpha_ls * gp:
                accepted = True
                break
            alpha_ls *= 0.5
        if accepted:
            z = trial
            hvp_stale = True
            lam_damp = max(lam_floor, lam_damp / omega) if alpha_ls == 1.0 else min(lam_ceil, 1.5 * lam_damp)
            F, g_z, _, _ = vg_beta(z)
            gP = proj_grad_joint(g_z, z[:theta_numel], theta_numel, S, lo, hi)
            if verbose:
                print(f"  cg={cg_iters}({status})  a={alpha_ls:.1e}  dF={F - history[-1]['F']:+.3e}", flush=True)
            if ckpt_path:
                atomic_save({"z": z.cpu(), "F": F, "gnorm": float(gP.norm()), "k": k,
                             "theta_numel": theta_numel, "S": S}, ckpt_path)
            rel_dF = abs(F - history[-1]["F"]) / max(1.0, abs(F))
            stall = stall + 1 if rel_dF < ftol else 0
            if stall >= plateau_patience:
                if verbose:
                    print(f"  F-plateau ({stall} steps rel_dF<{ftol:.0e}), stopping at |Pg|={float(gP.norm()):.3e}",
                          flush=True)
                break
        else:
            lam_damp = min(lam_ceil, 4.0 * lam_damp)
            if verbose:
                print(f"  cg={cg_iters}({status})  line-search FAILED -> lam={lam_damp:.3e}", flush=True)
            if lam_damp >= lam_ceil:
                if verbose:
                    print("  lam at ceiling, stopping", flush=True)
                break

    th, be = split(z)
    return th, proj_alpha(be), float(gP.norm()), F, history


# --------------------------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    n_fam = int(os.environ.get("FAMILIES", "256"))
    lam = float(os.environ.get("LAM", "0.03"))
    min_rate = float(os.environ.get("MIN_RATE", "0.05"))
    max_rate = float(os.environ.get("MAX_RATE", "2.0"))
    adam_steps = int(os.environ.get("ADAM", "60"))
    adam_lr = float(os.environ.get("ADAM_LR", "0.3"))
    newton_steps = int(os.environ.get("NEWTON", "40"))
    gtol = float(os.environ.get("GTOL", "1e-3"))
    cg_max = int(os.environ.get("CG_MAX", "60"))
    do_cert = os.environ.get("CERT", "1") == "1"
    cert_m = int(os.environ.get("CERT_M", "200"))
    do_fisher = os.environ.get("FISHER", "1") == "1"
    fisher_species = int(os.environ.get("FISHER_SPECIES", "0"))  # 0 = all
    seed = int(os.environ.get("SEED", "0"))
    init_theta = os.environ.get("INIT_THETA", "")
    out = os.environ.get("OUT", f"runs/bounded_joint_archaea_n{n_fam}_lam{lam}_{os.environ['SADDLE_DTYPE']}.pt")
    resume = os.environ.get("RESUME", "1") == "1"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    print(f"[build] archaea n_fam={n_fam if n_fam>0 else 'ALL'} lam={lam} box=[{min_rate},{max_rate}] "
          f"dtype={DTYPE}", flush=True)
    so = SolverOptions(**_CV_SO)
    so.validate()
    paths = _archaea_families(None if n_fam <= 0 else n_fam)
    model = GeneReconModel(_SP_TREE, [str(p) for p in paths], mode="specieswise",
                           device=DEV, solver_options=so)
    bs = model.batch_statics
    S = int(model.species_helpers["S"])
    theta_numel = 3 * S
    lo, hi = log2_rate_bounds(min_rate, max_rate)
    rp = make_reparam(S, floor_frac=float(os.environ.get("W_FLOOR_FRAC", "1e-2")))
    sp_parent = model.species_helpers["sp_parent"].to(DEV).long().reshape(-1)
    child = (sp_parent >= 0).nonzero(as_tuple=True)[0].contiguous()
    parent = sp_parent[child].contiguous()
    lap = make_tree_lap(child, parent, lam)
    print(f"[build] {len(bs)} batch(es)  S={S}  p_joint={theta_numel + S}  "
          f"w_floor=c={rp['c']:.3e} (=1/{round(1/rp['c'])})  [reparam w=c+(1-Sc)softmax(beta)]", flush=True)

    # ---- warm theta ----
    if init_theta and os.path.exists(init_theta):
        d = torch.load(init_theta, map_location=DEV, weights_only=False)
        theta0 = (d["theta"] if isinstance(d, dict) else d).to(DEV).to(DTYPE).reshape(S, 3).contiguous()
        print(f"[init] warm theta from {init_theta} (loss {d.get('loss','?') if isinstance(d,dict) else '?'})",
              flush=True)
    else:
        theta0 = torch.full((S, 3), math.log2(0.1), device=DEV, dtype=DTYPE)
        print("[init] theta = log2(0.1)", flush=True)
    clamp_log_rate_(theta0, min_rate=min_rate, max_rate=max_rate)
    # beta0: small gauge-fixed random -> w slightly non-uniform (the joint HVP needs non-uniform alpha).
    g = torch.Generator(device=DEV).manual_seed(seed)
    beta0 = proj_alpha(0.05 * torch.randn(S, generator=g, device=DEV, dtype=torch.float64))

    # vg is alpha-space; bounded_gauge_newton feeds it alpha=alpha_of_beta(beta). The 2nd arg only fixes
    # S/dtype/device (the live alpha is read from each call's z).
    vg = make_value_and_grad(bs, beta0, theta_shape=(S, 3), optimize_receiver=True,
                             tree_penalty=(lam, sp_parent))

    ckpt = out + ".ckpt"
    z0 = None
    if resume and os.path.exists(ckpt):
        c = torch.load(ckpt, map_location=DEV, weights_only=False)
        z0 = c["z"].to(DEV).double()
        print(f"[resume] from {ckpt}: k={c.get('k')} F={c.get('F'):.4f} |Pg|={c.get('gnorm'):.3e}", flush=True)
    if z0 is None:
        z0 = torch.cat([theta0.reshape(-1).double(), beta0])  # no warmup: Newton converges from cold beta

    # ---- bounded gauge Newton on z=[theta; beta] ----
    theta, beta, pg, F, hist = bounded_gauge_newton(
        bs, vg, rp, z0, theta_numel=theta_numel, S=S, lo=lo, hi=hi,
        lap=lap, child=child, parent=parent, lam=lam, max_cg=cg_max, gtol=gtol,
        max_newton=newton_steps, seed=seed, ckpt_path=ckpt)
    alpha = rp["alpha_of_beta"](beta)
    w = rp["w_of_beta"](beta)
    print(f"\n[SOLVE DONE] F={F:.6f}  |Pg|={pg:.4e}  w in [{float(w.min()):.3e},{float(w.max()):.3e}] "
          f"(floor {rp['c']:.2e})  ({time.time()-t0:.0f}s)", flush=True)

    rec = dict(theta=theta.cpu(), beta=beta.cpu(), alpha=alpha.cpu(), w=w.cpu(), loss=F, proj_gnorm=pg,
               lam=lam, box=(min_rate, max_rate), w_floor=rp["c"], n_families=len(paths),
               S=S, history=hist, dtype=os.environ["SADDLE_DTYPE"])
    atomic_save(rec, out)

    # ---- KKT certificate on the free joint subspace (theta box-free + beta gauge) ----
    if do_cert:
        g_z = vg(torch.cat([theta.reshape(-1).double(), alpha.double()]))[1].double()
        free_mask = (~binding_theta_mask(theta.reshape(-1).double(), g_z[:theta_numel], lo, hi)).double()
        P = make_P_free(free_mask, theta_numel)
        Hb = make_Hbeta(bs, theta.reshape(S, 3), beta, rp, lap, theta_numel, S)  # GN beta-Hessian
        print("[cert] free-subspace reduced-Hessian PD (deflated gauge Lanczos)...", flush=True)
        cert = certify_joint_min(bs, theta, beta, hvp=Hb, theta_numel=theta_numel, S=S,
                                 proj=P, m=cert_m, verbose=True)
        n_active = int((free_mask < 0.5).sum())
        rec.update(lam_min_free=cert["lam_min_gauge"], ritz_resid=cert["ritz_resid"],
                   cert_leak=cert["leak"], certified=bool(cert["pd"]), active=n_active,
                   frac_active=n_active / theta_numel)
        atomic_save(rec, out)

        # ---- receiver-weight Fisher / uncertainty (covariance of beta -> delta-method s.e. of w) ----
        if do_fisher:
            fisher_cg_max = int(os.environ.get("FISHER_CG_MAX", "400"))
            fisher_cg_tol = float(os.environ.get("FISHER_CG_TOL", "1e-6"))
            sp = None if fisher_species <= 0 else list(range(min(fisher_species, S)))
            _r = os.environ.get("FISHER_RIDGE")
            ridge = float(_r) if _r else (0.0 if cert["pd"] else max(1e-3, abs(cert["lam_min_gauge"]) + 1e-2))
            print(f"[fisher] receiver_information (ridge={ridge:.1e}, cg_max={fisher_cg_max}, "
                  f"{'all '+str(S) if sp is None else len(sp)} species)...", flush=True)
            fish = receiver_information(bs, theta, beta, hvp=Hb, theta_numel=theta_numel, S=S,
                                        proj=P, species=sp, ridge=ridge, cg_max=fisher_cg_max,
                                        cg_tol=fisher_cg_tol, verbose=True)
            rec.update(se_beta=fish["se_alpha"].cpu(), fisher_ridge=ridge,
                       fisher_species=(list(range(S)) if sp is None else sp),
                       fisher_cg_resid_max=max(fish["cg_resid"]))
            if sp is None:  # full covariance -> delta-method s.e. of the recipient probabilities w
                Jw = rp["dw_jac"](beta)                          # dw/dbeta [S,S]
                Sig_w = Jw @ fish["Sigma_aa"] @ Jw.T
                se_w = Sig_w.diagonal().clamp_min(0.0).sqrt()
                rec["se_w"] = se_w.cpu()
                print(f"[fisher] se_w (recipient prob) in [{float(se_w.min()):.3e}, {float(se_w.max()):.3e}]",
                      flush=True)
            print(f"[fisher] se_beta in [{float(fish['se_alpha'].min()):.3e}, "
                  f"{float(fish['se_alpha'].max()):.3e}]  max_cg_resid={max(fish['cg_resid']):.2e}", flush=True)

    atomic_save(rec, out)
    print(f"[saved] {out}  ({time.time()-t0:.0f}s total)", flush=True)
    return rec


if __name__ == "__main__":
    main()
