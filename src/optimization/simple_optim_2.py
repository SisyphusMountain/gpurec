#!/usr/bin/env python3
from __future__ import annotations

# ---- Imports -----------------------------------------------------------------
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.autograd.functional import vjp

from src.optimization.theta_optimizer import ThetaOptimizationProblem
from src.core.likelihood_2 import (
    E_step,
    Pi_step,
    compute_log_likelihood,
)

# ---- Records -----------------------------------------------------------------
@dataclass
class StepRecord:
    iteration: int
    theta: torch.Tensor
    rates: torch.Tensor
    negative_log_likelihood: float
    log_likelihood: float
    theta_step_inf: float
    grad_infinity_norm: float

# ---- Helpers: flatten/unflatten ----------------------------------------------
def _flatten_state(Pi: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    return torch.cat([Pi.reshape(-1), E.reshape(-1)])

def _build_unflatten(Pi_ref: torch.Tensor, E_ref: torch.Tensor):
    pi_numel = Pi_ref.numel()
    def unflatten(x_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        Pi_part = x_flat[:pi_numel].view_as(Pi_ref)
        E_part = x_flat[pi_numel:].view_as(E_ref)
        return Pi_part, E_part
    return unflatten

# ---- Fixed-point map H(x, θ) -------------------------------------------------
def _fixed_point_map(
    x_flat: torch.Tensor,
    theta: torch.Tensor,
    *,
    problem: ThetaOptimizationProblem,
    Pi_ref: torch.Tensor,
    E_ref: torch.Tensor,
) -> torch.Tensor:
    """H(x, θ) = flatten(F(Pi,E,θ), G(E,θ))."""
    unflatten = _build_unflatten(Pi_ref, E_ref)
    Pi_cur, E_cur = unflatten(x_flat)

    # E update (returns E_next + components)
    E_next, E_s1, E_s2, E_bar = E_step(
        E_cur,
        problem.species_helpers["s_P_indexes"],
        problem.species_helpers["s_C12_indexes"],
        problem.species_helpers["Recipients_mat"],
        theta,
        return_components=True,
    )

    # Pi update
    log_two = torch.tensor([math.log(2.0)], dtype=theta.dtype, device=theta.device)
    Pi_next = Pi_step(
        Pi_cur,
        problem.ccp_helpers,
        problem.species_helpers,
        problem.clade_species_map,
        E_cur,
        E_bar,
        E_s1,
        E_s2,
        theta,
        log_two,
    )
    return _flatten_state(Pi_next, E_next)

# ==============================================================================
# Fast adjoint: Flexible GMRES + matrix-free Jacobi
# ==============================================================================

@torch.no_grad()
def _estimate_diag_hutch(
    Av: Callable[[torch.Tensor], torch.Tensor],
    n: int, *, k: int, device, dtype, eps: float = 1e-8
) -> torch.Tensor:
    """Estimate diag(A) ≈ mean_i z ⊙ (A z) with k Rademacher probes."""
    acc = torch.zeros(n, device=device, dtype=dtype)
    for _ in range(k):
        z = torch.empty(n, device=device, dtype=dtype).bernoulli_(0.5).mul_(2).add_(-1)  # ±1
        acc += z * Av(z)
    d = (acc / float(k)).abs().clamp_min(eps)
    med = d.median()
    if med > 0:
        d = d / med
    return d

@torch.no_grad()
def _fgmres(
    Av: Callable[[torch.Tensor], torch.Tensor], b: torch.Tensor, *,
    M_apply: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    tol: float = 1e-6, restart: int = 40, maxit: int = 200,
    x0: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Flexible GMRES (right preconditioning), minimizes ||r||_2."""
    device, dtype = b.device, b.dtype
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    if M_apply is None:
        M_apply = lambda v: v

    it = 0
    while it < maxit:
        r = b - Av(x)
        beta = r.norm()
        if beta <= tol * b.norm().clamp_min(1.0):
            return x
        m = min(restart, maxit - it)
        V, Z = [], []
        H = torch.zeros((m+1, m), device=device, dtype=dtype)
        v1 = r / beta
        V.append(v1)
        happy = False
        for j in range(m):
            z_j = M_apply(V[j]); Z.append(z_j)
            w = Av(z_j)  # one VJP inside Av if Av = I - J^T
            # Modified Gram–Schmidt
            for i in range(j+1):
                hij = torch.dot(V[i], w)
                H[i, j] = hij
                w = w - hij * V[i]
            H[j+1, j] = w.norm()
            if H[j+1, j] < 1e-14:
                happy = True
                m = j + 1
                break
            V.append(w / H[j+1, j])
        e1 = torch.zeros(m+1, device=device, dtype=dtype); e1[0] = 1.0
        y = torch.linalg.lstsq(H[:m+1, :m], beta * e1[:m+1]).solution
        x = x + sum(y[i] * Z[i] for i in range(m))
        it += m
        if happy:
            return x
    return x

@torch.no_grad()
def _solve_adjoint_krylov(
    vjp_apply: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor, *,
    tol: float = 1e-6, restart: int = 40, maxit: int = 200,
    prec_hutch_probes: int = 8,
    warm_start: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Solve (I - J^T) x = rhs using only v -> J^T v. Right-preconditioned FGMRES with Jacobi.
    """
    device, dtype = rhs.device, rhs.dtype
    def Av(x):  # A x = (I - J^T) x
        return x - vjp_apply(x)
    d = _estimate_diag_hutch(Av, rhs.numel(), k=prec_hutch_probes, device=device, dtype=dtype)
    M_apply = lambda v: v / d
    return _fgmres(Av, rhs, M_apply=M_apply, tol=tol, restart=restart, maxit=maxit, x0=warm_start)

# ---- Implicit gradient via VJP + Krylov --------------------------------------
def _implicit_grad_vjp(
    problem: ThetaOptimizationProblem,
    theta: torch.Tensor,
    E_star: torch.Tensor,
    Pi_star: torch.Tensor,
    *,
    tol: float = 1e-6,
    krylov_restart: int = 40,
    krylov_maxit: int = 200,
    hutch_probes: int = 8,
    lam: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute dL/dθ at the fixed point using only VJP and FGMRES for (I - J_x^T) λ = ∂L/∂x.
    """
    # Build graph at fixed point
    Pi_var = Pi_star.detach().clone().requires_grad_(True)
    E_var  = E_star.detach().clone().requires_grad_(True)
    flat_state = _flatten_state(Pi_var, E_var)

    # VJP pullbacks (closures)
    def JxT_apply(v: torch.Tensor) -> torch.Tensor:
        v = v.to(device=flat_state.device, dtype=flat_state.dtype)
        _, JxT_v = vjp(
            lambda s: _fixed_point_map(s, theta, problem=problem, Pi_ref=Pi_var, E_ref=E_var),
            flat_state, v=v, create_graph=False
        )
        return JxT_v

    def JthetaT_apply(v: torch.Tensor) -> torch.Tensor:
        v = v.to(device=theta.device, dtype=theta.dtype)
        _, JthetaT_v = vjp(
            lambda th: _fixed_point_map(flat_state, th, problem=problem, Pi_ref=Pi_var, E_ref=E_var),
            theta, v=v, create_graph=False
        )
        return JthetaT_v

    # RHS b = ∂L/∂x (L depends only on Pi in the readout)
    L = compute_log_likelihood(Pi_var, problem.root_clade_id)
    dL_dPi, dL_dE = torch.autograd.grad(L, (Pi_var, E_var), retain_graph=False, allow_unused=True)
    if dL_dE is None:
        dL_dE = torch.zeros_like(E_var)
    b = _flatten_state(dL_dPi, dL_dE)

    # Solve (I - J_x^T) λ = b with FGMRES + Jacobi
    lam = _solve_adjoint_krylov(
        JxT_apply, b,
        tol=tol, restart=krylov_restart, maxit=krylov_maxit,
        prec_hutch_probes=hutch_probes, warm_start=lam
    )

    # Gradient in θ
    grad_theta = JthetaT_apply(lam)
    return grad_theta, lam

# ==============================================================================
# L-BFGS (two-loop) + non-monotone Armijo (GLL)
# ==============================================================================

def _lbfgs_two_loop(g, S_hist, Y_hist, H0=1.0):
    q = g.clone()
    alpha = []
    for s, y in reversed(list(zip(S_hist, Y_hist))):
        rho = 1.0 / (y @ s)
        a = rho * (s @ q)
        alpha.append(a); q = q - a * y
    r = H0 * q
    for (s, y), a in zip(zip(S_hist, Y_hist), reversed(alpha)):
        rho = 1.0 / (y @ s)
        beta = rho * (y @ r)
        r = r + s * (a - beta)
    return -r

def _nonmonotone_armijo(eval_fn, theta, f0, g0, p, window_vals,
                        c1=1e-4, tau=0.5, t0=1.0, t_min=1e-8, max_backtracks=8):
    """
    Minimize nLL: accept first t with
      f(θ+tp) <= max(window) + c1 * t * <g0, p>.
    Reject NaN/Inf trials automatically.
    """
    f_ref = max(window_vals) if window_vals else f0
    t = t0
    for _ in range(max_backtracks):
        f_try = float(eval_fn(theta + t * p))
        if not (math.isfinite(f_try)):
            t *= tau
            if t < t_min: break
            continue
        # sufficient decrease in nLL
        if f_try <= f_ref + c1 * t * float(g0 @ p):
            return t, f_try
        t *= tau
        if t < t_min: break
    return 0.0, f0


# ==============================================================================
# Public API: optimize_theta (signature preserved)
# ==============================================================================

def optimize_theta(
    species_tree_path: str,
    gene_tree_path: str,
    theta_init: torch.Tensor,
    *,
    steps: int = 200,
    lr: float = 0.2,                 # used as initial step scale for L-BFGS line-search (t0)
    adam_eps: float = 1e-8,          # kept for signature compatibility (unused)
    tol_theta: float = 1e-3,
    e_max_iters: int = 2000,
    pi_max_iters: int = 2000,
    e_tol: float = 1e-12,
    pi_tol: float = 1e-12,
    damping: float = 1.0,            # kept for signature compatibility (unused)
    vjp_max_iter: int = 500,         # FGMRES maxit
    vjp_tol: float = 1e-6,           # FGMRES tol
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> Dict[str, object]:
    """
    Curvature-aware L-BFGS (two-loop) + non-monotone Armijo on θ.
    Implicit gradients solved with FGMRES (+ Jacobi) using only VJPs.
    """
    # --- setup
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    theta = theta_init.detach().clone().to(device=device, dtype=dtype).requires_grad_(True)
    problem = ThetaOptimizationProblem(
        species_tree_path, gene_tree_path, device=device, dtype=dtype
    )
    # stash problem tolerances/iters
    problem.e_max_iters = e_max_iters
    problem.pi_max_iters = pi_max_iters

    # L-BFGS state
    S_hist: List[torch.Tensor] = []
    Y_hist: List[torch.Tensor] = []
    window_vals: List[float] = []
    prev_theta = theta.detach().clone()
    lam: Optional[torch.Tensor] = None  # adjoint warm-start
    history: List[StepRecord] = []

    # adaptive inner tolerances (loose early, tight late)
    loose_e  = max(e_tol, 1e-4)
    loose_pi = max(pi_tol, 1e-4)

    # one evaluation (NLL) + implicit gradient at current θ
    def eval_and_grad(loose: bool):
        problem.e_tol  = loose_e  if loose else e_tol
        problem.pi_tol = loose_pi if loose else pi_tol
        E, Pi, _ = problem.fixed_points(
            theta,
            e_max_iters=problem.e_max_iters, pi_max_iters=problem.pi_max_iters,
            e_tol=problem.e_tol, pi_tol=problem.pi_tol, warm_start=True
        )
        L = compute_log_likelihood(Pi, problem.root_clade_id)
        nll = -L
        gL, lam_out = _implicit_grad_vjp(
            problem, theta, E, Pi,
            tol=vjp_tol, krylov_restart=min(40, max(10, vjp_max_iter)),
            krylov_maxit=vjp_max_iter, hutch_probes=8, lam=lam
        )
        g_nll = -gL.detach()   # <-- SIGN FIX: gradient of nLL
        return nll, g_nll, lam_out, E.detach(), Pi.detach()

    # scalar evaluator for line search
    def f_scalar(th_try: torch.Tensor) -> float:
        with torch.no_grad():
            theta.data.copy_(th_try)
        problem.e_tol, problem.pi_tol = loose_e, loose_pi
        E, Pi, _ = problem.fixed_points(
            theta, e_max_iters=problem.e_max_iters, pi_max_iters=problem.pi_max_iters,
            e_tol=problem.e_tol, pi_tol=problem.pi_tol, warm_start=True
        )
        L = compute_log_likelihood(Pi, problem.root_clade_id)
        return float((-L).detach())

    # --- main loop
    for it in range(1, steps + 1):
        # (1) Evaluate and get gradient (loose tolerances)
        nll, g, lam, E_star, Pi_star = eval_and_grad(loose=True)
        f0 = float(nll.detach())

        # (2) L-BFGS direction
        if len(S_hist) == 0:
            p = -g
            H0 = 1.0
        else:
            s_last, y_last = S_hist[-1], Y_hist[-1]
            H0 = float((s_last @ y_last) / (y_last @ y_last))
            H0 = max(abs(H0), 1e-3)
            p = _lbfgs_two_loop(g, S_hist, Y_hist, H0=H0)

        # (3) Non-monotone Armijo (few backtracks)
        t_init = lr  # use lr as initial trial step
        t, f_try = _nonmonotone_armijo(f_scalar, theta.detach(), f0, g, p, window_vals,
                                       c1=1e-4, tau=0.5, t0=t_init, max_backtracks=6)
        if t == 0.0:
            t = 1e-3  # ultra-conservative fallback

        theta_new = (theta.detach() + t * p).requires_grad_(True)

        # (4) Accept step; tighten inner tolerances on accepted θ for accuracy
        with torch.no_grad():
            theta.data.copy_(theta_new)

        # Accurate eval + grad (tight tolerances)
        nll_new, g_new, lam, _, _ = eval_and_grad(loose=False)

        # L-BFGS memory update
        s = (theta.detach() - prev_theta).clone()
        y = g_new - g
        if (y @ s) > 1e-12:
            S_hist.append(s); Y_hist.append(y)
            if len(S_hist) > 20:
                S_hist.pop(0); Y_hist.pop(0)

        # bookkeeping
        prev_theta = theta.detach().clone()
        window_vals.append(float(nll_new.detach()))
        if len(window_vals) > 5:
            window_vals.pop(0)

        # history (uses f0 / nll_new)
        rates = torch.exp(theta.detach())
        step_inf = float(s.abs().max().item())
        grad_inf = float(g_new.abs().max().item())
        history.append(StepRecord(
            iteration=it,
            theta=theta.detach().cpu(),
            rates=rates.cpu(),
            negative_log_likelihood=float(nll_new.item()),
            log_likelihood=float((-nll_new).item()),
            theta_step_inf=step_inf,
            grad_infinity_norm=grad_inf,
        ))

        # stopping criteria
        if step_inf < tol_theta and g_new.norm().item() < 1e-3:
            print(f"[{it:03d}] Converged: nLL={float(nll_new):.6f}")
            break
        else:
            print(f"[{it:03d}] nLL={float(nll_new):.6f} | ||g||={g_new.norm().item():.3e} | step∞={step_inf:.2e}")

    # --- final report
    problem.e_tol, problem.pi_tol = e_tol, pi_tol
    E_fin, Pi_fin, _ = problem.fixed_points(
        theta, e_max_iters=problem.e_max_iters, pi_max_iters=problem.pi_max_iters,
        e_tol=problem.e_tol, pi_tol=problem.pi_tol, warm_start=True
    )
    L_fin = compute_log_likelihood(Pi_fin, problem.root_clade_id)
    return {
        "theta": theta.detach().cpu(),
        "rates": torch.exp(theta.detach()).cpu(),
        "log_likelihood": float(L_fin.item()),
        "negative_log_likelihood": float((-L_fin).item()),
        "history": history,
    }
