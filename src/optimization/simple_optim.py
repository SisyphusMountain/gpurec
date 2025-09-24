"""
Simple theta optimization using fixed-point implicit gradients (VJP only).

- Converges E*, Pi* for the current theta via likelihood_2 fixed-point steps.
- Computes the implicit gradient dL/dtheta using only autograd.functional.vjp:
    (I - J_x H)^T lambda = dL/dx,  grad_theta = (J_theta H)^T lambda
  where x = [Pi, E] and H(x, theta) = (F(Pi, E, theta), G(E, theta)).

This module keeps things minimal and explicit for clarity/debuggability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch.autograd.functional import vjp

from src.optimization.theta_optimizer import ThetaOptimizationProblem
from src.core.likelihood_2 import (
    E_step,
    Pi_step,
    compute_log_likelihood,
)

from collections import deque

class _ThetaQN:
    """
    Pseudo-curvature for theta (p=dim(theta) small).
    Stores (s_i=Δθ, y_i=Δg) and returns an L-BFGS two-loop direction.
    Also provides Barzilai-Borwein (BB2) step scaling as a fallback.
    """
    def __init__(self, max_hist: int = 10):
        self.max_hist = max_hist
        self.S: List[torch.Tensor] = []
        self.Y: List[torch.Tensor] = []
        self.g_prev: Optional[torch.Tensor] = None
        self.th_prev: Optional[torch.Tensor] = None

    def update(self, theta: torch.Tensor, g: torch.Tensor):
        # expects detached clones
        if self.th_prev is not None and self.g_prev is not None:
            s = (theta - self.th_prev).reshape(-1)
            y = (g - self.g_prev).reshape(-1)
            if torch.isfinite(s).all() and torch.isfinite(y).all():
                if (y @ s) > 1e-12:
                    self.S.append(s); self.Y.append(y)
                    if len(self.S) > self.max_hist:
                        self.S.pop(0); self.Y.pop(0)
        self.th_prev = theta.clone()
        self.g_prev  = g.clone()

    def lbfgs_direction(self, g: torch.Tensor, H0_min: float = 1e-3) -> torch.Tensor:
        """Return p ≈ -H^{-1} g via two-loop; if no history, return -g."""
        if len(self.S) == 0:
            return -g
        s_last, y_last = self.S[-1], self.Y[-1]
        H0 = float((s_last @ y_last) / (y_last @ y_last))
        H0 = max(abs(H0), H0_min)
        # two-loop
        q = g.clone()
        alpha: List[float] = []
        for s, y in zip(reversed(self.S), reversed(self.Y)):
            rho = 1.0 / (y @ s)
            a = rho * (s @ q)
            alpha.append(a)
            q = q - a * y
        r = H0 * q
        for (s, y), a in zip(zip(self.S, self.Y), reversed(alpha)):
            rho = 1.0 / (y @ s)
            beta = rho * (y @ r)
            r = r + s * (a - beta)
        return -r

    def bb2_scale(self, theta: torch.Tensor, g: torch.Tensor, lr_base: float,
                  lr_min: float = 1e-4, lr_max: float = 10.0) -> float:
        """Return lr_eff = lr_base * (s·y)/(y·y) if previous (s,y) exist."""
        if self.th_prev is None or self.g_prev is None:
            return lr_base
        s = (theta - self.th_prev).reshape(-1)
        y = (g - self.g_prev).reshape(-1)
        yy = (y @ y).item()
        if yy <= 1e-20:
            return lr_base
        alpha_bb2 = (s @ y).item() / yy
        alpha_bb2 = max(min(alpha_bb2, lr_max), lr_min)
        return lr_base * alpha_bb2



class _PseudoCurvMem:
    """
    Learns a linear map T: R^{p} -> R^{n} s.t. Δλ ≈ T Δθ using past pairs.
    p = len(theta) (here 3); n = dim(flat_state) for λ.
    Stores small history and fits T by ridge-LS (closed-form) each time we predict.
    """
    def __init__(self, max_hist: int = 6, ridge: float = 1e-6):
        self.max_hist = max_hist
        self.ridge = ridge
        self.theta_hist = deque()  # list of θ_k (detach clones)
        self.lam_hist   = deque()  # list of λ_k (detach clones)

    def update(self, theta_k: torch.Tensor, lam_k: torch.Tensor):
        th = theta_k.detach().clone()
        la = lam_k.detach().clone()
        if len(self.theta_hist) and th.shape != self.theta_hist[-1].shape:
            # ignore shape change
            return
        self.theta_hist.append(th)
        self.lam_hist.append(la)
        if len(self.theta_hist) > self.max_hist:
            self.theta_hist.popleft(); self.lam_hist.popleft()

    @torch.no_grad()
    def predict_lambda(self, theta_new: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Return predicted λ_new from current cache, or None if not enough info.
        Uses multi-secant T ≈ ΔΛ (ΔΘ)^T (ΔΘ ΔΘ^T + ρI)^{-1} and λ_pred = λ_last + T (θ_new-θ_last).
        """
        m = len(self.theta_hist)
        if m < 2:
            return None
        # Build ΔΘ ∈ R^{p×(m-1)}, ΔΛ ∈ R^{n×(m-1)}
        p = self.theta_hist[0].numel()
        n = self.lam_hist[0].numel()
        dTheta = torch.empty((p, m-1), device=theta_new.device, dtype=theta_new.dtype)
        dLambda = torch.empty((n, m-1), device=self.lam_hist[0].device, dtype=self.lam_hist[0].dtype)
        for i in range(m-1):
            dTheta[:, i]  = (self.theta_hist[i+1] - self.theta_hist[i]).reshape(-1)
            dLambda[:, i] = (self.lam_hist[i+1]   - self.lam_hist[i]).reshape(-1)
        # If last step tiny, fallback
        s = (theta_new.detach() - self.theta_hist[-1]).reshape(-1)
        if s.abs().max() < 1e-12:
            return self.lam_hist[-1].clone()
        # Ridge LS: T = ΔΛ ΔΘ^T (ΔΘ ΔΘ^T + ρI)^{-1}  (p×p inversion only; here p=3)
        G = dTheta @ dTheta.T     # p×p
        G = G + self.ridge * torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
        T = (dLambda @ dTheta.T) @ torch.linalg.solve(G, torch.eye(G.shape[0], device=G.device, dtype=G.dtype))
        dlam_pred = T @ s
        return self.lam_hist[-1].reshape(-1) + dlam_pred



## GMRES ##
# ---------- FGMRES + matrix-free Jacobi (uses only v -> J^T v) ----------
from typing import Callable, Optional
import torch, math

@torch.no_grad()
def _estimate_diag_hutch(Av: Callable[[torch.Tensor], torch.Tensor],
                         n: int, *, k: int, device, dtype,
                         eps: float = 1e-8) -> torch.Tensor:
    """Estimate diag(A) ≈ mean_i z ⊙ (A z) with k Rademacher probes; A is matrix-free."""
    acc = torch.zeros(n, device=device, dtype=dtype)
    for _ in range(k):
        z = torch.empty(n, device=device, dtype=dtype).bernoulli_(0.5).mul_(2).add_(-1)  # ±1
        acc += z * Av(z)
    d = (acc / float(k)).abs().clamp_min(eps)
    med = d.median()
    if med > 0: d = d / med                        # normalize scales
    return d

@torch.no_grad()
def _fgmres(Av: Callable[[torch.Tensor], torch.Tensor], b: torch.Tensor, *,
            M_apply: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
            tol: float = 1e-6, restart: int = 40, maxit: int = 200,
            x0: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Flexible GMRES (right-preconditioned), minimizing ||r||_2."""
    device, dtype = b.device, b.dtype
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    if M_apply is None: M_apply = lambda v: v

    it = 0
    while it < maxit:
        r = b - Av(x)
        beta = r.norm()
        if beta <= tol * b.norm().clamp_min(1.0): return x
        m = min(restart, maxit - it)
        V, Z = [], []
        H = torch.zeros((m+1, m), device=device, dtype=dtype)
        v1 = r / beta
        V.append(v1)
        happy = False
        for j in range(m):
            z_j = M_apply(V[j]); Z.append(z_j)
            w = Av(z_j)                               # one VJP inside Av
            for i in range(j+1):
                hij = torch.dot(V[i], w)
                H[i, j] = hij
                w = w - hij * V[i]
            H[j+1, j] = w.norm()
            if H[j+1, j] < 1e-14:                     # happy breakdown
                happy = True; m = j + 1; break
            V.append(w / H[j+1, j])
        e1 = torch.zeros(m+1, device=device, dtype=dtype); e1[0] = 1.0
        y = torch.linalg.lstsq(H[:m+1, :m], beta * e1[:m+1]).solution
        x = x + sum(y[i] * Z[i] for i in range(m))    # right-preconditioned update
        it += m
        if happy: return x
    return x

@torch.no_grad()
def _solve_adjoint_krylov(vjp_apply: Callable[[torch.Tensor], torch.Tensor],
                          rhs: torch.Tensor, *,
                          tol: float = 1e-6, restart: int = 40, maxit: int = 200,
                          hutch_probes: int = 8, warm_start: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Solve (I - J^T) x = rhs with FGMRES + Jacobi; only needs v -> J^T v."""
    device, dtype = rhs.device, rhs.dtype
    def Av(x): return x - vjp_apply(x)                # A = I - J^T
    d = _estimate_diag_hutch(Av, rhs.numel(), k=hutch_probes, device=device, dtype=dtype)
    M_apply = lambda v: v / d
    return _fgmres(Av, rhs, M_apply=M_apply, tol=tol, restart=restart, maxit=maxit, x0=warm_start)

## GMRES ##

@dataclass
class StepRecord:
    iteration: int
    theta: torch.Tensor
    rates: torch.Tensor
    negative_log_likelihood: float
    log_likelihood: float
    theta_step_inf: float
    grad_infinity_norm: float


def _flatten_state(Pi: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    return torch.cat([Pi.reshape(-1), E.reshape(-1)])


def _build_unflatten(Pi_ref: torch.Tensor, E_ref: torch.Tensor):
    pi_numel = Pi_ref.numel()

    def unflatten(x_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        Pi_part = x_flat[:pi_numel].view_as(Pi_ref)
        E_part = x_flat[pi_numel:].view_as(E_ref)
        return Pi_part, E_part

    return unflatten


def _fixed_point_map(
    x_flat: torch.Tensor,
    theta: torch.Tensor,
    *,
    problem: ThetaOptimizationProblem,
    Pi_ref: torch.Tensor,
    E_ref: torch.Tensor,
) -> torch.Tensor:
    """H(x, theta) = flatten(F(Pi,E,theta), G(E,theta))."""
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


def _implicit_grad_vjp(
    problem: ThetaOptimizationProblem,
    theta: torch.Tensor,
    E_star: torch.Tensor,
    Pi_star: torch.Tensor,
    *,
    max_iter: int = 500,
    tol: float = 1e-10,
    damping: float = 1.0,
    lam=None,
    use_gmres=False,
) -> torch.Tensor:
    """
    Compute dL/dtheta at the fixed point using only VJP (autograd.functional.vjp).

    Solves (I - J_x H)^T lambda = dL/dx by Neumann iteration:
      lambda_{k+1} = b + damping * (J_x H)^T lambda_k
    and then grad_theta = (J_theta H)^T lambda.
    """
    # Make differentiable state copies to build graph
    Pi_var = Pi_star.detach().clone().requires_grad_(True)
    E_var = E_star.detach().clone().requires_grad_(True)

    flat_state = _flatten_state(Pi_var, E_var)
    import time
    # Pullbacks using autograd.functional.vjp (you supply v each time)
    def apply_JxT(v: torch.Tensor) -> torch.Tensor:
        v = v.to(device=flat_state.device, dtype=flat_state.dtype)
        _, JxT_v = vjp(
            lambda s: _fixed_point_map(s, theta, problem=problem, Pi_ref=Pi_var, E_ref=E_var),
            flat_state,
            v=v,
            create_graph=False,
        )
        return JxT_v

    def apply_JthetaT(v: torch.Tensor) -> torch.Tensor:
        v = v.to(device=theta.device, dtype=theta.dtype)
        _, JthetaT_v = vjp(
            lambda th: _fixed_point_map(flat_state, th, problem=problem, Pi_ref=Pi_var, E_ref=E_var),
            theta,
            v=v,
            create_graph=False,
        )
        return JthetaT_v

    # RHS b = ∂L/∂x (L depends only on Pi in our readout)
    L = compute_log_likelihood(Pi_var, problem.root_clade_id)
    dL_dPi, dL_dE = torch.autograd.grad(L, (Pi_var, E_var), retain_graph=False, allow_unused=True)
    if dL_dE is None:
        dL_dE = torch.zeros_like(E_var)
    b = _flatten_state(dL_dPi, dL_dE)


    # FGMRES for (I - Jx^T) lambda = b  (one VJP per Krylov step; Jacobi adds 0 per-step VJPs)
    if use_gmres:
        lam = _solve_adjoint_krylov(
            vjp_apply=lambda v: apply_JxT(v),      # Jx^T v
            rhs=b,
            tol=tol,
            restart=40,
            maxit=max_iter,                        # reuse your vjp_max_iter
            hutch_probes=8,
            warm_start=lam                         # reuse previous λ as x0
        )
    else:
        # Neumann iteration for (I - Jx^T) lambda = b
        if lam is None:
            lam = torch.zeros_like(b)
        for it in range(max_iter):
            lam_next = b + damping * apply_JxT(lam)
            if torch.norm((lam_next - lam)/(lam+1e-10)) <= tol:
                lam = lam_next
                print(f"stopped neumann at {it}")
                break
            lam = lam_next
    grad_theta = apply_JthetaT(lam)
    return grad_theta, lam


def optimize_theta(
    species_tree_path: str,
    gene_tree_path: str,
    theta_init: torch.Tensor,
    *,
    steps: int = 200,
    lr: float = 0.2,
    adam_eps: float = 1e-8,
    tol_theta: float = 1e-3,
    e_max_iters: int = 2000,
    pi_max_iters: int = 2000,
    e_tol: float = 1e-12,
    pi_tol: float = 1e-12,
    damping: float = 1.0,
    vjp_max_iter: int = 500,
    vjp_tol: float = 1e-6,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
    use_gmres: bool = False,
    # --- NEW: pseudo-curvature options ---
    use_pseudocurv: bool = False,
    pseudocurv_max_hist: int = 6,
    pseudocurv_ridge: float = 1e-6,
    use_theta_pseudocurv: bool = False,         # NEW
    theta_mode: str = "lbfgs",                  # "lbfgs" or "bb2"
) -> Dict[str, object]:
    """
    Simple Adam loop using implicit gradient dL/dtheta (VJP-only).

    Args:
      ...
      use_gmres: if True, use FGMRES for adjoint; else Neumann.
      use_pseudocurv: if True, predict λ from past (Δθ, Δλ) before adjoint solve.
      pseudocurv_max_hist: number of (θ, λ) pairs to keep for the predictor.
      pseudocurv_ridge: ridge term for the small multi-secant fit.

    Returns:
      dict with final theta, rates, log_likelihood, negative_log_likelihood, and history.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    problem = ThetaOptimizationProblem(species_tree_path, gene_tree_path, device=device, dtype=dtype)

    theta = torch.nn.Parameter(theta_init.to(device=device, dtype=dtype).clone())
    # opt = torch.optim.Adam([theta], lr=lr, eps=adam_eps, betas=(0.7,0.9))
    opt = torch.optim.SGD([theta], lr=lr)
    # --- NEW: pseudo-curvature memory (optional) ---
    mem = _PseudoCurvMem(max_hist=pseudocurv_max_hist, ridge=pseudocurv_ridge) if use_pseudocurv else None
    theta_qn = _ThetaQN(max_hist=10) if use_theta_pseudocurv else None

    history: List[StepRecord] = []
    prev_theta = theta.detach().clone()
    lam = None

    import time
    for it in range(1, steps + 1):
        # Converge fixed points (warm-start handled inside problem)
        t0 = time.time()
        E, Pi, _ = problem.fixed_points(
            theta,
            e_max_iters=e_max_iters,
            pi_max_iters=pi_max_iters,
            e_tol=e_tol,
            pi_tol=pi_tol,
            warm_start=(it > 1),
        )
        t1 = time.time()
        print(f"fixed point time {t1 - t0:.3f}s")

        # Likelihood at fixed point
        L = compute_log_likelihood(Pi, problem.root_clade_id)
        print(f"{L=}")
        nll = -L

        # --- NEW: predict λ from past (Δθ, Δλ) before adjoint solve ---
        if use_pseudocurv and mem is not None:
            lam_pred = mem.predict_lambda(theta.detach())
            if lam_pred is not None:
                # adopt predicted shape if first time; else ensure same shape
                if lam is None or lam.numel() != lam_pred.numel():
                    lam = lam_pred.clone()
                else:
                    lam.copy_(lam_pred.view_as(lam))

        # Implicit gradient via VJP-only adjoint (Neumann or GMRES), warm-start with lam
        # grad_theta, lam = _implicit_grad_vjp(
        #     problem,
        #     theta,
        #     E_star=E,
        #     Pi_star=Pi,
        #     max_iter=vjp_max_iter,
        #     tol=vjp_tol,
        #     damping=damping,
        #     lam=lam,
        #     use_gmres=use_gmres,
        # )
        # t2 = time.time()
        # print(f"grad time {t2 - t1:.3f}s")

        # # Adam step in ascent on L (i.e., descent on nLL): set param.grad = -grad_theta
        # opt.zero_grad(set_to_none=True)
        # theta.grad = -grad_theta.detach().clone()
        # opt.step()
        # Implicit gradient of L; convert to gradient of nLL
        grad_theta, lam = _implicit_grad_vjp(
            problem,
            theta,
            E_star=E,
            Pi_star=Pi,
            max_iter=vjp_max_iter,
            tol=vjp_tol,
            damping=damping,
            lam=lam,
            use_gmres=use_gmres,
        )
        g_nll = (-grad_theta).detach()                 # ∇ nLL

        if use_theta_pseudocurv and theta_qn is not None:
            # curvature-informed direction / step for theta (no extra evals)
            if theta_mode.lower() == "lbfgs":
                p = theta_qn.lbfgs_direction(g_nll.reshape(-1)).view_as(theta)
                # single explicit update (no torch optimizer), clamp step∞ for safety
                step_inf_cap = 0.2
                t = lr
                step_inf = (t * p).abs().max().item()
                if step_inf > step_inf_cap:
                    t *= step_inf_cap / (step_inf + 1e-12)
                with torch.no_grad():
                    theta.data.add_(-t, g_nll.new_zeros(1))  # no-op to keep grad graph clean
                    theta.data.add_(t, p)                    # θ ← θ + t p
            else:  # "bb2"
                lr_eff = theta_qn.bb2_scale(theta.detach(), g_nll.detach(), lr)
                with torch.no_grad():
                    theta.data.add_(-lr_eff, g_nll)          # θ ← θ - lr_eff ∇nLL
        else:
            # original SGD/Adam path (kept for parity)
            opt.zero_grad(set_to_none=True)
            theta.grad = g_nll.clone()
            opt.step()

        # update θ-curvature memory AFTER applying the step
        if use_theta_pseudocurv and theta_qn is not None:
            theta_qn.update(theta.detach().clone(), g_nll.detach().clone())

        # --- NEW: update pseudo-curvature memory AFTER step ---
        if use_pseudocurv and mem is not None and lam is not None:
            mem.update(theta.detach().clone(), lam.detach().clone())

        # Bookkeeping
        theta_detached = theta.detach()
        diff = torch.max(torch.abs(theta_detached - prev_theta)).item()
        prev_theta = theta_detached.clone()
        rates = torch.exp(theta_detached)
        grad_inf = float(grad_theta.abs().max().item())

        history.append(
            StepRecord(
                iteration=it,
                theta=theta_detached.cpu(),
                rates=rates.cpu(),
                negative_log_likelihood=float(nll.item()),
                log_likelihood=float(L.item()),
                theta_step_inf=diff,
                grad_infinity_norm=grad_inf,
            )
        )

        if diff < tol_theta:
            break

    # Recompute likelihood at final theta
    E_final, Pi_final, _ = problem.fixed_points(
        theta,
        e_max_iters=e_max_iters,
        pi_max_iters=pi_max_iters,
        e_tol=e_tol,
        pi_tol=pi_tol,
        warm_start=True,
    )
    L_final = compute_log_likelihood(Pi_final, problem.root_clade_id)

    return {
        "theta": theta.detach().cpu(),
        "rates": torch.exp(theta.detach()).cpu(),
        "log_likelihood": float(L_final.item()),
        "negative_log_likelihood": float((-L_final).item()),
        "history": history,
    }
