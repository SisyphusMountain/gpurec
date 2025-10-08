# -*- coding: utf-8 -*-
"""
Implicit-gradient optimisation of reconciliation parameters (θ).

Key changes vs. your previous version:
- Fixed points (E*, Π*) are computed under torch.no_grad() -> small memory.
- Gradients use implicit differentiation with ONLY VJP calls (backward):
    (I - F_Π^T) v = φ,  φ = ∂ logL / ∂Π
    (I - G_E^T) w = q,  q = (∂F/∂E)^T v
    ∇θ logL = (∂F/∂θ)^T v + (∂G/∂θ)^T w
- Linear solves default to CG; detect non-SPD and fall back to GMRES.

You can increase Adam's LR safely to take bigger θ-steps without
backprop-through-FP memory blowups.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable

import math
import torch
from torch import Tensor
from torch import func as tfunc

# ---------------------------- dataset helpers ----------------------------

from src.core.ccp import (
    build_ccp_from_single_tree,
    build_ccp_helpers,
    build_clade_species_mapping,
    get_root_clade_id,
)

from src.core.tree_helpers import build_species_helpers
from src.core.likelihood import (
    E_fixed_point,
    Pi_fixed_point,
    compute_log_likelihood,
    Pi_step,
    E_step,
)

# -------------------------------------------------------------------------
# Dataclasses for logging
# -------------------------------------------------------------------------

@dataclass
class FixedPointInfo:
    iterations_E: int
    iterations_Pi: int

@dataclass
class LinearSolveStats:
    method: str
    iters: int
    rel_residual: float
    fallback_used: bool

@dataclass
class StepRecord:
    iteration: int
    theta: torch.Tensor
    rates: torch.Tensor
    negative_log_likelihood: float
    log_likelihood: float
    theta_step_inf: float
    grad_infinity_norm: float
    fp_info: FixedPointInfo
    gradient: torch.Tensor
    solve_stats_F: LinearSolveStats
    solve_stats_G: LinearSolveStats


# -------------------------------------------------------------------------
# Utilities: matrix-free Krylov solvers (CG + GMRES fallback)
# -------------------------------------------------------------------------

@torch.no_grad()
def _cg(
    Av: Callable[[Tensor], Tensor],
    b: Tensor,
    *,
    M: Optional[Callable[[Tensor], Tensor]] = None,
    tol: float = 1e-8,
    maxiter: int = 500,
    x0: Optional[Tensor] = None,
) -> Tuple[Tensor, LinearSolveStats, bool]:
    """
    Conjugate Gradients for SPD A. Uses only Av (no A^T).
    Returns (x, stats, success). If breakdown detected, success=False.
    """
    device, dtype = b.device, b.dtype
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - Av(x)
    z = r if M is None else M(r)
    p = z.clone()
    rz_old = float(torch.dot(r, z))
    bnorm = float(b.norm()) if b.numel() > 0 else 1.0
    bnorm = max(bnorm, 1.0)

    success = True
    iters = 0
    for k in range(1, maxiter + 1):
        Ap = Av(p)
        pAp = float(torch.dot(p, Ap))
        if pAp <= 0.0 or not math.isfinite(pAp):  # non-SPD or numerical breakdown
            success = False
            iters = k - 1
            break
        alpha = rz_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rel_res = float(r.norm()) / bnorm
        if rel_res <= tol:
            iters = k
            return x, LinearSolveStats("CG", iters, rel_res, fallback_used=False), True
        z = r if M is None else M(r)
        rz_new = float(torch.dot(r, z))
        beta = rz_new / max(rz_old, 1e-30)
        p = z + beta * p
        rz_old = rz_new
        iters = k

    rel_res = float(r.norm()) / bnorm
    return x, LinearSolveStats("CG", iters, rel_res, fallback_used=False), success


@torch.no_grad()
def _gmres(
    Av: Callable[[Tensor], Tensor],
    b: Tensor,
    *,
    M: Optional[Callable[[Tensor], Tensor]] = None,  # right preconditioner
    tol: float = 1e-8,
    restart: int = 40,
    maxiter: int = 500,
    x0: Optional[Tensor] = None,
) -> Tuple[Tensor, LinearSolveStats]:
    """
    Restarted GMRES with right preconditioning. Uses only Av.
    """
    device, dtype = b.device, b.dtype
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    Id = (lambda v: v)
    M = Id if M is None else M

    iters = 0
    bnorm = float(b.norm())
    bnorm = max(bnorm, 1.0)

    while iters < maxiter:
        r = b - Av(x)
        beta = float(r.norm())
        if beta / bnorm <= tol:
            return x, LinearSolveStats("GMRES", iters, beta / bnorm, fallback_used=False)

        m = min(restart, maxiter - iters)
        V: List[Tensor] = []
        Z: List[Tensor] = []
        H = torch.zeros((m + 1, m), device=device, dtype=dtype)

        v1 = r / (beta + 1e-30)
        V.append(v1)
        happy = False
        used = 0

        for j in range(m):
            z_j = M(V[j]); Z.append(z_j)
            w = Av(z_j)
            # Modified Gram-Schmidt
            for i in range(j + 1):
                H[i, j] = torch.dot(V[i], w)
                w = w - H[i, j] * V[i]
            H[j + 1, j] = w.norm()
            if float(H[j + 1, j]) < 1e-14:
                happy = True
                used = j + 1
                break
            V.append(w / (H[j + 1, j] + 1e-30))
            used = j + 1

        e1 = torch.zeros(used + 1, device=device, dtype=dtype); e1[0] = 1.0
        y = torch.linalg.lstsq(H[:used + 1, :used], beta * e1[:used + 1]).solution
        x = x + sum(y[i] * Z[i] for i in range(used))
        iters += used
        if happy:
            r = b - Av(x)
            rel = float(r.norm()) / bnorm
            return x, LinearSolveStats("GMRES", iters, rel, fallback_used=False)

    r = b - Av(x)
    rel = float(r.norm()) / bnorm
    return x, LinearSolveStats("GMRES", iters, rel, fallback_used=False)


@torch.no_grad()
def _estimate_diag_hutchinson(Av: Callable[[torch.Tensor], torch.Tensor],
                              n: int, *, k: int = 6,
                              device=None, dtype=None, eps: float = 1e-8) -> torch.Tensor:
    """Matrix-free estimate of diag(A) for a FLAT operator Av: ℝ^n→ℝ^n."""
    acc = torch.zeros(n, device=device, dtype=dtype)
    for _ in range(k):
        z = torch.empty(n, device=device, dtype=dtype).bernoulli_(0.5).mul_(2).add_(-1)  # ±1
        acc += z * Av(z)
    d = (acc / float(k)).abs().clamp_min(eps)
    med = d.median()
    if float(med) > 0:
        d = d / med
    return d



# -------------------------------------------------------------------------
# Implicit gradient: build VJP closures & solve the two transpose systems
# -------------------------------------------------------------------------

def _build_maps(problem, Pi_star: torch.Tensor, E_star: torch.Tensor, theta: torch.Tensor):
    """Unchanged signature; returns callables F and G."""
    species = problem.species_helpers
    ccp = problem.ccp_helpers
    clade_species_map = problem.clade_species_map

    Recipients_mat = species['Recipients_mat']
    sp_P = species['s_P_indexes']
    sp_C12 = species['s_C12_indexes']
    s_P_idx = species['s_P_indexes']

    log_2 = float(math.log(2.0))

    def Gfun(E: torch.Tensor, th: torch.Tensor) -> torch.Tensor:
        return E_step(E, sp_P, sp_C12, Recipients_mat, th)

    def Ffun(Pi: torch.Tensor, E: torch.Tensor, th: torch.Tensor) -> torch.Tensor:
        _, E_s1, E_s2, Ebar = E_step(E, s_P_idx, sp_C12, Recipients_mat, th, return_components=True)
        return Pi_step(Pi, ccp, species, clade_species_map, E, Ebar, E_s1, E_s2, th, log_2)

    return Ffun, Gfun



@torch.no_grad()
def implicit_grad_loglik_vjp_cg(
    problem,
    *,
    Pi_star: torch.Tensor,
    E_star: torch.Tensor,
    theta: torch.Tensor,
    cg_tol: float = 1e-8,
    cg_maxiter: int = 500,
    use_jacobi_prec: bool = True,
    gmres_restart: int = 40,
):
    """
    Return (∇_θ logL, statsF, statsG) using only VJPs.
    NOTE: Krylov vectors are FLAT; VJP inputs are reshaped on the fly.
    """
    device, dtype = theta.device, theta.dtype
    Ffun, Gfun = _build_maps(problem, Pi_star, E_star, theta)

    # Build VJP closures at the fixed point
    with torch.enable_grad():
        Pi_req = Pi_star.detach().requires_grad_(True)
        E_req  = E_star.detach().requires_grad_(True)
        th_req = theta.detach().requires_grad_(True)

        # VJPs
        _, vjpF = tfunc.vjp(Ffun, Pi_req, E_req, th_req)   # returns gPi, gE, gθ
        _, vjpG = tfunc.vjp(Gfun, E_req, th_req)           # returns gE, gθ

        # φ = ∂ logL / ∂Π (same shape as Π)
        loss = compute_log_likelihood(Pi_req, problem.root_clade_id)
        (phi_T,) = torch.autograd.grad(loss, Pi_req)

    # Shapes and sizes
    Pi_shape = Pi_star.shape
    E_shape  = E_star.shape
    nPi = Pi_star.numel()
    nE  = E_star.numel()

    # Flattened φ for Krylov
    phi = phi_T.reshape(-1)

    # Define FLAT operators A_F and A_G for Krylov
    def AF_flat(v_flat: torch.Tensor) -> torch.Tensor:
        vPi = v_flat.view(Pi_shape).contiguous()
        gPi, *_ = vjpF(vPi.clone())           # shape = Pi_shape
        return (vPi - gPi).reshape(-1)

    def AG_flat(w_flat: torch.Tensor) -> torch.Tensor:
        wE = w_flat.view(E_shape).contiguous()
        gE, *_ = vjpG(wE.clone())             # shape = E_shape
        return (wE - gE).reshape(-1)

    # Preconditioners (diagonal, flat)
    MF = MG = None
    if use_jacobi_prec:
        dF = _estimate_diag_hutchinson(AF_flat, nPi, k=6, device=device, dtype=dtype)
        MF = (lambda v: v / dF)

    # ---- Solve (I - F_Π^T) v = φ (flat) ----
    v_flat, statsF, okF = _cg(AF_flat, phi, M=MF, tol=cg_tol, maxiter=cg_maxiter)
    if not okF:
        v_flat, statsF = _gmres(AF_flat, phi, tol=cg_tol, restart=gmres_restart, maxiter=cg_maxiter)
        statsF.fallback_used = True  # type: ignore

    vPi = v_flat.view(Pi_shape)
    # q = (∂F/∂E)^T v  and gθ_F
    _, q_E, gθ_F = vjpF(vPi)
    q_flat = q_E.reshape(-1)

    if use_jacobi_prec:
        dG = _estimate_diag_hutchinson(AG_flat, nE, k=6, device=device, dtype=dtype)
        MG = (lambda v: v / dG)

    # ---- Solve (I - G_E^T) w = q (flat) ----
    w_flat, statsG, okG = _cg(AG_flat, q_flat, M=MG, tol=cg_tol, maxiter=cg_maxiter)
    if not okG:
        w_flat, statsG = _gmres(AG_flat, q_flat, tol=cg_tol, restart=gmres_restart, maxiter=cg_maxiter)
        statsG.fallback_used = True  # type: ignore

    wE = w_flat.view(E_shape)
    # gθ_G
    _, gθ_G = vjpG(wE)

    grad_theta = (gθ_F + gθ_G).detach()    # ∇θ logL
    return grad_theta, statsF, statsG



# -------------------------------------------------------------------------
# Problem wrapper (unchanged public API; fixed points under no_grad)
# -------------------------------------------------------------------------

class ThetaOptimizationProblem:
    """Wrap preprocessing and warm-starts for θ optimisation (implicit-grad ready)."""

    def __init__(
        self,
        species_tree_path: str,
        gene_tree_path: str,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.dtype = dtype

        self.ccp = build_ccp_from_single_tree(gene_tree_path)
        self.species_helpers = build_species_helpers(species_tree_path, device, dtype)
        self.ccp_helpers = build_ccp_helpers(self.ccp, device, dtype)

        clade_species = build_clade_species_mapping(self.ccp, self.species_helpers, device, dtype)
        self.clade_species_map = torch.log(clade_species)
        self.root_clade_id = get_root_clade_id(self.ccp)

        self._warm_E: Optional[Tensor] = None
        self._warm_Pi: Optional[Tensor] = None

    # --------------------------- fixed-point solves ---------------------------

    @torch.no_grad()
    def fixed_points(
        self,
        theta: Tensor,
        *,
        e_max_iters: int = 200,
        pi_max_iters: int = 200,
        e_tol: float = 1e-12,
        pi_tol: float = 1e-12,
        warm_start: bool = True,
    ) -> tuple[Tensor, Tensor, FixedPointInfo]:
        """Converge E* and Π* at θ (no graph; memory light)."""
        theta = theta.to(device=self.device, dtype=self.dtype)

        warm_E = self._warm_E if warm_start else None
        E_out = E_fixed_point(
            species_helpers=self.species_helpers,
            theta=theta,
            max_iters=e_max_iters,
            tolerance=e_tol,
            return_components=True,
            warm_start_E=warm_E,
            dtype=self.dtype,
            device=self.device,
        )
        E = E_out["E"]
        E_s1 = E_out["E_s1"]
        E_s2 = E_out["E_s2"]
        E_bar = E_out["E_bar"]

        warm_Pi = self._warm_Pi if warm_start else None
        Pi_out = Pi_fixed_point(
            ccp_helpers=self.ccp_helpers,
            species_helpers=self.species_helpers,
            clade_species_map=self.clade_species_map,
            E=E,
            Ebar=E_bar,
            E_s1=E_s1,
            E_s2=E_s2,
            theta=theta,
            max_iters=pi_max_iters,
            tolerance=pi_tol,
            warm_start_Pi=warm_Pi,
        )
        Pi = Pi_out["Pi"]

        # cache detaches as next warm start
        self._warm_E = E.detach()
        self._warm_Pi = Pi.detach()

        info = FixedPointInfo(
            iterations_E=int(E_out["iterations"]),
            iterations_Pi=int(Pi_out["iterations"]),
        )
        return E, Pi, info

    @torch.no_grad()
    def get_fixed_points(self) -> Tuple[Tensor, Tensor]:
        if self._warm_Pi is None or self._warm_E is None:
            raise RuntimeError("Call fixed_points() once before get_fixed_points().")
        return self._warm_Pi.detach(), self._warm_E.detach()

    @torch.no_grad()
    def set_initial_guess(self, *, Pi0: Optional[Tensor] = None, E0: Optional[Tensor] = None) -> None:
        if Pi0 is not None:
            self._warm_Pi = Pi0.detach()
        if E0 is not None:
            self._warm_E = E0.detach()

    @torch.no_grad()
    def one_step_predictor(self, theta_trial: Tensor) -> Tuple[Tensor, Tensor, float]:
        """Single forward application of E_step then Pi_step at theta_trial (no graph)."""
        if self._warm_Pi is None or self._warm_E is None:
            raise RuntimeError("Warm states empty; call fixed_points() first.")
        theta_trial = theta_trial.to(device=self.device, dtype=self.dtype)

        species = self.species_helpers
        ccp = self.ccp_helpers
        clade_species_map = self.clade_species_map

        Recipients_mat = species['Recipients_mat']
        sp_P   = species['s_P_indexes']
        sp_C12 = species['s_C12_indexes']

        E1, E_s1, E_s2, Ebar = E_step(self._warm_E, sp_P, sp_C12, Recipients_mat,
                                      theta_trial, return_components=True)
        Pi1 = Pi_step(self._warm_Pi, ccp, species, clade_species_map,
                      E1, Ebar, E_s1, E_s2, theta_trial, log_2=math.log(2.0))

        res_inf = max(float((E1 - self._warm_E).abs().max().item()),
                      float((Pi1 - self._warm_Pi).abs().max().item()))
        return Pi1, E1, res_inf

    # ------------------------------ likelihood ------------------------------

    @torch.no_grad()
    def log_likelihood(self, Pi: Tensor) -> Tensor:
        return compute_log_likelihood(Pi, self.root_clade_id)


# -------------------------------------------------------------------------
# Optimiser using implicit gradient (Adam outer loop)
# -------------------------------------------------------------------------

def optimize_theta_implicit(
    problem: ThetaOptimizationProblem,
    theta_init: Tensor,
    *,
    steps: int = 200,
    lr: float = 0.2,
    tol_theta: float = 1e-3,
    # fixed-point limits/tols:
    e_max_iters: int = 200,
    pi_max_iters: int = 200,
    e_tol: float = 1e-12,
    pi_tol: float = 1e-12,
    # linear solve (implicit grad):
    cg_tol: float = 1e-8,
    cg_maxiter: int = 500,
    gmres_restart: int = 40,
    # predictor to take bigger θ steps with one solve/iter:
    prescreen: bool = True,

) -> Dict[str, object]:
    """
    Adam on θ with implicit gradients. Each iteration:
      1) (no_grad) converge fixed points at current θ (warm start)
      2) (no_grad) evaluate NLL = -log L
      3) implicit ∇θ log L via VJP+CG (fallback GMRES)
      4) set θ.grad = -∇θ log L and Adam step
    """
    device, dtype = problem.device, problem.dtype

    theta = torch.nn.Parameter(theta_init.to(device=device, dtype=dtype).clone())
    opt = torch.optim.Adam([theta], lr=lr)

    history: List[StepRecord] = []
    prev_theta = theta.detach().clone()

    # initial FP solve
    with torch.no_grad():
        E_star, Pi_star, fp_info = problem.fixed_points(
            theta.detach(), e_max_iters=e_max_iters, pi_max_iters=pi_max_iters,
            e_tol=e_tol, pi_tol=pi_tol, warm_start=False
        )
        logL = problem.log_likelihood(Pi_star)
        nll = -logL

    for it in range(1, steps + 1):
        # optional predictor to help take bigger steps:
        if prescreen:
            # do one Picard step at a tentative θ_try = θ (no change); we reuse it to check residual scaling
            Pi_pred, E_pred, _ = problem.one_step_predictor(theta.detach())
            problem.set_initial_guess(Pi0=Pi_pred, E0=E_pred)

        # 1) solve fixed point at current θ (one solve per iteration)
        with torch.no_grad():
            E_star, Pi_star, fp_info = problem.fixed_points(
                theta.detach(), e_max_iters=e_max_iters, pi_max_iters=pi_max_iters,
                e_tol=e_tol, pi_tol=pi_tol, warm_start=True
            )
            logL = problem.log_likelihood(Pi_star)
            nll = -logL

        # 2) implicit gradient of log-likelihood (NO FP graph needed)
        g_logL, statsF, statsG = implicit_grad_loglik_vjp_cg(
            problem,
            Pi_star=Pi_star,
            E_star=E_star,
            theta=theta.detach(),
            cg_tol=cg_tol,
            cg_maxiter=cg_maxiter,
            use_jacobi_prec=False,      # <- turn off for now
            gmres_restart=gmres_restart,
        )


        # 3) set gradient of NLL = -∇θ logL
        opt.zero_grad(set_to_none=True)
        theta.grad = (-g_logL).clone()
        opt.step()

        # bookkeeping
        theta_detached = theta.detach()
        diff = float(torch.max(torch.abs(theta_detached - prev_theta)).item())
        prev_theta = theta_detached.clone()
        rates = torch.exp(theta_detached)
        grad_inf = float(theta.grad.detach().abs().max().item())

        history.append(
            StepRecord(
                iteration=it,
                theta=theta_detached.cpu(),
                rates=rates.cpu(),
                negative_log_likelihood=float(nll.item()),
                log_likelihood=float(logL.item()),
                theta_step_inf=diff,
                grad_infinity_norm=grad_inf,
                fp_info=fp_info,
                gradient=(-g_logL).cpu(),
                solve_stats_F=statsF,
                solve_stats_G=statsG,
            )
        )

        if diff < tol_theta:
            break

    # final consistency
    with torch.no_grad():
        _, Pi_final, _ = problem.fixed_points(
            theta.detach(),
            e_max_iters=e_max_iters, pi_max_iters=pi_max_iters,
            e_tol=e_tol, pi_tol=pi_tol, warm_start=True,
        )
        final_log_lik = problem.log_likelihood(Pi_final)

    return {
        "theta": theta.detach().cpu(),
        "rates": torch.exp(theta.detach()).cpu(),
        "log_likelihood": float(final_log_lik.item()),
        "negative_log_likelihood": float((-final_log_lik).item()),
        "history": history,
    }


# -------------------------------------------------------------------------
# (Optional) Big-step variant: trust-region L-BFGS with implicit gradients
# -------------------------------------------------------------------------
# If you want to keep taking *very* large steps with one solve/iter, you can
# drop back in the trust-region L-BFGS outer loop from earlier replies. The
# only change is that wherever a gradient is needed, you call
# `implicit_grad_loglik_vjp_cg(...)` and set theta.grad accordingly.

# End of file
