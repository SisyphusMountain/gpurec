"""Wave-based optimizer (uses wave forward + wave backward)."""
from __future__ import annotations

import math
from typing import List, Optional

import torch

from src.core.likelihood import (
    E_fixed_point,
    Pi_wave_forward,
    compute_log_likelihood,
)
from src.core.extract_parameters import extract_parameters, extract_parameters_uniform

from .types import FixedPointInfo, LinearSolveStats, StepRecord
from .implicit_grad import implicit_grad_loglik_vjp_wave


def optimize_theta_wave(
    wave_layout,
    species_helpers,
    root_clade_ids,
    unnorm_row_max,
    theta_init,
    *,
    transfer_mat_unnormalized=None,
    steps: int = 200,
    lr: float = 0.2,
    tol_theta: float = 1e-3,
    e_max_iters: int = 2000,
    e_tol: float = 1e-8,
    neumann_terms: int = 4,
    use_pruning: bool = True,
    pruning_threshold: float = 1e-6,
    cg_tol: float = 1e-8,
    cg_maxiter: int = 500,
    gmres_restart: int = 40,
    specieswise: bool = False,
    device=None,
    dtype=torch.float64,
    pibar_mode: str = 'uniform',
    optimizer: str = 'adam',
    momentum: float = 0.9,
):
    """Optimize theta using wave forward/backward + implicit gradient.

    Parameters
    ----------
    wave_layout : dict
        From build_wave_layout().
    species_helpers : dict
        Species tree helpers.
    root_clade_ids : Tensor
        Root clade IDs (original ordering) for likelihood computation.
    unnorm_row_max : Tensor [S]
        Row maxima of unnormalized transfer matrix.
    theta_init : Tensor [3] or [S, 3]
        Initial rate parameters (log-space).
    steps : int
        Maximum optimization iterations (ignored for 'lbfgs').
    lr : float
        Learning rate (ignored for 'lbfgs').
    optimizer : str
        'adam', 'sgd', or 'lbfgs'.
    momentum : float
        Momentum for SGD (default 0.9, ignored for adam/lbfgs).

    Returns
    -------
    dict with 'theta', 'rates', 'log_likelihood', 'negative_log_likelihood', 'history'.
    """
    if device is None:
        device = theta_init.device

    _THETA_MIN = math.log2(1e-10)

    # Precompute ancestors_T for uniform mode
    _ancestors_T = None
    if pibar_mode == 'uniform':
        anc_dense = species_helpers['ancestors_dense'].to(device=device, dtype=dtype)
        _ancestors_T = anc_dense.T.to_sparse_coo()

    def _extract(theta_d):
        if pibar_mode in ('dense', 'topk') and transfer_mat_unnormalized is not None:
            log_pS, log_pD, log_pL, transfer_mat, mt_raw = extract_parameters(
                theta_d, transfer_mat_unnormalized,
                genewise=False, specieswise=specieswise, pairwise=False,
            )
            mt = mt_raw.squeeze(-1) if mt_raw.ndim == 2 else mt_raw
        else:
            log_pS, log_pD, log_pL, transfer_mat, mt = extract_parameters_uniform(
                theta_d, unnorm_row_max, specieswise=specieswise,
            )
        return log_pS, log_pD, log_pL, transfer_mat, mt

    def _forward_backward(theta_d, warm_E):
        """Full forward + backward: returns (nll, grad_theta, history_record, warm_E)."""
        with torch.no_grad():
            log_pS, log_pD, log_pL, transfer_mat, mt = _extract(theta_d)

            E_out = E_fixed_point(
                species_helpers=species_helpers,
                log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
                transfer_mat=transfer_mat, max_transfer_mat=mt,
                max_iters=e_max_iters, tolerance=e_tol,
                warm_start_E=warm_E,
                dtype=dtype, device=device, pibar_mode=pibar_mode,
                ancestors_T=_ancestors_T,
            )

            Pi_out = Pi_wave_forward(
                wave_layout=wave_layout, species_helpers=species_helpers,
                E=E_out['E'], Ebar=E_out['E_bar'],
                E_s1=E_out['E_s1'], E_s2=E_out['E_s2'],
                log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
                transfer_mat=transfer_mat, max_transfer_mat=mt,
                device=device, dtype=dtype, pibar_mode=pibar_mode,
            )

            logL = compute_log_likelihood(Pi_out['Pi'], E_out['E'], root_clade_ids)
            nll = float(logL.sum().item())

        grad_theta, statsG = implicit_grad_loglik_vjp_wave(
            wave_layout, species_helpers,
            Pi_star_wave=Pi_out['Pi_wave_ordered'],
            Pibar_star_wave=Pi_out['Pibar_wave_ordered'],
            E_star=E_out['E'], E_s1=E_out['E_s1'],
            E_s2=E_out['E_s2'], Ebar=E_out['E_bar'],
            log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
            max_transfer_mat=mt,
            root_clade_ids_perm=wave_layout['root_clade_ids'],
            theta=theta_d,
            unnorm_row_max=unnorm_row_max,
            specieswise=specieswise,
            device=device, dtype=dtype,
            neumann_terms=neumann_terms,
            use_pruning=use_pruning,
            pruning_threshold=pruning_threshold,
            cg_tol=cg_tol, cg_maxiter=cg_maxiter,
            gmres_restart=gmres_restart,
            pibar_mode=pibar_mode,
            transfer_mat=transfer_mat,
            transfer_mat_unnormalized=transfer_mat_unnormalized,
            ancestors_T=_ancestors_T,
        )

        return nll, grad_theta, statsG, E_out

    # --- L-BFGS-B path (scipy) ---
    if optimizer == 'lbfgs':
        import numpy as np
        from scipy.optimize import minimize as scipy_minimize

        theta_t = theta_init.to(device=device, dtype=dtype).clone()
        theta_shape = theta_t.shape
        history: List[StepRecord] = []
        warm_E_ref = [None]
        eval_count = [0]

        def forward_and_grad(theta_flat_np):
            theta_flat = torch.from_numpy(theta_flat_np).to(device=device, dtype=dtype)
            theta_d = theta_flat.reshape(theta_shape).clamp(min=_THETA_MIN)

            nll, grad_theta, statsG, E_out = _forward_backward(theta_d, warm_E_ref[0])
            warm_E_ref[0] = E_out['E'].detach()

            eval_count[0] += 1
            fp_info = FixedPointInfo(iterations_E=int(E_out['iterations']),
                                     iterations_Pi=0)
            history.append(StepRecord(
                iteration=eval_count[0], theta=theta_d.detach().cpu(),
                rates=torch.exp2(theta_d.detach()).cpu(),
                negative_log_likelihood=nll, log_likelihood=-nll,
                theta_step_inf=0.0,
                grad_infinity_norm=float(grad_theta.abs().max().item()),
                fp_info=fp_info, gradient=grad_theta.cpu(),
                solve_stats_F=LinearSolveStats("wave_neumann", neumann_terms, 0.0, False),
                solve_stats_G=statsG,
            ))

            grad_np = grad_theta.reshape(-1).cpu().to(torch.float64).numpy()
            np.nan_to_num(grad_np, copy=False, nan=0.0)
            return float(nll), grad_np

        x0 = theta_t.reshape(-1).cpu().to(torch.float64).numpy()
        bounds = [(_THETA_MIN, None)] * len(x0)
        result = scipy_minimize(
            forward_and_grad, x0, method='L-BFGS-B', jac=True,
            bounds=bounds,
            options={'maxiter': steps, 'ftol': 1e-12, 'gtol': 1e-6},
        )

        theta_final = torch.from_numpy(result.x).to(device=device, dtype=dtype).reshape(theta_shape)
        return {
            "theta": theta_final.cpu(),
            "rates": torch.exp2(theta_final).cpu(),
            "negative_log_likelihood": result.fun,
            "log_likelihood": -result.fun,
            "history": history,
            "scipy_result": result,
        }

    # --- Iterative optimizers (adam, sgd) ---
    theta = torch.nn.Parameter(theta_init.to(device=device, dtype=dtype).clone())
    if optimizer == 'sgd':
        opt = torch.optim.SGD([theta], lr=lr, momentum=momentum)
    else:
        opt = torch.optim.Adam([theta], lr=lr)

    history: List[StepRecord] = []
    prev_theta = theta.detach().clone()
    warm_E = None

    for it in range(1, steps + 1):
        theta_d = theta.detach()

        nll, grad_theta, statsG, E_out = _forward_backward(theta_d, warm_E)
        warm_E = E_out['E'].detach()
        iters_E = int(E_out['iterations'])

        # Optimizer step (minimize NLL)
        opt.zero_grad(set_to_none=True)
        grad_clean = grad_theta.clone()
        grad_clean.nan_to_num_(nan=0.0)
        theta.grad = grad_clean
        opt.step()

        with torch.no_grad():
            theta.clamp_(min=_THETA_MIN)

        # Bookkeeping
        theta_detached = theta.detach()
        diff = float(torch.max(torch.abs(theta_detached - prev_theta)).item())
        prev_theta = theta_detached.clone()
        rates = torch.exp2(theta_detached)
        grad_inf = float(grad_theta.abs().max().item())

        fp_info = FixedPointInfo(iterations_E=iters_E, iterations_Pi=0)
        statsF_dummy = LinearSolveStats("wave_neumann", neumann_terms, 0.0, False)

        history.append(
            StepRecord(
                iteration=it,
                theta=theta_detached.cpu(),
                rates=rates.cpu(),
                negative_log_likelihood=nll,
                log_likelihood=-nll,
                theta_step_inf=diff,
                grad_infinity_norm=grad_inf,
                fp_info=fp_info,
                gradient=grad_theta.cpu(),
                solve_stats_F=statsF_dummy,
                solve_stats_G=statsG,
            )
        )

        if diff < tol_theta and it > 1:
            break

    return {
        "theta": theta.detach().cpu(),
        "rates": torch.exp2(theta.detach()).cpu(),
        "negative_log_likelihood": history[-1].negative_log_likelihood if history else float('nan'),
        "log_likelihood": history[-1].log_likelihood if history else float('nan'),
        "history": history,
    }
