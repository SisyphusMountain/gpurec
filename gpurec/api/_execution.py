import torch

from gpurec.api._batch_state import _BatchStatic
from gpurec.api._implicit_grad import implicit_grad_loglik_vjp_wave
from gpurec.core.inference.solver import nll_from_root_rows, solve_resident_e_pi


def theta_for_static(static: _BatchStatic, theta: torch.Tensor, *, genewise: bool) -> torch.Tensor:
    return theta.index_select(0, static.family_index_tensor) if genewise else theta


def evaluate_static_loss_grad(static: _BatchStatic, theta: torch.Tensor, *, need_grad: bool):
    with torch.no_grad():
        E, E_s1, E_s2, Ebar, root_rows, pi_wave, pibar_wave, pibar_row_max, log_pS, log_pD, log_pL, max_transfer_vec = solve_resident_e_pi(
            static, theta, warm_start_E=static.warm_E
        )
        loss = nll_from_root_rows(root_rows, E).detach()
        static.warm_E = E.detach()
        if not need_grad:
            return loss, None
        grad = implicit_grad_loglik_vjp_wave(
            static.wave_layout,
            static.species_helpers,
            Pi_star_wave=pi_wave,
            Pibar_star_wave=pibar_wave,
            E_star=E,
            Ebar=Ebar,
            E_s1=E_s1,
            E_s2=E_s2,
            log_pS=log_pS,
            log_pD=log_pD,
            log_pL=log_pL,
            max_transfer_mat=max_transfer_vec,
            theta=theta,
            family_idx=static.rate_family_idx,
            uniform_pibar_row_max=pibar_row_max,
            specieswise=static.specieswise,
            genewise=static.genewise,
            neumann_terms=static.solver_options.neumann_terms,
            bicgstab_max_iter=static.solver_options.bicgstab_max_iter,
            bicgstab_tol=static.solver_options.bicgstab_tol,
            bicgstab_breakdown_tol=static.solver_options.bicgstab_breakdown_tol,
            adjoint_pruning_threshold=static.solver_options.adjoint_pruning_threshold,
            use_adjoint_pruning=static.solver_options.use_adjoint_pruning,
            pibar_side_threshold=static.solver_options.pibar_side_threshold,
        ).detach()
    return loss, grad


def stream_batches(batch_statics: list[_BatchStatic], theta: torch.Tensor, *, genewise: bool, need_grad: bool):
    total = torch.zeros((), dtype=theta.dtype, device=theta.device)
    grad_total = torch.zeros_like(theta) if need_grad else None
    for static in batch_statics:
        theta_batch = theta_for_static(static, theta, genewise=genewise)
        loss_i, grad_i = evaluate_static_loss_grad(static, theta_batch, need_grad=need_grad)
        total = total + loss_i.to(device=theta.device, dtype=theta.dtype)
        if need_grad:
            if grad_i is None or grad_total is None:
                raise RuntimeError("missing batch gradient")
            if genewise:
                grad_total.index_add_(0, static.family_index_tensor, grad_i.to(device=theta.device, dtype=theta.dtype))
            else:
                grad_total.add_(grad_i.to(device=theta.device, dtype=theta.dtype))
    return total.detach(), None if grad_total is None else grad_total.detach()
