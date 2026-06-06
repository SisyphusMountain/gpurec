import torch

from gpurec.api._batch_state import (
    _BatchStatic,
    gmres_check_schedule_state_for_static,
    gmres_solution_cache_for_static,
)
from gpurec.api._implicit_grad import implicit_grad_loglik_vjp_wave
from gpurec.core.inference.solver import nll_from_root_rows, receiver_weights_are_uniform, solve_resident_e_pi


def theta_for_static(static: _BatchStatic, theta: torch.Tensor, *, genewise: bool) -> torch.Tensor:
    return theta.index_select(0, static.family_index_tensor) if genewise else theta


def evaluate_static_loss_grad(
    static: _BatchStatic,
    theta: torch.Tensor,
    receiver_weights: torch.Tensor,
    *,
    need_grad: bool,
):
    with torch.no_grad():
        (
            E,
            E_s1,
            E_s2,
            Ebar,
            root_rows,
            pi_wave,
            pibar_wave,
            pibar_row_max,
            log_pS,
            log_pD,
            log_pL,
            max_transfer_vec,
            receiver_log_probs,
        ) = solve_resident_e_pi(
            static, theta, receiver_weights, warm_start_E=static.warm_E
        )
        loss = nll_from_root_rows(root_rows, E).detach()
        static.warm_E = E.detach()
        if not need_grad:
            return loss, None, None
        use_receiver_weights = not receiver_weights_are_uniform(receiver_weights)
        gmres_check_schedule_state = gmres_check_schedule_state_for_static(static)
        grad_theta, grad_receiver = implicit_grad_loglik_vjp_wave(
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
            receiver_log_probs=receiver_log_probs,
            use_receiver_weights=use_receiver_weights,
            theta=theta,
            receiver_weights=receiver_weights,
            family_idx=static.rate_family_idx,
            uniform_pibar_row_max=pibar_row_max,
            specieswise=static.specieswise,
            genewise=static.genewise,
            neumann_terms=static.solver_options.neumann_terms,
            self_loop_solver=static.solver_options.self_loop_solver,
            gmres_tol=static.solver_options.gmres_tol,
            gmres_check_interval=static.solver_options.gmres_check_interval,
            gmres_check_schedule=gmres_check_schedule_state[0],
            gmres_validate_check_schedule=gmres_check_schedule_state[1],
            gmres_trust_check_schedule=static.solver_options.gmres_trust_check_schedule,
            gmres_trusted_schedule_safety_margin=static.solver_options.gmres_trusted_schedule_safety_margin,
            gmres_solution_cache=gmres_solution_cache_for_static(static),
            gmres_solution_cache_min_iterations=static.solver_options.gmres_solution_cache_min_iterations,
            gmres_preconditioner=static.solver_options.gmres_preconditioner,
            gmres_diagonal_preconditioner_floor=static.solver_options.gmres_diagonal_preconditioner_floor,
            bicgstab_max_iter=static.solver_options.bicgstab_max_iter,
            bicgstab_tol=static.solver_options.bicgstab_tol,
            bicgstab_breakdown_tol=static.solver_options.bicgstab_breakdown_tol,
            adjoint_pruning_threshold=static.solver_options.adjoint_pruning_threshold,
            use_adjoint_pruning=static.solver_options.use_adjoint_pruning,
            pibar_side_threshold=static.solver_options.pibar_side_threshold,
        )
        grad_theta = grad_theta.detach()
        grad_receiver = grad_receiver.detach()
    return loss, grad_theta, grad_receiver


def stream_batches(
    batch_statics: list[_BatchStatic],
    theta: torch.Tensor,
    receiver_weights: torch.Tensor,
    *,
    genewise: bool,
    need_grad: bool,
):
    total = torch.zeros((), dtype=theta.dtype, device=theta.device)
    grad_total = torch.zeros_like(theta) if need_grad else None
    grad_receiver_total = torch.zeros_like(receiver_weights) if need_grad else None
    for static in batch_statics:
        theta_batch = theta_for_static(static, theta, genewise=genewise)
        loss_i, grad_i, grad_receiver_i = evaluate_static_loss_grad(
            static,
            theta_batch,
            receiver_weights,
            need_grad=need_grad,
        )
        total = total + loss_i.to(device=theta.device, dtype=theta.dtype)
        if need_grad:
            if grad_i is None or grad_total is None or grad_receiver_i is None or grad_receiver_total is None:
                raise RuntimeError("missing batch gradient")
            if genewise:
                grad_total.index_add_(0, static.family_index_tensor, grad_i.to(device=theta.device, dtype=theta.dtype))
            else:
                grad_total.add_(grad_i.to(device=theta.device, dtype=theta.dtype))
            grad_receiver_total.add_(
                grad_receiver_i.to(device=receiver_weights.device, dtype=receiver_weights.dtype)
            )
    return (
        total.detach(),
        None if grad_total is None else grad_total.detach(),
        None if grad_receiver_total is None else grad_receiver_total.detach(),
    )
