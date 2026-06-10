import torch

from gpurec.api._batch_state import _BatchStatic
from gpurec.api._implicit_grad import implicit_grad_loglik_vjp_wave
from gpurec.core.inference.solver import (
    nll_from_root_rows,
    nll_vector_from_root_rows,
    receiver_weights_are_uniform,
    solve_resident_e_pi,
)


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


def evaluate_static_convergence(
    static: _BatchStatic,
    theta: torch.Tensor,
    receiver_weights: torch.Tensor,
    *,
    pi_iters_high: int,
    neumann_terms: int | None = None,
):
    """Per-family solver-convergence diagnostics for one batch (batch-local index).

    Runs ONE forward solve at a high ``pi_iters`` (so the forward fixed point is
    converged — required for the backward signal to be meaningful), capturing the
    final Pi update size, then a no-grad backward self-loop pass measuring the last
    Neumann increment. Returns ``(forward_resid, backward_relres, backward_vk_mag)``,
    each a 1-D float tensor of length ``n_families_in_batch``.
    """
    with torch.no_grad():
        C = int(static.wave_layout["leaf_species_index"].numel())
        pi_residual = torch.zeros(C, device=theta.device, dtype=torch.float32)
        (
            E, E_s1, E_s2, Ebar, root_rows, pi_wave, pibar_wave, pibar_row_max,
            log_pS, log_pD, log_pL, max_transfer_vec, receiver_log_probs,
        ) = solve_resident_e_pi(
            static, theta, receiver_weights,
            warm_start_E=None, pi_iters=pi_iters_high, pi_residual_out=pi_residual,
        )
        fam_local = static.wave_layout["family_idx"].to(device=theta.device, dtype=torch.long)
        n_fam = int(static.family_index_tensor.numel())
        forward_resid = torch.zeros(n_fam, device=theta.device, dtype=torch.float32)
        forward_resid.scatter_reduce_(0, fam_local, pi_residual, reduce="amax", include_self=True)

        use_receiver_weights = not receiver_weights_are_uniform(receiver_weights)
        nt = static.solver_options.neumann_terms if neumann_terms is None else int(neumann_terms)
        backward_relres, backward_vk_mag = implicit_grad_loglik_vjp_wave(
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
            neumann_terms=nt,
            self_loop_solver="neumann",  # diagnostic always measures Neumann convergence
            adjoint_pruning_threshold=static.solver_options.adjoint_pruning_threshold,
            use_adjoint_pruning=static.solver_options.use_adjoint_pruning,
            pibar_side_threshold=static.solver_options.pibar_side_threshold,
            collect_backward_relres=True,
        )
    return forward_resid, backward_relres, backward_vk_mag


def evaluate_static_loss_vector_grad(
    static: _BatchStatic,
    theta: torch.Tensor,
    receiver_weights: torch.Tensor,
    *,
    need_grad: bool,
    update_warm_start: bool,
):
    if not static.genewise:
        raise ValueError("per-family loss vectors require genewise mode")
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
        loss_vec = nll_vector_from_root_rows(root_rows, E).detach()
        if update_warm_start:
            static.warm_E = E.detach()
        if not need_grad:
            return loss_vec, None, None
        use_receiver_weights = not receiver_weights_are_uniform(receiver_weights)
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
            bicgstab_max_iter=static.solver_options.bicgstab_max_iter,
            bicgstab_tol=static.solver_options.bicgstab_tol,
            bicgstab_breakdown_tol=static.solver_options.bicgstab_breakdown_tol,
            adjoint_pruning_threshold=static.solver_options.adjoint_pruning_threshold,
            use_adjoint_pruning=static.solver_options.use_adjoint_pruning,
            pibar_side_threshold=static.solver_options.pibar_side_threshold,
        )
        grad_theta = grad_theta.detach()
        grad_receiver = grad_receiver.detach()
    return loss_vec, grad_theta, grad_receiver


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


def stream_genewise_loss_vector_grad(
    batch_statics: list[_BatchStatic],
    theta: torch.Tensor,
    receiver_weights: torch.Tensor,
    *,
    need_grad: bool,
    update_warm_starts: bool = False,
):
    if theta.ndim < 1:
        raise ValueError("genewise theta must have a family batch dimension")
    loss_total = torch.empty((int(theta.shape[0]),), dtype=theta.dtype, device=theta.device)
    grad_total = torch.zeros_like(theta) if need_grad else None
    grad_receiver_total = torch.zeros_like(receiver_weights) if need_grad else None
    for static in batch_statics:
        if not static.genewise:
            raise ValueError("per-family loss vectors require genewise mode")
        theta_batch = theta_for_static(static, theta, genewise=True)
        loss_i, grad_i, grad_receiver_i = evaluate_static_loss_vector_grad(
            static,
            theta_batch,
            receiver_weights,
            need_grad=need_grad,
            update_warm_start=update_warm_starts,
        )
        loss_total.index_copy_(
            0,
            static.family_index_tensor,
            loss_i.to(device=theta.device, dtype=theta.dtype),
        )
        if need_grad:
            if grad_i is None or grad_total is None or grad_receiver_i is None or grad_receiver_total is None:
                raise RuntimeError("missing batch gradient")
            grad_total.index_add_(0, static.family_index_tensor, grad_i.to(device=theta.device, dtype=theta.dtype))
            grad_receiver_total.add_(
                grad_receiver_i.to(device=receiver_weights.device, dtype=receiver_weights.dtype)
            )
    return (
        loss_total.detach(),
        None if grad_total is None else grad_total.detach(),
        None if grad_receiver_total is None else grad_receiver_total.detach(),
    )
