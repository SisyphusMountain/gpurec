import torch

from gpurec.api._implicit_grad import implicit_grad_loglik_vjp_wave
from gpurec.core.inference.solver import nll_from_root_rows, receiver_weights_are_uniform, solve_resident_e_pi


class _GeneReconFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta: torch.Tensor, receiver_weights: torch.Tensor, static):
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
            loss = nll_from_root_rows(root_rows, E)

        ctx.save_for_backward(
            theta,
            receiver_weights,
            pi_wave,
            pibar_wave,
            E,
            E_s1,
            E_s2,
            Ebar,
            log_pS,
            log_pD,
            log_pL,
            max_transfer_vec,
            pibar_row_max,
            receiver_log_probs,
        )
        ctx.static = static
        static.warm_E = E.detach()
        return loss

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        (
            theta,
            receiver_weights,
            Pi_star_wave,
            Pibar_star_wave,
            E_star,
            E_s1,
            E_s2,
            Ebar,
            log_pS,
            log_pD,
            log_pL,
            max_transfer_vec,
            uniform_pibar_row_max,
            receiver_log_probs,
        ) = ctx.saved_tensors
        static = ctx.static
        use_receiver_weights = not receiver_weights_are_uniform(receiver_weights)
        grad_theta, grad_receiver_weights = implicit_grad_loglik_vjp_wave(
            static.wave_layout,
            static.species_helpers,
            Pi_star_wave=Pi_star_wave,
            Pibar_star_wave=Pibar_star_wave,
            E_star=E_star,
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
            uniform_pibar_row_max=uniform_pibar_row_max,
            specieswise=static.specieswise,
            genewise=static.genewise,
            neumann_terms=static.solver_options.neumann_terms,
            bicgstab_max_iter=static.solver_options.bicgstab_max_iter,
            bicgstab_tol=static.solver_options.bicgstab_tol,
            bicgstab_breakdown_tol=static.solver_options.bicgstab_breakdown_tol,
            adjoint_pruning_threshold=static.solver_options.adjoint_pruning_threshold,
            use_adjoint_pruning=static.solver_options.use_adjoint_pruning,
            pibar_side_threshold=static.solver_options.pibar_side_threshold,
        )
        return (
            grad_theta * grad_output,
            grad_receiver_weights * grad_output,
            None,
        )


class _GeneReconFullLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta: torch.Tensor, receiver_weights: torch.Tensor, model):
        loss, grad_theta, grad_receiver = model._stream_batches(theta, receiver_weights, need_grad=True)
        if grad_theta is None or grad_receiver is None:
            raise RuntimeError("missing streamed gradient")
        ctx.save_for_backward(
            grad_theta.to(device=theta.device, dtype=theta.dtype),
            grad_receiver.to(device=receiver_weights.device, dtype=receiver_weights.dtype),
        )
        return loss.to(device=theta.device, dtype=theta.dtype)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        grad_theta, grad_receiver = ctx.saved_tensors
        return (
            grad_theta * grad_output.to(device=grad_theta.device, dtype=grad_theta.dtype),
            grad_receiver * grad_output.to(device=grad_receiver.device, dtype=grad_receiver.dtype),
            None,
        )
