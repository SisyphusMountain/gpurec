import torch

from gpurec.api._implicit_grad import implicit_grad_loglik_vjp_wave
from gpurec.core.inference.solver import (
    nll_from_root_rows,
    origination_grad_from_root_rows,
    origination_weights_are_uniform,
    receiver_weights_are_uniform,
    solve_resident_e_pi,
)
from gpurec.core.parameters.extract_parameters import origination_log_probs_from_weights


def _origination_log_probs(origination_weights, like):
    """Fp64 head probabilities, or ``(None, None)`` for the uniform fast path."""
    if origination_weights_are_uniform(origination_weights):
        return None, None
    o_lp = origination_log_probs_from_weights(
        origination_weights.to(device=like.device, dtype=torch.float64)
    )
    return o_lp, torch.exp2(o_lp)


class _GeneReconFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta: torch.Tensor, receiver_weights: torch.Tensor, origination_weights: torch.Tensor, static):
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
            o_lp, o_p = _origination_log_probs(origination_weights, theta)
            loss = nll_from_root_rows(root_rows, E, origination_log_probs=o_lp, origination_probs=o_p)

        centered_pi_state = getattr(static, "centered_pi_forward_state", None)
        ctx.centered_pi_forward = centered_pi_state is not None
        if centered_pi_state is None:
            pi_offset = torch.empty((0,), device=pi_wave.device, dtype=torch.float64)
            pibar_offset = torch.empty((0,), device=pi_wave.device, dtype=torch.float64)
        else:
            # Save the exact offsets used by this forward.  Keeping only a
            # pointer on ``static`` is unsafe when another forward runs before
            # this autograd context's backward and replaces the static state.
            pi_offset = centered_pi_state.pi_offset
            pibar_offset = centered_pi_state.pibar_offset
        ctx.save_for_backward(
            theta,
            receiver_weights,
            origination_weights,
            root_rows,
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
            pi_offset,
            pibar_offset,
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
            origination_weights,
            root_rows,
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
            pi_offset,
            pibar_offset,
        ) = ctx.saved_tensors
        static = ctx.static
        use_receiver_weights = not receiver_weights_are_uniform(receiver_weights)
        o_lp, o_p = _origination_log_probs(origination_weights, theta)
        centered_kwargs = (
            {"pi_offset": pi_offset, "pibar_offset": pibar_offset}
            if bool(getattr(ctx, "centered_pi_forward", False))
            else {}
        )
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
            self_loop_solver=static.solver_options.self_loop_solver,
            bicgstab_max_iter=static.solver_options.bicgstab_max_iter,
            bicgstab_tol=static.solver_options.bicgstab_tol,
            bicgstab_breakdown_tol=static.solver_options.bicgstab_breakdown_tol,
            adjoint_pruning_threshold=static.solver_options.adjoint_pruning_threshold,
            use_adjoint_pruning=static.solver_options.use_adjoint_pruning,
            pibar_side_threshold=static.solver_options.pibar_side_threshold,
            origination_log_probs=o_lp,
            origination_probs=o_p,
            **centered_kwargs,
        )
        # The centered and absolute kernels return cotangents in the primal
        # matrix dtype.  Match the configured autograd input dtype explicitly.
        grad_theta = grad_theta.to(device=theta.device, dtype=theta.dtype)
        grad_receiver_weights = grad_receiver_weights.to(
            device=receiver_weights.device, dtype=receiver_weights.dtype
        )
        grad_origination = None
        if ctx.needs_input_grad[2]:
            grad_origination = origination_grad_from_root_rows(
                root_rows, E_star, origination_weights
            )
            grad_origination = grad_origination * grad_output.to(
                device=grad_origination.device, dtype=grad_origination.dtype
            )
        return (
            grad_theta * grad_output.to(device=grad_theta.device, dtype=grad_theta.dtype),
            grad_receiver_weights
            * grad_output.to(
                device=grad_receiver_weights.device,
                dtype=grad_receiver_weights.dtype,
            ),
            grad_origination,
            None,
        )


class _GeneReconFullLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta: torch.Tensor, receiver_weights: torch.Tensor, origination_weights: torch.Tensor, model):
        need_origination_grad = bool(origination_weights.requires_grad)
        loss, grad_theta, grad_receiver, grad_origination = model._stream_batches(
            theta,
            receiver_weights,
            origination_weights,
            need_grad=True,
            need_origination_grad=need_origination_grad,
        )
        if grad_theta is None or grad_receiver is None:
            raise RuntimeError("missing streamed gradient")
        ctx.save_for_backward(
            grad_theta.to(device=theta.device, dtype=theta.dtype),
            grad_receiver.to(device=receiver_weights.device, dtype=receiver_weights.dtype),
        )
        ctx.grad_origination = (
            None
            if grad_origination is None
            else grad_origination.to(device=origination_weights.device, dtype=origination_weights.dtype)
        )
        # The streamed likelihood head deliberately carries the scalar loss in
        # fp64 for both representations; preserve that dtype.
        return loss.to(device=theta.device)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        grad_theta, grad_receiver = ctx.saved_tensors
        grad_origination = ctx.grad_origination
        return (
            grad_theta * grad_output.to(device=grad_theta.device, dtype=grad_theta.dtype),
            grad_receiver * grad_output.to(device=grad_receiver.device, dtype=grad_receiver.dtype),
            None
            if grad_origination is None
            else grad_origination * grad_output.to(device=grad_origination.device, dtype=grad_origination.dtype),
            None,
        )
