"""Extract the four expected event counts during one ordinary gradient pass.

This is an experiment-only hook around the final implicit-gradient tail.  It leaves
GPURec's returned theta gradient unchanged and records the adjoints immediately
before their event-probability softmax fold.  Minus those adjoints are the positive
S/D/L/T counts for the observed family plus survival-conditioning ghosts.
"""
from __future__ import annotations

import torch

import gpurec.api._implicit_grad as implicit_grad
from gpurec.core.kernels.e_step import e_step_triton_autograd
from gpurec.core.parameters.extract_parameters import (
    as_family_param,
    extract_parameters_weighted_receivers,
    resolve_accumulator_dtype,
)


_ORIGINAL_TAIL = implicit_grad._e_adjoint_and_theta_vjp
_SINK: list[torch.Tensor] = []


def _tail_with_counts(
    E_star, log_pS, log_pD, log_pL, max_transfer_mat, receiver_log_probs, use_receiver_weights,
    grad_E, grad_Ebar, grad_E_s1, grad_E_s2,
    grad_log_pD, grad_log_pS, grad_max_transfer_mat, grad_receiver_log_probs,
    n_fam, theta, receiver_weights, species_helpers, *, specieswise, genewise,
    leaf_fm_log, drop_norm, e_adjoint_max_iter, e_adjoint_tol, cache, origination_probs,
    accumulator_dtype, event_counts_out=None,
):
    if drop_norm:
        raise ValueError("event counts require the survival-conditioned gradient")
    if grad_receiver_log_probs is not None:
        raise ValueError("event counts require frozen receiver weights")
    if e_adjoint_max_iter is None:
        e_adjoint_max_iter = implicit_grad.SolverOptions().e_adjoint_max_iter
    accumulator_dtype = resolve_accumulator_dtype(accumulator_dtype, fallback=E_star.dtype)
    topology_args = (
        species_helpers["sp_parent"], species_helpers["sp_child1"], species_helpers["sp_child2"],
        species_helpers["sp_height"], int(species_helpers["compact_level_ptr"].numel()) - 1,
    )

    # This is the production E-adjoint solve.  No second Pi-side sweep is added.
    E_req = E_star.detach().requires_grad_(True)
    with torch.enable_grad():
        triton_E_from_E, E_s1_from_E, E_s2_from_E, Ebar_from_E = e_step_triton_autograd(
            E_req, log_pS, log_pD, log_pL, max_transfer_mat, receiver_log_probs, *topology_args,
            use_receiver_weights=use_receiver_weights, leaf_fm_log=leaf_fm_log,
        )
        denom = implicit_grad._likelihood_log2_survival(
            E_req, origination_probs, accumulator_dtype=accumulator_dtype,
        )
        direct_obj = denom.sum() if E_req.shape[0] == n_fam else (n_fam * denom).sum()
        (aux_to_e,) = torch.autograd.grad(
            (direct_obj, E_s1_from_E, E_s2_from_E, Ebar_from_E), E_req,
            grad_outputs=(torch.ones_like(direct_obj), grad_E_s1, grad_E_s2, grad_Ebar),
            retain_graph=True,
        )
    q_E = grad_E + aux_to_e
    E_shape = E_star.shape

    def adjoint_operator(w_flat):
        w_e = w_flat.view(E_shape).contiguous()
        with torch.enable_grad():
            (g_e,) = torch.autograd.grad(triton_E_from_E, E_req, grad_outputs=w_e, retain_graph=True)
        return (w_e - g_e).reshape(-1)

    w_e = implicit_grad._neumann_e_adjoint(
        adjoint_operator, q_E.reshape(-1), max_iter=e_adjoint_max_iter, tol=e_adjoint_tol,
    ).view(E_shape)

    # Ask autograd for the ordinary theta gradient and for the adjoints of its
    # intermediate free log-probabilities in the SAME parameter E-step/VJP.
    S = int(species_helpers["S"])
    family_rows = int(E_star.shape[0])
    theta_req = theta.detach().requires_grad_(True)
    receiver_req = receiver_weights.detach().requires_grad_(True)
    with torch.enable_grad():
        l_s, l_d, l_l, mt, receiver_lp = extract_parameters_weighted_receivers(
            theta_req, receiver_req, species_helpers,
            specieswise=specieswise, genewise=genewise, uniform_fast=not use_receiver_weights,
            accumulator_dtype=accumulator_dtype,
        )
        parameter_loss = (
            (as_family_param(l_s, family_rows, S) * grad_log_pS).sum()
            + (as_family_param(l_d, family_rows, S) * grad_log_pD).sum()
            + (mt * grad_max_transfer_mat).sum()
        )
        E_from_params, _, _, Ebar_from_params = e_step_triton_autograd(
            E_star.detach(), l_s, l_d, l_l, mt, receiver_lp, *topology_args,
            use_receiver_weights=use_receiver_weights, leaf_fm_log=leaf_fm_log,
        )
        theta_gradient, a_s, a_d, a_l, a_mt = torch.autograd.grad(
            (parameter_loss, Ebar_from_params, E_from_params),
            (theta_req, l_s, l_d, l_l, mt),
            grad_outputs=(torch.ones_like(parameter_loss), grad_Ebar, w_e),
        )
    # A common shift of every transfer-recipient log probability is the free
    # family log-pT direction, hence sum its matrix adjoint across recipients.
    a_t = a_mt.reshape(theta.shape[0], -1).sum(dim=1)
    adjoint_matrix = torch.stack(
        (a_s.reshape(-1), a_d.reshape(-1), a_l.reshape(-1), a_t), dim=1,
    ).detach()
    _SINK.append(adjoint_matrix)
    return theta_gradient, None


def counts_and_gradient(model, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return CPU float64 per-family NLL, theta gradient, and positive S/D/L/T counts."""
    _SINK.clear()
    implicit_grad._e_adjoint_and_theta_vjp = _tail_with_counts
    try:
        values, gradient, _ = model.genewise_loss_vector_and_grad(
            theta=theta.to(device=model.theta.device, dtype=model.theta.dtype).contiguous(), need_grad=True,
        )
    finally:
        implicit_grad._e_adjoint_and_theta_vjp = _ORIGINAL_TAIL
    if len(_SINK) != len(model.batch_statics):
        raise RuntimeError(f"count hook fired {len(_SINK)} times for {len(model.batch_statics)} batches")
    counts_device = torch.zeros(theta.shape[0], 4, dtype=model.theta.dtype, device=model.theta.device)
    for static, adjoints in zip(model.batch_statics, _SINK):
        counts_device[static.family_index_tensor] = -adjoints
    return (
        values.detach().double().cpu(), gradient.detach().double().cpu(),
        counts_device.detach().double().cpu(),
    )
