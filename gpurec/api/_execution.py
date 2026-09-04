import os

import torch

from gpurec.api import _failure_dump
from gpurec.api._batch_state import _BatchStatic
from gpurec.api._implicit_grad import implicit_grad_loglik_vjp_wave
from gpurec.core.inference.solver import (
    nll_from_root_rows,
    nll_vector_from_root_rows,
    origination_grad_from_root_rows,
    origination_weights_are_uniform,
    receiver_weights_are_uniform,
    solve_resident_e_pi,
)
from gpurec.core.parameters.extract_parameters import (
    origination_log_probs_from_weights,
    resolve_accumulator_dtype,
)


def _backward_offsets(static: _BatchStatic) -> dict[str, torch.Tensor]:
    """Return the row gauges from the just-completed Pi solve."""
    state = getattr(static, "pi_forward_state", None)
    if state is None:
        raise RuntimeError("Pi forward did not publish its row-offset state")
    return {
        "pi_offset": state.pi_offset,
        "pibar_offset": state.pibar_offset,
        # Which rows the exact forward could not hold in one row scale. The adjoint has to make
        # the same call the forward did, so it travels with the gauges rather than being
        # re-derived. ``None`` on every non-exact forward, and on an exact one that flagged
        # nothing -- which is the ordinary case, and keeps the backward on its usual path.
        "wide_row": state.wide_row if state.wide_row_total > 0 else None,
    }


def theta_for_static(static: _BatchStatic, theta: torch.Tensor, *, genewise: bool) -> torch.Tensor:
    return theta.index_select(0, static.family_index_tensor) if genewise else theta


def origination_weights_for_static(static: _BatchStatic, origination_weights: torch.Tensor) -> torch.Tensor:
    """Select each batch family's own ``ω_g`` row from a per-family ``[G,S]`` weight tensor.

    Mirrors ``theta_for_static``: identity for a global ``[S]`` weight (``ndim == 1``); for a
    per-family ``[G,S]`` weight, selects the batch's rows via ``static.family_index_tensor`` so
    ``origination_weights_static[i]`` lines up with family ``i`` of this batch's ``root_rows``. A
    no-op for a single batch spanning all families (identity index_select), but required for
    multi-batch runs where a batch only covers a subset of families.
    """
    return (
        origination_weights.index_select(0, static.family_index_tensor)
        if origination_weights.ndim == 2
        else origination_weights
    )


def _origination_log_probs(
    origination_weights: torch.Tensor,
    *,
    like: torch.Tensor,
    accumulator_dtype: torch.dtype,
):
    """Head probabilities, or ``(None, None)`` for the uniform fast path."""
    if origination_weights_are_uniform(origination_weights):
        return None, None
    accumulator_dtype = resolve_accumulator_dtype(
        accumulator_dtype,
        fallback=like.dtype,
    )
    # Match the configured head before the softmax so its direct origination
    # gradient differentiates the exact function returned by the forward pass.
    o_lp = origination_log_probs_from_weights(
        origination_weights.to(device=like.device, dtype=accumulator_dtype),
        accumulator_dtype=accumulator_dtype,
    )
    return o_lp, torch.exp2(o_lp)


def _static_accumulator_dtype(static: _BatchStatic, fallback: torch.dtype) -> torch.dtype:
    return resolve_accumulator_dtype(
        getattr(static, "accumulator_dtype", None),
        fallback=fallback,
    )


def _stream_accumulator_dtype(
    batch_statics: list[_BatchStatic],
    fallback: torch.dtype,
) -> torch.dtype:
    if not batch_statics:
        return resolve_accumulator_dtype(None, fallback=fallback)
    dtype = _static_accumulator_dtype(batch_statics[0], fallback)
    for static in batch_statics[1:]:
        if _static_accumulator_dtype(static, fallback) != dtype:
            raise ValueError("all streamed batches must share one accumulator dtype")
    return dtype


def evaluate_static_loss_grad(
    static: _BatchStatic,
    theta: torch.Tensor,
    receiver_weights: torch.Tensor,
    origination_weights: torch.Tensor,
    *,
    need_grad: bool,
    need_origination_grad: bool = False,
):
    accumulator_dtype = _static_accumulator_dtype(static, theta.dtype)
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
        origination_weights_static = origination_weights_for_static(static, origination_weights)
        o_lp, o_p = _origination_log_probs(
            origination_weights_static,
            like=theta,
            accumulator_dtype=accumulator_dtype,
        )
        loss = nll_from_root_rows(
            root_rows,
            E,
            origination_log_probs=o_lp,
            origination_probs=o_p,
            accumulator_dtype=accumulator_dtype,
        ).detach()
        static.warm_E = E.detach()
        if not need_grad:
            return loss, None, None, None
        use_receiver_weights = not receiver_weights_are_uniform(receiver_weights)
        # Opt-in adjoint warm-start (GPUREC_WARM_ADJOINT): reuse the previous call's per-wave Pi-adjoint
        # as the Neumann initial guess (cached in-place on static.warm_v). Default off -> behaviour unchanged.
        if os.environ.get("GPUREC_WARM_ADJOINT") and getattr(static, "warm_adjoint_ok", True):
            if static.warm_v is None:
                static.warm_v = {}
            _warm_v = static.warm_v
        else:
            _warm_v = None
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
            leaf_fm_log=static.leaf_fm_log,
            uniform_pibar_row_max=pibar_row_max,
            specieswise=static.specieswise,
            genewise=static.genewise,
            neumann_terms=static.solver_options.neumann_terms,
            neumann_term_tol=static.solver_options.neumann_term_tol,
            adjoint_self_loop=static.solver_options.adjoint_self_loop,
            e_adjoint_max_iter=static.solver_options.e_adjoint_max_iter,
            e_adjoint_tol=static.solver_options.e_adjoint_tol,
            adjoint_pruning_threshold=static.solver_options.adjoint_pruning_threshold,
            use_adjoint_pruning=static.solver_options.use_adjoint_pruning,
            pibar_side_threshold=static.solver_options.pibar_side_threshold,
            warm_v=_warm_v,
            reserved_scratch_bytes=(static.warm_scratch_reserved_bytes if _warm_v is not None else None),
            origination_log_probs=o_lp,
            origination_probs=o_p,
            accumulator_dtype=accumulator_dtype,
            **_backward_offsets(static),
        )
        grad_theta = grad_theta.detach()
        grad_receiver = grad_receiver.detach()
        grad_origination = (
            origination_grad_from_root_rows(
                root_rows,
                E,
                origination_weights_static,
                accumulator_dtype=accumulator_dtype,
            ).detach()
            if need_origination_grad
            else None
        )
    return loss, grad_theta, grad_receiver, grad_origination


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
        accumulator_dtype = _static_accumulator_dtype(static, theta.dtype)
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
            leaf_fm_log=static.leaf_fm_log,
            uniform_pibar_row_max=pibar_row_max,
            specieswise=static.specieswise,
            genewise=static.genewise,
            neumann_terms=nt,
            neumann_term_tol=static.solver_options.neumann_term_tol,
            adjoint_self_loop=static.solver_options.adjoint_self_loop,
            adjoint_pruning_threshold=static.solver_options.adjoint_pruning_threshold,
            use_adjoint_pruning=static.solver_options.use_adjoint_pruning,
            pibar_side_threshold=static.solver_options.pibar_side_threshold,
            collect_backward_relres=True,
            accumulator_dtype=accumulator_dtype,
            **_backward_offsets(static),
        )
    return forward_resid, backward_relres, backward_vk_mag


def evaluate_static_loss_vector_grad(
    static: _BatchStatic,
    theta: torch.Tensor,
    receiver_weights: torch.Tensor,
    origination_weights: torch.Tensor,
    *,
    need_grad: bool,
    update_warm_start: bool,
    need_origination_grad: bool = False,
):
    if not static.genewise:
        raise ValueError("per-family loss vectors require genewise mode")
    accumulator_dtype = _static_accumulator_dtype(static, theta.dtype)
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
        origination_weights_static = origination_weights_for_static(static, origination_weights)
        o_lp, o_p = _origination_log_probs(
            origination_weights_static,
            like=theta,
            accumulator_dtype=accumulator_dtype,
        )
        loss_vec = nll_vector_from_root_rows(
            root_rows,
            E,
            origination_log_probs=o_lp,
            origination_probs=o_p,
            accumulator_dtype=accumulator_dtype,
        ).detach()
        if update_warm_start:
            static.warm_E = E.detach()
        if not need_grad:
            return loss_vec, None, None, None
        use_receiver_weights = not receiver_weights_are_uniform(receiver_weights)
        # Opt-in adjoint warm-start (GPUREC_WARM_ADJOINT): reuse the previous call's per-wave Pi-adjoint
        # as the Neumann initial guess (cached in-place on static.warm_v). Default off -> behaviour unchanged.
        if os.environ.get("GPUREC_WARM_ADJOINT") and getattr(static, "warm_adjoint_ok", True):
            if static.warm_v is None:
                static.warm_v = {}
            _warm_v = static.warm_v
        else:
            _warm_v = None
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
            leaf_fm_log=static.leaf_fm_log,
            uniform_pibar_row_max=pibar_row_max,
            specieswise=static.specieswise,
            genewise=static.genewise,
            neumann_terms=static.solver_options.neumann_terms,
            neumann_term_tol=static.solver_options.neumann_term_tol,
            adjoint_self_loop=static.solver_options.adjoint_self_loop,
            e_adjoint_max_iter=static.solver_options.e_adjoint_max_iter,
            e_adjoint_tol=static.solver_options.e_adjoint_tol,
            adjoint_pruning_threshold=static.solver_options.adjoint_pruning_threshold,
            use_adjoint_pruning=static.solver_options.use_adjoint_pruning,
            pibar_side_threshold=static.solver_options.pibar_side_threshold,
            warm_v=_warm_v,
            reserved_scratch_bytes=(static.warm_scratch_reserved_bytes if _warm_v is not None else None),
            origination_log_probs=o_lp,
            origination_probs=o_p,
            accumulator_dtype=accumulator_dtype,
            **_backward_offsets(static),
        )
        grad_theta = grad_theta.detach()
        grad_receiver = grad_receiver.detach()
        grad_origination = (
            origination_grad_from_root_rows(
                root_rows,
                E,
                origination_weights_static,
                accumulator_dtype=accumulator_dtype,
            ).detach()
            if need_origination_grad
            else None
        )
    if _failure_dump.is_enabled():
        # Synchronising, so it only runs for a driver that asked for dumps. A non-finite loss or
        # gradient here means the batch is already broken before any curvature probe touches it.
        broken = [
            text
            for text in (
                _failure_dump.nonfinite_summary("loss_vec", loss_vec),
                _failure_dump.nonfinite_summary("grad_theta", grad_theta),
                _failure_dump.nonfinite_summary("grad_receiver", grad_receiver),
            )
            if text
        ]
        if broken:
            print(f"[gpurec] non-finite gradient output on a "
                  f"{len(static.family_indices)}-family batch: {'; '.join(broken)}")
            print(_failure_dump.describe_forward_state(static, theta, receiver_weights))
    return loss_vec, grad_theta, grad_receiver, grad_origination


def stream_batches(
    batch_statics: list[_BatchStatic],
    theta: torch.Tensor,
    receiver_weights: torch.Tensor,
    origination_weights: torch.Tensor,
    *,
    genewise: bool,
    need_grad: bool,
    need_origination_grad: bool = False,
):
    loss_dtype = _stream_accumulator_dtype(batch_statics, theta.dtype)
    total = torch.zeros((), dtype=loss_dtype, device=theta.device)
    grad_total = torch.zeros_like(theta) if need_grad else None
    grad_receiver_total = torch.zeros_like(receiver_weights) if need_grad else None
    grad_origination_total = (
        torch.zeros_like(origination_weights) if (need_grad and need_origination_grad) else None
    )
    for static in batch_statics:
        theta_batch = theta_for_static(static, theta, genewise=genewise)
        loss_i, grad_i, grad_receiver_i, grad_origination_i = evaluate_static_loss_grad(
            static,
            theta_batch,
            receiver_weights,
            origination_weights,
            need_grad=need_grad,
            need_origination_grad=need_origination_grad,
        )
        total = total + loss_i.to(device=theta.device, dtype=loss_dtype)
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
            if grad_origination_total is not None and grad_origination_i is not None:
                grad_origination_i = grad_origination_i.to(
                    device=origination_weights.device, dtype=origination_weights.dtype
                )
                if origination_weights.ndim == 2:  # per-family [G,S]: scatter batch-local rows into full [G,S]
                    grad_origination_total.index_add_(0, static.family_index_tensor, grad_origination_i)
                else:  # 1-D global [S]: plain sum (byte-for-bit, specieswise)
                    grad_origination_total.add_(grad_origination_i)
    return (
        total.detach(),
        None if grad_total is None else grad_total.detach(),
        None if grad_receiver_total is None else grad_receiver_total.detach(),
        None if grad_origination_total is None else grad_origination_total.detach(),
    )


def stream_genewise_loss_vector_grad(
    batch_statics: list[_BatchStatic],
    theta: torch.Tensor,
    receiver_weights: torch.Tensor,
    origination_weights: torch.Tensor,
    *,
    need_grad: bool,
    update_warm_starts: bool = False,
    need_origination_grad: bool = False,
):
    if theta.ndim < 1:
        raise ValueError("genewise theta must have a family batch dimension")
    loss_dtype = _stream_accumulator_dtype(batch_statics, theta.dtype)
    loss_total = torch.empty((int(theta.shape[0]),), dtype=loss_dtype, device=theta.device)
    grad_total = torch.zeros_like(theta) if need_grad else None
    grad_receiver_total = torch.zeros_like(receiver_weights) if need_grad else None
    grad_origination_total = (
        torch.zeros_like(origination_weights) if (need_grad and need_origination_grad) else None
    )
    for static in batch_statics:
        if not static.genewise:
            raise ValueError("per-family loss vectors require genewise mode")
        theta_batch = theta_for_static(static, theta, genewise=True)
        loss_i, grad_i, grad_receiver_i, grad_origination_i = evaluate_static_loss_vector_grad(
            static,
            theta_batch,
            receiver_weights,
            origination_weights,
            need_grad=need_grad,
            update_warm_start=update_warm_starts,
            need_origination_grad=need_origination_grad,
        )
        loss_total.index_copy_(
            0,
            static.family_index_tensor,
            loss_i.to(device=theta.device, dtype=loss_dtype),
        )
        if need_grad:
            if grad_i is None or grad_total is None or grad_receiver_i is None or grad_receiver_total is None:
                raise RuntimeError("missing batch gradient")
            grad_total.index_add_(0, static.family_index_tensor, grad_i.to(device=theta.device, dtype=theta.dtype))
            grad_receiver_total.add_(
                grad_receiver_i.to(device=receiver_weights.device, dtype=receiver_weights.dtype)
            )
            if grad_origination_total is not None and grad_origination_i is not None:
                grad_origination_i = grad_origination_i.to(
                    device=origination_weights.device, dtype=origination_weights.dtype
                )
                if origination_weights.ndim == 2:  # per-family [G,S]: scatter batch-local rows into full [G,S]
                    grad_origination_total.index_add_(0, static.family_index_tensor, grad_origination_i)
                else:  # 1-D global [S]: plain sum (byte-for-bit, specieswise)
                    grad_origination_total.add_(grad_origination_i)
    return (
        loss_total.detach(),
        None if grad_total is None else grad_total.detach(),
        None if grad_receiver_total is None else grad_receiver_total.detach(),
        None if grad_origination_total is None else grad_origination_total.detach(),
    )
