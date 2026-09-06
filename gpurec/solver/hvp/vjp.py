"""Root-score VJP shared by the analytic exact-Hessian implementation."""

from __future__ import annotations

import torch

from gpurec.api._implicit_grad import implicit_grad_loglik_vjp_wave
from gpurec.core.inference.solver import receiver_weights_are_uniform
from gpurec.core.parameters.extract_parameters import resolve_accumulator_dtype


@torch.no_grad()
def vjp_root_to_theta(
    static,
    sv,
    seed_root,
    theta,
    receiver_weights,
    *,
    drop_norm=True,
    neumann_terms=None,
    use_pruning=None,
    e_adjoint_tol=None,
    cache=None,
    origination_log_probs=None,
    origination_probs=None,
    reserved_scratch_bytes=None,
    leaf_fm_log=None,
):
    """Apply a root-score cotangent and return ``(grad_theta, grad_receiver)``.

    This is a thin adapter over the production implicit backward. ``cache`` collects the primal
    adjoint state consumed by the exact Hessian's tangent-adjoint sweep.
    """
    so = static.solver_options
    pi_state = sv["pi_state"]
    pi_state.validate(
        sv["pi_wave"], sv["pibar_wave"], sv["pibar_row_max"], check_values=False
    )
    return implicit_grad_loglik_vjp_wave(
        static.wave_layout,
        static.species_helpers,
        Pi_star_wave=sv["pi_wave"],
        Pibar_star_wave=sv["pibar_wave"],
        E_star=sv["E"],
        E_s1=sv["E_s1"],
        E_s2=sv["E_s2"],
        Ebar=sv["Ebar"],
        log_pS=sv["log_pS"],
        log_pD=sv["log_pD"],
        log_pL=sv["log_pL"],
        max_transfer_mat=sv["max_transfer"],
        receiver_log_probs=sv["receiver_log_probs"],
        use_receiver_weights=not receiver_weights_are_uniform(receiver_weights),
        need_receiver_grad=True,
        forward_gene_split=None,
        theta=theta,
        receiver_weights=receiver_weights,
        uniform_pibar_row_max=sv["pibar_row_max"],
        family_idx=static.rate_family_idx,
        leaf_fm_log=leaf_fm_log,
        specieswise=static.specieswise,
        genewise=static.genewise,
        neumann_terms=int(so.neumann_terms if neumann_terms is None else neumann_terms),
        neumann_term_tol=float(so.neumann_term_tol),
        e_adjoint_max_iter=so.e_adjoint_max_iter,
        e_adjoint_tol=(so.e_adjoint_tol if e_adjoint_tol is None else e_adjoint_tol),
        adjoint_pruning_threshold=so.adjoint_pruning_threshold,
        use_adjoint_pruning=bool(
            so.use_adjoint_pruning if use_pruning is None else use_pruning
        ),
        pibar_side_threshold=so.pibar_side_threshold,
        seed_root=seed_root,
        drop_norm=drop_norm,
        cache=cache,
        origination_log_probs=origination_log_probs,
        origination_probs=origination_probs,
        accumulator_dtype=resolve_accumulator_dtype(
            getattr(static, "accumulator_dtype", None),
            fallback=sv["pi_wave"].dtype,
        ),
        reserved_scratch_bytes=reserved_scratch_bytes,
        pi_offset=pi_state.pi_offset,
        pibar_offset=pi_state.pibar_offset,
    )
