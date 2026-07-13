"""Gauss-Newton / Fisher Hessian-vector product:  M v = J^T B J v.

``J = d(Pi_root)/dtheta`` (forward tangent, ``forward_tangent.py``);
``B_i = ln2 (diag(q_i) - q_i q_i^T)`` is the posterior/Fisher covariance of the root softmax
``q_i = softmax2(Pi_root[i,:])`` (PSD, so M is PSD and CG always converges);
``J^T`` is the existing wave adjoint, reused here with a *custom root seed* and with the loss's
explicit E-norm term dropped (it is not part of ``d(Pi_root)/dtheta``).

``vjp_root_to_theta`` is a thin wrapper over the production
``implicit_grad_loglik_vjp_wave`` (``gpurec.api._implicit_grad``): it unpacks ``static``/``sv``
and forwards, supplying a custom ``seed_root`` and ``drop_norm=True`` (the two knobs that turn the
real backward into ``J^T B``). The single implementation is regression-gated by the real backward
(``test_optim_golden``) and the exact-HVP FD gate (``test_genewise_hvp``).
"""

from __future__ import annotations

import torch

from gpurec.core.inference.logspace import logsumexp2 as _logsumexp2
from gpurec.api._implicit_grad import _safe_exp2_ratio, implicit_grad_loglik_vjp_wave
from gpurec.core.parameters.extract_parameters import resolve_accumulator_dtype

from gpurec.solver.forward_tangent import jvp_root_scores, DEFAULT_SELF_MAX_ITER

_LN2 = 0.6931471805599453


@torch.no_grad()
def vjp_root_to_theta(static, sv, seed_root, theta, receiver_weights, *, drop_norm=True,
                      neumann_terms=None, use_pruning=None, bicgstab_tol=None, cache=None,
                      origination_log_probs=None, origination_probs=None,
                      reserved_scratch_bytes=None):
    """J^T applied to a root-score cotangent ``seed_root`` [n_root, S] -> ``(grad_theta [S,3], grad_col)``.

    Thin wrapper over the production ``implicit_grad_loglik_vjp_wave``: it unpacks ``static``/``sv``
    and forwards. With ``seed_root=None`` the loss seed ``-softmax2(Pi_root)`` is used and ``drop_norm``
    should be False to reproduce the real gradient (regression path); the GGN path passes a custom
    ``seed_root`` and ``drop_norm=True`` (drop the loss's explicit E-norm term, which is not part of
    d(Pi_root)/dtheta). ``neumann_terms``/``use_pruning``/``bicgstab_tol`` override the solver options
    so the adjoint can be made convergent + unpruned to match the convergent Jvp (M = J^T B J
    symmetric). ``cache`` collects per-wave adjoint state for the exact-HVP tangent sweep.
    ``reserved_scratch_bytes`` mirrors the gradient path's memory-gate reservation (see
    ``_execution.py``): the caller sources it from ``static.warm_scratch_reserved_bytes`` only
    when warm-adjoint is resident for this static, else ``None`` (cold path unchanged) -- this
    wrapper just forwards it through to the gated fast path.
    """
    so = static.solver_options
    pi_state = sv["pi_state"]
    pi_state.validate(
        sv["pi_wave"], sv["pibar_wave"], sv["pibar_row_max"],
        check_values=False,
    )
    return implicit_grad_loglik_vjp_wave(
        static.wave_layout, static.species_helpers,
        Pi_star_wave=sv["pi_wave"], Pibar_star_wave=sv["pibar_wave"],
        E_star=sv["E"], E_s1=sv["E_s1"], E_s2=sv["E_s2"], Ebar=sv["Ebar"],
        log_pS=sv["log_pS"], log_pD=sv["log_pD"], log_pL=sv["log_pL"],
        max_transfer_mat=sv["max_transfer"], receiver_log_probs=sv["receiver_log_probs"],
        theta=theta, receiver_weights=receiver_weights,
        uniform_pibar_row_max=sv["pibar_row_max"], family_idx=static.rate_family_idx,
        specieswise=static.specieswise, genewise=static.genewise,
        neumann_terms=int(so.neumann_terms if neumann_terms is None else neumann_terms),
        self_loop_solver=so.self_loop_solver,
        bicgstab_max_iter=so.bicgstab_max_iter,
        bicgstab_tol=(so.bicgstab_tol if bicgstab_tol is None else bicgstab_tol),
        bicgstab_breakdown_tol=so.bicgstab_breakdown_tol,
        adjoint_pruning_threshold=so.adjoint_pruning_threshold,
        use_adjoint_pruning=bool(so.use_adjoint_pruning if use_pruning is None else use_pruning),
        pibar_side_threshold=so.pibar_side_threshold,
        e_adjoint_solver=so.e_adjoint_solver,
        seed_root=seed_root, drop_norm=drop_norm, cache=cache,
        origination_log_probs=origination_log_probs, origination_probs=origination_probs,
        accumulator_dtype=resolve_accumulator_dtype(
            getattr(static, "accumulator_dtype", None),
            fallback=sv["pi_wave"].dtype,
        ),
        reserved_scratch_bytes=reserved_scratch_bytes,
        pi_offset=pi_state.pi_offset,
        pibar_offset=pi_state.pibar_offset,
    )


def make_ggn_hvp(static, theta, receiver_weights, sv, *, self_tol=None,
                 self_max_iter=DEFAULT_SELF_MAX_ITER,
                 vjp_neumann_terms=None, vjp_use_pruning=None, vjp_bicgstab_tol=None):
    """Return hvp(v_vec) computing the GGN/Fisher product M v in theta-space (flat 3S).

    Defaults use the solver's production adjoint settings (neumann_terms, pruning, bicgstab tol);
    the wave self-loop already converges within those terms, so M is unchanged vs the convergent
    settings (pass overrides only to force a convergent/unpruned adjoint for symmetry checks).
    """
    S = int(static.species_helpers["S"])
    root_ids = static.wave_layout["root_clade_ids"]
    root_Pi = sv["pi_wave"].index_select(0, root_ids)
    accumulator_dtype = resolve_accumulator_dtype(
        getattr(static, "accumulator_dtype", None),
        fallback=root_Pi.dtype,
    )
    root_head = root_Pi.to(dtype=accumulator_dtype)
    root_lse = _logsumexp2(root_head, dim=-1, keepdim=True)
    q = _safe_exp2_ratio(root_head, root_lse)  # posterior softmax2 per root row

    def hvp(v_vec):
        v = v_vec.reshape(S, 3).to(theta.dtype)
        t = jvp_root_scores(static, theta, v, sv, self_tol=self_tol, self_max_iter=self_max_iter)
        t_head = t.to(dtype=accumulator_dtype)
        u = _LN2 * q * (
            t_head - (q * t_head).sum(dim=-1, keepdim=True)
        )  # B t  (PSD Fisher covariance)
        gt, _gc = vjp_root_to_theta(static, sv, u, theta, receiver_weights, drop_norm=True,
                                    neumann_terms=vjp_neumann_terms, use_pruning=vjp_use_pruning,
                                    bicgstab_tol=vjp_bicgstab_tol)
        return gt.reshape(-1)

    return hvp
