"""Gauss-Newton / Fisher Hessian-vector product:  M v = J^T B J v.

``J = d(Pi_root)/dtheta`` (forward tangent, ``newton/forward_tangent.py``);
``B_i = ln2 (diag(q_i) - q_i q_i^T)`` is the posterior/Fisher covariance of the root softmax
``q_i = softmax2(Pi_root[i,:])`` (PSD, so M is PSD and CG always converges);
``J^T`` is the existing wave adjoint, reused here with a *custom root seed* and with the loss's
explicit E-norm term dropped (it is not part of ``d(Pi_root)/dtheta``).

``vjp_root_to_theta`` is a faithful copy of ``implicit_grad_loglik_vjp_wave``
(kbench/api/_implicit_grad.py) parametrized by ``seed_root`` and ``drop_norm``; kept in ``newton/``
so the frozen ``kbench/api`` is not edited. ``_check_vjp_matches_golden`` regresses it against the
real backward (seed = -q, norm kept).
"""

from __future__ import annotations

import math

import torch

from gpurec.core.inference.logspace import logsumexp2 as _logsumexp2, log2_survival as _log2_survival
from gpurec.core.inference.solver import receiver_weights_are_uniform
from gpurec.core.kernels.dts_fused import compute_dts_forward
from gpurec.core.kernels.e_step import e_step_triton_autograd
from gpurec.core.kernels.wave_backward import (
    active_mask_from_rhs_absmax_fused,
    dts_cross_backward_accum_fused,
    uniform_cross_pibar_vjp_tree_from_ud_fused,
    wave_backward_uniform_fused,
)
from gpurec.core.parameters.extract_parameters import (
    as_family_param, as_family_species, extract_parameters_weighted_receivers,
)
from gpurec.api._implicit_grad import _bicgstab, _safe_exp2_ratio

from gpurec.optim.forward_tangent import jvp_root_scores

_LN2 = 0.6931471805599453


@torch.no_grad()
def vjp_root_to_theta(static, sv, seed_root, theta, receiver_weights, *, drop_norm=True,
                      neumann_terms=None, use_pruning=None, bicgstab_tol=None, cache=None,
                      origination_log_probs=None, origination_probs=None):
    """J^T applied to a root-score cotangent ``seed_root`` [n_root, S] -> grad_theta [S, 3].

    With ``seed_root=None`` the loss seed ``-softmax2(Pi_root)`` is used and ``drop_norm`` should be
    False to reproduce the real gradient (regression path). ``neumann_terms``/``use_pruning``
    override the solver options so the adjoint can be made convergent + unpruned to match the
    convergent Jvp (so M = J^T B J is symmetric).
    """
    so = static.solver_options
    wave_layout = static.wave_layout
    species_helpers = static.species_helpers
    family_idx = static.rate_family_idx
    specieswise, genewise = static.specieswise, static.genewise
    neumann_terms = int(so.neumann_terms if neumann_terms is None else neumann_terms)
    use_pruning = bool(so.use_adjoint_pruning if use_pruning is None else use_pruning)
    self_loop_solver = so.self_loop_solver
    # S7: derive from the base alpha's non-uniformity, exactly as production
    # (solver.py:27, _execution.py:48). At a non-uniform base the weighted receiver
    # paths must be LIVE or the backward/cache diverges (E-adjoint -> 1e18).
    use_receiver_weights = not receiver_weights_are_uniform(receiver_weights)

    Pi_star_wave = sv["pi_wave"]
    Pibar_star_wave = sv["pibar_wave"]
    E_star, E_s1, E_s2, Ebar = sv["E"], sv["E_s1"], sv["E_s2"], sv["Ebar"]
    log_pS, log_pD, log_pL = sv["log_pS"], sv["log_pD"], sv["log_pL"]
    max_transfer_mat, receiver_log_probs = sv["max_transfer"], sv["receiver_log_probs"]
    uniform_pibar_row_max = sv["pibar_row_max"]

    C, S = Pi_star_wave.shape
    device, dtype = Pi_star_wave.device, Pi_star_wave.dtype
    item_rows = int(E_star.shape[0])
    E_item, Ebar_item, log_pS_item, log_pD_item, max_transfer_item = (
        as_family_species(x, S, item_rows) for x in (E_star, Ebar, log_pS, log_pD, max_transfer_mat)
    )
    log_pD_param, log_pS_param = (as_family_param(x, item_rows, S) for x in (log_pD, log_pS))
    DL_item = 1.0 + log_pD_item + E_item
    SL1_item = log_pS_item + as_family_species(E_s2, S, item_rows)
    SL2_item = log_pS_item + as_family_species(E_s1, S, item_rows)
    accumulated_rhs = torch.zeros(C, S, device=device, dtype=dtype)
    grad_log_pD, grad_log_pS = (torch.zeros_like(x) for x in (log_pD_param, log_pS_param))
    grad_max_transfer_mat = torch.zeros_like(max_transfer_item)
    grad_receiver_log_probs = torch.zeros((S,), device=device, dtype=dtype)
    grad_E_acc, grad_Ebar_acc, grad_E_s1_acc, grad_E_s2_acc = (
        torch.zeros_like(x) for x in (E_star, Ebar, E_star, E_star)
    )
    root_ids = wave_layout["root_clade_ids"]
    root_Pi = Pi_star_wave.index_select(0, root_ids)
    # origination-weighted numerator seed: softmax over (root_Pi + log2 origination_prob). The uniform
    # default (origination_log_probs is None) reproduces the legacy -softmax2(root_Pi) seed bit-for-bit.
    root_Pi_w = root_Pi if origination_log_probs is None else root_Pi + origination_log_probs
    root_lse = _logsumexp2(root_Pi_w, dim=-1, keepdim=True)
    if seed_root is None:
        seed_root = -_safe_exp2_ratio(root_Pi_w, root_lse)
    accumulated_rhs.index_copy_(0, root_ids, seed_root.to(dtype))

    def _scatter_accum(acc, item_rows_for_wave, contrib):
        if contrib.dtype != acc.dtype:
            contrib = contrib.to(dtype=acc.dtype)
        if int(item_rows) == 1:
            if acc.ndim == 1:
                acc[0] += contrib.sum()
            elif int(acc.shape[1]) == 1:
                acc[0, 0] += contrib.sum()
            else:
                acc[0] += contrib.sum(dim=0)
            return
        if acc.ndim == 1:
            acc.index_add_(0, item_rows_for_wave, contrib.sum(dim=1))
        elif int(acc.shape[1]) == 1:
            acc[:, 0].index_add_(0, item_rows_for_wave, contrib.sum(dim=1))
        else:
            acc.index_add_(0, item_rows_for_wave, contrib)

    sp_child1 = species_helpers["sp_child1"]
    sp_child2 = species_helpers["sp_child2"]
    compact_level_ptr = species_helpers["compact_level_ptr"]
    compact_level_parents = species_helpers["compact_level_parents"]
    compact_level_child1 = species_helpers["compact_level_child1"]
    compact_level_child2 = species_helpers["compact_level_child2"]
    leaf_species_idx = wave_layout["leaf_species_index"].to(device=device, dtype=torch.int32).contiguous()

    for meta in reversed(wave_layout["wave_metas"]):
        ws = int(meta["start"])
        W = int(meta["W"])
        rhs_k = accumulated_rhs[ws : ws + W]
        active_mask = active_mask_from_rhs_absmax_fused(
            rhs_k, so.adjoint_pruning_threshold, use_pruning=use_pruning,
        ).contiguous()
        has_splits = bool(meta.get("has_splits", "sl" in meta))
        has_leaf_term = int(meta.get("phase", 1 if not has_splits else 2)) == 1
        dts_r = (
            compute_dts_forward(
                Pi_star_wave.detach(), Pibar_star_wave.detach(), meta["sl"], meta["sr"],
                sp_child1, sp_child2, W, meta["reduce_idx"], log_pD_param, log_pS_param,
                family_idx=family_idx, log_split_probs=meta.get("log_split_probs"),
                n_eq1=meta.get("n_eq1"), eq1_reduce_idx=meta.get("eq1_reduce_idx"),
                ge2_ptr=meta.get("ge2_ptr"), ge2_parent_ids=meta.get("ge2_parent_ids"),
                ge2_max_fanout=meta.get("ge2_max_fanout"), active_parent_rows=active_mask,
                family_offset=ws,
            )
            if has_splits else None
        )
        v_k, aw0, aw1, aw2, aw345, aw3, aw4 = wave_backward_uniform_fused(
            Pi_star_wave, Pibar_star_wave, ws, W, S, dts_r, rhs_k, max_transfer_item,
            DL_item, Ebar_item, E_item, SL1_item, SL2_item, receiver_log_probs,
            sp_child1, sp_child2, None, neumann_terms=neumann_terms,
            leaf_species_idx=leaf_species_idx, leaf_logp=log_pS_item, has_leaf_term=has_leaf_term,
            active_mask=active_mask, sp_parent=species_helpers["sp_parent"],
            max_ancestor_depth=int(species_helpers["max_ancestor_depth"]),
            pibar_row_max=uniform_pibar_row_max, family_idx=family_idx, family_indexed_consts=True,
            compact_level_ptr=species_helpers["compact_level_ptr"],
            compact_level_parents=compact_level_parents, compact_level_child1=compact_level_child1,
            compact_level_child2=compact_level_child2, grad_receiver_log_probs=grad_receiver_log_probs,
            use_receiver_weights=use_receiver_weights, self_loop_solver=self_loop_solver,
            return_last_increment=False,
        )
        if cache is not None:
            # per-wave adjoint state for the exact-HVP tangent sweep (theta fixed across CG).
            # With pruning on, the precompute kernel skips zeroing v_k rows for inactive rows
            # (uninitialized memory); the primal never reads them, but the second-order
            # contraction reads all rows -> sanitize with the row mask.
            row_active = (active_mask.reshape(W, -1) != 0).any(dim=1)
            v_clean = torch.where(row_active.unsqueeze(1), v_k, torch.zeros_like(v_k))
            cache.setdefault("waves", []).append(dict(
                ws=ws, W=W, v=v_clean, dts_r=dts_r, active_mask=active_mask,
                has_splits=has_splits, has_leaf_term=has_leaf_term, meta=meta,
            ))
        item_rows_for_wave = family_idx[ws : ws + W]
        _scatter_accum(grad_log_pD, item_rows_for_wave, aw0)
        _scatter_accum(grad_log_pS, item_rows_for_wave, aw345)
        _scatter_accum(grad_E_acc, item_rows_for_wave, aw0 + aw2)
        _scatter_accum(grad_Ebar_acc, item_rows_for_wave, aw1)
        _scatter_accum(grad_E_s1_acc, item_rows_for_wave, aw4)
        _scatter_accum(grad_E_s2_acc, item_rows_for_wave, aw3)
        _scatter_accum(grad_max_transfer_mat, item_rows_for_wave, aw2)
        if has_splits and dts_r is not None:
            sl, sr = meta["sl"], meta["sr"]
            grad_Pibar_l, grad_Pibar_r, pibar_side_active, _pD, _pS = dts_cross_backward_accum_fused(
                Pi_star_wave, Pibar_star_wave, v_k, ws, sl, sr, meta["reduce_idx"],
                meta.get("log_split_probs", sl.new_zeros((int(sl.numel()),), dtype=Pi_star_wave.dtype)),
                log_pD_param, log_pS_param, sp_child1, sp_child2, accumulated_rhs, S,
                active_mask=active_mask, merge_s_term=True, grad_log_pD=grad_log_pD,
                grad_log_pS=grad_log_pS, grad_mt=grad_max_transfer_mat, accum_param_reductions=True,
                accum_mt_reduction=True, output_pibar_ud=True, output_pibar_side_active=True,
                pibar_side_threshold=so.pibar_side_threshold, mt_squeezed=max_transfer_item,
                pibar_row_max=uniform_pibar_row_max,
                grad_mt_two_stage=bool(grad_max_transfer_mat.ndim == 2 and int(grad_max_transfer_mat.shape[0]) == 1),
                grad_mt_two_stage_tile_splits=128, skip_inactive_pibar_output_zero=True, family_idx=family_idx,
            )
            uniform_cross_pibar_vjp_tree_from_ud_fused(
                Pi_star_wave, receiver_log_probs, grad_Pibar_l, grad_Pibar_r, sl, sr, accumulated_rhs, S,
                active_mask=active_mask, reduce_idx=meta["reduce_idx"], pibar_row_max=uniform_pibar_row_max,
                skip_zero_sides=True, side_active=pibar_side_active, compact_level_ptr=compact_level_ptr,
                compact_level_parents=compact_level_parents, compact_level_child1=compact_level_child1,
                compact_level_child2=compact_level_child2, grad_receiver_log_probs=grad_receiver_log_probs,
                use_receiver_weights=use_receiver_weights, side_active_threshold=so.pibar_side_threshold,
            )

    if cache is not None:
        cache["accum"] = dict(
            grad_E=grad_E_acc, grad_Ebar=grad_Ebar_acc, grad_E_s1=grad_E_s1_acc,
            grad_E_s2=grad_E_s2_acc, grad_log_pD=grad_log_pD, grad_log_pS=grad_log_pS,
            grad_mc=grad_max_transfer_mat, grad_col=grad_receiver_log_probs,
        )
    return _e_adjoint_and_theta_vjp(
        E_star, log_pS, log_pD, log_pL, max_transfer_mat, receiver_log_probs, use_receiver_weights,
        grad_E_acc, grad_Ebar_acc, grad_E_s1_acc, grad_E_s2_acc,
        grad_log_pD, grad_log_pS, grad_max_transfer_mat, grad_receiver_log_probs,
        int(root_ids.numel()), theta, receiver_weights, species_helpers,
        specieswise=specieswise, genewise=genewise, drop_norm=drop_norm,
        bicgstab_max_iter=so.bicgstab_max_iter,
        bicgstab_tol=(so.bicgstab_tol if bicgstab_tol is None else bicgstab_tol),
        bicgstab_breakdown_tol=so.bicgstab_breakdown_tol,
        cache=cache,
        origination_probs=origination_probs,
    )


def _e_adjoint_and_theta_vjp(
    E_star, log_pS, log_pD, log_pL, max_transfer_mat, receiver_log_probs, use_receiver_weights,
    grad_E, grad_Ebar, grad_E_s1, grad_E_s2, grad_log_pD, grad_log_pS, grad_max_transfer_mat,
    grad_receiver_log_probs, n_fam, theta, receiver_weights, species_helpers, *, specieswise, genewise,
    drop_norm, bicgstab_max_iter=500, bicgstab_tol=None, bicgstab_breakdown_tol=None,
    cache=None, origination_probs=None,
):
    topology_args = (
        species_helpers["sp_parent"], species_helpers["sp_child1"], species_helpers["sp_child2"],
        int(species_helpers["max_ancestor_depth"]),
    )
    E_req = E_star.detach().requires_grad_(True)
    with torch.enable_grad():
        triton_E_from_E, E_s1_from_E, E_s2_from_E, Ebar_from_E = e_step_triton_autograd(
            E_req, log_pS, log_pD, log_pL, max_transfer_mat, receiver_log_probs, *topology_args,
            use_receiver_weights=use_receiver_weights,
        )
        aux_outputs = (E_s1_from_E, E_s2_from_E, Ebar_from_E)
        aux_grads = (grad_E_s1, grad_E_s2, grad_Ebar)
        if not drop_norm:
            denom = _log2_survival(E_req, origination_probs)
            direct_obj = denom.sum() if E_req.shape[0] == n_fam else (n_fam * denom).sum()
            aux_outputs = (direct_obj, *aux_outputs)
            aux_grads = (torch.ones_like(direct_obj), *aux_grads)
        (aux_to_e,) = torch.autograd.grad(aux_outputs, E_req, grad_outputs=aux_grads, retain_graph=True)
    q_E = grad_E + aux_to_e
    E_shape = E_star.shape
    q_flat = q_E.reshape(-1)

    def AG_flat(w_flat):
        wE = w_flat.view(E_shape).contiguous()
        with torch.enable_grad():
            (gE,) = torch.autograd.grad(triton_E_from_E, E_req, grad_outputs=wE.clone(), retain_graph=True)
        return (wE - gE).reshape(-1)

    wE = _bicgstab(AG_flat, q_flat, max_iter=bicgstab_max_iter, tol=bicgstab_tol,
                   breakdown_tol=bicgstab_breakdown_tol).view(E_shape)
    if cache is not None:
        cache["e_side"] = dict(q_E=q_E, wE=wE, aux_to_e=aux_to_e)

    theta_req = theta.detach().requires_grad_(True)
    col_req = receiver_weights.detach().requires_grad_(True)
    with torch.enable_grad():
        log_pS_r, log_pD_r, log_pL_r, mt_r, receiver_log_probs_r = extract_parameters_weighted_receivers(
            theta_req, col_req, species_helpers, specieswise=specieswise, genewise=genewise,
            uniform_fast=not use_receiver_weights,
        )
        S = int(species_helpers["S"])
        item_rows = int(E_star.shape[0])
        log_pS_param = as_family_param(log_pS_r, item_rows, S)
        log_pD_param = as_family_param(log_pD_r, item_rows, S)
        param_loss = (
            (log_pS_param * grad_log_pS).sum() + (log_pD_param * grad_log_pD).sum()
            + (mt_r * grad_max_transfer_mat).sum() + (receiver_log_probs_r * grad_receiver_log_probs).sum()
        )
        E_from_params, _, _, Ebar_from_params = e_step_triton_autograd(
            E_star.detach(), log_pS_r, log_pD_r, log_pL_r, mt_r, receiver_log_probs_r, *topology_args,
            use_receiver_weights=use_receiver_weights,
        )
        grad_theta, grad_col = torch.autograd.grad(
            (param_loss, Ebar_from_params, E_from_params), (theta_req, col_req),
            grad_outputs=(torch.ones_like(param_loss), grad_Ebar, wE),
        )
    return grad_theta, grad_col


def make_ggn_hvp(static, theta, receiver_weights, sv, *, self_tol=None, self_max_iter=200,
                 vjp_neumann_terms=None, vjp_use_pruning=None, vjp_bicgstab_tol=None):
    """Return hvp(v_vec) computing the GGN/Fisher product M v in theta-space (flat 3S).

    Defaults use the solver's production adjoint settings (neumann_terms, pruning, bicgstab tol);
    the wave self-loop already converges within those terms, so M is unchanged vs the convergent
    settings (pass overrides only to force a convergent/unpruned adjoint for symmetry checks).
    """
    S = int(static.species_helpers["S"])
    root_ids = static.wave_layout["root_clade_ids"]
    root_Pi = sv["pi_wave"].index_select(0, root_ids)
    root_lse = _logsumexp2(root_Pi, dim=-1, keepdim=True)
    q = _safe_exp2_ratio(root_Pi, root_lse)  # posterior softmax2 per root row

    def hvp(v_vec):
        v = v_vec.reshape(S, 3).to(theta.dtype)
        t = jvp_root_scores(static, theta, v, sv, self_tol=self_tol, self_max_iter=self_max_iter)
        u = _LN2 * q * (t - (q * t).sum(dim=-1, keepdim=True))  # B t  (PSD Fisher covariance)
        gt, _gc = vjp_root_to_theta(static, sv, u, theta, receiver_weights, drop_norm=True,
                                    neumann_terms=vjp_neumann_terms, use_pruning=vjp_use_pruning,
                                    bicgstab_tol=vjp_bicgstab_tol)
        return gt.reshape(-1)

    return hvp
