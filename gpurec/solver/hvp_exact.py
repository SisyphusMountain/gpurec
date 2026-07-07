"""Analytic exact-Hessian HVP (forward-over-reverse) — orchestrator.

Per outer Newton point: ``build_point_cache`` runs the production backward ONCE (the verified
``vjp_root_to_theta`` loop) while caching per-wave adjoints ``v_k``/``dts_r`` and the E-side
adjoint ``wE`` — theta is fixed across all CG iterations, so the cache amortizes. Each
``hvp(u)`` then costs one tangent-forward sweep + one tangent-adjoint sweep (same solve
operators, modified seeds) + the second-order contraction kernels (e_step_so / wave_so / dts_so).

Status: point-cache + gradient reproduction (build step 2). The tangent-adjoint sweep
(steps 3-5) composes on top of this cache.
"""

from __future__ import annotations

import torch

from gpurec.api._implicit_grad import _bicgstab, _safe_exp2_ratio
from gpurec.core.inference.logspace import logsumexp2 as _logsumexp2, survival_from_E as _survival_from_E
from gpurec.core.inference.solver import receiver_weights_are_uniform
from gpurec.core.kernels.dts_so import dts_backward_so
from gpurec.core.kernels.e_step import e_step_triton_autograd
from gpurec.core.kernels.e_step_so import e_step_backward_so
from gpurec.core.kernels.wave_backward import (
    dts_cross_backward_accum_fused, uniform_cross_pibar_vjp_tree_from_ud_fused,
    wave_backward_uniform_fused,
)
from gpurec.core.kernels.wave_so import wave_backward_so
from gpurec.core.parameters.extract_parameters import (
    as_family_param, as_family_species, extract_parameters_weighted_receivers,
)
from gpurec.solver.forward_tangent import jvp_root_scores, wave_step_constants
from gpurec.solver.ggn import vjp_root_to_theta

_LN2 = 0.6931471805599453


def _single_static(static):
    """The exact HVP is single-batch; accept either a single static or a ``batch_statics`` list."""
    if isinstance(static, (list, tuple)):
        if len(static) != 1:
            raise NotImplementedError(
                "exact HVP is single-batch; multi-batch HVP is not ported (Phase 3 is single-batch)"
            )
        return static[0]
    return static


@torch.no_grad()
def build_point_cache(static, theta, col_weights, sv, *, origination_log_probs=None,
                      origination_probs=None):
    """Run the production-configured backward once, caching per-wave (v_k, dts_r, active_mask)
    and the E-side adjoint. Returns (grad_theta, grad_col, cache)."""
    static = _single_static(static)
    cache: dict = {}
    grad_theta, grad_col = vjp_root_to_theta(
        static, sv, None, theta, col_weights, drop_norm=False, cache=cache,
        origination_log_probs=origination_log_probs, origination_probs=origination_probs,
    )
    return grad_theta, grad_col, cache


def _scatter_accum(acc, item_rows_for_wave, contrib, item_rows):
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


def _head_seed_tangents(root_Pi, E_star, omega, t_root, dE, u_omega, dtype):
    """Origination-weighted head-HVP seed tangents (double-backward on the small NLL aggregation head
    ``h = sum_fam nll(root_rows, E, omega)``; correct by construction and consistent with the kernels'
    seed convention -- the hand-coded uniform ``d_seed`` equals ``ds_root`` here at uniform omega).

    Returns ``(ds_root, ds_E_survival, Hv_omega)``:
      ds_root        = d(dnll/droot) along (t_root, u_omega)  -> root-row HVP seed (with the omega cross-term),
      ds_E_survival  = d(dnll/dE)   along (dE, u_omega)       -> survival E-seed tangent (replaces dg_norm),
      Hv_omega       = d(dnll/domega) along (t_root, dE, u_omega) -> the omega row of ``H u``.
    """
    from gpurec.core.inference.solver import nll_vector_from_root_rows
    from gpurec.core.parameters.extract_parameters import origination_log_probs_from_weights

    rP = root_Pi.detach().requires_grad_(True)
    Ev = E_star.detach().requires_grad_(True)
    om = omega.detach().to(device=rP.device, dtype=rP.dtype).requires_grad_(True)
    with torch.enable_grad():
        olp = origination_log_probs_from_weights(om)
        op = torch.exp2(olp)
        nll = nll_vector_from_root_rows(rP, Ev, origination_log_probs=olp, origination_probs=op).sum()
        s_root, s_E, s_om = torch.autograd.grad(nll, (rP, Ev, om), create_graph=True)
        inner = ((s_root * t_root).sum() + (s_E * dE).sum() + (s_om * u_omega.to(s_om.dtype)).sum())
        ds_root, ds_E, Hv_om = torch.autograd.grad(inner, (rP, Ev, om))
    return ds_root.to(dtype), ds_E.to(dtype), Hv_om.to(dtype)


def make_exact_hvp(static, theta, col_weights, sv, *, cache=None, debug_out=None,
                   tangent_self_iters=None, origination_log_probs=None, origination_probs=None,
                   origination_weights=None):
    """Analytic exact-Hessian HVP. Builds the per-point adjoint cache once (if not given) and
    returns ``hvp(u_vec) -> H u`` (flat 3S). Runs in the dtype of ``theta``/``sv``.

    ``tangent_self_iters`` sets the FIXED per-wave self-loop iteration count for the tangent
    forward sweep (sync-free; see ``jvp_root_scores``). Resolution order: this argument, then
    the ``NEWTON_TANGENT_SELF_ITERS`` env var, then ``solver_options.pi_iters`` (matching the
    primal forward truncation). Not hardcoded — change it per run via the env var or the arg.
    """
    import os

    static = _single_static(static)
    so = static.solver_options
    if tangent_self_iters is None:
        _env = os.environ.get("NEWTON_TANGENT_SELF_ITERS")
        if _env:
            tangent_self_iters = int(_env)
        elif getattr(so, "pi_iters", None):
            # Match the primal forward's truncation so the HVP is the exact
            # Hessian of the *truncated* objective.
            tangent_self_iters = int(so.pi_iters)
        else:
            # No primal pi_iters to match (ad-hoc caller). 16 is documented as
            # non-convergent on the representative fixture (+33 NLL, ~2.6x grad
            # bias), so fall back to the validated floor of 64 and warn rather
            # than silently returning a wrong curvature.
            import warnings
            tangent_self_iters = 64
            warnings.warn(
                "hvp_exact: no tangent_self_iters / NEWTON_TANGENT_SELF_ITERS / "
                "solver_options.pi_iters provided; defaulting to 64 (the validated "
                "convergence floor). Pass tangent_self_iters to match your primal "
                "forward truncation.",
                RuntimeWarning, stacklevel=2,
            )
    # How often to run free_cuda_cache_if_tight() in the reverse sweep. It is a blocking
    # cudaMemGetInfo round-trip (~7.9us) and on a large-free GPU never empties the pool, so
    # firing once per wave is ~142 wasted driver calls/HVP. Gate it to every K waves; it stays
    # load-bearing on the big fixtures (memory pressure builds gradually, so checking every K
    # waves still trips the gate in time). 1 = every wave (old behaviour); change per run via env.
    _fc_env = os.environ.get("NEWTON_FREE_CACHE_EVERY")
    free_cache_every = int(_fc_env) if _fc_env else 32
    free_cache_every = max(1, free_cache_every)
    sh, wl = static.species_helpers, static.wave_layout
    S = int(sh["S"])
    # S7: derive the weighted-receiver flag from the base alpha (kill the False hardcodes).
    # uniform base -> uniform_fast path (the legacy theta-only behaviour, bit-for-bit);
    # non-uniform base -> weighted paths LIVE so the backward/cache + col-cotangent are finite.
    use_receiver_weights = not receiver_weights_are_uniform(col_weights)
    use_col_weights = use_receiver_weights
    item_idx = static.rate_family_idx
    c1, c2, parent = sh["sp_child1"], sh["sp_child2"], sh["sp_parent"]
    mad = int(sh["max_ancestor_depth"])
    leaf_state_idx = wl["leaf_species_index"].to(device=theta.device, dtype=torch.int32).contiguous()
    root_ids = wl["root_clade_ids"]
    n_fam = int(root_ids.numel())
    dtype = sv["pi_wave"].dtype
    C = int(sv["pi_wave"].shape[0])
    E_star = sv["E"]
    G = int(E_star.shape[0])

    # Turn ON the origination head whenever origination_weights are supplied (even UNIFORM omega=0):
    # the omega curvature at uniform omega is nonzero and is exactly what the joint gate must capture.
    # Derive the log-probs from the weights (shape-generic: [S] specieswise/global, [G,S] genewise) so
    # build_point_cache runs the same weighted head as the tangent sweep -- at uniform omega this is
    # numerically identical to the default uniform forward, so the point cache stays consistent; we are
    # only enabling the head double-backward that produces Hv_omega.
    if origination_weights is not None and origination_log_probs is None:
        from gpurec.core.parameters.extract_parameters import origination_log_probs_from_weights
        origination_log_probs = origination_log_probs_from_weights(origination_weights)
        origination_probs = torch.exp2(origination_log_probs)

    if cache is None:
        _, _, cache = build_point_cache(static, theta, col_weights, sv,
                                        origination_log_probs=origination_log_probs,
                                        origination_probs=origination_probs)
    acc = cache["accum"]
    wE = cache["e_side"]["wE"]

    cst = wave_step_constants(sv, S)
    prm = sv["pibar_row_max"]
    col = sv["receiver_log_probs"]
    item = lambda t: as_family_species(t, S, G)
    pS_m, pD_m, pL_m = item(sv["log_pS"]), item(sv["log_pD"]), item(sv["log_pL"])

    # e-step autograd graph at (E*, P): reused for all linear-in-cotangent transposed products
    E_req = E_star.detach().requires_grad_(True)
    with torch.enable_grad():
        E_new_g, E_s1_g, E_s2_g, Ebar_g = e_step_triton_autograd(
            E_req, sv["log_pS"], sv["log_pD"], sv["log_pL"], sv["max_transfer"], col,
            parent, c1, c2, mad, use_receiver_weights=use_receiver_weights,
        )

    def jt_E(g_new):
        with torch.enable_grad():
            (out,) = torch.autograd.grad(E_new_g, E_req, grad_outputs=g_new.contiguous(),
                                         retain_graph=True)
        return out

    def aux_T(g_s1, g_s2, g_ebar):
        with torch.enable_grad():
            (out,) = torch.autograd.grad((E_s1_g, E_s2_g, Ebar_g), E_req,
                                         grad_outputs=(g_s1, g_s2, g_ebar), retain_graph=True)
        return out

    norm = _survival_from_E(E_star, keepdim=True)  # uniform survival; consumed only in the origination-uniform tangent below
    fam_factor = 1.0 if G == n_fam else float(n_fam)

    zeros_state = lambda: torch.zeros_like(E_star)

    # e-step head VJP at fixed cotangents (used both for the u-independent primal cotangents
    # below and, per-u, for the tangent cotangents with g_new=dwE)
    def e_bwd_params(g_new, g_ebar):
        with torch.enable_grad():
            pS_r = pS_m.detach().requires_grad_(True)
            pD_r = pD_m.detach().requires_grad_(True)
            pL_r = pL_m.detach().requires_grad_(True)
            mc_r = item(sv["max_transfer"].squeeze(-1)).detach().requires_grad_(True)
            col_r = col.detach().requires_grad_(True)
            En, _, _, Eb = e_step_triton_autograd(
                E_star.detach(), pS_r, pD_r, pL_r, mc_r, col_r,
                parent, c1, c2, mad, use_receiver_weights=use_receiver_weights,
            )
            outs = torch.autograd.grad((En, Eb), (pS_r, pD_r, pL_r, mc_r, col_r),
                                       grad_outputs=(g_new, g_ebar), allow_unused=True)
        return tuple(torch.zeros_like(z) if o is None else o
                     for o, z in zip(outs, (pS_m, pD_m, pL_m, pS_m, col)))

    # ---- u-INDEPENDENT setup (theta fixed across all CG iterations): primal cotangents and the
    # smooth head graph + first-order grad g1 are built ONCE here, not per hvp(u). The head's
    # forward graph is retained (create_graph) so each hvp(u) only adds phi2 + one backward. ----
    base_p = e_bwd_params(wE, acc["grad_Ebar"])
    # Genewise: log_pS/log_pD are per-family scalars broadcast across all S species, so their e-step
    # cotangent base_p[.] ([G,S]) must be summed to per-family [G,1] BEFORE adding the already-per-family
    # DTS cotangent acc[...] ([G,1]) -- otherwise the [G,1] term broadcasts over S and the head
    # contraction (pS_hp[G,1] * cot_pS).sum(), which sums the species axis, multiplies it by S (the ~Sx
    # HVP bug). Specieswise/global keep the per-species form BIT-FOR-BIT (there S IS the parameter axis;
    # everything is [1,S]). pL has no acc term and mc's acc grad_mc is genuinely per-species [G,S], so
    # both are correct unchanged in every mode.
    if static.genewise:
        cot_pS = acc["grad_log_pS"] + base_p[0].sum(dim=-1, keepdim=True)
        cot_pD = acc["grad_log_pD"] + base_p[1].sum(dim=-1, keepdim=True)
    else:
        cot_pS = acc["grad_log_pS"] + as_family_param(base_p[0], G, S)
        cot_pD = acc["grad_log_pD"] + as_family_param(base_p[1], G, S)
    cot_pL = base_p[2]
    cot_mc = acc["grad_mc"] + base_p[3]
    cot_col = acc["grad_col"] + base_p[4]
    theta_req = theta.detach().requires_grad_(True)
    col_req = col_weights.detach().requires_grad_(True)
    _head_grad_ctx = torch.enable_grad()
    _head_grad_ctx.__enter__()
    pS_h, pD_h, pL_h, mt_h, col_h = extract_parameters_weighted_receivers(
        theta_req, col_req, sh, specieswise=static.specieswise, genewise=static.genewise,
        uniform_fast=not use_receiver_weights,
    )
    pS_hp = as_family_param(pS_h, G, S)
    pD_hp = as_family_param(pD_h, G, S)
    pL_hi = as_family_species(pL_h, S, G)
    mt_hi = as_family_species(mt_h.squeeze(-1) if mt_h.ndim == pS_h.ndim + 1 else mt_h, S, G)
    phi1 = ((pS_hp * cot_pS).sum() + (pD_hp * cot_pD).sum() + (pL_hi * cot_pL).sum()
            + (mt_hi * cot_mc).sum() + (col_h * cot_col).sum())
    # S8: grad phi1 w.r.t. BOTH (theta_req, col_req). g1_col carries the col-row of the first-order
    # head VJP (through col_h(col_req) = log_softmax and mt_h(col_req) via receiver_norm). Autograd
    # returns each partial independently, so g1_theta is bit-for-bit the legacy single-target grad
    # (the u_alpha=0 regression is preserved). create_graph so the second grad below can pass
    # through the softmax/receiver_norm curvature ONCE (no double-count with the kernels' dcol-linear
    # cotangent). col_req always requires_grad here, so allow_unused is False.
    g1_theta, g1_col = torch.autograd.grad(phi1, (theta_req, col_req), create_graph=True)
    _head_grad_ctx.__exit__(None, None, None)

    def hvp(u_vec):
        # Joint split: u = [u_theta (theta_numel); u_alpha (S)]. The theta-milestone harness still
        # passes a length-(theta_numel) vector (u_alpha implicitly 0); accept both. theta_shape is
        # explicit (do NOT assume [S,3]).
        u_vec = u_vec.to(theta.dtype)
        theta_numel = theta.numel()
        u = u_vec[:theta_numel].reshape(theta.shape)
        # BASE layout: [u_theta (theta_numel); u_alpha (S)?; u_omega (omega_numel)?]. Alpha BEFORE omega
        # -- the codebase convention (origination_curvature.py consumes z=[theta;alpha;omega]). The omega
        # size is keyed off the origination parameter, so this is uniform across modes: genewise
        # omega_numel=G*S -> u_omega reshapes to [G,S]; specieswise/global omega_numel=S -> [S].
        #   n_tail == 0                 -> theta-only, returns theta_numel BIT-FOR-BIT (_verify_hvp gate);
        #   n_tail == S                 -> [theta; alpha], omega implicitly 0 (_verify_hvp_recv path);
        #   n_tail == S + omega_numel   -> full [theta; alpha; omega].
        omega_numel = origination_weights.numel() if origination_weights is not None else 0
        n_tail = u_vec.numel() - theta_numel
        has_omega = (omega_numel > 0 and n_tail == S + omega_numel)
        joint = n_tail >= S
        if has_omega:
            u_alpha = u_vec[theta_numel:theta_numel + S].contiguous()
            u_omega = u_vec[theta_numel + S:theta_numel + S + omega_numel].reshape(origination_weights.shape)
        else:
            u_alpha = (u_vec[theta_numel:theta_numel + S].contiguous() if joint
                       else torch.zeros(S, device=theta.device, dtype=theta.dtype))
            # u_omega is only consumed when the origination head is active; keep its shape matched to
            # origination_weights ([G,S] genewise / [S] otherwise) so _head_seed_tangents contracts cleanly.
            u_omega = (torch.zeros_like(origination_weights) if origination_weights is not None
                       else torch.zeros(S, device=theta.device, dtype=theta.dtype))
        with torch.no_grad():
            # S3/S7: at a NON-UNIFORM base the tangent forward MUST go through the weighted path
            # (param_jvp_weighted + use_col_weights), consistent with the weighted primal fixed
            # point, or the tangent E-adjoint diverges (1e18 / NaN). u_alpha=0 then gives dcol=0 ->
            # the pure-theta tangent at the non-uniform base. At a UNIFORM base keep alpha=None so
            # the legacy uniform theta-only tangent is reproduced BIT-FOR-BIT (regression guard).
            _alpha = col_weights if use_receiver_weights else None
            _u_alpha = u_alpha if use_receiver_weights else None
            t_root, full = jvp_root_scores(static, theta, u, sv, return_full=True,
                                           keep_d_dts=False, self_iters=tangent_self_iters,
                                           alpha=_alpha, u_alpha=_u_alpha)
            dcst = full["dcst"]
            dPi, dPibar = full["dPi"], full["dPibar"]
            dpS_m, dpD_m, dpL_m = item(full["dlog_pS"]), item(full["dlog_pD"]), item(full["dlog_pL"])
            dmc_m = item(full["dmax_coupling"].squeeze(-1))
            # S4: the alpha (col) tangent seed = softmax-Jacobian . u_alpha, exposed by S3's
            # weighted jvp. At a uniform base the weighted path is off and there is no dcol key
            # -> dcol=None (the e_step_backward_so dcol slot then zero-fills; bit-for-bit legacy).
            dcol = full.get("dreceiver_log_probs") if use_col_weights else None
            dE, dEbar_e = full["dE"], full["dEbar"]
            dE_s1, dE_s2 = full["dE_s1"], full["dE_s2"]

            # tangent of the loss seed -q on root rows. Uniform origination: hand-coded (bit-for-bit
            # legacy). Weighted origination: the autograd head gives the weighted root-seed tangent
            # (incl. the omega cross-term), the survival E-seed tangent, and the omega row of H u.
            root_Pi = sv["pi_wave"].index_select(0, root_ids)
            Hv_omega = ds_E_surv = None
            if origination_log_probs is None:
                q = _safe_exp2_ratio(root_Pi, _logsumexp2(root_Pi, dim=-1, keepdim=True))
                d_seed = -_LN2 * q * (t_root - (q * t_root).sum(dim=-1, keepdim=True))
            else:
                d_seed, ds_E_surv, Hv_omega = _head_seed_tangents(
                    root_Pi, E_star, origination_weights, t_root, dE, u_omega, dtype)

            d_rhs = torch.zeros((C, S), device=theta.device, dtype=dtype)
            d_rhs.index_copy_(0, root_ids, d_seed.to(dtype))

            d_gpD = torch.zeros_like(acc["grad_log_pD"])
            d_gpS = torch.zeros_like(acc["grad_log_pS"])
            d_gE, d_gEbar, d_gEs1, d_gEs2 = (zeros_state() for _ in range(4))
            d_gmc = torch.zeros_like(acc["grad_mc"])
            d_gcol = torch.zeros((S,), device=theta.device, dtype=dtype)

            from gpurec.solver.value_and_grad import free_cuda_cache_if_tight

            for _wi, wave in enumerate(cache["waves"]):  # already reverse order
                if _wi % free_cache_every == 0:
                    free_cuda_cache_if_tight()
                ws, W = wave["ws"], wave["W"]
                meta = wave["meta"]
                v_k = wave["v"]
                dts_r = wave["dts_r"]
                # recompute d_dts per wave from the cached (pruned) dts_r: storing all of them
                # would cost another Pi-sized buffer; one tangent launch per wave is cheap
                if dts_r is not None:
                    from gpurec.core.kernels.dts_tangent import compute_dts_tangent
                    d_dts = compute_dts_tangent(
                        sv["pi_wave"], sv["pibar_wave"], dPi, dPibar, meta["sl"], meta["sr"],
                        c1, c2, W, meta["reduce_idx"], cst["pd_param"], cst["ps_param"],
                        dcst["dpd_param"], dcst["dps_param"], dts_r, item_idx,
                        log_split_probs=meta.get("log_split_probs"), n_eq1=meta.get("n_eq1"),
                        eq1_reduce_idx=meta.get("eq1_reduce_idx"), ge2_ptr=meta.get("ge2_ptr"),
                        ge2_parent_ids=meta.get("ge2_parent_ids"),
                        ge2_max_fanout=meta.get("ge2_max_fanout"), item_offset=ws,
                    )
                else:
                    d_dts = None
                has_leaf = wave["has_leaf_term"]
                # (a) second-order contraction at fixed v_k; d_rhs folds the wave's rhs cotangent
                # into d_Av so it IS the solve seed (= d_rhs[ws:ws+W] + d_Av), no host add
                d_Av, c_aw0, c_aw1, c_aw2, c_aw345, c_aw3, c_aw4, c_gcol = wave_backward_so(
                    sv["pi_wave"], dPi, sv["pibar_wave"], dPibar, v_k, ws, W, S,
                    prm, cst["mc"], cst["dl"], dcst["dDL"], cst["ebar"], dcst["dEbar"],
                    cst["e"], dcst["dE"], cst["sl1"], dcst["dSL1"], cst["sl2"], dcst["dSL2"],
                    col, c1, c2, parent, mad, dts_r, d_dts,
                    leaf_state_idx=leaf_state_idx, leaf_logp=cst["leaf"], dleaf_logp=dcst["dleaf"],
                    item_idx=item_idx, has_leaf_term=has_leaf, use_col_weights=use_col_weights,
                    d_rhs=d_rhs, dcol=dcol,
                )
                # S5: accumulate the wave-SO col-cotangent (tangent of the wave self-loop
                # receiver-grad). Zero when use_col_weights is off -> bit-for-bit legacy.
                if use_col_weights:
                    d_gcol = d_gcol + c_gcol
                # (b) tangent-adjoint solve with the SAME operator and cached mask
                seed = d_Av
                dv, l_aw0, l_aw1, l_aw2, l_aw345, l_aw3, l_aw4 = wave_backward_uniform_fused(
                    sv["pi_wave"], sv["pibar_wave"], ws, W, S, dts_r, seed, cst["mc"],
                    cst["dl"], cst["ebar"], cst["e"], cst["sl1"], cst["sl2"], col,
                    c1, c2, None, neumann_terms=int(so.neumann_terms),
                    leaf_species_idx=leaf_state_idx, leaf_logp=cst["leaf"], has_leaf_term=has_leaf,
                    active_mask=wave["active_mask"], sp_parent=parent, max_ancestor_depth=mad,
                    pibar_row_max=prm, family_idx=item_idx, family_indexed_consts=True,
                    compact_level_ptr=sh["compact_level_ptr"],
                    compact_level_parents=sh["compact_level_parents"],
                    compact_level_child1=sh["compact_level_child1"],
                    compact_level_child2=sh["compact_level_child2"],
                    grad_receiver_log_probs=d_gcol, use_receiver_weights=use_receiver_weights,
                    self_loop_solver=so.self_loop_solver, return_last_increment=False,
                )
                aw0 = c_aw0 + l_aw0
                aw1 = c_aw1 + l_aw1
                aw2 = c_aw2 + l_aw2
                aw345 = c_aw345 + l_aw345
                aw3 = c_aw3 + l_aw3
                aw4 = c_aw4 + l_aw4
                if debug_out is not None:
                    debug_out.setdefault("wave_trace", []).append(
                        (ws, float(d_Av.abs().max()), float(dv.abs().max()),
                         float(d_rhs.abs().max())))
                rows_i = item_idx[ws:ws + W]
                _scatter_accum(d_gpD, rows_i, aw0, G)
                _scatter_accum(d_gpS, rows_i, aw345, G)
                _scatter_accum(d_gE, rows_i, aw0 + aw2, G)
                _scatter_accum(d_gEbar, rows_i, aw1, G)
                _scatter_accum(d_gEs1, rows_i, aw4, G)
                _scatter_accum(d_gEs2, rows_i, aw3, G)
                _scatter_accum(d_gmc, rows_i, aw2, G)
                if dts_r is not None:
                    # C^T dv via the frozen kernels (linear in v)
                    gl, gr, side_act, _p1, _p2 = dts_cross_backward_accum_fused(
                        sv["pi_wave"], sv["pibar_wave"], dv, ws, meta["sl"], meta["sr"],
                        meta["reduce_idx"],
                        meta.get("log_split_probs", meta["sl"].new_zeros((int(meta["sl"].numel()),), dtype=dtype)),
                        cst["pd_param"], cst["ps_param"], c1, c2, d_rhs, S,
                        active_mask=wave["active_mask"], merge_s_term=True,
                        grad_log_pD=d_gpD, grad_log_pS=d_gpS, grad_mt=d_gmc,
                        accum_param_reductions=True, accum_mt_reduction=True,
                        output_pibar_ud=True, output_pibar_side_active=True,
                        pibar_side_threshold=so.pibar_side_threshold, mt_squeezed=cst["mc"],
                        pibar_row_max=prm,
                        grad_mt_two_stage=bool(d_gmc.ndim == 2 and int(d_gmc.shape[0]) == 1),
                        grad_mt_two_stage_tile_splits=128, skip_inactive_pibar_output_zero=True,
                        family_idx=item_idx,
                    )
                    uniform_cross_pibar_vjp_tree_from_ud_fused(
                        sv["pi_wave"], col, gl, gr, meta["sl"], meta["sr"], d_rhs, S,
                        active_mask=wave["active_mask"], reduce_idx=meta["reduce_idx"],
                        pibar_row_max=prm, skip_zero_sides=True, side_active=side_act,
                        compact_level_ptr=sh["compact_level_ptr"],
                        compact_level_parents=sh["compact_level_parents"],
                        compact_level_child1=sh["compact_level_child1"],
                        compact_level_child2=sh["compact_level_child2"],
                        grad_receiver_log_probs=d_gcol, use_receiver_weights=use_receiver_weights,
                        side_active_threshold=so.pibar_side_threshold,
                    )
                    # d(C^T) v_k contraction at fixed v_k
                    dts_backward_so(
                        sv["pi_wave"], dPi, sv["pibar_wave"], dPibar, v_k, ws, meta, S,
                        cst["pd_param"], cst["ps_param"], dcst["dpd_param"], dcst["dps_param"],
                        cst["mc"], dcst["dMC"], col, c1, c2, parent, mad, prm, item_idx,
                        d_rhs, d_gpD, d_gpS, d_gmc, d_gcol,
                        compact_level_ptr=sh["compact_level_ptr"],
                        compact_level_parents=sh["compact_level_parents"],
                        compact_level_child1=sh["compact_level_child1"],
                        compact_level_child2=sh["compact_level_child2"],
                        use_col_weights=use_col_weights, dcol=dcol,
                    )

            # ---- E-side ---- (the big tangent buffers are no longer needed)
            del dPi, dPibar
            full.clear()
            free_cuda_cache_if_tight()
            x_args = (E_star.contiguous(), E_star.contiguous(), sv["E_s1"], sv["E_s2"],
                      sv["Ebar"], pS_m, pD_m, pL_m, col.contiguous(),
                      parent, c1, c2, mad)
            # S4: the 9th dx slot is dcol (e_step_backward_so signature ...,dlog_pL, dcol). Was
            # hardcoded None (col tangent dead); thread the S3 seed so the e-step SO contraction
            # carries the alpha->E col dependence. None at a uniform base (legacy bit-for-bit).
            dx = (dE, dE, dE_s1, dE_s2, dEbar_e, dpS_m, dpD_m, dpL_m, dcol)
            zero_g = zeros_state()
            # tangent of aux_to_e: linear part + contraction + norm-term closed form
            aux_lin = aux_T(d_gEs1, d_gEs2, d_gEbar)
            so_aux = e_step_backward_so(*x_args, zero_g, acc["grad_E_s1"], acc["grad_E_s2"],
                                        acc["grad_Ebar"], *dx, use_col_weights=use_col_weights)
            if origination_log_probs is None:
                e2E = torch.exp2(E_star)
                dnorm = -_LN2 * (e2E * dE).mean(dim=-1, keepdim=True)
                dg_norm = fam_factor * (-_LN2 * e2E * dE / (S * norm) + e2E * dnorm / (S * norm * norm))
            else:
                dg_norm = ds_E_surv  # weighted + omega-coupled survival tangent (autograd head)
            dq_E = d_gE + aux_lin + so_aux[0] + dg_norm
            # tangent E-adjoint solve: same operator, new rhs
            so_w = e_step_backward_so(*x_args, wE, zero_g, zero_g, zero_g, *dx,
                                      use_col_weights=use_col_weights)
            rhs_E = (dq_E + so_w[0]).reshape(-1)
            E_shape = E_star.shape

            def AG_flat(w_flat):
                gE = jt_E(w_flat.view(E_shape))
                return (w_flat.view(E_shape) - gE).reshape(-1)

            dwE = _bicgstab(AG_flat, rhs_E, max_iter=so.bicgstab_max_iter,
                            tol=so.bicgstab_tol, breakdown_tol=so.bicgstab_breakdown_tol
                            ).view(E_shape)
            if debug_out is not None:
                debug_out.update(
                    d_gE=d_gE.clone(), d_gpD=d_gpD.clone(), d_gpS=d_gpS.clone(),
                    d_gmc=d_gmc.clone(), d_gEbar=d_gEbar.clone(), d_gEs1=d_gEs1.clone(),
                    d_gEs2=d_gEs2.clone(), dq_E=dq_E.clone(), dwE=dwE.clone(),
                    d_gcol=d_gcol.clone(),
                )

            # tangent param-cotangents from the e-step head: linear (tangent cotangents,
            # g_new=dwE) + contraction at fixed cotangents (g_new=wE, g_ebar=grad_Ebar_acc).
            # e_bwd_params and the primal cotangents/head graph are hoisted (u-independent).
            lin_p = e_bwd_params(dwE, d_gEbar)
            so_p = e_step_backward_so(*x_args, wE, zero_g, zero_g, acc["grad_Ebar"], *dx,
                                      use_col_weights=use_col_weights)
            # so_p outputs: (d_grad_E, d_grad_pS, d_grad_pD, d_grad_pL, d_grad_mc, d_grad_col)

            if static.genewise:  # per-family log_pS/pD: sum the per-species tangent cotangents (see phi1 above)
                d_cot_pS = d_gpS + (lin_p[0] + so_p[1]).sum(dim=-1, keepdim=True)
                d_cot_pD = d_gpD + (lin_p[1] + so_p[2]).sum(dim=-1, keepdim=True)
            else:
                d_cot_pS = d_gpS + as_family_param(lin_p[0] + so_p[1], G, S)
                d_cot_pD = d_gpD + as_family_param(lin_p[1] + so_p[2], G, S)
            d_cot_pL = lin_p[2] + so_p[3]
            d_cot_mc = d_gmc + lin_p[3] + so_p[4]
            d_cot_col = d_gcol + lin_p[4] + so_p[5]

        # ---- smooth parameter head (autograd; forward graph + g1 hoisted, retained) ----
        with torch.enable_grad():
            phi2 = ((pS_hp * d_cot_pS).sum() + (pD_hp * d_cot_pD).sum() + (pL_hi * d_cot_pL).sum()
                    + (mt_hi * d_cot_mc).sum() + (col_h * d_cot_col).sum())
            # S8: grad w.r.t. BOTH (theta_req, col_req) of
            #   (g1_theta * u_theta).sum() + (g1_col * u_alpha).sum() + phi2.
            # - out_theta = H_tt u_theta + H_ta u_alpha (theta row of H u).
            # - out_col   = H_at u_theta + H_aa u_alpha (alpha row of H u).
            # The alpha-alpha softmax curvature + receiver_norm->max_transfer curvature are captured
            # HERE ONCE (col_h(col_req)/mt_h(col_req) differentiated twice via g1's create_graph);
            # the kernels carry ONLY the dcol-LINEAR cotangent (d_cot_col), so no double-count.
            # u_theta = u (the reshaped theta tangent); with u_alpha=0 the (g1_col*u_alpha) term
            # vanishes and out_theta is bit-for-bit the legacy grad (regression guard). retain_graph
            # so the hoisted forward graph + g1 survive for the next hvp(u) call.
            if not joint:
                # legacy theta-only contract: EXACT legacy graph -- grad of (g1_theta*u).sum()+phi2
                # w.r.t. theta_req ONLY (no (g1_col*u_alpha) node, no col grad target). g1_theta is
                # autograd's independent partial of phi1, so it equals the old single-target g1
                # bit-for-bit -> out reproduces the theta-only HVP BIT-FOR-BIT (the _verify_hvp
                # uniform gate + the non-uniform milestone depend on this).
                (out_theta,) = torch.autograd.grad((g1_theta * u).sum() + phi2, theta_req,
                                                   retain_graph=True)
                return out_theta.reshape(-1)
            # joint contract: grad of (g1_theta*u_theta).sum()+(g1_col*u_alpha).sum()+phi2 w.r.t.
            # BOTH (theta_req, col_req) -> (out_theta=H_tt u_t+H_ta u_a, out_col=H_at u_t+H_aa u_a).
            head = (g1_theta * u).sum() + (g1_col * u_alpha).sum() + phi2
            out_theta, out_col = torch.autograd.grad(head, (theta_req, col_req), retain_graph=True)
        if has_omega:
            # Full [theta; alpha; omega] contract (BASE order, alpha BEFORE omega -- matches
            # origination_curvature.py's z=[theta;alpha;omega]): the omega row is the head omega-Hessian .
            # (t_root, dE, u_omega) from _head_seed_tangents (omega is head-only -> no kernel/adjoint
            # term). Hv_omega is [G,S] genewise / [S] otherwise; flatten in place.
            return torch.cat([out_theta.reshape(-1), out_col.reshape(-1), Hv_omega.reshape(-1)])
        return torch.cat([out_theta.reshape(-1), out_col.reshape(-1)])

    return hvp

